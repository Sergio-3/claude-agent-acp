// Stays the first import: importing the test module re-collects its suites here,
// and they only pass with the `vi.mock` it declares for the agent SDK — that mock
// has to register before anything else pulls in `../acp-agent.js`. The SDK import
// below sits between the two so the guard test sees the real module if the order
// ever breaks.
import { mockSessionState, userEcho, wrapQuery } from "./acp-agent.test.js";
import { deleteSession } from "@anthropic-ai/claude-agent-sdk";
import { describe, it, expect, vi } from "vitest";
import { SessionNotification } from "@agentclientprotocol/sdk";
import { ClaudeAcpAgent, getAvailableSlashCommands, type AcpClient } from "../acp-agent.js";
import { Pushable } from "../utils.js";
import { randomUUID } from "crypto";

describe("module graph wiring", () => {
  it("resolves the agent SDK to acp-agent.test.ts's mock", () => {
    expect(
      vi.isMockFunction(deleteSession),
      "./acp-agent.test.js must stay the first import of this file: its vi.mock for " +
        "@anthropic-ai/claude-agent-sdk has to register before ../acp-agent.js pulls the SDK in. " +
        "With the imports reordered, the suites re-collected from acp-agent.test.ts run against " +
        "the real SDK and fail with unrelated-looking errors.",
    ).toBe(true);
  });
});

// `injectSession` below intentionally deviates from the upstream PR, which used a
// plain `function* empty() {}`. Since `ensureConsumer` drains the query stream, an
// immediately-returning generator ends the stream right away and the following
// `agent.prompt()` fails with SESSION_ENDED_MESSAGE — keep the pending generator.

describe("/remote-control command", () => {
  function createAgentWithCapture() {
    const updates: string[] = [];
    const mockClient = {
      sessionUpdate: async (params: SessionNotification) => {
        const u = params.update;
        if (u.sessionUpdate === "agent_message_chunk" && u.content.type === "text") {
          updates.push(u.content.text);
        }
      },
    } as unknown as AcpClient;
    const agent = new ClaudeAcpAgent(mockClient, { log: () => {}, error: () => {} });
    return { agent, updates };
  }

  function injectSession(agent: ClaudeAcpAgent, sessionId: string, enableRemoteControl: any) {
    // Stays open like a real query stream: `/remote-control` starts the consumer
    // (so the bridge's turns are drained), and a generator that returned would
    // end the session the moment the command ran.
    // eslint-disable-next-line require-yield -- an open stream yields nothing; it just stays pending
    async function* pending() {
      await new Promise(() => {});
    }
    const gen = Object.assign(pending(), {
      interrupt: vi.fn(),
      close: vi.fn(),
      supportedCommands: vi.fn().mockResolvedValue([]),
      enableRemoteControl,
    });
    agent.sessions[sessionId] = mockSessionState({
      query: gen as any,
      input: new Pushable(),
    });
    return agent.sessions[sessionId]!;
  }

  it("connects, surfaces the session URL, then disconnects on a second invocation", async () => {
    const { agent, updates } = createAgentWithCapture();
    const enableRemoteControl = vi
      .fn()
      .mockResolvedValueOnce({ session_url: "https://claude.ai/code/session_abc" })
      .mockResolvedValueOnce(undefined);
    injectSession(agent, "s1", enableRemoteControl);

    const r1 = await agent.prompt({
      sessionId: "s1",
      prompt: [{ type: "text", text: "/remote-control my-session" }],
    });
    expect(r1.stopReason).toBe("end_turn");
    expect(enableRemoteControl).toHaveBeenNthCalledWith(1, true, "my-session");
    expect(updates.join("\n")).toContain("https://claude.ai/code/session_abc");
    expect(agent.sessions["s1"]!.remoteControlActive).toBe(true);

    const r2 = await agent.prompt({
      sessionId: "s1",
      prompt: [{ type: "text", text: "/rc" }],
    });
    expect(r2.stopReason).toBe("end_turn");
    expect(enableRemoteControl).toHaveBeenNthCalledWith(2, false, undefined);
    expect(updates.join("\n")).toContain("Remote Control disconnected");
    expect(agent.sessions["s1"]!.remoteControlActive).toBe(false);
  });

  it("reports an error when the bridge fails", async () => {
    const { agent, updates } = createAgentWithCapture();
    const enableRemoteControl = vi.fn().mockRejectedValue(new Error("Remote Control is disabled"));
    injectSession(agent, "s1", enableRemoteControl);

    const r = await agent.prompt({
      sessionId: "s1",
      prompt: [{ type: "text", text: "/remote-control" }],
    });
    expect(r.stopReason).toBe("end_turn");
    expect(updates.join("\n")).toContain("Remote Control failed: Remote Control is disabled");
    expect(agent.sessions["s1"]!.remoteControlActive).toBe(false);
  });

  it("advertises remote-control and rc so clients forward them", () => {
    const names = getAvailableSlashCommands([]).map((c) => c.name);
    expect(names).toContain("remote-control");
    expect(names).toContain("rc");
  });

  it("does not duplicate remote-control when the SDK already reports it", () => {
    const names = getAvailableSlashCommands([
      { name: "remote-control", description: "from sdk" } as any,
    ]).map((c) => c.name);
    expect(names.filter((n) => n === "remote-control")).toHaveLength(1);
  });

  it("explains when the SDK build lacks Remote Control support", async () => {
    const { agent, updates } = createAgentWithCapture();
    injectSession(agent, "s1", undefined);

    const r = await agent.prompt({
      sessionId: "s1",
      prompt: [{ type: "text", text: "/rc" }],
    });
    expect(r.stopReason).toBe("end_turn");
    expect(updates.join("\n")).toContain("Remote Control isn't available");
  });
});

/** A user message from the other client: a uuid we never pushed, human origin. */
function remoteMessage(text: string, extra: Record<string, any> = {}) {
  return {
    type: "user",
    message: { role: "user", content: text },
    parent_tool_use_id: null,
    uuid: randomUUID(),
    origin: { kind: "human" },
    ...extra,
  };
}

describe("mirroring input from a Remote Control client", () => {
  /** messageIds of the mirrored chunks from the most recent `mirroredTexts` run,
   *  so the grouping assertion can read them without changing its return type. */
  let mirroredIds: (string | null | undefined)[] = [];

  /** Drives one turn whose stream carries our own prompt echo followed by
   *  `foreign` — a user message this adapter never pushed, standing in for text
   *  typed on claude.ai/code or the mobile app. Returns the user-visible text of
   *  every `user_message_chunk` the client received. */
  async function mirroredTexts(foreign: Record<string, any>, sessionOverrides: object = {}) {
    const chunks: string[] = [];
    mirroredIds = [];
    const agent = new ClaudeAcpAgent(
      {
        sessionUpdate: async (notification: SessionNotification) => {
          const update = notification.update;
          if (update.sessionUpdate === "user_message_chunk" && update.content.type === "text") {
            chunks.push(update.content.text);
            mirroredIds.push(update.messageId);
          }
        },
      } as unknown as AcpClient,
      { log: () => {}, error: () => {} },
    );
    const input = new Pushable<any>();
    const generator = (async function* () {
      const iter = input[Symbol.asyncIterator]();
      const { value: pushed } = await iter.next();
      yield userEcho(pushed);
      yield { session_id: "test-session", ...foreign };
      yield {
        type: "result",
        subtype: "success",
        stop_reason: null,
        is_error: false,
        result: "",
        errors: [],
        duration_ms: 0,
        duration_api_ms: 0,
        num_turns: 1,
        total_cost_usd: 0,
        usage: {
          input_tokens: 1,
          output_tokens: 1,
          cache_read_input_tokens: 0,
          cache_creation_input_tokens: 0,
        },
        modelUsage: {},
        permission_denials: [],
        uuid: randomUUID(),
        session_id: "test-session",
      };
      yield { type: "system", subtype: "session_state_changed", state: "idle" };
    })();
    agent.sessions["test-session"] = mockSessionState({
      query: wrapQuery(generator),
      input,
      remoteControlActive: true,
      ...sessionOverrides,
    });

    await agent.prompt({
      sessionId: "test-session",
      prompt: [{ type: "text", text: "typed locally in the editor" }],
    });
    return chunks;
  }

  it("surfaces text typed on the remote client so the local feed stays complete", async () => {
    expect(await mirroredTexts(remoteMessage("sent from my phone"))).toEqual([
      "sent from my phone",
    ]);
  });

  it("surfaces it even when the bridge marks the message as a replay", async () => {
    // Whether remote input arrives flagged as a replay is the CLI's choice; the
    // uuid we never pushed is what identifies it either way.
    expect(await mirroredTexts(remoteMessage("from the train", { isReplay: true }))).toEqual([
      "from the train",
    ]);
  });

  it("accepts a single text block as well as string content", async () => {
    const message = remoteMessage("");
    message.message = { role: "user", content: [{ type: "text", text: "block form" }] } as any;
    expect(await mirroredTexts(message)).toEqual(["block form"]);
  });

  it("never echoes back our own prompt, which the client already rendered", async () => {
    // The local prompt's echo carries a uuid this adapter pushed, so it must stay
    // out of the feed even while the bridge is attached.
    const texts = await mirroredTexts(remoteMessage("sent from my phone"));
    expect(texts).not.toContain("typed locally in the editor");
  });

  it("stays silent while no Remote Control bridge is attached", async () => {
    // The guard that keeps session/load's history replay from being emitted twice.
    expect(
      await mirroredTexts(remoteMessage("sent from my phone"), { remoteControlActive: false }),
    ).toEqual([]);
  });

  it("ignores synthetic messages and non-human origins", async () => {
    expect(await mirroredTexts(remoteMessage("synthetic", { isSynthetic: true }))).toEqual([]);
    expect(
      await mirroredTexts(remoteMessage("notification", { origin: { kind: "task-notification" } })),
    ).toEqual([]);
  });

  it("ignores whitespace-only remote input", async () => {
    expect(await mirroredTexts(remoteMessage("   \n  "))).toEqual([]);
  });

  it("carries the remote message's own id so it renders as a separate message", async () => {
    // Without an id of its own the chunk is appended to the local user's most
    // recent message — always the wrong bubble, since the CLI only hands us a
    // Remote Control turn once a LATER local turn writes to the input.
    const message = remoteMessage("sent from my phone");
    await mirroredTexts(message);
    expect(mirroredIds).toEqual([message.uuid]);
  });
});

describe("a remote turn arriving under the local cancel latch", () => {
  /** Captures both halves of a mirrored turn: the remote user's text and the
   *  model's reply to it. */
  function capturingAgent() {
    const userChunks: string[] = [];
    const agentChunks: string[] = [];
    const agent = new ClaudeAcpAgent(
      {
        sessionUpdate: async (notification: SessionNotification) => {
          const update = notification.update;
          if (update.sessionUpdate === "user_message_chunk" && update.content.type === "text") {
            userChunks.push(update.content.text);
          }
          if (update.sessionUpdate === "agent_message_chunk" && update.content.type === "text") {
            agentChunks.push(update.content.text);
          }
        },
      } as unknown as AcpClient,
      { log: () => {}, error: () => {} },
    );
    return { agent, userChunks, agentChunks };
  }

  const assistantAnswer = (text: string) => ({
    type: "assistant",
    parent_tool_use_id: null,
    uuid: randomUUID(),
    session_id: "test-session",
    message: {
      id: "msg_answer",
      role: "assistant",
      model: "claude-sonnet-4-5",
      content: [{ type: "text", text }],
      usage: {
        input_tokens: 1,
        output_tokens: 1,
        cache_read_input_tokens: 0,
        cache_creation_input_tokens: 0,
      },
    },
  });

  const resultFrame = (overrides: Record<string, any> = {}) => ({
    type: "result",
    subtype: "success",
    stop_reason: null,
    is_error: false,
    result: "",
    errors: [],
    duration_ms: 0,
    duration_api_ms: 0,
    num_turns: 1,
    total_cost_usd: 0,
    usage: {
      input_tokens: 1,
      output_tokens: 1,
      cache_read_input_tokens: 0,
      cache_creation_input_tokens: 0,
    },
    modelUsage: {},
    permission_denials: [],
    uuid: randomUUID(),
    session_id: "test-session",
    ...overrides,
  });

  /** The issue-#453 shape: no streamed deltas and zero output tokens, so the
   *  answer reaches the client only as the result's fallback text. */
  const replayedResult = (text: string) =>
    resultFrame({
      result: text,
      usage: {
        input_tokens: 10,
        output_tokens: 0,
        cache_read_input_tokens: 0,
        cache_creation_input_tokens: 0,
      },
    });

  const idle = { type: "system", subtype: "session_state_changed", state: "idle" };
  const running = { type: "system", subtype: "session_state_changed", state: "running" };

  /** A background Task/Agent-tool subagent starting: its liveness is what makes
   *  the turn's result defer (hold the turn open) instead of settle it. */
  const subagentStarted = (taskId: string) => ({
    type: "system",
    subtype: "task_started",
    task_id: taskId,
    tool_use_id: `toolu_${taskId}`,
    description: "Explore the project",
    subagent_type: "Explore",
    uuid: randomUUID(),
    session_id: "test-session",
  });

  const taskNotification = (taskId: string) => ({
    type: "system",
    subtype: "task_notification",
    task_id: taskId,
    tool_use_id: `toolu_${taskId}`,
    status: "completed",
    output_file: "",
    summary: "done",
    uuid: randomUUID(),
    session_id: "test-session",
  });

  /** A `command_lifecycle` frame (CLIs 2.1.206+) reporting `state` for a
   *  uuid-stamped command. */
  const lifecycleFrame = (commandUuid: string, state: string) => ({
    type: "command_lifecycle",
    command_uuid: commandUuid,
    state,
    uuid: randomUUID(),
    session_id: "test-session",
  });

  // Poll across timer turns, so the test can wait for the consumer to reach a
  // state (a turn became active) without coupling to its scheduling.
  const waitFor = async (cond: () => boolean) => {
    for (let i = 0; i < 200; i++) {
      if (cond()) return;
      await new Promise((r) => setTimeout(r, 0));
    }
    throw new Error("waitFor timed out");
  };

  it("mirrors the remote turn although a cancel is latched", async () => {
    // The defect this pins: the cancel latch used to drop every user/assistant
    // frame, and a Remote Control turn always reaches the consumer under it —
    // the CLI holds the turn back until we next write to the input, and clients
    // cancel before every prompt (Zed's run_turn does). The user was left with
    // the turn's permission dialogs and tool calls (which take other paths) and
    // none of the prose around them.
    const { agent, userChunks, agentChunks } = capturingAgent();
    const remote = remoteMessage("sent from my phone");
    const input = new Pushable<any>();
    const generator = (async function* () {
      const iter = input[Symbol.asyncIterator]();
      const { value: pushed } = await iter.next();
      // Flushed by that write, ahead of our own echo.
      yield { session_id: "test-session", ...remote };
      yield assistantAnswer("answered on the phone");
      yield userEcho(pushed); // activates the local turn, clearing the latch
      yield resultFrame();
      yield idle;
    })();
    agent.sessions["test-session"] = mockSessionState({
      query: wrapQuery(generator),
      input,
      remoteControlActive: true,
    });

    await agent.cancel({ sessionId: "test-session" });
    expect(agent.sessions["test-session"]!.cancelled).toBe(true);

    await agent.prompt({
      sessionId: "test-session",
      prompt: [{ type: "text", text: "typed locally in the editor" }],
    });

    expect(userChunks).toEqual(["sent from my phone"]);
    expect(agentChunks).toEqual(["answered on the phone"]);
  });

  it("still drops the stragglers of the cancelled local turn", async () => {
    // The other side of the same guard: a frame that belongs to the local turn
    // the user just cancelled must stay out of the feed — with a bridge
    // attached as much as without one.
    const { agent, agentChunks } = capturingAgent();
    let release!: () => void;
    const cancelled = new Promise<void>((resolve) => {
      release = resolve;
    });
    const input = new Pushable<any>();
    const generator = (async function* () {
      const iter = input[Symbol.asyncIterator]();
      const { value: pushed } = await iter.next();
      yield userEcho(pushed); // activates the local turn
      await cancelled;
      yield assistantAnswer("straggler of the cancelled turn");
      yield idle; // settles the turn "cancelled"
    })();
    agent.sessions["test-session"] = mockSessionState({
      query: wrapQuery(generator),
      input,
      remoteControlActive: true,
    });

    const turn = agent.prompt({
      sessionId: "test-session",
      prompt: [{ type: "text", text: "something long" }],
    });
    await waitFor(() => !!agent.sessions["test-session"]!.activeTurn);
    await agent.cancel({ sessionId: "test-session" });
    release();

    expect((await turn).stopReason).toBe("cancelled");
    expect(agentChunks).toEqual([]);
  });

  it("keeps the answer of a turn cancelled while queued off the feed", async () => {
    // A cancelled QUEUED turn is settled and dropped from the queue right away,
    // so `activeTurn` is null while the SDK — which the interrupt left it
    // running — streams its answer. That answer belongs to a prompt the client
    // was already told "cancelled" and must not reach it just because a bridge
    // is attached.
    const { agent, agentChunks } = capturingAgent();
    let release!: () => void;
    const cancelled = new Promise<void>((resolve) => {
      release = resolve;
    });
    const input = new Pushable<any>();
    const generator = (async function* () {
      const iter = input[Symbol.asyncIterator]();
      const { value: dead } = await iter.next();
      await cancelled; // the client cancels before the SDK picked the turn up
      const { value: live } = await iter.next();
      // The interrupt reported the first turn as still queued, so the SDK runs
      // it anyway and its cycle arrives ahead of the live one.
      yield userEcho(dead);
      yield assistantAnswer("answer to the cancelled prompt");
      yield resultFrame();
      yield idle;
      yield userEcho(live);
      yield assistantAnswer("answer to the live prompt");
      yield resultFrame();
      yield idle;
    })();
    agent.sessions["test-session"] = mockSessionState({
      query: wrapQuery(generator),
      input,
      remoteControlActive: true,
    });

    const dead = agent.prompt({
      sessionId: "test-session",
      prompt: [{ type: "text", text: "first" }],
    });
    await agent.cancel({ sessionId: "test-session" });
    await expect(dead).resolves.toMatchObject({ stopReason: "cancelled" });
    const live = agent.prompt({
      sessionId: "test-session",
      prompt: [{ type: "text", text: "second" }],
    });
    release();

    await expect(live).resolves.toMatchObject({ stopReason: "end_turn" });
    expect(agentChunks).toEqual(["answer to the live prompt"]);
  });

  it("keeps it off the feed even after a result cleared the orphan bookkeeping", async () => {
    // The orphan lanes cannot stand in for "a dead turn may still speak": on a
    // msg_lifecycle_v1 CLI, `recordResultForOrphanCommands` drops every
    // `started` entry at ANY result while a turn is active — including the
    // cancelled active turn's own result, which runs before the cancelled
    // break. So the dispatched queued turn's entry is gone by the time its
    // answer arrives, and only a record of the cancel itself still says it.
    const { agent, agentChunks } = capturingAgent();
    let release!: () => void;
    const cancelled = new Promise<void>((resolve) => {
      release = resolve;
    });
    const input = new Pushable<any>();
    const generator = (async function* () {
      const iter = input[Symbol.asyncIterator]();
      const { value: active } = await iter.next();
      yield userEcho(active); // turn A is the active one
      const { value: dispatched } = await iter.next();
      yield lifecycleFrame(dispatched.uuid, "started"); // turn B: queued, running
      await cancelled;
      yield resultFrame(); // A's own result: deletes B's `started` entry
      yield idle; // settles A "cancelled" — activeTurn is null from here
      yield assistantAnswer("answer to the cancelled prompt"); // B's answer
      yield resultFrame();
      yield idle;
    })();
    agent.sessions["test-session"] = mockSessionState({
      query: wrapQuery(generator),
      input,
      remoteControlActive: true,
      msgLifecycleV1: true,
    });

    const active = agent.prompt({
      sessionId: "test-session",
      prompt: [{ type: "text", text: "first" }],
    });
    const dispatched = agent.prompt({
      sessionId: "test-session",
      prompt: [{ type: "text", text: "second" }],
    });
    await waitFor(() =>
      (agent.sessions["test-session"]!.turnQueue ?? []).some((turn: any) => turn.commandStarted),
    );
    await agent.cancel({ sessionId: "test-session" });
    release();

    await expect(active).resolves.toMatchObject({ stopReason: "cancelled" });
    await expect(dispatched).resolves.toMatchObject({ stopReason: "cancelled" });
    expect(agentChunks).toEqual([]);
  });

  it("keeps the in-flight cycle of an inline-settled held turn off the feed", async () => {
    // cancel() settles a turn held open for its background subagents inline —
    // `activeTurn` null, no orphan entry seeded anywhere — and the latch stays
    // set because the session is not quiescent. The interrupted cycle's prose
    // belongs to a prompt already answered "cancelled".
    const { agent, agentChunks } = capturingAgent();
    let release!: () => void;
    const afterCancel = new Promise<void>((resolve) => {
      release = resolve;
    });
    const input = new Pushable<any>();
    const generator = (async function* () {
      const iter = input[Symbol.asyncIterator]();
      const { value: pushed } = await iter.next();
      yield userEcho(pushed);
      yield running;
      yield subagentStarted("agent-1");
      yield resultFrame(); // the turn defers instead of settling
      yield idle;
      yield taskNotification("agent-1"); // the model wakes for the summary
      yield running;
      await afterCancel;
      yield assistantAnswer("summary of the interrupted cycle");
      yield idle;
    })();
    agent.sessions["test-session"] = mockSessionState({
      query: wrapQuery(generator),
      input,
      remoteControlActive: true,
    });

    const turn = agent.prompt({
      sessionId: "test-session",
      prompt: [{ type: "text", text: "explore" }],
    });
    await waitFor(
      () =>
        !!agent.sessions["test-session"]!.activeTurn?.deferredSettle &&
        agent.sessions["test-session"]!.lastSessionState === "running",
    );
    await agent.cancel({ sessionId: "test-session" });
    release();

    await expect(turn).resolves.toMatchObject({ stopReason: "cancelled" });
    expect(agent.sessions["test-session"]!.cancelled).toBe(true);
    expect(agentChunks).toEqual([]);
  });

  it("does not let mirrored prose consume the local turn's result fallback", async () => {
    // The mirrored answer is another client's, so it must not count as this
    // session's delivered answer: a local turn that streams nothing and reports
    // zero output tokens (cache replay, non-streaming backend) shows its answer
    // only through the issue-#453 result-text fallback, which a stale delivery
    // record would suppress.
    const { agent, agentChunks } = capturingAgent();
    const input = new Pushable<any>();
    const generator = (async function* () {
      const iter = input[Symbol.asyncIterator]();
      const { value: pushed } = await iter.next();
      yield { session_id: "test-session", ...remoteMessage("sent from my phone") };
      yield assistantAnswer("answered on the phone");
      yield userEcho(pushed);
      yield replayedResult("the local turn's only answer");
      yield idle;
    })();
    agent.sessions["test-session"] = mockSessionState({
      query: wrapQuery(generator),
      input,
      remoteControlActive: true,
    });

    await agent.cancel({ sessionId: "test-session" });
    await agent.prompt({
      sessionId: "test-session",
      prompt: [{ type: "text", text: "typed locally in the editor" }],
    });

    expect(agentChunks).toEqual(["answered on the phone", "the local turn's only answer"]);
  });

  it("keeps the record across the client's next pre-prompt cancel", async () => {
    // The record must be set-only, never recomputed: by the time the client
    // sends its usual cancel before the NEXT prompt, the dead turn has settled
    // and the queue is empty, so a recomputing cancel would read "nothing
    // outstanding" and clear it — while the turn it was recorded for is still
    // streaming its answer. That is the ordinary Zed sequence (cancel, then
    // cancel again before the follow-up), not an exotic one.
    const { agent, agentChunks } = capturingAgent();
    let release!: () => void;
    const cancelled = new Promise<void>((resolve) => {
      release = resolve;
    });
    const input = new Pushable<any>();
    const generator = (async function* () {
      const iter = input[Symbol.asyncIterator]();
      const { value: dead } = await iter.next();
      yield userEcho(dead);
      await cancelled;
      yield idle; // settles the cancelled turn: the queue is empty from here
      const { value: live } = await iter.next();
      yield assistantAnswer("answer of the turn the user cancelled");
      yield userEcho(live);
      yield assistantAnswer("answer of the live turn");
      yield resultFrame();
      yield idle;
    })();
    agent.sessions["test-session"] = mockSessionState({
      query: wrapQuery(generator),
      input,
      remoteControlActive: true,
    });

    const dead = agent.prompt({
      sessionId: "test-session",
      prompt: [{ type: "text", text: "first" }],
    });
    await waitFor(() => !!agent.sessions["test-session"]!.activeTurn);
    await agent.cancel({ sessionId: "test-session" });
    release();
    await expect(dead).resolves.toMatchObject({ stopReason: "cancelled" });

    // The client's pre-prompt cancel, on a session that only looks quiet.
    await agent.cancel({ sessionId: "test-session" });
    await agent.prompt({ sessionId: "test-session", prompt: [{ type: "text", text: "second" }] });

    expect(agentChunks).toEqual(["answer of the live turn"]);
  });

  it("mirrors after a bridge is enabled while an earlier cancel is still latched", async () => {
    // `/remote-control` answers before the turn machinery, so it creates no
    // turn: without a clear of its own, the record of the cancel that preceded
    // it would stand for the rest of the session and drop every mirrored frame
    // — in the feature's headline sequence (stop what the editor was doing,
    // switch the bridge on, continue on the phone).
    const { agent, userChunks, agentChunks } = capturingAgent();
    let releaseIdle!: () => void;
    let releaseRemote!: () => void;
    const cancelled = new Promise<void>((resolve) => {
      releaseIdle = resolve;
    });
    const bridged = new Promise<void>((resolve) => {
      releaseRemote = resolve;
    });
    const input = new Pushable<any>();
    const generator = (async function* () {
      const iter = input[Symbol.asyncIterator]();
      const { value: pushed } = await iter.next();
      yield userEcho(pushed);
      await cancelled;
      yield idle; // settles the cancelled turn
      await bridged;
      yield { session_id: "test-session", ...remoteMessage("sent from my phone") };
      yield assistantAnswer("answered on the phone");
    })();
    agent.sessions["test-session"] = mockSessionState({
      query: Object.assign(wrapQuery(generator), {
        enableRemoteControl: vi.fn().mockResolvedValue({ session_url: "https://claude.ai/code/x" }),
      }),
      input,
      remoteControlActive: false,
    });

    const turn = agent.prompt({
      sessionId: "test-session",
      prompt: [{ type: "text", text: "run something long" }],
    });
    await waitFor(() => !!agent.sessions["test-session"]!.activeTurn);
    await agent.cancel({ sessionId: "test-session" });
    releaseIdle();
    await expect(turn).resolves.toMatchObject({ stopReason: "cancelled" });
    expect(agent.sessions["test-session"]!.cancelledTurnMayStillEmit).toBe(true);

    await agent.prompt({ sessionId: "test-session", prompt: [{ type: "text", text: "/rc" }] });
    expect(agent.sessions["test-session"]!.remoteControlActive).toBe(true);
    releaseRemote();
    await agent.sessions["test-session"]!.consumer;

    // The latch is still set — no turn has run since the cancel — so these
    // frames reach the client only because enabling the bridge lifted the
    // record. (agentChunks also carries the command's own confirmation text.)
    expect(agent.sessions["test-session"]!.cancelled).toBe(true);
    expect(userChunks).toEqual(["sent from my phone"]);
    expect(agentChunks.at(-1)).toBe("answered on the phone");
  });

  it("mirrors again once a local turn has run since the cancel", async () => {
    // The record of a cancel lives exactly as long as the latch: a turn that
    // activates afterwards is the one output is attributed to, so nothing is
    // left over from the old cancel. Without that clear the first mid-turn
    // cancel of a session would silence the mirror for good.
    const { agent, userChunks, agentChunks } = capturingAgent();
    let releaseSecond!: () => void;
    const afterFirst = new Promise<void>((resolve) => {
      releaseSecond = resolve;
    });
    const input = new Pushable<any>();
    const generator = (async function* () {
      const iter = input[Symbol.asyncIterator]();
      const { value: first } = await iter.next();
      yield userEcho(first);
      await afterFirst;
      yield idle; // settles the cancelled turn
      const { value: second } = await iter.next();
      yield userEcho(second); // a live turn again: the record is cleared here
      yield resultFrame();
      yield idle;
      const { value: third } = await iter.next();
      yield { session_id: "test-session", ...remoteMessage("sent from my phone") };
      yield assistantAnswer("answered on the phone");
      yield userEcho(third);
      yield resultFrame();
      yield idle;
    })();
    agent.sessions["test-session"] = mockSessionState({
      query: wrapQuery(generator),
      input,
      remoteControlActive: true,
    });

    const first = agent.prompt({
      sessionId: "test-session",
      prompt: [{ type: "text", text: "a" }],
    });
    await waitFor(() => !!agent.sessions["test-session"]!.activeTurn);
    await agent.cancel({ sessionId: "test-session" });
    releaseSecond();
    await expect(first).resolves.toMatchObject({ stopReason: "cancelled" });

    await agent.prompt({ sessionId: "test-session", prompt: [{ type: "text", text: "b" }] });

    // The client's usual pre-prompt cancel, now on a session with nothing left.
    await agent.cancel({ sessionId: "test-session" });
    await agent.prompt({ sessionId: "test-session", prompt: [{ type: "text", text: "c" }] });

    expect(userChunks).toEqual(["sent from my phone"]);
    expect(agentChunks).toEqual(["answered on the phone"]);
  });

  it("mirrors after a cancel that left a hold's subagents running", async () => {
    // cancel() skips the interrupt when a held turn sat on a quiet SDK, so its
    // subagents survive, and lifts the latch because nothing is left over. The
    // record of the cancel has to lift with it — otherwise the mirror stays
    // silent for the rest of the session after any follow-up sent during a
    // subagent run.
    const { agent, userChunks, agentChunks } = capturingAgent();
    let releaseAfterCancel!: () => void;
    const afterCancel = new Promise<void>((resolve) => {
      releaseAfterCancel = resolve;
    });
    const input = new Pushable<any>();
    const generator = (async function* () {
      const iter = input[Symbol.asyncIterator]();
      const { value: held } = await iter.next();
      yield userEcho(held);
      yield running;
      yield subagentStarted("agent-1");
      yield resultFrame(); // the turn defers
      yield idle; // and the SDK goes quiet: the skip's precondition
      await afterCancel;
      yield taskNotification("agent-1"); // the surviving subagent reports
      yield resultFrame({ origin: { kind: "task-notification" } });
      yield idle;
      const { value: next } = await iter.next();
      yield { session_id: "test-session", ...remoteMessage("sent from my phone") };
      yield assistantAnswer("answered on the phone");
      yield userEcho(next);
      yield resultFrame();
      yield idle;
    })();
    agent.sessions["test-session"] = mockSessionState({
      query: wrapQuery(generator),
      input,
      remoteControlActive: true,
    });

    const held = agent.prompt({
      sessionId: "test-session",
      prompt: [{ type: "text", text: "explore" }],
    });
    await waitFor(
      () =>
        !!agent.sessions["test-session"]!.activeTurn?.deferredSettle &&
        agent.sessions["test-session"]!.lastSessionState === "idle",
    );
    await agent.cancel({ sessionId: "test-session" });
    await expect(held).resolves.toMatchObject({ stopReason: "cancelled" });
    expect(agent.sessions["test-session"]!.cancelled).toBe(false);
    expect(agent.sessions["test-session"]!.query.interrupt).not.toHaveBeenCalled();
    releaseAfterCancel();

    await agent.cancel({ sessionId: "test-session" });
    await agent.prompt({
      sessionId: "test-session",
      prompt: [{ type: "text", text: "typed locally in the editor" }],
    });

    expect(userChunks).toEqual(["sent from my phone"]);
    expect(agentChunks).toEqual(["answered on the phone"]);
  });

  it("leaves the latch untouched for a session with no bridge attached", async () => {
    // The exception is scoped to Remote Control: with no bridge there is no
    // other client whose turn could be mirrored, so every frame under the latch
    // is the local session's own and stays dropped, whatever the turn state.
    const { agent, userChunks, agentChunks } = capturingAgent();
    const input = new Pushable<any>();
    const generator = (async function* () {
      const iter = input[Symbol.asyncIterator]();
      const { value: pushed } = await iter.next();
      yield { session_id: "test-session", ...remoteMessage("sent from my phone") };
      yield assistantAnswer("answered on the phone");
      yield userEcho(pushed);
      yield resultFrame();
      yield idle;
    })();
    agent.sessions["test-session"] = mockSessionState({
      query: wrapQuery(generator),
      input,
      remoteControlActive: false,
    });

    await agent.cancel({ sessionId: "test-session" });
    await agent.prompt({
      sessionId: "test-session",
      prompt: [{ type: "text", text: "typed locally in the editor" }],
    });

    expect(userChunks).toEqual([]);
    expect(agentChunks).toEqual([]);
  });
});

describe("Remote Control starts the stream consumer", () => {
  /** `/remote-control` answers without going through the turn machinery, so it
   *  returns before `prompt()` would start the consumer. Enabling a bridge as the
   *  first thing in a session must still start it: the CLI puts the bridge's
   *  turns on the stream (the remote message as a replayed user message, its
   *  answer as normal assistant output), and with nothing draining the stream
   *  they sit in the SDK's buffer until an unrelated prompt happens to start it. */
  it("starts the consumer when a bridge is enabled before any prompt", async () => {
    const agent = new ClaudeAcpAgent(
      { sessionUpdate: vi.fn(async () => {}) } as unknown as AcpClient,
      { log: () => {}, error: () => {} },
    );
    // eslint-disable-next-line require-yield -- an open stream yields nothing; it just stays pending
    async function* pending() {
      await new Promise(() => {});
    }
    agent.sessions["s1"] = mockSessionState({
      query: Object.assign(pending(), {
        interrupt: vi.fn(),
        close: vi.fn(),
        supportedCommands: vi.fn().mockResolvedValue([]),
        enableRemoteControl: vi.fn().mockResolvedValue({ session_url: "https://claude.ai/code/x" }),
      }) as any,
      input: new Pushable(),
    });
    expect(agent.sessions["s1"]!.consumer).toBeUndefined();

    await agent.prompt({ sessionId: "s1", prompt: [{ type: "text", text: "/rc" }] });

    expect(agent.sessions["s1"]!.remoteControlActive).toBe(true);
    expect(agent.sessions["s1"]!.consumer).toBeDefined();
  });
});
