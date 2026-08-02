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
