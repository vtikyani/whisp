// Talks to Python through the QWebChannel object "bridge" (see bridge.py).
//
//   await call("settings.get")          -> value, or throws BridgeError
//   const off = on("engine.status", fn) -> subscribe to pushed events
//
// Slow methods answer {"pending": id}; their result arrives later as a
// "call.result" event. Both shapes resolve the same Promise here.

export class BridgeError extends Error {
  constructor(message) {
    super(message || "Something went wrong.");
    this.name = "BridgeError";
  }
}

let channelObject = null;
const listeners = new Map();
const waiting = new Map();
const earlyResults = new Map();

export function connectBridge() {
  return new Promise((resolve, reject) => {
    const transport = globalThis.qt && globalThis.qt.webChannelTransport;
    if (typeof QWebChannel !== "function" || !transport) {
      reject(new Error("The window's connection to Whisp isn't available."));
      return;
    }
    new QWebChannel(transport, (channel) => {
      channelObject = channel.objects.bridge;
      channelObject.event.connect(dispatch);
      resolve();
    });
  });
}

function settle(entry, result) {
  if ("error" in result) entry.reject(new BridgeError(result.error));
  else entry.resolve(result.ok);
}

function dispatch(name, payloadJson) {
  let payload;
  try {
    payload = JSON.parse(payloadJson);
  } catch {
    console.error(`Unreadable "${name}" event`);
    return;
  }
  if (name === "call.result") {
    const entry = waiting.get(payload.id);
    if (entry) {
      waiting.delete(payload.id);
      settle(entry, payload);
    } else {
      earlyResults.set(payload.id, payload);
    }
    return;
  }
  const handlers = listeners.get(name);
  if (!handlers) return;
  for (const handler of [...handlers]) {
    try {
      handler(payload);
    } catch (error) {
      console.error(`Handler for "${name}" failed: ${error && error.message}`);
    }
  }
}

export function call(method, params = {}) {
  return new Promise((resolve, reject) => {
    if (!channelObject) {
      reject(new BridgeError("Whisp isn't connected yet."));
      return;
    }
    channelObject.call(method, JSON.stringify(params), (reply) => {
      let result;
      try {
        result = JSON.parse(reply);
      } catch {
        reject(new BridgeError());
        return;
      }
      if (result && "pending" in result) {
        const entry = { resolve, reject };
        const early = earlyResults.get(result.pending);
        if (early) {
          earlyResults.delete(result.pending);
          settle(entry, early);
        } else {
          waiting.set(result.pending, entry);
        }
        return;
      }
      settle({ resolve, reject }, result || { error: "" });
    });
  });
}

export function on(name, handler) {
  let handlers = listeners.get(name);
  if (!handlers) {
    handlers = new Set();
    listeners.set(name, handlers);
  }
  handlers.add(handler);
  return () => handlers.delete(handler);
}
