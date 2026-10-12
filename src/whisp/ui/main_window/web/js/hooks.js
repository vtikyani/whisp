import { useCallback, useEffect, useRef, useState } from "preact/hooks";
import { on } from "./bridge.js";

/** Subscribe to a bridge event for the component's lifetime. */
export function useEvent(name, handler) {
  const latest = useRef(handler);
  latest.current = handler;
  useEffect(() => on(name, (payload) => latest.current(payload)), [name]);
}

/** Load async data; stale responses are ignored, so the newest request always wins. */
export function useResource(fetcher, deps = []) {
  const [state, setState] = useState({ data: undefined, error: null, loading: true });
  const token = useRef(0);
  const load = useCallback(async () => {
    const id = ++token.current;
    setState((previous) => ({ ...previous, loading: true }));
    try {
      const data = await fetcher();
      if (id === token.current) setState({ data, error: null, loading: false });
    } catch (error) {
      if (id === token.current) setState((previous) => ({ data: previous.data, error, loading: false }));
    }
  }, deps);
  useEffect(() => {
    load();
  }, [load]);
  useEffect(() => () => {
    token.current += 1;
  }, []);
  return { ...state, reload: load };
}

export function useDebounced(value, delay) {
  const [debounced, setDebounced] = useState(value);
  useEffect(() => {
    const timer = setTimeout(() => setDebounced(value), delay);
    return () => clearTimeout(timer);
  }, [value, delay]);
  return debounced;
}

/** Calls fn at most once per `delay` ms, always ending with the latest call. */
export function useThrottledCallback(fn, delay) {
  const latest = useRef(fn);
  latest.current = fn;
  const timer = useRef(null);
  const pending = useRef(false);
  useEffect(() => () => clearTimeout(timer.current), []);
  return useCallback(() => {
    if (timer.current) {
      pending.current = true;
      return;
    }
    latest.current();
    timer.current = setTimeout(function tick() {
      if (pending.current) {
        pending.current = false;
        latest.current();
        timer.current = setTimeout(tick, delay);
      } else {
        timer.current = null;
      }
    }, delay);
  }, [delay]);
}
