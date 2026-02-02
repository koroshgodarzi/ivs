# graph/monitoring.py
import json, time, logging
from copy import deepcopy
from typing import Any, Callable, Dict

LOG_KEYS = [
    "generated_query",
    "query_results",
    "error_message",
    "summary_context",
    "retry_count",
    "validation_result",
    "retrieved_schema",
    "messages",
]

def _truncate(s: Any, n: int = 400) -> Any:
    if isinstance(s, str) and len(s) > n:
        return s[:n] + "…"
    return s

def _state_view(state: Dict[str, Any]) -> Dict[str, Any]:
    """Make a compact, JSON-serializable view of GraphState."""
    view: Dict[str, Any] = {}

    # messages: only keep last user + last assistant (or last 2)
    msgs = state.get("messages") or []
    if isinstance(msgs, list) and msgs:
        view["messages_tail"] = [
            {"role": m.get("role"), "content": _truncate(m.get("content", ""), 300)}
            for m in msgs[-2:]
        ]
        view["messages_len"] = len(msgs)

    # generated_query: only last query + count
    gq = state.get("generated_query") or []
    if isinstance(gq, list) and gq:
        view["generated_query_last"] = _truncate(gq[-1], 500)
        view["generated_query_count"] = len(gq)

    # error_message: only last error + count
    em = state.get("error_message") or []
    if isinstance(em, list) and em:
        view["error_last"] = _truncate(em[-1], 500)
        view["error_count"] = len(em)

    # query_results: truncate
    if state.get("query_results") is not None:
        view["query_results"] = _truncate(state.get("query_results"), 500)

    # other small fields
    for k in ["retry_count", "validation_result", "retrieved_schema", "summary_context"]:
        if k in state:
            view[k] = _truncate(state.get(k), 700)

    return view

def _diff(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k in set(before) | set(after):
        if before.get(k) != after.get(k):
            out[k] = {"before": before.get(k), "after": after.get(k)}
    return out

def configure_jsonl_logger(path: str = "workflow_state.jsonl") -> logging.Logger:
    logger = logging.getLogger("workflow")
    logger.setLevel(logging.INFO)
    if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        fh = logging.FileHandler(path, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(fh)
    return logger

def with_state_logging(node_name: str, fn: Callable[..., Any], logger: logging.Logger):
    def wrapped(state: Dict[str, Any], *args, **kwargs):
        start = time.time()
        before_full = deepcopy(state)
        before_view = _state_view(before_full)

        try:
            result = fn(state, *args, **kwargs)
            # LangGraph nodes often mutate and return state; handle either
            final_state = result if isinstance(result, dict) else state
            after_view = _state_view(final_state)

            logger.info(json.dumps({
                "ts": time.time(),
                "node": node_name,
                "ok": True,
                "ms": int((time.time() - start) * 1000),
                "retry_count": final_state.get("retry_count"),
                "diff": _diff(before_view, after_view),
                "after": after_view,  # remove if too noisy
            }, ensure_ascii=False))
            return result

        except Exception as e:
            logger.info(json.dumps({
                "ts": time.time(),
                "node": node_name,
                "ok": False,
                "ms": int((time.time() - start) * 1000),
                "error": repr(e),
                "before": before_view,
            }, ensure_ascii=False))
            raise
    return wrapped
