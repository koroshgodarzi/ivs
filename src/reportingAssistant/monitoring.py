import time
import os
import logging
from copy import deepcopy
from typing import Any, Callable, Dict
from datetime import datetime
# --- IMPORT RUNNABLECONFIG ---
from langchain_core.runnables import RunnableConfig

def _truncate(s: Any, n: int = 400) -> Any:
    if isinstance(s, str) and len(s) > n:
        return s[:n] + "…"
    return s

def _state_view(state: Dict[str, Any]) -> Dict[str, Any]:
    view: Dict[str, Any] = {}
    
    msgs = state.get("messages") or []
    if isinstance(msgs, list) and msgs:
        view["messages_tail"] = [
            {"role": m.get("role") if isinstance(m, dict) else getattr(m, 'role', 'unknown'), 
             "content": _truncate(m.get("content", "") if isinstance(m, dict) else getattr(m, 'content', ''), 300)}
            for m in msgs
        ]
        view["messages_len"] = len(msgs)

    # ... (rest of your _state_view logic remains the same)
    gq = state.get("generated_query")
    if gq: view["generated_query"] = _truncate(gq, 500)
    
    return view

def _diff(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out = {}
    for k in set(before) | set(after):
        if before.get(k) != after.get(k):
            out[k] = {"before": before.get(k), "after": after.get(k)}
    return out

def configure_text_logger(path: str) -> logging.Logger:
    path = os.path.join('..', 'output', path, "workflow_state_LLM_based.log")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    logger = logging.getLogger("workflow")
    logger.setLevel(logging.INFO)
    if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        fh = logging.FileHandler(path, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(fh)
    logger._initial_state_logged = False 
    return logger

def _format_diff(diff: Dict[str, Dict[str, Any]]) -> str:
    if not diff: return "  (no state changes)"
    lines = []
    for key, change in diff.items():
        lines.append(f"  - {key}:\n      before: {change['before']}\n      after : {change['after']}")
    return "\n".join(lines)

# --- CORRECTED WRAPPER ---
def with_state_logging(
    node_name: str,
    fn: Callable[..., Any],
    logger: logging.Logger,
):
    # LangGraph needs to see the type hint 'RunnableConfig' to pass the config object
    def wrapped(state: Dict[str, Any], config: RunnableConfig):
        start = time.time()

        # Extract model name for logging purposes so you can see it in your file!
        model_name = config.get("configurable", {}).get("model_name", "unknown")

        if not getattr(logger, "_initial_state_logged", False):
            ts = datetime.utcnow().isoformat(timespec="seconds") + "Z"
            initial_view = _state_view(state)
            header = (
                f"[{ts}] WORKFLOW START | Model: {model_name}\n"
                f"Initial messages: {initial_view.get('messages_tail', [])}\n"
                f"{'-' * 80}"
            )
            logger.info(header)
            logger._initial_state_logged = True

        before_view = _state_view(deepcopy(state))

        try:
            # Pass BOTH state and config to the actual node function
            result = fn(state, config)
            
            final_state = result if isinstance(result, dict) else state
            after_view = _state_view(final_state)
            diff = _diff(before_view, after_view)
            ts = datetime.utcnow().isoformat(timespec="seconds") + "Z"
            elapsed_ms = int((time.time() - start) * 1000)

            log_lines = [
                f"[{ts}] NODE: {node_name} (Model: {model_name})",
                f"  status : OK",
                f"  elapsed: {elapsed_ms} ms",
                "  state changes:",
                _format_diff(diff),
                "-" * 80
            ]
            logger.info("\n".join(log_lines))
            return result

        except Exception as e:
            ts = datetime.utcnow().isoformat(timespec="seconds") + "Z"
            logger.info(
                f"[{ts}] NODE: {node_name}\n  status : ERROR\n  error : {repr(e)}\n{'-' * 80}"
            )
            raise

    return wrapped