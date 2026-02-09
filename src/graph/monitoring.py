# graph/monitoring.py
import time
import os
import logging
from copy import deepcopy
from typing import Any, Callable, Dict
from datetime import datetime


def _truncate(s: Any, n: int = 400) -> Any:
    if isinstance(s, str) and len(s) > n:
        return s[:n] + "…"
    return s


def _state_view(state: Dict[str, Any]) -> Dict[str, Any]:
    view: Dict[str, Any] = {}

    msgs = state.get("messages") or []
    if isinstance(msgs, list) and msgs:
        view["messages_tail"] = [
            {"role": m.get("role"), "content": _truncate(m.get("content", ""), 300)}
            for m in msgs
        ]
        view["messages_len"] = len(msgs)

    gq = state.get("generated_query") or []
    if isinstance(gq, list) and gq:
        view["generated_query_last"] = _truncate(gq[-1], 500)
        view["generated_query_count"] = len(gq)

    em = state.get("error_message") or []
    if isinstance(em, list) and em:
        view["error_last"] = _truncate(em[-1], 500)
        view["error_count"] = len(em)

    qe = state.get("query_explanation") or []
    if isinstance(qe, list) and qe:
        view["query_explanation_last"] = _truncate(qe[-1], 700)
        view["query_explanation_count"] = len(qe)

    if state.get("query_results") is not None:
        view["query_results"] = _truncate(state.get("query_results"), 500)

    for k in ["validation_result", "retrieved_schema", "summary_context"]:
        if k in state:
            view[k] = _truncate(state.get(k), 700)

    return view


def _diff(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out = {}
    for k in set(before) | set(after):
        if before.get(k) != after.get(k):
            out[k] = {
                "before": before.get(k),
                "after": after.get(k),
            }
    return out


def configure_text_logger(path: str = os.path.join('..', 'output', "workflow_state.log")) -> logging.Logger:
    logger = logging.getLogger("workflow")
    logger.setLevel(logging.INFO)

    if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        fh = logging.FileHandler(path, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(fh)

    # marker so we only write initial state once
    logger._initial_state_logged = False  # type: ignore[attr-defined]

    return logger


def _format_diff(diff: Dict[str, Dict[str, Any]]) -> str:
    if not diff:
        return "  (no state changes)"

    lines = []
    for key, change in diff.items():
        lines.append(f"  - {key}:")
        lines.append(f"      before: {change['before']}")
        lines.append(f"      after : {change['after']}")
    return "\n".join(lines)


def with_state_logging(
    node_name: str,
    fn: Callable[..., Any],
    logger: logging.Logger,
):
    def wrapped(state: Dict[str, Any], *args, **kwargs):
        start = time.time()

        # ---- INITIAL WORKFLOW HEADER (messages only) ----
        if not getattr(logger, "_initial_state_logged", False):
            ts = datetime.utcnow().isoformat(timespec="seconds") + "Z"
            initial_view = _state_view(state)

            header = (
                f"[{ts}] WORKFLOW START\n"
                f"Initial messages:\n"
                f"{initial_view.get('messages_tail', [])}\n"
                f"{'-' * 80}"
            )
            logger.info(header)
            logger._initial_state_logged = True  # type: ignore[attr-defined]

        before_view = _state_view(deepcopy(state))

        try:
            result = fn(state, *args, **kwargs)
            final_state = result if isinstance(result, dict) else state
            after_view = _state_view(final_state)

            diff = _diff(before_view, after_view)
            ts = datetime.utcnow().isoformat(timespec="seconds") + "Z"
            elapsed_ms = int((time.time() - start) * 1000)

            log_lines = [
                f"[{ts}] NODE: {node_name}",
                f"  status : OK",
                f"  elapsed: {elapsed_ms} ms",
            ]

            # retry_count only for error_handler
            if node_name == "error_handler":
                log_lines.append(
                    f"  retry_count: {final_state.get('retry_count')}"
                )

            log_lines.append("  state changes:")
            log_lines.append(_format_diff(diff))
            log_lines.append("-" * 80)

            logger.info("\n".join(log_lines))
            return result

        except Exception as e:
            ts = datetime.utcnow().isoformat(timespec="seconds") + "Z"
            elapsed_ms = int((time.time() - start) * 1000)

            log_entry = (
                f"[{ts}] NODE: {node_name}\n"
                f"  status : ERROR\n"
                f"  elapsed: {elapsed_ms} ms\n"
                f"  error  : {repr(e)}\n"
                f"  state before:\n"
                f"{before_view}\n"
                f"{'-' * 80}"
            )

            logger.info(log_entry)
            raise

    return wrapped
