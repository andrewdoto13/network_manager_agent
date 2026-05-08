"""UI utilities for running and displaying agent output."""

import json
import re
import time
from datetime import datetime
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from .config import PROJECT_ROOT


def _parse_tool_result(content: str) -> dict:
    """Parse a run_code tool result into stdout and value fields.

    run_code returns: "[stdout]\\n{stdout}\\n[/stdout]\\n{value}"
    or just "[stdout]\\n{stdout}\\n[/stdout]"
    or just the raw value (no [stdout] marker).
    """
    match = re.match(r"^\[stdout\]\n(.*?)\n\[/stdout\](?:\n(.+))?$", content, re.DOTALL)
    if match:
        return {"stdout": match.group(1), "value": match.group(2)}
    return {"stdout": "", "value": content}


def _try_parse_value(value):
    """Try to parse a string value as JSON, handling numpy repr fallback."""
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        pass
    try:
        fixed = value.replace("np.int64(", "").replace("np.float64(", "")
        fixed = fixed.replace("np.bool_(True)", "true").replace("np.bool_(False)", "false")
        fixed = fixed.replace("True", "true").replace("False", "false").replace("None", "null")
        fixed = re.sub(r"'([^']*)'", r'"\1"', fixed)
        return json.loads(fixed)
    except (json.JSONDecodeError, TypeError):
        return value


_VALUE_MAX = 2000


def _truncate_value(value, max_len=_VALUE_MAX):
    s = str(value)
    if len(s) > max_len:
        return s[:max_len] + f"\n... [truncated, {len(s) - max_len} more chars]"
    return s


def _format_code(code: str) -> str:
    """Format run_code submitted code as an indented block."""
    lines = code.strip().split("\n")
    return "\n".join(f"    {line}" for line in lines)


def _format_dict_table(records: list[dict]) -> str:
    """Format list[dict] as an ASCII table."""
    if not records:
        return "    (empty result)"
    keys = list(records[0].keys())
    widths = {k: max(len(k), max(len(str(r.get(k, ""))) for r in records)) for k in keys}
    header = "    " + " | ".join(k.ljust(widths[k]) for k in keys)
    separator = "   " + "-+-".join("-" * widths[k] for k in keys)
    rows = []
    for rec in records:
        row = "    " + " | ".join(str(rec.get(k, "")).ljust(widths[k]) for k in keys)
        rows.append(row)
    return f"{header}\n{separator}\n" + "\n".join(rows)


def _format_result(value) -> str:
    """Format run_code result for console display."""
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return _format_result(parsed)
        except (json.JSONDecodeError, TypeError):
            return f"    result: {value}"
    if isinstance(value, list) and value and isinstance(value[0], dict):
        return _format_dict_table(value)
    if isinstance(value, (dict, list)):
        return "    result:\n" + "\n".join(
            f"    {line}" for line in json.dumps(value, indent=2, default=str).split("\n")
        )
    return f"    result: {value}"


def _fmt_ts(dt: datetime) -> str:
    return dt.strftime("%H:%M:%S.%f")[:-3]


def _fmt_iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _session_header(f_txt, f_jsonl, dt: datetime):
    ts = _fmt_ts(dt)
    f_txt.write(f"\n--- Session: {ts} ---\n\n")
    f_jsonl.write(json.dumps({"session": ts}) + "\n")


def _log_human(f_txt, f_jsonl, step: int, dt: datetime, msg: HumanMessage):
    ts = _fmt_ts(dt)
    content = msg.content.strip()
    f_txt.write(f"[{ts}] STEP {step} | HUMAN INPUT\n")
    f_txt.write(f"  {content}\n\n")
    f_jsonl.write(json.dumps({
        "step": step,
        "timestamp": _fmt_iso(dt),
        "actions": [{"type": "human_input", "content": content}]
    }) + "\n")


def _log_tool_call(f_txt, f_jsonl, step: int, dt: datetime, node: str,
                   tool_name: str, tool_args: dict):
    ts = _fmt_ts(dt)
    f_txt.write(f"[{ts}] STEP {step} | {node}\n")
    f_txt.write(f"  TOOL CALL: {tool_name}\n")
    if tool_name == "run_code":
        code = tool_args.get("code", "")
        f_txt.write(f"    ```\n{_format_code(code)}\n    ```\n\n")
    else:
        f_txt.write(f"    -> {tool_name}({json.dumps(tool_args, default=str)})\n\n")
    f_jsonl.write(json.dumps({
        "step": step,
        "node": node,
        "timestamp": _fmt_iso(dt),
        "actions": [{"type": "tool_call", "tool": tool_name, "args": tool_args}]
    }) + "\n")


def _log_tool_result(f_txt, f_jsonl, step: int, dt: datetime, node: str,
                     duration_ms: int, tool_name: str, stdout: str,
                     value, error: str = None):
    ts = _fmt_ts(dt)
    f_txt.write(f"[{ts}] STEP {step} | {node} ({duration_ms // 1000}s)")
    if error:
        f_txt.write(", ERROR")
    f_txt.write("\n")
    f_txt.write(f"  TOOL RESULT: {tool_name}\n")
    if stdout:
        f_txt.write(f"    [stdout]\n{stdout}\n    [/stdout]\n")
    if value is not None:
        if tool_name == "run_code":
            f_txt.write(f"{_truncate_value(_format_result(value))}\n")
        else:
            f_txt.write(f"    result: {value}\n")
    f_txt.write("\n")
    f_jsonl.write(json.dumps({
        "step": step,
        "node": node,
        "timestamp": _fmt_iso(dt),
        "duration_ms": duration_ms,
        "actions": [{
            "type": "tool_result",
            "tool": tool_name,
            "result": {"stdout": stdout, "value": _truncate_value(_try_parse_value(value)), "error": error}
        }]
    }) + "\n")


def _log_ai_output(f_txt, f_jsonl, step: int, dt: datetime, node: str,
                   duration_ms: int, content: str):
    ts = _fmt_ts(dt)
    f_txt.write(f"[{ts}] STEP {step} | {node} ({duration_ms // 1000}s)\n")
    f_txt.write("  AI OUTPUT\n")
    f_txt.write(f"    {content.strip()}\n\n")
    f_jsonl.write(json.dumps({
        "step": step,
        "node": node,
        "timestamp": _fmt_iso(dt),
        "duration_ms": duration_ms,
        "actions": [{"type": "ai_output", "content": content.strip()}]
    }) + "\n")


def _log_summary(f_txt, f_jsonl, dt: datetime, summary: str):
    ts = _fmt_ts(dt)
    f_txt.write(f"[{ts}] RUNNING SUMMARY\n")
    f_txt.write(f"  {summary.strip()}\n\n")
    f_jsonl.write(json.dumps({
        "timestamp": _fmt_iso(dt),
        "actions": [{"type": "summary", "content": summary.strip()}]
    }) + "\n")


def _log_stop(f_txt, f_jsonl, step: int, dt: datetime, total_ms: int, reason: str):
    ts = _fmt_ts(dt)
    f_txt.write(f"[{ts}] COMPLETE (total: {total_ms // 1000}s)\n")
    f_txt.write(f"Stopped: {reason}\n")
    f_jsonl.write(json.dumps({
        "step": step,
        "timestamp": _fmt_iso(dt),
        "stop_reason": reason,
        "total_duration_ms": total_ms
    }) + "\n")


def run_agent(
    agent,
    inputs: dict,
    config: dict,
    output_dir: Path | None = None,
    max_steps: int | None = None,
):
    """Run the agent and stream output to console and log files in real-time.

    Args:
        agent: Compiled LangGraph agent.
        inputs: Agent input dict with 'messages', 'county_specialty_thresholds', etc.
        config: LangGraph config dict with thread_id.
        output_dir: Directory to save action log. Defaults to project-root logs/.
        max_steps: Maximum number of node steps before auto-stopping. None = unlimited.
    """
    print("Starting Agent Execution...\n")

    # Initialize log files in per-thread directory
    thread_id = config['configurable']['thread_id']
    log_dir = (output_dir or PROJECT_ROOT / "logs") / f"thread_{thread_id}"
    log_dir.mkdir(parents=True, exist_ok=True)
    txt_path = log_dir / "log.txt"
    jsonl_path = log_dir / "log.jsonl"

    # Determine if we're appending to an existing log
    append_mode = txt_path.exists()
    session_start = time.time()
    step_start = session_start

    stopped_reason = None
    step = 0

    with open(txt_path, "a", encoding="utf-8") as f_txt, \
         open(jsonl_path, "a", encoding="utf-8") as f_jsonl:

        if append_mode:
            _session_header(f_txt, f_jsonl, datetime.now())

        # Log initial human input
        human_msgs = [m for m in inputs.get("messages", []) if isinstance(m, HumanMessage)]
        if human_msgs:
            _log_human(f_txt, f_jsonl, step, datetime.now(), human_msgs[-1])
            step += 1

        try:
            for chunk in agent.stream(inputs, stream_mode="updates", config=config):
                if stopped_reason:
                    break
                for node_name, output in chunk.items():
                    if output is None:
                        continue

                    # Check step budget before processing
                    if max_steps and step > max_steps:
                        stopped_reason = f"max_steps ({max_steps})"
                        break

                    # Console Output
                    print(f"\n>>>> NODE: {node_name} <<<<")

                    if isinstance(output, dict) and "messages" in output:
                        for m in output["messages"]:
                            if isinstance(m, AIMessage) and m.tool_calls:
                                # Tool call from AI
                                for tc in m.tool_calls:
                                    tool_name = tc['name']
                                    tool_args = tc['args']

                                    # Console
                                    if tool_name == "run_code":
                                        print("   --- TOOL CALL: run_code ---")
                                        print(_format_code(tool_args.get("code", "")))
                                    else:
                                        print("   --- TOOL CALL ---")
                                        print(f"   -> {tool_name}({json.dumps(tool_args, default=str)})")

                                    # File log
                                    _log_tool_call(f_txt, f_jsonl, step, datetime.now(),
                                                 node_name, tool_name, tool_args)

                            elif isinstance(m, ToolMessage):
                                # Tool result
                                elapsed_ms = int((time.time() - step_start) * 1000)
                                step_start = time.time()

                                content = m.content
                                parsed = _parse_tool_result(content) if isinstance(content, str) else {"stdout": "", "value": content}
                                stdout = parsed.get("stdout", "")
                                value = parsed.get("value", None)

                                # Check for error in stdout
                                error = None
                                if stdout and "Error:" in stdout:
                                    error = stdout

                                # Console
                                print(f"   --- TOOL RESULT ({m.name}) ---")
                                if stdout:
                                    print(f"    [stdout]\n{stdout}\n    [/stdout]")
                                if value is not None:
                                    if m.name == "run_code":
                                        print(_format_result(value))
                                    else:
                                        print(f"    result: {value}")
                                print()

                                # File log
                                _log_tool_result(f_txt, f_jsonl, step, datetime.now(),
                                               node_name, elapsed_ms, m.name,
                                               stdout, value, error)

                                step += 1

                            elif isinstance(m, AIMessage):
                                # AI text output
                                elapsed_ms = int((time.time() - step_start) * 1000)
                                step_start = time.time()

                                content = m.content
                                if isinstance(content, list):
                                    content = " ".join(str(c) for c in content)
                                if content.strip():
                                    # Console
                                    print("   --- CURRENT CONTENT ---")
                                    m.pretty_print()
                                    print()

                                    # File log
                                    _log_ai_output(f_txt, f_jsonl, step, datetime.now(),
                                                 node_name, elapsed_ms, content)

                                step += 1

                            elif isinstance(m, HumanMessage):
                                # Console
                                print("   --- CURRENT CONTENT ---")
                                m.pretty_print()
                                print()

                                # File log
                                _log_human(f_txt, f_jsonl, step, datetime.now(), m)
                                step += 1

        except KeyboardInterrupt:
            stopped_reason = "user interrupted (Ctrl+C)"

        total_ms = int((time.time() - session_start) * 1000)

        # Write running summary
        try:
            state = agent.get_state(config)
            summary = state.values.get("summary", "")
            if isinstance(summary, str) and summary.strip():
                _log_summary(f_txt, f_jsonl, datetime.now(), summary)
        except Exception:
            pass

        # Write stop marker
        if not stopped_reason:
            stopped_reason = "execution complete"
        _log_stop(f_txt, f_jsonl, step, datetime.now(), total_ms, stopped_reason)

    if stopped_reason:
        print(f"\n--- Stopped at step {step - 1}: {stopped_reason} ---")
        print(f"Checkpoint saved. Resume with --thread-id {config['configurable']['thread_id']}")
    else:
        print(f"\nExecution Complete. Detailed history saved to '{txt_path}'")
