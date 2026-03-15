"""
Round-based debate runner: orchestrates rounds, summary agent, and transcript logging.
Prints transcript to the terminal with a unique color per agent.
Shows an animated "agent is working ..." while each agent runs.
"""

import random
import sys
import threading
from pathlib import Path

_WORKING_COLOR = "\033[90m"

# ANSI color codes (bright, distinct). No red to avoid confusion with errors.
_TERM_RESET = "\033[0m"
_AGENT_COLORS = [
    "\033[92m",   # bright green
    "\033[93m",   # bright yellow
    "\033[94m",   # bright blue
    "\033[95m",   # bright magenta
    "\033[96m",   # bright cyan
    "\033[97m",   # bright white
    "\033[32m",   # green
    "\033[33m",   # yellow
    "\033[34m",   # blue
    "\033[35m",   # magenta
    "\033[36m",   # cyan
    "\033[38;5;208m",  # orange (256-color)
    "\033[38;5;39m",  # light blue (256-color)
]
_USER_COLOR = "\033[97m"      # bright white for [User]
_SUMMARY_COLOR = "\033[93m"   # bright yellow for [summary]
_ROUND_HEADER_COLOR = "\033[90m"  # dim for -- Round N --


def _read_multiline(prompt: str) -> str:
    """
    Read multiple lines from stdin until the user enters an empty line.
    Allows pasting or typing multi-line content. Final empty line signals end of input.
    """
    print(prompt)
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip() == "":
            break
        lines.append(line)
    return "\n".join(lines).strip()


def _build_agent_color_map(debate_names: list[str], summary_name: str | None) -> dict[str, str]:
    """Assign a unique ANSI color to each agent, User, and summary."""
    color_map: dict[str, str] = {"User": _USER_COLOR, "summary": _SUMMARY_COLOR}
    if summary_name:
        color_map[summary_name] = _SUMMARY_COLOR
    for i, name in enumerate(debate_names):
        if name not in color_map:
            color_map[name] = _AGENT_COLORS[i % len(_AGENT_COLORS)]
    return color_map


def _run_with_working_animation(agent_name: str, run_fn):
    """Run run_fn() while showing an animated 'agent_name is working ...' line. Returns run_fn() result."""
    stop = threading.Event()
    done = [None]

    def animate():
        dots = 0
        while not stop.wait(0.25):
            frame = "  " + agent_name + " is working " + "." * (dots % 3 + 1) + "   "
            sys.stdout.write(f"\r{_WORKING_COLOR}{frame}{_TERM_RESET}")
            sys.stdout.flush()
            dots += 1
        sys.stdout.write("\r" + " " * (len(agent_name) + 25) + "\r")
        sys.stdout.flush()

    def run():
        try:
            done[0] = run_fn()
        except Exception as e:
            done[0] = e

    t = threading.Thread(target=run, daemon=True)
    t.start()
    anim = threading.Thread(target=animate, daemon=True)
    anim.start()
    t.join()
    stop.set()
    anim.join(timeout=1.5)
    sys.stdout.write("\r" + " " * 60 + "\r")
    sys.stdout.flush()
    if isinstance(done[0], Exception):
        raise done[0]
    return done[0]


def _expand_newlines(s: str) -> str:
    """Replace literal \\n in string (e.g. from repr of dict) with real newlines for display."""
    if not s or not isinstance(s, str):
        return s
    return s.replace("\\n", "\n")


def _agent_provider_model(council, agent_name: str) -> str:
    """Return '(provider / model)' for the agent, or '' if not available."""
    try:
        agent = council.get_agent(agent_name)
        llm = getattr(agent, "llm", None)
        if llm is None:
            return ""
        provider = getattr(llm, "provider_name", "") or ""
        model = getattr(llm, "model_name", "") or ""
        if provider or model:
            return f" ({provider} / {model})"
    except Exception:
        pass
    return ""


class RoundRunner:
    """
    Runs the council in rounds. User provides initial topic; debate agents speak
    in random order each round; summary agent summarizes each round and at end.
    Transcript (broadcasts, tool calls, whispers) is written to a file and echoed
    to the terminal with a unique color per agent.
    """

    def __init__(self, council, transcript_path: str | Path = "debate_transcript.txt"):
        self._council = council
        self._transcript_path = Path(transcript_path).resolve()
        self._agent_colors: dict[str, str] = {}

    def _print_line(self, text: str, role: str) -> None:
        """Print one transcript line to the terminal with the role's color."""
        if role == "":
            color = _ROUND_HEADER_COLOR
        else:
            color = self._agent_colors.get(role, _TERM_RESET)
        print(f"{color}{text}{_TERM_RESET}", flush=True)

    def _write_round(self, round_number: int, round_lines: list[str]) -> None:
        with open(self._transcript_path, "a", encoding="utf-8") as f:
            f.write(f"-- Round {round_number}\n")
            f.write("\n".join(round_lines))
            if round_lines and not round_lines[-1].endswith("\n"):
                f.write("\n")
            f.write("\n")
            f.flush()

    def _write_final_summary(self, summary_text: str) -> None:
        with open(self._transcript_path, "a", encoding="utf-8") as f:
            f.write("-- Final summary\n")
            f.write(summary_text)
            if summary_text and not summary_text.endswith("\n"):
                f.write("\n")
            f.write("\n")
            f.flush()

    def _build_turn_prompt(self, round_summaries: list[str]) -> str:
        parts = []
        if round_summaries:
            parts.append("Previous round summaries:")
            for i, s in enumerate(round_summaries, 1):
                parts.append(f"Round {i}: {s}")
            parts.append("")
        parts.append(
            "Review the forum and any whispers to you. Reason, use tools if needed "
            "(e.g. whisper to another agent for a private note), and end by broadcasting your opinion."
        )
        return "\n".join(parts)

    def run_debate(self) -> None:
        council = self._council
        debate_names = council.debate_agent_names
        if not debate_names:
            raise ValueError("Council has no debate agents.")
        summary_name = getattr(council, "_summary_agent_name", None)

        initial = _read_multiline(
            "Enter your thought, idea, or topic to discuss (start of round 1).\n"
            "You can paste multiple lines; press Enter on an empty line when done:"
        )
        if not initial:
            initial = "(No topic given)"
        council.forum.broadcast("User", initial)

        print("Enter number of rounds to run (or press Enter to prompt after each round; type /quit to end):")
        rounds_input = input().strip()
        try:
            max_rounds = int(rounds_input) if rounds_input else None
            if max_rounds is not None and max_rounds < 1:
                max_rounds = None
        except ValueError:
            max_rounds = None
        fixed_rounds_mode = max_rounds is not None
        if fixed_rounds_mode:
            print(f"Running {max_rounds} round(s). No prompt between rounds.\n")

        self._agent_colors = _build_agent_color_map(debate_names, summary_name)
        round_summaries: list[str] = []
        round_number = 1

        self._transcript_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._transcript_path, "a", encoding="utf-8") as f:
            f.flush()
        print(f"Transcript will be saved to: {self._transcript_path}", flush=True)

        while True:
            order = list(debate_names)
            random.shuffle(order)
            round_lines: list[str] = []
            self._print_line(f"-- Round {round_number} --", "")
            user_block = f"[User]\n{initial}"
            round_lines.append(user_block)
            self._print_line(user_block, "User")

            try:
                for agent_name in order:
                    turn_tool_calls: list[tuple[str, dict, str]] = []
                    turn_whispers: list[tuple[str, str]] = []

                    def on_tool_call(tool_name: str, arguments: dict, result: str) -> None:
                        turn_tool_calls.append((tool_name, arguments, result))
                        if tool_name == "whisper":
                            to_agent = arguments.get("to_agent", "")
                            msg = arguments.get("message", "")
                            turn_whispers.append((to_agent, msg))

                    provider_model = _agent_provider_model(council, agent_name)
                    header = f"[{agent_name}]{provider_model}"
                    round_lines.append(f"[{agent_name}]")
                    self._print_line(header, agent_name)

                    prompt = self._build_turn_prompt(round_summaries)

                    def do_turn():
                        return council.run_turn(
                            agent_name,
                            prompt,
                            short_term_limit=10,
                            on_tool_call=on_tool_call,
                        )

                    try:
                        reply = _run_with_working_animation(agent_name, do_turn)
                    except Exception as e:
                        reply = f"(Error during turn: {e})"

                    for tname, targs, tresult in turn_tool_calls:
                        line = f"<tool-call>{tname}({targs}) -> {tresult}</tool-call>"
                        line = _expand_newlines(line)
                        round_lines.append(line)
                        for part in line.split("\n"):
                            self._print_line(part, agent_name)
                    for to_agent, msg in turn_whispers:
                        line = f"Whisper to {to_agent}: {msg}"
                        line = _expand_newlines(line)
                        round_lines.append(line)
                        for part in line.split("\n"):
                            self._print_line(part, agent_name)
                    reply_display = _expand_newlines(reply) if reply else ""
                    round_lines.append(reply_display or reply)
                    if reply:
                        for part in reply_display.split("\n"):
                            self._print_line(part, agent_name)
                    round_lines.append("")
                    self._print_line("", agent_name)

                round_content = "\n".join(round_lines)
                if summary_name:
                    summary_prompt = (
                        "You are summarizing this round. Based on the following discussion, "
                        "list important points and main positions. Be concise.\n\n"
                        f"Discussion:\n{round_content}"
                    )
                    summary_header_line = f"[summary]{_agent_provider_model(council, summary_name)}"
                    self._print_line(summary_header_line, summary_name)

                    def do_summary():
                        return council.run_turn(summary_name, summary_prompt)

                    try:
                        summary_text = _run_with_working_animation(summary_name, do_summary)
                    except Exception as e:
                        summary_text = f"(Error during summary: {e})"
                    if summary_text:
                        self._print_line(summary_text, summary_name)
                    round_summaries.append(summary_text)
                    summary_block = f"[summary]\n{summary_text}"
                    round_lines.append(summary_block)
                else:
                    round_lines.append("[summary]\n(No summary agent set)")
                    self._print_line("[summary]\n(No summary agent set)", "summary")
            finally:
                self._write_round(round_number, round_lines)

            round_number += 1

            if fixed_rounds_mode:
                if round_number > max_rounds:
                    break
                initial = "(No new context)"
                continue

            user_input = _read_multiline(
                "\nAdd context for the next round, or type /quit to end the debate.\n"
                "You can paste multiple lines; press Enter on an empty line when done:"
            )
            if user_input.lower().strip() == "/quit":
                break
            if user_input:
                council.forum.broadcast("User", user_input)
                initial = user_input
            else:
                initial = "(No new context)"

        if summary_name and round_summaries:
            final_prompt = (
                "Summarize the entire debate across all rounds: what was discussed, "
                "for and opposing views, important points. Give a consolidated view.\n\n"
                "Round summaries:\n"
            )
            for i, s in enumerate(round_summaries, 1):
                final_prompt += f"\nRound {i}: {s}\n"
            self._print_line("\n-- Final summary --", "")
            final_header = f"[summary]{_agent_provider_model(council, summary_name)}"
            if final_header.strip():
                self._print_line(final_header, summary_name)

            def do_final_summary():
                return council.run_turn(summary_name, final_prompt)

            try:
                final_summary = _run_with_working_animation(summary_name, do_final_summary)
            except Exception as e:
                final_summary = f"(Error during final summary: {e})"
            if final_summary:
                self._print_line(final_summary, summary_name)
            self._write_final_summary(final_summary)
        else:
            self._write_final_summary("(No summary agent or no rounds)")
        print(f"\nTranscript written to {self._transcript_path}")
