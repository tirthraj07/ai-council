"""
Round-based debate runner: orchestrates rounds, summary agent, and transcript logging.
"""

import random
from pathlib import Path


class RoundRunner:
    """
    Runs the council in rounds. User provides initial topic; debate agents speak
    in random order each round; summary agent summarizes each round and at end.
    Transcript (broadcasts, tool calls, whispers) is written to a file.
    """

    def __init__(self, council, transcript_path: str | Path = "debate_transcript.txt"):
        self._council = council
        self._transcript_path = Path(transcript_path)

    def _write_round(self, round_number: int, round_lines: list[str]) -> None:
        with open(self._transcript_path, "a", encoding="utf-8") as f:
            f.write(f"-- Round {round_number}\n")
            f.write("\n".join(round_lines))
            if round_lines and not round_lines[-1].endswith("\n"):
                f.write("\n")
            f.write("\n")

    def _write_final_summary(self, summary_text: str) -> None:
        with open(self._transcript_path, "a", encoding="utf-8") as f:
            f.write("-- Final summary\n")
            f.write(summary_text)
            if summary_text and not summary_text.endswith("\n"):
                f.write("\n")
            f.write("\n")

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

        print("Enter your thought, idea, or topic to discuss (start of round 1):")
        initial = input().strip()
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

        round_summaries: list[str] = []
        round_number = 1

        while True:
            order = list(debate_names)
            random.shuffle(order)
            round_lines: list[str] = []
            round_lines.append(f"[User]\n{initial}")

            for agent_name in order:
                turn_tool_calls: list[tuple[str, dict, str]] = []
                turn_whispers: list[tuple[str, str]] = []

                def on_tool_call(tool_name: str, arguments: dict, result: str) -> None:
                    turn_tool_calls.append((tool_name, arguments, result))
                    if tool_name == "whisper":
                        to_agent = arguments.get("to_agent", "")
                        msg = arguments.get("message", "")
                        turn_whispers.append((to_agent, msg))

                prompt = self._build_turn_prompt(round_summaries)
                reply = council.run_turn(
                    agent_name,
                    prompt,
                    short_term_limit=10,
                    on_tool_call=on_tool_call,
                )

                round_lines.append(f"[{agent_name}]")
                for tname, targs, tresult in turn_tool_calls:
                    round_lines.append(f"<tool-call>{tname}({targs}) -> {tresult}</tool-call>")
                for to_agent, msg in turn_whispers:
                    round_lines.append(f"Whisper to {to_agent}: {msg}")
                round_lines.append(reply)
                round_lines.append("")

            round_content = "\n".join(round_lines)
            if summary_name:
                summary_prompt = (
                    "You are summarizing this round. Based on the following discussion, "
                    "list important points and main positions. Be concise.\n\n"
                    f"Discussion:\n{round_content}"
                )
                summary_text = council.run_turn(summary_name, summary_prompt)
                round_summaries.append(summary_text)
                round_lines.append(f"[summary]\n{summary_text}")
            else:
                round_lines.append("[summary]\n(No summary agent set)")

            self._write_round(round_number, round_lines)
            round_number += 1

            if fixed_rounds_mode:
                if round_number > max_rounds:
                    break
                initial = "(No new context)"
                continue

            print("\nAdd context for the next round, or type /quit to end the debate:")
            user_input = input().strip()
            if user_input.lower() == "/quit":
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
            final_summary = council.run_turn(summary_name, final_prompt)
            self._write_final_summary(final_summary)
            print("\n-- Final summary\n")
            print(final_summary)
        else:
            self._write_final_summary("(No summary agent or no rounds)")
        print(f"\nTranscript written to {self._transcript_path}")
