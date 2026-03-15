import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from src.config import build_council_from_config
from src.council import RoundRunner


def main():
    base_dir = Path(__file__).parent
    load_dotenv(base_dir / ".env")
    default_config = base_dir / "config" / "council.yaml"

    parser = argparse.ArgumentParser(description="Run the council debate from YAML config.")
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=None,
        help=f"Path to council YAML config. Default: {default_config}",
    )
    args = parser.parse_args()

    config_path = (
        args.config
        or (os.environ.get("COUNCIL_CONFIG") and Path(os.environ["COUNCIL_CONFIG"]))
        or default_config
    )

    council, options = build_council_from_config(config_path, base_dir=base_dir)
    transcript_path = options["transcript_path"]

    print("Debate agents:", council.debate_agent_names)
    print("Summary agent:", getattr(council, "_summary_agent_name", None))
    print("Config:", options["config_path"])

    runner = RoundRunner(council, transcript_path=transcript_path)
    runner.run_debate()


if __name__ == "__main__":
    main()
