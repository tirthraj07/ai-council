"""Remove persistent memory (ChromaDB) and debate transcript. Run from project root."""

import os
import shutil
from pathlib import Path

# Paths matching config/council.yaml
CHROMA_DIR = Path("data/chroma")
TRANSCRIPT_FILE = Path("debate_transcript.txt")


def main() -> None:
    base = Path(__file__).resolve().parent.parent
    os.chdir(base)
    if CHROMA_DIR.is_dir():
        shutil.rmtree(CHROMA_DIR)
        print(f"  Removed {CHROMA_DIR}")
    if TRANSCRIPT_FILE.is_file():
        TRANSCRIPT_FILE.unlink()
        print(f"  Removed {TRANSCRIPT_FILE}")
    print("Done. You can restart the council from scratch.")


if __name__ == "__main__":
    main()
