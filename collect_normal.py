"""Backend selector. Original PDDL implementation is preserved verbatim in collect_pddl.py."""
import argparse
import importlib
import sys

# Preserve imports used by the existing OpenAI/Gemini collectors and tests.
from collect_pddl import collect, digest, save_json, select_games

BACKENDS = {"pddl": "collect_pddl", "openai": "collect_llm", "gemini": "collect_gemini", "local": "collect_local"}


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--agent", choices=BACKENDS, default="pddl")
    choice, remaining = parser.parse_known_args(arguments)
    if any(value in remaining for value in ("-h", "--help")):
        print("Backend selector: --agent {pddl,openai,gemini,local} (default: pddl). Options below belong to the selected backend.\n")
    module = importlib.import_module(BACKENDS[choice.agent])
    previous = sys.argv
    try:
        sys.argv = [previous[0], *remaining]
        return module.main()
    finally:
        sys.argv = previous


if __name__ == "__main__":
    raise SystemExit(main())
