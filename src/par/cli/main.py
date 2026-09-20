from __future__ import annotations

from par.cli import demo, doctor
from par.env import load_env


def main(argv: list[str] | None = None) -> int:
    import argparse

    load_env()
    parser = argparse.ArgumentParser(prog="par")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("doctor", help="Check runtime component readiness")
    subparsers.add_parser("demo", help="Run the live end-to-end demo (mock robot, no hardware needed)")

    args = parser.parse_args(argv)
    if args.command == "doctor":
        return doctor.run()
    if args.command == "demo":
        return demo.run()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
