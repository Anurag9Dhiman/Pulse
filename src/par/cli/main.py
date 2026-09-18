from __future__ import annotations

from par.cli import doctor


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="par")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("doctor", help="Check runtime component readiness")

    args = parser.parse_args(argv)
    if args.command == "doctor":
        return doctor.run()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
