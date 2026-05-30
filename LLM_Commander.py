#!/usr/bin/env python3
"""Thin runtime entrypoint for guarded refactored implementation (M0+M1)."""

from __future__ import annotations

from runtime_loop import main as runtime_main


def main() -> None:
    runtime_main()


if __name__ == "__main__":
    main()
