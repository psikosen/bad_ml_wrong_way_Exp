"""Toy execution tool for agent experiments.

Provides a safe, sandboxed arithmetic executor.
Safety: no network access, no system calls, no file I/O.
Only handles simple arithmetic expressions.
"""

import re
from typing import Tuple


# Allowlisted operations
ALLOWED_OPS = {"+", "-", "*"}


class ToyExecutor:
    """Safe arithmetic executor for agent experiments.

    Only supports: "compute <int> <op> <int>" where op in {+, -, *}.
    """

    def execute(self, command: str) -> Tuple[str, int, bool]:
        """Execute a safe command.

        Returns: (output_text, exit_code, success)
        """
        command = command.strip()

        # Parse "compute a op b" format
        match = re.match(r"compute\s+(\d+)\s*([+\-*])\s*(\d+)", command)
        if not match:
            return "[EXEC ERROR] Unsupported command format. Use: compute <int> <op> <int>", 1, False

        a = int(match.group(1))
        op = match.group(2)
        b = int(match.group(3))

        if op not in ALLOWED_OPS:
            return f"[EXEC ERROR] Operator '{op}' not allowed.", 1, False

        if op == "+":
            result = a + b
        elif op == "-":
            result = a - b
        elif op == "*":
            result = a * b

        return f"[EXEC OK] {a} {op} {b} = {result}", 0, True

    def dry_run(self, command: str) -> str:
        """Preview what would be executed without running it."""
        match = re.match(r"compute\s+(\d+)\s*([+\-*])\s*(\d+)", command.strip())
        if not match:
            return "[DRY RUN] Invalid command format."
        return f"[DRY RUN] Would compute: {match.group(1)} {match.group(2)} {match.group(3)}"
