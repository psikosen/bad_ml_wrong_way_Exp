"""Teacher/oracle tool for agent experiments.

Provides expert trajectories and critiques for distillation (E5).
The teacher is rule-based for reproducibility.
"""

from typing import Dict, List, Optional, Tuple


class TeacherOracle:
    """Rule-based teacher that produces action sequences and critiques."""

    def get_trajectory(self, task: str) -> Tuple[List[str], str]:
        """Return (plan_steps, recommended_action) for a given task.

        Returns:
            plan: list of plan tokens e.g. ["PLAN", "SEARCH", "CHECK", "ANSWER", "END"]
            action: the immediate next action to take
        """
        task_lower = task.lower()

        if any(kw in task_lower for kw in ["compute", "calculate", "execute", "run"]):
            return ["PLAN", "EXEC", "CHECK", "ANSWER", "END"], "EXEC"

        if any(kw in task_lower for kw in ["recall", "stored", "memory", "retrieve", "saved"]):
            return ["PLAN", "MEMORY_READ", "CHECK", "ANSWER", "END"], "MEMORY_READ"

        if any(kw in task_lower for kw in ["look up", "search", "find", "latest"]):
            return ["PLAN", "SEARCH", "CHECK", "ANSWER", "END"], "SEARCH"

        if any(kw in task_lower for kw in ["teach", "consult", "guidance", "help"]):
            return ["PLAN", "TEACHER", "CHECK", "ANSWER", "END"], "TEACHER"

        return ["PLAN", "ANSWER", "END", "PAD", "PAD"], "ANSWER"

    def critique(self, task: str, agent_action: str) -> Dict:
        """Critique an agent's action choice.

        Returns a dict with 'correct', 'expected_action', and 'feedback'.
        """
        _, expected = self.get_trajectory(task)
        correct = agent_action == expected
        feedback = "Correct action." if correct else f"Expected {expected}, got {agent_action}."
        return {
            "correct": correct,
            "expected_action": expected,
            "agent_action": agent_action,
            "feedback": feedback,
        }

    def generate_trace(self, task: str) -> List[Dict]:
        """Generate a full Thought/Action/Observation trace for distillation."""
        plan, action = self.get_trajectory(task)
        steps = []
        for t, step_token in enumerate(plan):
            if step_token == "PAD":
                break
            steps.append({
                "t": t,
                "thought": f"Step {t}: considering {step_token}",
                "action": step_token if step_token not in ("PLAN", "CHECK", "END") else None,
                "observation": f"Completed {step_token}" if step_token != "PLAN" else "Planning started",
                "reward": 0.0 if t < len(plan) - 1 else 1.0,
            })
        return steps
