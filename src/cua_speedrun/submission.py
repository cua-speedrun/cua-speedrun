"""The submission contract: a folder with exactly two scripts.

    init.py   runs once, untimed, before any task is revealed.
              contract: `python init.py` exits 0 when ready.
    agent.py  runs once per task, timed.
              contract: `python agent.py <env_url> <task_description>`

Nothing else is required. Agents live in agents/.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from cua_speedrun.specs import content_hash


@dataclass
class Submission:
    submission_dir: Path
    init_script: Path
    agent_script: Path
    fingerprint: str

    @classmethod
    def load(cls, submission_dir: Path) -> "Submission":
        submission_dir = Path(submission_dir).resolve()
        init_script = submission_dir / "init.py"
        agent_script = submission_dir / "agent.py"
        missing = [p.name for p in (init_script, agent_script) if not p.is_file()]
        if missing:
            raise ValueError(
                f"{submission_dir} is not a valid submission: missing {', '.join(missing)}. "
                "A submission is a folder containing init.py and agent.py."
            )
        return cls(
            submission_dir=submission_dir,
            init_script=init_script,
            agent_script=agent_script,
            fingerprint=content_hash([init_script, agent_script]),
        )
