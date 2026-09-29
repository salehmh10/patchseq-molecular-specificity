"""Wall-clock runtime controls for the 150-minute fast-paper pipeline."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class RuntimeCheckpoint:
    name: str
    timestamp: str
    elapsed_minutes: float
    remaining_minutes: float
    level: int


class RuntimeGovernor:
    """Select predefined compute levels without weakening mandatory design gates."""

    def __init__(
        self,
        total_minutes: float = 150.0,
        start_time: str | datetime | None = None,
        log_path: str | Path | None = None,
    ) -> None:
        self.total_minutes = float(total_minutes)
        if isinstance(start_time, str):
            self.start_time = datetime.fromisoformat(start_time)
        else:
            self.start_time = start_time or datetime.now().astimezone()
        self.log_path = Path(log_path) if log_path else None
        self.level = 0

    def elapsed_minutes(self) -> float:
        now = datetime.now(self.start_time.tzinfo) if self.start_time.tzinfo else datetime.now()
        return max(0.0, (now - self.start_time).total_seconds() / 60.0)

    def remaining_minutes(self) -> float:
        return max(0.0, self.total_minutes - self.elapsed_minutes())

    def can_start(self, expected_minutes: float, reserve_minutes: float = 5.0) -> bool:
        return expected_minutes + reserve_minutes <= self.remaining_minutes()

    def choose_level(self, projected_modeling_minutes: float) -> int:
        remaining_for_compute = max(0.0, self.remaining_minutes() - 5.0)
        if projected_modeling_minutes <= min(100.0, remaining_for_compute):
            self.level = 0
        elif projected_modeling_minutes * 0.65 <= remaining_for_compute:
            self.level = 1
        else:
            self.level = 2
        return self.level

    def checkpoint(self, name: str) -> RuntimeCheckpoint:
        checkpoint = RuntimeCheckpoint(
            name=name,
            timestamp=datetime.now().astimezone().isoformat(),
            elapsed_minutes=round(self.elapsed_minutes(), 4),
            remaining_minutes=round(self.remaining_minutes(), 4),
            level=self.level,
        )
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(asdict(checkpoint), sort_keys=True) + "\n")
        return checkpoint

    @property
    def settings(self) -> dict[str, int]:
        presets = {
            0: {"mlp_seeds": 3, "bootstrap": 1000, "permutations": 5000, "shap_cap": -1},
            1: {"mlp_seeds": 2, "bootstrap": 500, "permutations": 2000, "shap_cap": 2000},
            2: {"mlp_seeds": 1, "bootstrap": 300, "permutations": 1000, "shap_cap": 1000},
        }
        return presets[self.level]
