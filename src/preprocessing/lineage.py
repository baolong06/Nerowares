"""Lineage metadata per ADR-004."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from datetime import datetime, timezone

@dataclass
class Lineage:
    dataset_id: str
    paradigm: str
    subject_id: str
    session_id: str
    run_id: str
    device: str
    channels: int
    sampling_rate_hz: float
    stimulus_id: str
    stimulus_type: str
    label_source: str
    preprocessing_profile: str
    source_path: str
    created_at: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

def build_lineage(**kwargs) -> Lineage:
    kwargs.setdefault("created_at", datetime.now(timezone.utc).isoformat())
    return Lineage(**kwargs)
