"""
Local risk/intersection event logger for v3.

The server is not implemented yet, so this module writes the same data contract
locally:

- event_records/<intersection_event_id>/risk_events.jsonl
- event_records/<intersection_event_id>/intersection_event.jsonl
- event_records/<intersection_event_id>/snapshots/*.jpg
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import cv2

from safety_decision_reboot import ComplianceSeverity, IntersectionPhase


LOGGABLE_CAUTION_REASON_CODES = {
    "pedestrian_near_crosswalk",
    "pedestrian_near_crosswalk_in_intersection",
    "structure_uncertain",
    "right_turn_not_rightmost_lane",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


@dataclass(frozen=True)
class EventLoggerConfig:
    event_root_dir: Path
    device_id: str = "edge-001"
    vehicle_id: str = "test-car-01"
    enable_snapshots: bool = True


@dataclass
class _IntersectionSession:
    event_id: str
    started_at: str
    start_frame: int
    scenario: str
    event_dir: Path
    risk_log_path: Path
    intersection_log_path: Path
    snapshot_dir: Path
    phase_sequence: list[str] = field(default_factory=list)
    timeline: list[dict[str, Any]] = field(default_factory=list)
    snapshots: list[dict[str, Any]] = field(default_factory=list)
    event_codes: set[str] = field(default_factory=set)
    highest_severity: str = "INFO"
    final_decision: str = "UNKNOWN"
    evidence_summary: dict[str, Any] = field(default_factory=dict)
    last_timeline_signature: tuple | None = None


class LocalEventLogger:
    """Write local JSONL logs and representative snapshots."""

    def __init__(self, config: EventLoggerConfig):
        self.config = config
        self.config.event_root_dir.mkdir(parents=True, exist_ok=True)
        self._session: _IntersectionSession | None = None
        self._last_risk_signature: tuple | None = None

    def update(self, frame_index: int, frame, decision, context, detections=None) -> None:
        phase = context.phase.value
        scenario = context.scenario.value
        events = list(getattr(decision, "compliance_events", []) or [])
        event_codes = tuple(event.code for event in events)
        decision_level = decision.level.value
        reason_code = decision.reason_code
        evidence = dict(getattr(decision, "evidence", {}) or {})

        in_intersection_context = context.phase in (IntersectionPhase.APPROACH, IntersectionPhase.IN_INTERSECTION)
        if in_intersection_context and self._session is None:
            event_id = self._new_event_id()
            event_dir = self.config.event_root_dir / event_id
            snapshot_dir = event_dir / "snapshots"
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            self._session = _IntersectionSession(
                event_id=event_id,
                started_at=_now_iso(),
                start_frame=frame_index,
                scenario=scenario,
                event_dir=event_dir,
                risk_log_path=event_dir / "risk_events.jsonl",
                intersection_log_path=event_dir / "intersection_event.jsonl",
                snapshot_dir=snapshot_dir,
            )

        if self._session is not None and in_intersection_context:
            self._update_session(frame_index, frame, decision_level, reason_code, phase, scenario, events, evidence)

        loggable_caution = decision_level == "CAUTION" and reason_code in LOGGABLE_CAUTION_REASON_CODES
        risk_signature = (event_codes, decision_level, reason_code, phase, scenario)
        if (events or loggable_caution) and risk_signature != self._last_risk_signature:
            snapshot = self._save_snapshot(frame_index, frame, event_codes[0] if event_codes else reason_code, detections)
            session = self._session
            risk_log_path = session.risk_log_path if session is not None else self.config.event_root_dir / "risk_events_orphan.jsonl"
            self._append_jsonl(
                risk_log_path,
                {
                    "record_type": "risk_event_change" if events else "caution_event_change",
                    "device_id": self.config.device_id,
                    "vehicle_id": self.config.vehicle_id,
                    "intersection_event_id": session.event_id if session is not None else None,
                    "timestamp": _now_iso(),
                    "frame_index": frame_index,
                    "phase": phase,
                    "scenario": scenario,
                    "decision": decision_level,
                    "reason_code": reason_code,
                    "events": [self._event_to_dict(event) for event in events],
                    "evidence": evidence,
                    "detections": [self._detection_to_dict(det) for det in detections or []],
                    "snapshot_path": snapshot["path"] if snapshot else None,
                },
            )
            if self._session is not None and snapshot is not None:
                self._session.snapshots.append(snapshot)
        self._last_risk_signature = risk_signature

        if self._session is not None and not in_intersection_context:
            self.finish(frame_index)

    def finish(self, frame_index: int | None = None) -> None:
        if self._session is None:
            return

        session = self._session
        ended_at = _now_iso()
        event = {
            "record_type": "intersection_pass_event",
            "device_id": self.config.device_id,
            "vehicle_id": self.config.vehicle_id,
            "intersection_event_id": session.event_id,
            "started_at": session.started_at,
            "ended_at": ended_at,
            "start_frame": session.start_frame,
            "end_frame": frame_index,
            "scenario": session.scenario,
            "final_decision": session.final_decision,
            "highest_severity": session.highest_severity,
            "phase_sequence": session.phase_sequence,
            "event_codes": sorted(session.event_codes),
            "evidence_summary": session.evidence_summary,
            "snapshots": session.snapshots,
            "timeline": session.timeline,
            "event_dir": str(session.event_dir),
            "risk_log_path": str(session.risk_log_path),
        }
        self._append_jsonl(session.intersection_log_path, event)
        print(
            "[EVENT] intersection_saved "
            f"id={session.event_id} events={len(session.event_codes)} "
            f"snapshots={len(session.snapshots)} path={session.event_dir}"
        )
        self._session = None

    def _update_session(self,
                        frame_index: int,
                        frame,
                        decision_level: str,
                        reason_code: str,
                        phase: str,
                        scenario: str,
                        events,
                        evidence: dict[str, Any]) -> None:
        session = self._session
        if session is None:
            return

        session.scenario = scenario
        session.final_decision = decision_level
        if phase not in session.phase_sequence:
            session.phase_sequence.append(phase)

        for event in events:
            session.event_codes.add(event.code)
            session.highest_severity = self._max_severity(session.highest_severity, event.severity.value)

        self._merge_evidence_summary(session.evidence_summary, evidence)

        event_codes = tuple(event.code for event in events)
        signature = (phase, decision_level, reason_code, event_codes)
        if signature != session.last_timeline_signature:
            session.timeline.append(
                {
                    "frame_index": frame_index,
                    "timestamp": _now_iso(),
                    "phase": phase,
                    "decision": decision_level,
                    "reason_code": reason_code,
                    "events": list(event_codes),
                    "evidence": self._compact_evidence(evidence),
                }
            )
            session.last_timeline_signature = signature

    def _save_snapshot(self, frame_index: int, frame, reason: str, detections=None) -> dict[str, Any] | None:
        if not self.config.enable_snapshots or frame is None:
            return None
        session = self._session
        snapshot_dir = session.snapshot_dir if session is not None else self.config.event_root_dir
        name = f"{self.config.device_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_f{frame_index}_{reason}.jpg"
        path = snapshot_dir / self._safe_name(name)
        snapshot_frame = frame.copy()
        self._draw_detections(snapshot_frame, detections or [])
        ok = cv2.imwrite(str(path), snapshot_frame)
        if not ok:
            return None
        return {
            "frame_index": frame_index,
            "reason": reason,
            "path": str(path),
            "detections": [self._detection_to_dict(det) for det in detections or []],
        }

    def _new_event_id(self) -> str:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return f"{self.config.device_id}-{stamp}-{uuid4().hex[:8]}"

    @staticmethod
    def _append_jsonl(path: Path, obj: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")

    @staticmethod
    def _event_to_dict(event) -> dict[str, Any]:
        return {
            "code": event.code,
            "severity": event.severity.value,
            "title": event.title,
            "legal_basis": [basis.code for basis in getattr(event, "legal_basis", [])],
        }

    @staticmethod
    def _detection_to_dict(det) -> dict[str, Any]:
        return {
            "class": getattr(det, "cls_name", ""),
            "confidence": float(getattr(det, "conf", 0.0)),
            "box": [int(v) for v in getattr(det, "box", (0, 0, 0, 0))],
            "distance_m": getattr(det, "distance_m", None),
        }

    @staticmethod
    def _draw_detections(vis, detections) -> None:
        colors = {
            "pedestrian": (0, 255, 255),
            "vehicle": (255, 90, 90),
            "traffic_light_vehicle": (0, 80, 255),
            "traffic_light_pedestrian": (0, 180, 255),
            "crosswalk": (0, 255, 0),
            "left_turn_sign": (255, 180, 0),
        }
        h, w = vis.shape[:2]
        for det in detections:
            try:
                x1, y1, x2, y2 = [int(v) for v in getattr(det, "box", (0, 0, 0, 0))]
            except Exception:
                continue
            x1, x2 = max(0, min(w - 1, x1)), max(0, min(w - 1, x2))
            y1, y2 = max(0, min(h - 1, y1)), max(0, min(h - 1, y2))
            cls_name = str(getattr(det, "cls_name", "object"))
            color = colors.get(cls_name, (255, 255, 255))
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            label = f"{cls_name} {float(getattr(det, 'conf', 0.0)):.2f}"
            distance_m = getattr(det, "distance_m", None)
            if distance_m is not None:
                label += f" {float(distance_m):.1f}m"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            label_y0 = max(0, y1 - th - 8)
            cv2.rectangle(vis, (x1, label_y0), (min(w - 1, x1 + tw + 8), label_y0 + th + 8), color, -1)
            cv2.putText(
                vis,
                label,
                (x1 + 4, label_y0 + th + 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    @staticmethod
    def _max_severity(current: str, new: str) -> str:
        order = {
            ComplianceSeverity.INFO.value: 0,
            ComplianceSeverity.WARNING.value: 1,
            ComplianceSeverity.VIOLATION_SUSPECTED.value: 2,
        }
        return new if order.get(new, 0) > order.get(current, 0) else current

    @staticmethod
    def _merge_evidence_summary(summary: dict[str, Any], evidence: dict[str, Any]) -> None:
        bool_fields = [
            "pedestrian_on_crosswalk",
            "pedestrian_near_crosswalk",
            "pedestrian_approaching_crosswalk",
            "pedestrian_leaving_crosswalk",
            "stop_completed_on_red",
        ]
        for field in bool_fields:
            summary[field] = bool(summary.get(field, False) or evidence.get(field, False))

        if evidence.get("vehicle_signal") == "RED":
            summary["red_signal_seen"] = True
        summary["rightmost_lane"] = evidence.get("lane_role_rightmost", summary.get("rightmost_lane"))
        summary["last_vehicle_signal"] = evidence.get("vehicle_signal", summary.get("last_vehicle_signal"))
        summary["last_signal_source"] = evidence.get("vehicle_signal_source", summary.get("last_signal_source"))
        summary["last_stop_source"] = evidence.get("stop_completed_source", summary.get("last_stop_source"))
        if evidence.get("stop_completed_source") == "auto_motion" and evidence.get("vehicle_signal") == "RED":
            summary["min_stop_motion_score"] = LocalEventLogger._min_optional(
                summary.get("min_stop_motion_score"),
                evidence.get("stop_motion_score"),
            )

    @staticmethod
    def _compact_evidence(evidence: dict[str, Any]) -> dict[str, Any]:
        keys = [
            "vehicle_signal",
            "vehicle_signal_source",
            "stop_completed_on_red",
            "stop_completed_source",
            "stop_motion_score",
            "pedestrian_on_crosswalk",
            "pedestrian_near_crosswalk",
            "pedestrian_approaching_crosswalk",
            "pedestrian_leaving_crosswalk",
            "lane_role_rightmost",
            "lane_role_confidence",
        ]
        return {key: evidence.get(key) for key in keys if key in evidence}

    @staticmethod
    def _min_optional(left, right):
        if right is None:
            return left
        if left is None:
            return right
        return min(left, right)

    @staticmethod
    def _safe_name(name: str) -> str:
        keep = []
        for ch in name:
            keep.append(ch if ch.isalnum() or ch in "._-" else "_")
        return "".join(keep)
