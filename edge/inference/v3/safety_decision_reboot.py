"""
v3 explainable safety-decision rules.

This module converts road-structure facts into a structured decision:
STOP, CAUTION, GO, or UNKNOWN.  It also records why the decision was made,
which legal basis is related, and which compliance-risk events were observed.

Important scope note:
Camera perception cannot legally confirm a violation.  ComplianceEvent records
"suspected risk" evidence only, so UI and logs must avoid final legal wording.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import cv2

from korean_overlay import draw_korean_lines


class DecisionLevel(Enum):
    STOP = "STOP"
    CAUTION = "CAUTION"
    GO = "GO"
    UNKNOWN = "UNKNOWN"


class IntersectionPhase(Enum):
    APPROACH = "APPROACH"
    IN_INTERSECTION = "IN_INTERSECTION"
    UNKNOWN = "UNKNOWN"


class Scenario(Enum):
    RIGHT_TURN = "RIGHT_TURN"
    UNPROTECTED_LEFT = "UNPROTECTED_LEFT"
    STRAIGHT = "STRAIGHT"
    UNKNOWN = "UNKNOWN"


class SignalState(Enum):
    RED = "RED"
    YELLOW = "YELLOW"
    GREEN = "GREEN"
    UNKNOWN = "UNKNOWN"
    NONE = "NONE"


class ComplianceSeverity(Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    VIOLATION_SUSPECTED = "VIOLATION_SUSPECTED"


@dataclass(frozen=True)
class LegalBasis:
    code: str
    title: str
    summary: str


@dataclass
class ComplianceEvent:
    code: str
    severity: ComplianceSeverity
    title: str
    description: str
    legal_basis: list[LegalBasis] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass
class SafetyDecision:
    level: DecisionLevel
    reason_code: str
    short_reason: str
    explanation: str
    legal_basis: list[LegalBasis] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)
    compliance_events: list[ComplianceEvent] = field(default_factory=list)


@dataclass
class IntersectionContext:
    phase: IntersectionPhase = IntersectionPhase.UNKNOWN
    scenario: Scenario = Scenario.UNKNOWN
    vehicle_signal: SignalState = SignalState.UNKNOWN
    vehicle_signal_source: str = "manual"
    pedestrian_signal: SignalState = SignalState.UNKNOWN
    right_turn_signal: SignalState = SignalState.NONE
    stop_completed_on_red: bool = False
    stop_completed_source: str = "manual"
    stop_motion_score: float = 0.0
    stop_stable_frames: int = 0

    @classmethod
    def from_args(cls, args) -> "IntersectionContext":
        return cls(
            phase=parse_phase(getattr(args, "phase", "unknown")),
            scenario=parse_scenario(getattr(args, "scenario", "unknown")),
            vehicle_signal=parse_signal(getattr(args, "vehicle_signal", "unknown")),
            pedestrian_signal=parse_signal(getattr(args, "pedestrian_signal", "unknown")),
            right_turn_signal=parse_signal(getattr(args, "right_turn_signal", "none")),
            stop_completed_on_red=bool(getattr(args, "stop_completed_on_red", False)),
        )


RTA_ARTICLE_25 = LegalBasis(
    code="RTA_ART_25",
    title="Road Traffic Act Article 25: intersection passage",
    summary="For a right turn, a vehicle should proceed slowly along the right edge of the road.",
)

RTA_ARTICLE_27 = LegalBasis(
    code="RTA_ART_27",
    title="Road Traffic Act Article 27: pedestrian protection",
    summary=(
        "If a pedestrian is crossing or is about to cross a crosswalk, "
        "the vehicle should stop and protect the pedestrian."
    ),
)

RTA_SIGNAL_OBEDIENCE = LegalBasis(
    code="RTA_SIGNAL",
    title="Road Traffic Act: signal obedience",
    summary="At a red signal, a vehicle should stop before the stop line, crosswalk, or intersection.",
)

RTA_UNPROTECTED_LEFT_PLACEHOLDER = LegalBasis(
    code="RTA_UNPROTECTED_LEFT_TBD",
    title="Unprotected-left-turn basis placeholder",
    summary="Detailed opposing-vehicle and signal rules will be added later.",
)


def parse_phase(value) -> IntersectionPhase:
    key = str(value or "unknown").strip().lower()
    mapping = {
        "approach": IntersectionPhase.APPROACH,
        "in_intersection": IntersectionPhase.IN_INTERSECTION,
        "in-intersection": IntersectionPhase.IN_INTERSECTION,
        "intersection": IntersectionPhase.IN_INTERSECTION,
        "unknown": IntersectionPhase.UNKNOWN,
    }
    return mapping.get(key, IntersectionPhase.UNKNOWN)


def parse_scenario(value) -> Scenario:
    key = str(value or "unknown").strip().lower()
    mapping = {
        "right_turn": Scenario.RIGHT_TURN,
        "right-turn": Scenario.RIGHT_TURN,
        "unprotected_left": Scenario.UNPROTECTED_LEFT,
        "unprotected-left": Scenario.UNPROTECTED_LEFT,
        "left_turn": Scenario.UNPROTECTED_LEFT,
        "left-turn": Scenario.UNPROTECTED_LEFT,
        "straight": Scenario.STRAIGHT,
        "unknown": Scenario.UNKNOWN,
    }
    return mapping.get(key, Scenario.UNKNOWN)


def parse_signal(value) -> SignalState:
    key = str(value or "unknown").strip().lower()
    mapping = {
        "red": SignalState.RED,
        "yellow": SignalState.YELLOW,
        "green": SignalState.GREEN,
        "unknown": SignalState.UNKNOWN,
        "none": SignalState.NONE,
        "off": SignalState.NONE,
        "auto": SignalState.UNKNOWN,
    }
    return mapping.get(key, SignalState.UNKNOWN)


class SafetyDecisionEngine:
    """Phase/scenario separated rule engine."""

    def decide(self, state, context: IntersectionContext) -> SafetyDecision:
        if context.scenario == Scenario.RIGHT_TURN:
            if context.phase == IntersectionPhase.APPROACH:
                return self.decide_right_turn_approach(state, context)
            if context.phase == IntersectionPhase.IN_INTERSECTION:
                return self.decide_right_turn_in_intersection(state, context)
            return self.decide_unknown(state, context, "right_turn_unknown_phase")

        if context.scenario == Scenario.UNPROTECTED_LEFT:
            if context.phase == IntersectionPhase.APPROACH:
                return self.decide_unprotected_left_approach(state, context)
            if context.phase == IntersectionPhase.IN_INTERSECTION:
                return self.decide_unprotected_left_in_intersection(state, context)
            return self.decide_unknown(state, context, "unprotected_left_unknown_phase")

        return self.decide_unknown(state, context, "unsupported_or_unknown_scenario")

    def decide_right_turn_approach(self, state, context: IntersectionContext) -> SafetyDecision:
        evidence = _evidence(state, context)
        events = _collect_compliance_events(state, context, evidence)

        # Approach phase uses signal rules strongly because the vehicle has not
        # committed to the intersection yet.
        if context.right_turn_signal == SignalState.RED:
            return _decision(
                DecisionLevel.STOP,
                "right_turn_signal_red",
                "right-turn signal red",
                "Dedicated right-turn signal is red, so right-turn entry should stop.",
                [RTA_SIGNAL_OBEDIENCE],
                evidence,
                events,
            )

        if context.vehicle_signal == SignalState.RED and not context.stop_completed_on_red:
            return _decision(
                DecisionLevel.STOP,
                "red_signal_stop_required",
                "red signal stop required",
                "Vehicle signal is red and red-stop completion was not confirmed.",
                [RTA_SIGNAL_OBEDIENCE],
                evidence,
                events,
            )

        if state.pedestrian_on_crosswalk:
            return _decision(
                DecisionLevel.STOP,
                "pedestrian_on_crosswalk",
                "pedestrian on crosswalk",
                "A pedestrian is detected inside the visible crosswalk area.",
                [RTA_ARTICLE_27],
                evidence,
                events,
            )

        if state.pedestrian_near_crosswalk:
            return _decision(
                DecisionLevel.CAUTION,
                "pedestrian_near_crosswalk",
                "pedestrian near crosswalk",
                "A pedestrian is near the visible crosswalk area and may enter it.",
                [RTA_ARTICLE_27],
                evidence,
                events,
            )

        if state.lane_role_rightmost is False and state.lane_role_confidence >= 0.45:
            return _decision(
                DecisionLevel.CAUTION,
                "right_turn_not_rightmost_lane",
                "not rightmost lane",
                "The current lane is estimated not to be the rightmost lane for a right turn.",
                [RTA_ARTICLE_25],
                evidence,
                events,
            )

        if state.crosswalk_present:
            if not state.path_available:
                return _decision(
                    DecisionLevel.CAUTION,
                    "structure_uncertain",
                    "crosswalk with uncertain path",
                    "A crosswalk is visible, but the ego/path structure is not reliable enough.",
                    [RTA_ARTICLE_27],
                    evidence,
                    events,
                )
            return _decision(
                DecisionLevel.CAUTION,
                "approaching_crosswalk",
                "approaching crosswalk",
                "A crosswalk is visible on approach; pedestrian presence should be checked.",
                [RTA_ARTICLE_27],
                evidence,
                events,
            )

        return _decision(
            DecisionLevel.GO,
            "clear_right_turn_approach",
            "no right-turn risk basis",
            "No current signal, pedestrian, lane, or crosswalk risk basis was observed.",
            [],
            evidence,
            events,
        )

    def decide_right_turn_in_intersection(self, state, context: IntersectionContext) -> SafetyDecision:
        evidence = _evidence(state, context)
        events = _collect_compliance_events(state, context, evidence)

        # Inside the intersection, do not stop only because vehicle_signal is
        # red.  The active basis becomes pedestrian/crosswalk risk.
        if state.pedestrian_on_crosswalk:
            return _decision(
                DecisionLevel.STOP,
                "pedestrian_on_crosswalk_in_intersection",
                "pedestrian on crosswalk",
                "A pedestrian is detected inside the crosswalk while the vehicle is in the intersection.",
                [RTA_ARTICLE_27],
                evidence,
                events,
            )

        if state.pedestrian_near_crosswalk:
            return _decision(
                DecisionLevel.CAUTION,
                "pedestrian_near_crosswalk_in_intersection",
                "pedestrian near crosswalk",
                "A pedestrian is near the crosswalk while the vehicle is passing through the intersection.",
                [RTA_ARTICLE_27],
                evidence,
                events,
            )

        if state.crosswalk_present:
            return _decision(
                DecisionLevel.CAUTION,
                "visible_crosswalk_in_intersection",
                "visible crosswalk",
                "A crosswalk is visible inside the intersection, so pedestrian risk remains relevant.",
                [RTA_ARTICLE_27],
                evidence,
                events,
            )

        return _decision(
            DecisionLevel.GO,
            "clear_right_turn_in_intersection",
            "no intersection risk basis",
            "No pedestrian or visible-crosswalk risk basis was observed inside the intersection.",
            [],
            evidence,
            events,
        )

    def decide_unprotected_left_approach(self, state, context: IntersectionContext) -> SafetyDecision:
        evidence = _evidence(state, context)
        events = _collect_compliance_events(state, context, evidence)
        if state.pedestrian_on_crosswalk:
            return _decision(
                DecisionLevel.STOP,
                "pedestrian_on_crosswalk_unprotected_left",
                "pedestrian on crosswalk",
                "A pedestrian is detected on a crosswalk during unprotected-left approach.",
                [RTA_ARTICLE_27, RTA_UNPROTECTED_LEFT_PLACEHOLDER],
                evidence,
                events,
            )
        return _decision(
            DecisionLevel.CAUTION,
            "unprotected_left_rules_placeholder",
            "unprotected-left rules incomplete",
            "Opposing-vehicle logic is not implemented yet, so this scenario stays conservative.",
            [RTA_UNPROTECTED_LEFT_PLACEHOLDER],
            evidence,
            events,
        )

    def decide_unprotected_left_in_intersection(self, state, context: IntersectionContext) -> SafetyDecision:
        evidence = _evidence(state, context)
        events = _collect_compliance_events(state, context, evidence)
        if state.pedestrian_on_crosswalk:
            return _decision(
                DecisionLevel.STOP,
                "pedestrian_on_crosswalk_unprotected_left_in_intersection",
                "pedestrian on crosswalk",
                "A pedestrian is detected on a crosswalk during unprotected-left traversal.",
                [RTA_ARTICLE_27, RTA_UNPROTECTED_LEFT_PLACEHOLDER],
                evidence,
                events,
            )
        return _decision(
            DecisionLevel.CAUTION,
            "unprotected_left_in_intersection_placeholder",
            "unprotected-left rules incomplete",
            "Opposing-vehicle tracking is not implemented yet, so this scenario stays conservative.",
            [RTA_UNPROTECTED_LEFT_PLACEHOLDER],
            evidence,
            events,
        )

    def decide_unknown(self, state, context: IntersectionContext, reason_code: str) -> SafetyDecision:
        return _decision(
            DecisionLevel.UNKNOWN,
            reason_code,
            "scenario or phase unknown",
            "Safety decision requires a known phase and maneuver scenario.",
            [],
            _evidence(state, context),
        )


def _evidence(state, context: IntersectionContext) -> dict[str, Any]:
    return {
        "phase": context.phase.value,
        "scenario": context.scenario.value,
        "vehicle_signal": context.vehicle_signal.value,
        "vehicle_signal_source": context.vehicle_signal_source,
        "pedestrian_signal": context.pedestrian_signal.value,
        "right_turn_signal": context.right_turn_signal.value,
        "stop_completed_on_red": context.stop_completed_on_red,
        "stop_completed_source": context.stop_completed_source,
        "stop_motion_score": round(float(context.stop_motion_score), 3),
        "stop_stable_frames": int(context.stop_stable_frames),
        "path_available": state.path_available,
        "path_source": state.path_source,
        "path_confidence": round(float(state.path_confidence), 3),
        "path_age_frames": state.path_age_frames,
        "crosswalk_present": state.crosswalk_present,
        "pedestrian_on_crosswalk": state.pedestrian_on_crosswalk,
        "pedestrian_near_crosswalk": state.pedestrian_near_crosswalk,
        "pedestrian_approaching_crosswalk": getattr(state, "pedestrian_approaching_crosswalk", False),
        "pedestrian_leaving_crosswalk": getattr(state, "pedestrian_leaving_crosswalk", False),
        "active_pedestrian_count": state.active_pedestrian_count,
        "near_pedestrian_count": state.near_pedestrian_count,
        "approaching_pedestrian_count": getattr(state, "approaching_pedestrian_count", 0),
        "leaving_pedestrian_count": getattr(state, "leaving_pedestrian_count", 0),
        "stationary_pedestrian_count": getattr(state, "stationary_pedestrian_count", 0),
        "unknown_motion_pedestrian_count": getattr(state, "unknown_motion_pedestrian_count", 0),
        "nearest_pedestrian_crosswalk_distance_px": getattr(
            state,
            "nearest_pedestrian_crosswalk_distance_px",
            None,
        ),
        "stop_line_present": state.stop_line_present,
        "stop_line_y_ratio": state.stop_line_y_ratio,
        "lane_role_rightmost": state.lane_role_rightmost,
        "lane_role_leftmost": state.lane_role_leftmost,
        "lane_role_confidence": round(float(state.lane_role_confidence), 3),
        "lane_role_left_score": round(float(getattr(state, "lane_role_left_score", 0.0)), 3),
        "lane_role_right_score": round(float(getattr(state, "lane_role_right_score", 0.0)), 3),
        "left_adjacent_vehicle": state.left_adjacent_vehicle,
        "right_adjacent_vehicle": state.right_adjacent_vehicle,
        "vehicle_count": state.vehicle_count,
        "pedestrian_count": state.pedestrian_count,
    }


def _decision(
    level: DecisionLevel,
    reason_code: str,
    short_reason: str,
    explanation: str,
    legal_basis: list[LegalBasis],
    evidence: dict[str, Any],
    compliance_events: list[ComplianceEvent] | None = None,
) -> SafetyDecision:
    return SafetyDecision(
        level=level,
        reason_code=reason_code,
        short_reason=short_reason,
        explanation=explanation,
        legal_basis=legal_basis,
        evidence=evidence,
        compliance_events=compliance_events or [],
    )


def _collect_compliance_events(state, context: IntersectionContext, evidence: dict[str, Any]) -> list[ComplianceEvent]:
    """Collect every simultaneously satisfied compliance-risk event."""
    events: list[ComplianceEvent] = []

    if context.scenario == Scenario.RIGHT_TURN:
        if context.phase == IntersectionPhase.APPROACH:
            if context.right_turn_signal == SignalState.RED:
                events.append(_event_right_turn_signal_red(evidence))
            if context.vehicle_signal == SignalState.RED and not context.stop_completed_on_red:
                events.append(_event_red_no_stop(evidence))
            if state.lane_role_rightmost is False and state.lane_role_confidence >= 0.45:
                events.append(_event_right_turn_lane(evidence))

        if state.pedestrian_on_crosswalk:
            events.append(_event_pedestrian_protection(evidence))
        elif state.pedestrian_near_crosswalk:
            events.append(_event_pedestrian_near(evidence))

    elif context.scenario == Scenario.UNPROTECTED_LEFT:
        if state.pedestrian_on_crosswalk:
            events.append(_event_pedestrian_protection(evidence))
        elif state.pedestrian_near_crosswalk:
            events.append(_event_pedestrian_near(evidence))

    return events


def _event_red_no_stop(evidence: dict[str, Any]) -> ComplianceEvent:
    return ComplianceEvent(
        code="RED_SIGNAL_NO_STOP_SUSPECTED",
        severity=ComplianceSeverity.VIOLATION_SUSPECTED,
        title="Red signal no-stop risk",
        description="Vehicle-signal red was provided, but stop completion was not confirmed.",
        legal_basis=[RTA_SIGNAL_OBEDIENCE],
        evidence=evidence,
    )


def _event_right_turn_signal_red(evidence: dict[str, Any]) -> ComplianceEvent:
    return ComplianceEvent(
        code="RIGHT_TURN_SIGNAL_RED_RISK",
        severity=ComplianceSeverity.VIOLATION_SUSPECTED,
        title="Right-turn red-signal risk",
        description="Dedicated right-turn signal was red while the maneuver was right turn.",
        legal_basis=[RTA_SIGNAL_OBEDIENCE],
        evidence=evidence,
    )


def _event_pedestrian_protection(evidence: dict[str, Any]) -> ComplianceEvent:
    return ComplianceEvent(
        code="PEDESTRIAN_PROTECTION_RISK",
        severity=ComplianceSeverity.VIOLATION_SUSPECTED,
        title="Pedestrian protection risk",
        description="A pedestrian was detected inside the visible crosswalk area.",
        legal_basis=[RTA_ARTICLE_27],
        evidence=evidence,
    )


def _event_pedestrian_near(evidence: dict[str, Any]) -> ComplianceEvent:
    return ComplianceEvent(
        code="PEDESTRIAN_NEAR_CROSSWALK_RISK",
        severity=ComplianceSeverity.WARNING,
        title="Pedestrian near-crosswalk risk",
        description="A pedestrian was detected near the crosswalk area.",
        legal_basis=[RTA_ARTICLE_27],
        evidence=evidence,
    )


def _event_right_turn_lane(evidence: dict[str, Any]) -> ComplianceEvent:
    return ComplianceEvent(
        code="RIGHT_TURN_LANE_RISK",
        severity=ComplianceSeverity.WARNING,
        title="Right-turn lane risk",
        description="Current lane was estimated not to be the rightmost lane.",
        legal_basis=[RTA_ARTICLE_25],
        evidence=evidence,
    )


def draw_safety_decision(vis, decision: SafetyDecision, context: IntersectionContext, phase_result=None, road_state=None) -> None:
    """Draw only the essential Korean safety information."""
    h, w = vis.shape[:2]
    x0 = max(6, w - 330)
    y0 = 30
    panel_w = min(324, w - x0 - 6)
    panel_h = 190

    colors = {
        DecisionLevel.STOP: (40, 40, 230),
        DecisionLevel.CAUTION: (0, 210, 255),
        DecisionLevel.GO: (60, 210, 60),
        DecisionLevel.UNKNOWN: (180, 180, 180),
    }
    color = colors.get(decision.level, (180, 180, 180))
    cv2.rectangle(vis, (x0, y0), (x0 + panel_w, y0 + panel_h), (0, 0, 0), -1)
    cv2.rectangle(vis, (x0, y0), (x0 + panel_w, y0 + panel_h), color, 1)

    event = _events_label(decision.compliance_events)
    law = _legal_label(decision.legal_basis[0].code) if decision.legal_basis else "해당 없음"
    phase = _phase_label(context.phase)
    scenario = _scenario_label(context.scenario)
    signal = _signal_display_label(context)
    stop = _stop_display_label(context)
    reason = _reason_label(decision.reason_code)
    phase_hint = _phase_hint(phase_result)
    lines = [
        (f"판단: {_level_label(decision.level)}", color),
        (f"상황: {phase} / {scenario} / 신호 {signal}", (235, 235, 235)),
        (f"정지: {stop}", (235, 235, 235)),
        (f"이유: {reason}", (235, 235, 235)),
        (f"근거: {law}", (235, 235, 235)),
        (f"위반의심: {event}", (235, 235, 235)),
    ]
    if phase_hint:
        lines.append((phase_hint, (210, 255, 255)))
    lane_hint = _lane_hint(road_state)
    if lane_hint:
        lines.append((lane_hint, (210, 255, 210)))
    ped_motion_hint = _pedestrian_motion_hint(road_state)
    if ped_motion_hint:
        lines.append((ped_motion_hint, (255, 230, 180)))

    draw_korean_lines(vis, lines, (x0 + 10, y0 + 9), font_size=14, line_gap=19)


def _level_label(level: DecisionLevel) -> str:
    return {
        DecisionLevel.STOP: "정지",
        DecisionLevel.CAUTION: "주의",
        DecisionLevel.GO: "진행 가능",
        DecisionLevel.UNKNOWN: "판단 불가",
    }.get(level, "판단 불가")


def _phase_label(phase: IntersectionPhase) -> str:
    return {
        IntersectionPhase.APPROACH: "교차로 진입 전",
        IntersectionPhase.IN_INTERSECTION: "교차로 내부",
        IntersectionPhase.UNKNOWN: "알 수 없음",
    }.get(phase, "알 수 없음")


def _scenario_label(scenario: Scenario) -> str:
    return {
        Scenario.RIGHT_TURN: "우회전",
        Scenario.UNPROTECTED_LEFT: "비보호좌회전",
        Scenario.STRAIGHT: "직진",
        Scenario.UNKNOWN: "알 수 없음",
    }.get(scenario, "알 수 없음")


def _signal_label(signal: SignalState) -> str:
    return {
        SignalState.RED: "적색",
        SignalState.YELLOW: "황색",
        SignalState.GREEN: "녹색",
        SignalState.UNKNOWN: "알 수 없음",
        SignalState.NONE: "없음",
    }.get(signal, "알 수 없음")


def _signal_display_label(context: IntersectionContext) -> str:
    if context.phase == IntersectionPhase.IN_INTERSECTION:
        return "판단 제외"
    if context.vehicle_signal == SignalState.RED and context.vehicle_signal_source == "approach_red_hold":
        return "적색(유지)"
    return _signal_label(context.vehicle_signal)


def _stop_display_label(context: IntersectionContext) -> str:
    if context.vehicle_signal != SignalState.RED or context.phase != IntersectionPhase.APPROACH:
        return "판단 제외"
    if context.stop_completed_on_red:
        if context.stop_completed_source == "auto_motion":
            return f"완료({context.stop_stable_frames}f)"
        if context.stop_completed_source == "manual":
            return "완료(수동)"
        return "완료"
    return f"미완료({context.stop_motion_score:.2f})"


def _legal_label(code: str) -> str:
    return {
        "RTA_ART_25": "도로교통법 제25조",
        "RTA_ART_27": "도로교통법 제27조",
        "RTA_SIGNAL": "신호 준수 의무",
        "RTA_UNPROTECTED_LEFT_TBD": "비보호좌회전 기준 추가 예정",
    }.get(code, code)


def _event_label(code: str) -> str:
    return {
        "RED_SIGNAL_NO_STOP_SUSPECTED": "적색 일시정지 미확인",
        "RIGHT_TURN_SIGNAL_RED_RISK": "우회전 적색 신호",
        "PEDESTRIAN_PROTECTION_RISK": "보행자 보호 위험",
        "PEDESTRIAN_NEAR_CROSSWALK_RISK": "횡단보도 주변 보행자",
        "RIGHT_TURN_LANE_RISK": "우회전 차로 확인 필요",
    }.get(code, code)


def _events_label(events: list[ComplianceEvent]) -> str:
    if not events:
        return "없음"
    labels = [_event_label(event.code) for event in events[:3]]
    if len(events) > 3:
        labels.append(f"+{len(events) - 3}")
    return ", ".join(labels)


def _reason_label(reason_code: str) -> str:
    return {
        "right_turn_signal_red": "우회전 신호가 적색",
        "red_signal_stop_required": "적색 신호 일시정지 필요",
        "pedestrian_on_crosswalk": "횡단보도 위 보행자",
        "pedestrian_near_crosswalk": "횡단보도 주변 보행자",
        "right_turn_not_rightmost_lane": "우회전 차로 불확실",
        "structure_uncertain": "횡단보도 주변 구조 불확실",
        "approaching_crosswalk": "횡단보도 접근 중",
        "clear_right_turn_approach": "위험 근거 없음",
        "pedestrian_on_crosswalk_in_intersection": "교차로 내 횡단보도 보행자",
        "pedestrian_near_crosswalk_in_intersection": "교차로 내 횡단보도 주변 보행자",
        "visible_crosswalk_in_intersection": "교차로 내 횡단보도 확인",
        "clear_right_turn_in_intersection": "교차로 내부 위험 근거 없음",
        "unprotected_left_rules_placeholder": "비보호좌회전 기준 미완성",
        "unprotected_left_in_intersection_placeholder": "비보호좌회전 기준 미완성",
    }.get(reason_code, reason_code)


def _phase_hint(phase_result) -> str:
    if phase_result is None:
        return ""
    bottom = "-" if phase_result.crosswalk_bottom_y_ratio is None else f"{phase_result.crosswalk_bottom_y_ratio:.2f}"
    state = {
        "UNKNOWN": "알 수 없음",
        "APPROACH": "접근 중",
        "NEAR_CROSSWALK": "횡단보도 근접",
        "IN_INTERSECTION": "교차로 내부",
    }.get(getattr(phase_result.internal_state, "value", ""), getattr(phase_result.internal_state, "value", ""))
    return f"자동단계: {state}  횡단보도위치:{bottom}"


def _lane_hint(road_state) -> str:
    if road_state is None:
        return ""
    left = _tri_state_label(getattr(road_state, "lane_role_leftmost", None))
    right = _tri_state_label(getattr(road_state, "lane_role_rightmost", None))
    conf = float(getattr(road_state, "lane_role_confidence", 0.0))
    left_score = float(getattr(road_state, "lane_role_left_score", 0.0))
    right_score = float(getattr(road_state, "lane_role_right_score", 0.0))
    adj = []
    if getattr(road_state, "left_adjacent_vehicle", False):
        adj.append("좌측차량")
    if getattr(road_state, "right_adjacent_vehicle", False):
        adj.append("우측차량")
    adj_text = ",".join(adj) if adj else "없음"
    source = _lane_role_source_label(getattr(road_state, "lane_role_source", "unavailable"))
    age = int(getattr(road_state, "lane_role_age_frames", 0))
    age_text = f"+{age}f" if source == "진입전" and age > 0 else ""
    return (
        f"차로: 좌 {left}({left_score:.2f}) / 우 {right}({right_score:.2f}) "
        f"/ {source}{age_text} / 신뢰 {conf:.2f} / 옆차량 {adj_text}"
    )


def _pedestrian_motion_hint(road_state) -> str:
    if road_state is None:
        return ""
    approaching = int(getattr(road_state, "approaching_pedestrian_count", 0))
    leaving = int(getattr(road_state, "leaving_pedestrian_count", 0))
    stationary = int(getattr(road_state, "stationary_pedestrian_count", 0))
    unknown = int(getattr(road_state, "unknown_motion_pedestrian_count", 0))
    if approaching + leaving + stationary + unknown <= 0:
        return ""
    nearest = getattr(road_state, "nearest_pedestrian_crosswalk_distance_px", None)
    nearest_text = "-" if nearest is None else f"{float(nearest):.0f}px"
    return (
        f"보행자이동: 접근 {approaching} / 이탈 {leaving} / 정지 {stationary} "
        f"/ 불명 {unknown} / 거리 {nearest_text}"
    )


def _lane_role_source_label(source: str) -> str:
    return {
        "current": "현재",
        "cached_before_intersection": "진입전",
        "unavailable": "없음",
    }.get(source, source)


def _tri_state_label(value) -> str:
    if value is True:
        return "예"
    if value is False:
        return "아니오"
    return "모름"
