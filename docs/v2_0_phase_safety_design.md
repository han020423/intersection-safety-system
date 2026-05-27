# v2_0 Phase-Based Safety Decision Design

이 문서는 `v2_0` 리부트 작업에서 다음 단계로 구현할 **교차로 단계 분리형 안전 판단 구조**를 정리한다.

목표는 단순한 `STOP / CAUTION / GO` 표시가 아니라, 다음 질문에 답할 수 있는 판단 엔진을 만드는 것이다.

```text
1. 지금 교차로 진입 전인가, 통과 중인가?
2. 어떤 요소 때문에 멈추거나 주의해야 하는가?
3. 도로교통법상 어떤 의무와 연결되는가?
4. 모델이 실제로 본 근거는 무엇인가?
```

---

## 1. 현재 v2_0의 위치

`edge/inference/v2_0`은 기존 판단 로직을 그대로 재사용하지 않고 새로 만든 인지/구조화 단계다.

현재 구현된 핵심 요소:

```text
YOLO 객체 검출
road_v4 BiSeNet segmentation
ego lane reboot
lane role 판단
관심주행영역(path corridor)
crosswalk active/near zone
보행자 foot-point 기반 active/near 판정
```

현재 아직 없는 것:

```text
RoadStructureState
IntersectionPhase
IntersectionContext
SafetyDecision
도로교통법 근거 기반 설명
STOP / CAUTION / GO 판단
```

따라서 다음 작업은 **화면에 그려지는 인식 결과를 판단 가능한 상태값으로 정리하고, 교차로 단계별 규칙으로 판단하는 것**이다.

---

## 2. 이전 프로그램은 어떻게 했는가

이전 프로그램의 주요 파일:

```text
edge/inference/intersection_demo.py
edge/inference/postprocess.py
edge/inference/state_machine.py
edge/inference/ARCHITECTURE.md
edge/inference/v2/intersection_demo_v2.py
edge/inference/v2/road_v4_postprocess.py
```

이전 전체 흐름:

```text
YOLO + BiSeNet
-> postprocess.py / road_v4_postprocess.py
-> SceneContext
-> IntersectionFSM
-> STOP / CAUTION / GO
```

### 2.1 이전 FSM 상태

이전 `IntersectionFSM`은 다음 5개 상태를 사용했다.

```text
LANE_TRACKING
CROSSWALK_APPROACH
ENTERING_INTERSECTION
INTERSECTION_TRACKING
RELOCK_LANE
```

의미:

| 이전 상태 | 의미 | 현재 설계에서의 대응 |
|---|---|---|
| `LANE_TRACKING` | 일반 차선 추적 | `APPROACH` 이전 또는 일반 주행 |
| `CROSSWALK_APPROACH` | 횡단보도/정지선 접근 | `APPROACH` |
| `ENTERING_INTERSECTION` | 교차로 진입 시작 | `APPROACH -> IN_INTERSECTION` 전환 구간 |
| `INTERSECTION_TRACKING` | 교차로 내부 주행 | `IN_INTERSECTION` |
| `RELOCK_LANE` | 통과 후 차선 재확보 | `EXITING` 또는 일반 주행 복귀 |

### 2.2 이전 상태 전이 방식

이전 프로그램은 phase를 수동 입력받지 않고, 영상 인식 결과로 자동 추정했다.

```text
CROSSWALK_APPROACH 진입:
- crosswalk zone 픽셀이 증가
- stop line이 화면 하단 쪽에 보임

ENTERING_INTERSECTION 진입:
- stop_line_y_ratio > 0.85
- crosswalk/stop line이 사라짐
- lane confidence가 낮아짐

INTERSECTION_TRACKING 진입:
- lane confidence가 여러 프레임 낮음

RELOCK_LANE 진입:
- lane confidence 회복
```

좋은 점:

```text
교차로 접근/진입/내부/재확보라는 시간 흐름을 추적했다.
단일 프레임 튐을 막기 위해 low/high confidence count를 사용했다.
SceneContext로 판단 입력값을 한 번 모았다.
```

문제점:

```text
진입 전 판단과 통과 중 판단이 함수 단위로 분리되지 않았다.
모든 판단이 _decide() 하나의 우선순위 규칙에 섞여 있었다.
near crosswalk 보행자를 무조건 STOP으로 처리해 과도했다.
우회전/비보호좌회전 scenario가 명확히 분리되지 않았다.
도로교통법 근거와 모델 감지 근거가 구조화되어 있지 않았다.
path corridor가 centerline 기준 고정폭이라 도로 구조 변화에 약했다.
```

---

## 3. 새 설계의 핵심

새 구조는 이전 FSM의 “단계 추적 아이디어”만 가져오고, 판단 엔진은 다시 설계한다.

권장 흐름:

```text
YOLO + BiSeNet + EgoLane
-> RoadStructureState
-> IntersectionContext
-> Phase-specific SafetyRule
-> SafetyDecision
```

중요한 분리:

```text
RoadStructureState:
  모델이 본 도로 구조와 객체 관계만 정리한다.
  STOP/CAUTION을 직접 말하지 않는다.

IntersectionContext:
  지금 어떤 교차로 상황인지 정리한다.
  예: 진입 전, 통과 중, 우회전, 비보호좌회전, 신호 상태.

SafetyDecision:
  RoadStructureState + IntersectionContext를 보고 판단한다.
  법적 근거, 판단 이유, 감지 근거를 같이 보관한다.
```

---

## 4. IntersectionPhase

처음에는 2단계만 필수로 구현한다.

```python
class IntersectionPhase(Enum):
    APPROACH = "APPROACH"
    IN_INTERSECTION = "IN_INTERSECTION"
    UNKNOWN = "UNKNOWN"
```

추후 필요하면 추가:

```python
EXITING = "EXITING"
```

### 4.1 APPROACH

의미:

```text
교차로 또는 횡단보도에 진입하기 전.
핵심 질문은 "지금 진입해도 되는가?"이다.
```

주요 판단 요소:

```text
차량 신호
정지선
진입 전 횡단보도
보행자 active/near
현재 차로 위치
우회전/비보호좌회전 의도
```

대표 판단:

```text
전방 신호가 적색인가?
정지선/횡단보도 전에 일시정지해야 하는가?
관심주행영역 위 횡단보도에 보행자가 있는가?
횡단보도 주변 보행자가 통행하려는 상황인가?
우회전인데 오른쪽 차로 조건을 만족하는가?
비보호좌회전인데 왼쪽 차로 조건을 만족하는가?
```

### 4.2 IN_INTERSECTION

의미:

```text
이미 교차로 내부에 진입한 상태.
핵심 질문은 "계속 진행해도 되는가?"이다.
```

주요 판단 요소:

```text
회전 후 만나게 되는 횡단보도
진행 방향 보행자
비보호좌회전 시 대향 차량
우회전 시 우측/전방 보행자
교차로 내부에서 진행 경로를 막는 객체
관심주행영역 신뢰도
```

주의:

```text
교차로 내부에서는 차선이 사라지는 것이 자연스러울 수 있다.
따라서 path unavailable 자체만으로 STOP을 만들면 안 된다.
다만 판단 신뢰도 저하 또는 CAUTION 근거는 될 수 있다.
```

---

## 5. Scenario

교차로 phase와 별개로 “무슨 행동을 하는 중인가”를 분리한다.

```python
class Scenario(Enum):
    RIGHT_TURN = "RIGHT_TURN"
    UNPROTECTED_LEFT = "UNPROTECTED_LEFT"
    STRAIGHT = "STRAIGHT"
    UNKNOWN = "UNKNOWN"
```

처음 구현은 수동 옵션으로 시작한다.

```bash
--phase approach
--phase in_intersection
--scenario right_turn
--scenario unprotected_left
--scenario straight
```

나중에 방향지시등/CAN 입력이 들어오면:

```text
right turn signal -> Scenario.RIGHT_TURN 후보
left turn signal  -> Scenario.UNPROTECTED_LEFT 후보
```

---

## 6. RoadStructureState

`RoadStructureState`는 인식 결과를 안전 판단이 읽기 쉽게 요약한 구조체다.

초기 필드 제안:

```python
@dataclass
class RoadStructureState:
    frame_index: int = 0

    path_available: bool = False
    path_source: str = "unavailable"      # lane / cached_lane / unavailable
    path_confidence: float = 0.0
    path_age_frames: int = 0

    crosswalk_present: bool = False
    crosswalk_on_path: bool = False
    crosswalk_near_path: bool = False
    crosswalk_confidence: float = 0.0

    active_pedestrian_count: int = 0
    near_pedestrian_count: int = 0
    pedestrian_on_active_crosswalk: bool = False
    pedestrian_near_crosswalk: bool = False

    stop_line_present: bool = False
    stop_line_y_ratio: Optional[float] = None

    lane_role_leftmost: Optional[bool] = None
    lane_role_rightmost: Optional[bool] = None
    left_adjacent_vehicle: bool = False
    right_adjacent_vehicle: bool = False

    object_vehicle_count: int = 0
    object_pedestrian_count: int = 0
```

필드 의미:

| 필드 | 의미 |
|---|---|
| `path_available` | 관심주행영역을 쓸 수 있는가 |
| `path_source` | `lane`, `cached_lane`, `unavailable` |
| `path_confidence` | 관심주행영역 신뢰도 |
| `crosswalk_present` | 화면에 횡단보도 mask가 있는가 |
| `crosswalk_on_path` | 횡단보도가 관심주행영역과 직접 겹치는가 |
| `pedestrian_on_active_crosswalk` | 보행자가 active zone에 있는가 |
| `pedestrian_near_crosswalk` | 보행자가 near zone에 있는가 |
| `stop_line_present` | 정지선이 보이는가 |
| `lane_role_leftmost/rightmost` | 현재 차로가 좌/우 끝 차로인지 |

생성 입력:

```text
PathCorridor
CrosswalkZone
CrosswalkPedestrianStatus
EgoLaneResult.role
seg_mask stop_line class
YOLO detections
```

---

## 7. IntersectionContext

`IntersectionContext`는 판단 상황을 설명한다.

```python
@dataclass
class IntersectionContext:
    phase: IntersectionPhase = IntersectionPhase.UNKNOWN
    scenario: Scenario = Scenario.UNKNOWN

    vehicle_signal: SignalState = SignalState.UNKNOWN
    pedestrian_signal: SignalState = SignalState.UNKNOWN
    turn_signal: TurnSignal = TurnSignal.UNKNOWN

    stop_completed_on_red: bool = False
    manual_phase: bool = True
    manual_scenario: bool = True
```

초기에는 대부분 수동 입력으로 둔다.

이유:

```text
자동 phase 추정은 오탐이 많을 수 있다.
방향지시등/CAN 입력 전에는 scenario 자동 판단이 약하다.
먼저 phase별 판단 규칙이 맞는지 검증해야 한다.
```

---

## 8. SafetyDecision

판단 결과는 단순 문자열이 아니라, 법적 근거와 감지 근거를 함께 담아야 한다.

```python
class DecisionLevel(Enum):
    STOP = "STOP"
    CAUTION = "CAUTION"
    GO = "GO"
    UNKNOWN = "UNKNOWN"


@dataclass
class LegalBasis:
    code: str
    title: str
    summary: str
    needs_exact_review: bool = False


@dataclass
class SafetyDecision:
    level: DecisionLevel
    reason_code: str
    short_reason: str
    explanation: str
    legal_basis: list[LegalBasis]
    evidence: dict
```

예:

```python
SafetyDecision(
    level=DecisionLevel.STOP,
    reason_code="pedestrian_on_active_crosswalk",
    short_reason="횡단보도 위 보행자",
    explanation=(
        "관심주행영역과 겹치는 횡단보도에 보행자가 있어 "
        "보행자 통행을 방해할 수 있으므로 일시정지해야 합니다."
    ),
    legal_basis=[RTA_ARTICLE_27_PEDESTRIAN_PROTECTION],
    evidence={
        "crosswalk_on_path": True,
        "active_pedestrian_count": 1,
        "path_source": "lane",
        "path_confidence": 0.74,
    },
)
```

---

## 9. 법적 근거 코드

정확한 조문 번호와 문구는 최종 구현 전에 국가법령정보센터 기준으로 다시 확인해야 한다.

초기 코드 설계:

```python
RTA_ARTICLE_27_1 = LegalBasis(
    code="RTA_ART_27_1",
    title="도로교통법 제27조 제1항: 보행자 보호",
    summary="횡단보도를 통행 중이거나 통행하려는 보행자가 있으면 횡단보도 앞에서 일시정지해야 함.",
    needs_exact_review=True,
)

RTA_ARTICLE_27_2 = LegalBasis(
    code="RTA_ART_27_2",
    title="도로교통법 제27조 제2항: 교차로 회전 시 보행자 보호",
    summary="교차로에서 좌회전 또는 우회전할 때 보행자의 통행을 방해해서는 안 됨.",
    needs_exact_review=True,
)

RTA_SIGNAL_OBEDIENCE = LegalBasis(
    code="RTA_SIGNAL",
    title="도로교통법 신호 준수 의무",
    summary="차량 신호가 적색이면 정지선, 횡단보도 또는 교차로 직전에 정지해야 함.",
    needs_exact_review=True,
)
```

주의:

```text
법적 근거는 코드에 하드코딩하더라도 summary는 짧게 둔다.
최종 발표/문서에는 반드시 실제 조문 원문과 날짜를 확인한다.
```

---

## 10. Phase별 판단 규칙

### 10.1 APPROACH 규칙

진입 전에는 “진입해도 되는가?”를 판단한다.

우선순위 제안:

| 우선순위 | 조건 | 판단 | reason_code |
|---:|---|---|---|
| 1 | `vehicle_signal == RED` and not `stop_completed_on_red` | STOP | `red_signal_stop_required` |
| 2 | `crosswalk_on_path` and `pedestrian_on_active_crosswalk` | STOP | `pedestrian_on_active_crosswalk` |
| 3 | `crosswalk_present` and `pedestrian_near_crosswalk` | CAUTION | `pedestrian_near_crosswalk` |
| 4 | `scenario == RIGHT_TURN` and `lane_role_rightmost is False` | CAUTION | `right_turn_not_rightmost_lane` |
| 5 | `scenario == UNPROTECTED_LEFT` and `lane_role_leftmost is False` | CAUTION | `left_turn_not_leftmost_lane` |
| 6 | `crosswalk_on_path` and no pedestrian | CAUTION | `approaching_crosswalk` |
| 7 | path unavailable but crosswalk/stopline exists | CAUTION | `structure_uncertain` |
| 8 | no risk evidence | GO | `clear_approach` |

설명 예:

```text
STOP / pedestrian_on_active_crosswalk
차량 진행 경로와 겹치는 횡단보도 영역에 보행자가 있어 보행자 보호 의무가 발생합니다.
```

```text
CAUTION / pedestrian_near_crosswalk
보행자가 횡단보도 주변 영역에 있어 통행하려는 상황일 수 있습니다.
아직 관심주행영역 위 active zone은 아니므로 STOP 확정보다는 주의로 표시합니다.
```

### 10.2 IN_INTERSECTION 규칙

통과 중에는 “계속 진행해도 되는가?”를 판단한다.

우선순위 제안:

| 우선순위 | 조건 | 판단 | reason_code |
|---:|---|---|---|
| 1 | `pedestrian_on_active_crosswalk` | STOP | `pedestrian_in_turn_crosswalk` |
| 2 | `scenario in (RIGHT_TURN, UNPROTECTED_LEFT)` and `pedestrian_near_crosswalk` | CAUTION | `pedestrian_near_exit_crosswalk` |
| 3 | `scenario == UNPROTECTED_LEFT` and oncoming vehicle risk | CAUTION/STOP | `oncoming_vehicle_conflict` |
| 4 | path unavailable and crosswalk present | CAUTION | `path_uncertain_in_intersection` |
| 5 | no risk evidence | GO/UNKNOWN | `clear_in_intersection` |

주의:

```text
IN_INTERSECTION에서 path unavailable은 흔한 상황이다.
따라서 이것만으로 STOP하지 않는다.
보행자, 횡단보도, 대향 차량 등 실제 충돌 가능 근거와 함께 CAUTION으로 처리한다.
```

---

## 11. 이전 FSM과 새 설계 비교

| 항목 | 이전 프로그램 | 새 v2_0 설계 |
|---|---|---|
| 단계 추적 | 5-state FSM | `APPROACH / IN_INTERSECTION` 중심 |
| 판단 함수 | `_decide()` 하나 | phase별 rule 함수 |
| 법적 근거 | reason 문자열만 있음 | `LegalBasis` 구조화 |
| 보행자 near | 대부분 STOP | 보수적으로 CAUTION 우선 |
| path corridor | centerline 고정폭 | ego lane 기반 관심주행영역 + short cache |
| scenario | 우회전 중심, 약함 | `RIGHT_TURN / UNPROTECTED_LEFT / STRAIGHT` 분리 |
| 설명 가능성 | 낮음 | reason + law + evidence |

---

## 12. 구현 파일 제안

새 코드는 `edge/inference/v2_0` 안에서 기존 reboot 모듈과 분리한다.

```text
edge/inference/v2_0/road_state_reboot.py
  - RoadStructureState
  - build_road_structure_state()
  - stop_line 추출 요약

edge/inference/v2_0/intersection_context_reboot.py
  - IntersectionPhase
  - Scenario
  - SignalState
  - IntersectionContext
  - CLI 문자열 parse helper

edge/inference/v2_0/safety_decision_reboot.py
  - DecisionLevel
  - LegalBasis
  - SafetyDecision
  - SafetyDecisionEngine
  - apply_approach_rules()
  - apply_in_intersection_rules()

edge/inference/v2_0/legal_basis_kr.py
  - 도로교통법 근거 상수
```

`intersection_demo_v2_0.py` 연결 흐름:

```text
ego_lane = ego_lane_pipeline.update(...)
path_corridor = path_corridor_pipeline.update(...)
crosswalk_zone = estimate_crosswalk_zone(...)
crosswalk_peds = evaluate_crosswalk_pedestrians(...)

road_state = build_road_structure_state(
    ego_lane,
    path_corridor,
    crosswalk_zone,
    crosswalk_peds,
    seg_mask,
    detections,
)

context = build_intersection_context_from_args(args)
decision = safety_engine.decide(road_state, context)

draw_road_structure(...)
draw_safety_decision(...)
```

---

## 13. CLI 옵션 제안

초기 수동 검증용:

```bash
--phase approach
--phase in_intersection
--phase unknown

--scenario right_turn
--scenario unprotected_left
--scenario straight
--scenario unknown

--vehicle-signal red
--vehicle-signal yellow
--vehicle-signal green
--vehicle-signal unknown

--pedestrian-signal red
--pedestrian-signal green
--pedestrian-signal unknown

--stop-completed-on-red
--no-safety-decision
```

처음에는 자동 phase 추정 옵션을 넣지 않는다.

이유:

```text
판단 규칙 검증과 phase 자동 추정을 동시에 하면 디버깅이 어려워진다.
먼저 수동 phase/scenario로 규칙이 맞는지 확인한다.
```

---

## 14. 화면 표시 제안

시각화는 짧고 명확해야 한다.

상단 판단 뱃지:

```text
STOP | APPROACH | RIGHT_TURN
횡단보도 위 보행자
근거: 도로교통법 제27조 보행자 보호
```

하단 디버그:

```text
path:lane pconf:0.68 age:0
crosswalk:on_path activeP:1 nearP:0
law:RTA_ART_27_1 reason:pedestrian_on_active_crosswalk
```

로그/JSON은 자세히:

```json
{
  "decision": "STOP",
  "phase": "APPROACH",
  "scenario": "RIGHT_TURN",
  "reason_code": "pedestrian_on_active_crosswalk",
  "legal_basis": ["RTA_ART_27_1"],
  "evidence": {
    "path_source": "lane",
    "path_confidence": 0.68,
    "crosswalk_on_path": true,
    "active_pedestrian_count": 1,
    "near_pedestrian_count": 0
  }
}
```

---

## 15. 구현 순서

권장 순서:

```text
1. RoadStructureState dataclass 추가
2. PathCorridor/CrosswalkZone/CrosswalkPedestrianStatus를 RoadStructureState로 요약
3. stop_line_present, stop_line_y_ratio 추출 추가
4. IntersectionPhase, Scenario, SignalState, IntersectionContext 추가
5. CLI 옵션으로 phase/scenario/signal 입력
6. SafetyDecision, LegalBasis, SafetyDecisionEngine 추가
7. APPROACH 규칙만 먼저 구현
8. IN_INTERSECTION 규칙 구현
9. 화면에 decision/reason/legal/evidence 표시
10. 테스트 영상 기준표 작성
```

중요 원칙:

```text
인식/구조화 단계에서는 STOP/CAUTION을 만들지 않는다.
법적 판단은 SafetyDecisionEngine에서만 만든다.
phase별 규칙은 반드시 함수로 분리한다.
모르는 상태는 GO가 아니라 UNKNOWN 또는 CAUTION 쪽으로 처리한다.
near crosswalk는 무조건 STOP이 아니라 phase/scenario에 따라 CAUTION부터 시작한다.
```

---

## 16. 테스트 기준표 필요

나중에 구현 후 다음 표를 만들어야 한다.

| video | frame range | phase | scenario | expected | reason |
|---|---:|---|---|---|---|
| `v4.mp4` | TBD | APPROACH | RIGHT_TURN | CAUTION/STOP | pedestrian/crosswalk |
| `v4.mp4` | TBD | IN_INTERSECTION | RIGHT_TURN | CAUTION | path/crosswalk uncertainty |
| TBD | TBD | APPROACH | UNPROTECTED_LEFT | CAUTION | lane role / oncoming vehicle |

이 기준표가 없으면 화면을 보며 감으로 튜닝하게 된다.

---

## 17. 최종 방향

이전 프로그램은 교차로 상태 FSM을 갖고 있었지만, 판단 규칙이 한 함수에 섞여 있고 법적 설명이 약했다.

v2_0에서는 다음 방향으로 간다.

```text
이전 FSM의 단계 추적 아이디어는 참고한다.
판단 로직은 phase별로 새로 작성한다.
도로교통법 근거와 감지 근거를 SafetyDecision에 구조화한다.
진입 전은 "진입 허용 판단"으로 본다.
통과 중은 "계속 진행 가능 판단"으로 본다.
```

