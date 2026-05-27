# v3 위험 판단 기준표

이 문서는 발표, 보고서, 코드 유지보수를 위해 현재 `v3` 안전 판단 로직을 정리한 기준표다.
카메라 기반 인식 결과이므로 모든 법규 관련 표현은 "위반 확정"이 아니라 "위반 의심", "법규 리스크", "보호 필요"로 해석한다.

## 1. 전체 판단 흐름

```text
YOLO 객체 검출 + BiSeNet 도로 구조 segmentation
-> RoadStructureState 생성
-> IntersectionPhase 결정
-> IntersectionContext 생성
-> SafetyDecisionEngine 판단
-> 화면 표시 + 음성 안내 + 교차로 단위 이벤트 저장
```

판단 엔진은 `STOP / CAUTION / GO / UNKNOWN` 중 하나를 출력한다.

| 단계 | 입력 | 역할 | 출력 |
|---|---|---|---|
| 인식 | 영상 프레임 | 차량, 보행자, 신호등, 횡단보도, 차선 검출 | detections, seg_mask |
| 도로 상태 요약 | ego lane, path corridor, crosswalk, detections | 판단에 필요한 값만 구조화 | `RoadStructureState` |
| 교차로 단계 | crosswalk 위치/변화 | 진입 전/교차로 내부 구분 | `IntersectionPhase` |
| 상황 정보 | 단계, 우회전/좌회전, 신호, 정지완료 | 규칙 판단 조건 구성 | `IntersectionContext` |
| 안전 판단 | state + context | 위험도/이유/근거/이벤트 생성 | `SafetyDecision` |
| 기록/안내 | decision + evidence | 운전자 안내, 로그, 스냅샷 저장 | voice, JSONL, JPG |

## 2. 판단 등급 정의

| 등급 | 의미 | 운전자 안내 의미 | 로그 의미 |
|---|---|---|---|
| `STOP` | 즉시 정지 또는 정지 유지가 필요한 상황 | 멈춰야 함 | 위험 이벤트 또는 보호 필요 기록 |
| `CAUTION` | 통과 가능성을 배제하지 않지만 주의가 필요한 상황 | 서행/확인 필요 | 의미 있는 주의 상황은 기록 |
| `GO` | 현재 인식 근거상 위험 조건이 없음 | 진행 가능 | 보통 위험 로그 없음 |
| `UNKNOWN` | 단계/상황이 불명확해 판단 불가 | 수동 확인 필요 | 판단 불가 상태 |

## 3. 주요 상태값 정의

| 값 | 의미 | 생성 근거 | 판단에서 쓰임 |
|---|---|---|---|
| `phase` | 교차로 진입 전/내부/불명 | 횡단보도 mask의 위치와 변화 | 신호 판단 적용 범위 결정 |
| `scenario` | 우회전/비보호좌회전/직진/불명 | CLI 또는 추후 방향지시등 입력 | 규칙 분기 |
| `vehicle_signal` | 차량 신호 상태 | YOLO 신호등 crop 색상 추정 + smoothing | APPROACH 적색 정지 판단 |
| `right_turn_signal` | 우회전 전용 신호 | 현재는 수동/옵션 중심 | 우회전 적색 정지 판단 |
| `stop_completed_on_red` | 적색 신호에서 정지 완료 여부 | 하단 도로 영역 optical flow/프레임 움직임 | 적색 미정지 의심 이벤트 |
| `crosswalk_present` | 관련 횡단보도 보임 | BiSeNet crosswalk mask | 보행자 보호/주의 판단 |
| `pedestrian_on_crosswalk` | 보행자가 횡단보도 내부에 있음 | 보행자 bbox 발 위치와 crosswalk mask | STOP |
| `pedestrian_near_crosswalk` | 보행자가 횡단보도 주변에 있음 | bbox 발 위치와 crosswalk 주변 거리 | CAUTION |
| `pedestrian_approaching_crosswalk` | 보행자가 횡단보도 쪽으로 접근 중 | 거리 변화 + 이동 벡터 방향 | 음성/설명 근거 |
| `lane_role_rightmost` | 현재 차로가 우측 끝 차로인지 | 차선 색상/주변차선/옆 차량/캐시 점수 | 우회전 차로 리스크 |
| `path_available` | 관심주행영역 생성 가능 여부 | ego lane 기반 path corridor | 구조 불확실 판단 보조 |

## 4. 우회전 APPROACH 판단 기준

`APPROACH`는 교차로에 진입하기 전 상태다. 이 단계에서는 신호 판단을 강하게 적용한다.
아래 표는 우선순위 순서다. 위 조건이 만족되면 아래 조건보다 먼저 최종 판단 사유가 된다.
단, `ComplianceEvent`는 동시에 만족되는 위험을 모두 수집한다.

| 우선순위 | 조건 | 판단 | reason_code | 감지 근거 | 법적 근거 | 이벤트 코드 | 음성 안내 |
|---:|---|---|---|---|---|---|---|
| 1 | `right_turn_signal == RED` | `STOP` | `right_turn_signal_red` | 우회전 전용 신호 적색 | 신호 준수 의무 | `RIGHT_TURN_SIGNAL_RED_RISK` | 우회전 신호가 적색입니다. 정지하세요. |
| 2 | `vehicle_signal == RED` 그리고 `stop_completed_on_red == False` | `STOP` | `red_signal_stop_required` | 차량 신호 적색 + 정지 완료 미확인 | 신호 준수 의무 | `RED_SIGNAL_NO_STOP_SUSPECTED` | 적색 신호입니다. 일시정지하세요. |
| 3 | `pedestrian_on_crosswalk == True` | `STOP` | `pedestrian_on_crosswalk` | 보행자 발 위치가 횡단보도 내부 | 도로교통법 제27조 | `PEDESTRIAN_PROTECTION_RISK` | 횡단보도에 보행자가 있습니다. 정지하세요. |
| 4 | `pedestrian_near_crosswalk == True` | `CAUTION` | `pedestrian_near_crosswalk` | 보행자가 횡단보도 주변 | 도로교통법 제27조 | `PEDESTRIAN_NEAR_CROSSWALK_RISK` | 횡단보도 주변 보행자에 주의하세요. |
| 5 | `lane_role_rightmost == False` 그리고 `lane_role_confidence >= 0.45` | `CAUTION` | `right_turn_not_rightmost_lane` | 우측 끝 차로가 아닐 가능성 | 도로교통법 제25조 | `RIGHT_TURN_LANE_RISK` | 우회전 차로가 아닐 수 있습니다. 차로를 확인하세요. |
| 6 | `crosswalk_present == True` 그리고 `path_available == False` | `CAUTION` | `structure_uncertain` | 횡단보도는 보이나 주행 구조가 불확실 | 도로교통법 제27조 | 이벤트 없음, selected caution 로그 | 별도 음성 없음 |
| 7 | `crosswalk_present == True` | `CAUTION` | `approaching_crosswalk` | 횡단보도 접근 중 | 도로교통법 제27조 | 기본 위험 이벤트 없음 | 별도 음성 없음 |
| 8 | 위 조건 없음 | `GO` | `clear_right_turn_approach` | 위험 근거 없음 | 없음 | 없음 | 없음 |

주의:
- `red_signal_stop_required`가 최종 reason이어도, 동시에 보행자가 주변에 있으면 `PEDESTRIAN_NEAR_CROSSWALK_RISK`도 같이 기록된다.
- 음성 안내는 여러 위험요소를 합쳐 말한다. 예: "적색 신호입니다. 일시정지하세요. 보행자가 횡단보도로 접근 중입니다. 주의하세요."
- `approaching_crosswalk`는 단순 횡단보도 접근 안내라서 위험 이벤트로 기록하지 않는다.

## 5. 우회전 IN_INTERSECTION 판단 기준

`IN_INTERSECTION`은 이미 교차로 내부에 들어온 상태다. 이 단계에서는 차량 신호 적색만으로 정지 판단하지 않는다.
핵심은 보이는 횡단보도와 보행자 보호다.

| 우선순위 | 조건 | 판단 | reason_code | 감지 근거 | 법적 근거 | 이벤트 코드 | 음성 안내 |
|---:|---|---|---|---|---|---|---|
| 1 | `pedestrian_on_crosswalk == True` | `STOP` | `pedestrian_on_crosswalk_in_intersection` | 교차로 내부에서 횡단보도 위 보행자 | 도로교통법 제27조 | `PEDESTRIAN_PROTECTION_RISK` | 횡단보도에 보행자가 있습니다. 정지하세요. |
| 2 | `pedestrian_near_crosswalk == True` | `CAUTION` | `pedestrian_near_crosswalk_in_intersection` | 횡단보도 주변 보행자 | 도로교통법 제27조 | `PEDESTRIAN_NEAR_CROSSWALK_RISK` | 횡단보도 주변 보행자에 주의하세요. |
| 3 | `crosswalk_present == True` | `CAUTION` | `visible_crosswalk_in_intersection` | 교차로 내부에서 횡단보도 보임 | 도로교통법 제27조 | 없음 | 없음 |
| 4 | 위 조건 없음 | `GO` | `clear_right_turn_in_intersection` | 보행자/횡단보도 위험 근거 없음 | 없음 | 없음 | 없음 |

주의:
- 교차로 내부에서는 `vehicle_signal == RED`만으로 `STOP`을 만들지 않는다.
- 교차로 내부에서는 `path_available == False`를 위험 판단 근거로 쓰지 않는다.

## 6. 비보호좌회전 판단 기준

현재 v3에서는 비보호좌회전 로직이 완성 단계가 아니다. 보행자 보호만 실제 규칙으로 반영하고, 대향 차량 접근 판단은 추후 확장 대상으로 남겨두었다.

| 단계 | 조건 | 판단 | reason_code | 법적 근거 | 상태 |
|---|---|---|---|---|---|
| APPROACH | `pedestrian_on_crosswalk == True` | `STOP` | `pedestrian_on_crosswalk_unprotected_left` | 도로교통법 제27조 | 구현됨 |
| APPROACH | 그 외 | `CAUTION` | `unprotected_left_rules_placeholder` | 비보호좌회전 기준 추가 예정 | placeholder |
| IN_INTERSECTION | `pedestrian_on_crosswalk == True` | `STOP` | `pedestrian_on_crosswalk_unprotected_left_in_intersection` | 도로교통법 제27조 | 구현됨 |
| IN_INTERSECTION | 그 외 | `CAUTION` | `unprotected_left_in_intersection_placeholder` | 비보호좌회전 기준 추가 예정 | placeholder |

추후 추가할 항목:
- 대향 차량 검출
- 대향 차량 접근 방향/속도
- 좌회전 예상 경로와 대향 차량 경로 충돌 가능성
- 좌회전 신호/비보호 표지 조건

## 7. 위반 의심 이벤트 기준

`ComplianceEvent`는 법적 확정이 아니라 기록용 위험 이벤트다.

| 이벤트 코드 | 심각도 | 발생 조건 | 법적 근거 | 설명 |
|---|---|---|---|---|
| `RED_SIGNAL_NO_STOP_SUSPECTED` | `VIOLATION_SUSPECTED` | APPROACH + 우회전 + 차량 신호 적색 + 정지 완료 미확인 | 신호 준수 의무 | 적색 신호에서 일시정지 여부가 확인되지 않음 |
| `RIGHT_TURN_SIGNAL_RED_RISK` | `VIOLATION_SUSPECTED` | APPROACH + 우회전 + 우회전 전용 신호 적색 | 신호 준수 의무 | 우회전 전용 적색 신호 진입 리스크 |
| `PEDESTRIAN_PROTECTION_RISK` | `VIOLATION_SUSPECTED` | 보행자가 횡단보도 내부에 있음 | 도로교통법 제27조 | 보행자 보호 필요 |
| `PEDESTRIAN_NEAR_CROSSWALK_RISK` | `WARNING` | 보행자가 횡단보도 주변에 있음 | 도로교통법 제27조 | 보행자가 통행하려는 상태일 가능성 |
| `RIGHT_TURN_LANE_RISK` | `WARNING` | 우회전인데 우측 끝 차로가 아닐 가능성 | 도로교통법 제25조 | 우회전 차로 위치 확인 필요 |

## 8. 로그 저장 기준

교차로 이벤트는 교차로 단위 폴더로 저장한다.

```text
event_records/<intersection_event_id>/
  risk_events.jsonl
  intersection_event.jsonl
  snapshots/*.jpg
```

| 파일 | 저장 내용 | 목적 |
|---|---|---|
| `risk_events.jsonl` | 위험 이벤트 변화, selected caution 변화 | 디버깅, 서버 전송 후보 |
| `intersection_event.jsonl` | 교차로 통과 1회 요약 | 서버 업로드 기준 데이터 |
| `snapshots/*.jpg` | 위험 순간 대표 이미지 | 사후 확인, 앱 표시 |

스냅샷에는 운전자용 한글 판단 패널을 넣지 않는다. 대신 다음 근거를 남긴다.

```text
도로/차선/횡단보도 시각화
YOLO 객체 박스
객체 라벨/신뢰도/거리
```

`risk_events.jsonl`에 저장되는 selected caution:

| reason_code | 저장 이유 |
|---|---|
| `pedestrian_near_crosswalk` | 보행자 보호 관련 주의 |
| `pedestrian_near_crosswalk_in_intersection` | 교차로 내부 보행자 주의 |
| `structure_uncertain` | 횡단보도 주변에서 주행 구조 불확실 |
| `right_turn_not_rightmost_lane` | 우회전 차로 위치 리스크 |

저장하지 않는 단순 주의:

| reason_code | 제외 이유 |
|---|---|
| `approaching_crosswalk` | 단순 횡단보도 접근 정보 |
| `visible_crosswalk_in_intersection` | 단순 횡단보도 보임 |

## 9. 음성 안내 기준

음성 안내는 가장 높은 위험 하나만 말하지 않고, 동시에 만족된 위험 요소를 우선순위대로 합쳐 말한다.

| 위험 요소 | 안내 문구 |
|---|---|
| 횡단보도 위 보행자 | 횡단보도에 보행자가 있습니다. 정지하세요. |
| 우회전 신호 적색 | 우회전 신호가 적색입니다. 정지하세요. |
| 차량 적색 신호 + 정지 미확인 | 적색 신호입니다. 일시정지하세요. |
| 횡단보도 접근 보행자 | 보행자가 횡단보도로 접근 중입니다. 주의하세요. |
| 횡단보도 주변 보행자 | 횡단보도 주변 보행자에 주의하세요. |
| 우회전 차로 불확실 | 우회전 차로가 아닐 수 있습니다. 차로를 확인하세요. |
| 그 외 STOP | 위험 상황입니다. 정지하세요. |

예시:

```text
적색 신호입니다. 일시정지하세요.
적색 신호입니다. 일시정지하세요. 보행자가 횡단보도로 접근 중입니다. 주의하세요.
횡단보도에 보행자가 있습니다. 정지하세요. 적색 신호입니다. 일시정지하세요.
```

## 10. 발표용 한 장 요약

```text
v3 판단 로직은 카메라 인식 결과를 바로 STOP/GO로 바꾸지 않고,
도로 상태값으로 요약한 뒤 교차로 단계별 규칙을 적용한다.

교차로 진입 전:
신호, 일시정지 여부, 횡단보도 보행자, 우회전 차로 여부를 판단한다.

교차로 내부:
차량 신호보다는 보행자와 횡단보도 보호 여부를 중심으로 판단한다.

결과:
STOP / CAUTION / GO / UNKNOWN과 함께
판단 이유, 감지 근거, 법적 근거, 위반 의심 이벤트, 음성 안내, 스냅샷 로그를 구조화한다.

주의:
본 시스템은 카메라 기반 보조 장치이므로 법규 위반을 확정하지 않고,
"위반 의심" 또는 "법규 리스크"로 기록한다.
```
