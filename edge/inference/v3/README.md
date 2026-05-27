# v3 Lightweight BiSeNetV2 Visualizer

`v3`는 `v2_0`의 인지 시각화 로직을 그대로 가져오고, segmentation 모델만 새로 학습한 경량 BiSeNetV2로 교체한 버전이다.

## 목적

```text
YOLO 객체 검출
+ 새 BiSeNetV2 road_v4 segmentation
+ v2_0 차선/ego-lane/path-corridor 시각화 로직
-> 새 모델이 기존 후처리 흐름에서 정상 동작하는지 확인
```

## Safety decision reboot

`v3` now includes an explainable rule layer on top of YOLO, BiSeNet,
ego lane, path corridor, and crosswalk-zone perception.

Detailed decision tables for presentation/report/code maintenance are in
`DECISION_CRITERIA.md`.

Flow:

```text
ego_lane + path_corridor + crosswalk mask + YOLO detections
-> RoadStructureState
-> IntersectionPhaseEstimator when --phase auto
-> IntersectionContext from auto phase or CLI override
-> SafetyDecisionEngine
-> SafetyDecision + ComplianceEvent list
-> compact overlay
```

Main CLI options:

```bash
python edge/inference/v3/intersection_demo_v3.py --source edge/inference/v4.mp4 --scenario right_turn --phase approach --vehicle-signal red --save edge/inference/v3/smoke_safety.mp4
```

Decision options:

```text
--phase auto/approach/in_intersection/unknown
--scenario right_turn/unprotected_left/straight/unknown
--vehicle-signal auto/red/yellow/green/unknown/none
--pedestrian-signal auto/red/green/unknown/none
--right-turn-signal auto/red/yellow/green/unknown/none
--stop-completed-on-red
--no-auto-stop-check
--event-output-dir edge/inference/v3/event_records
--no-event-log
--no-event-snapshot
--voice-alert off/console/pyttsx3
--voice-cooldown-sec 5
--voice-rate 185
--voice-volume 1.0
--no-safety-decision
```

While `--show` is running, the test context can be changed without restarting:

```text
a = phase auto
1 = phase approach
2 = phase in_intersection
0 = phase unknown
r/y/g/u/n = vehicle signal red/yellow/green/unknown/none
v = vehicle signal auto
t = toggle right-turn signal red/none
s = toggle stop_completed_on_red
3 = scenario right_turn
4 = scenario unprotected_left
q or ESC = quit
```

Important scope:

```text
The system records suspected legal/compliance risk only.
It must not describe a camera-based result as a final legal violation.
```

Screen overlay is intentionally minimal:

```text
판단 / 상황 / 이유 / 법적 근거 / 위반의심
자동단계 and crosswalk position only when --phase auto is used
lane-role estimate: leftmost/rightmost/adjacent vehicle/confidence
one compact performance line at the bottom
```

Older ego-lane, crosswalk, and phase debug strings are hidden by default so the
demo can be read as a safety-judgment screen instead of a raw debug screen.

Lane-role estimate:

```text
The system does not treat "not visible" as "not present".
Lane role is decided by evidence scoring and recent-frame voting.
Outside lane boundary / adjacent vehicle -> edge score down.
Visible edge / yellow boundary / stable evidence -> edge score up.
If the ego boundary is yellow, outside-boundary and adjacent-vehicle penalties
are reduced because the outside marking may be centerline-side or road-edge
context rather than a usable same-direction lane.
The latest 15 frames are voted, so one-frame lane-mask noise does not
immediately change leftmost/rightmost.  A stable visible edge with score 0.40+
can become leftmost/rightmost after repeated votes.
When the phase is 자동 `교차로 진입 전`, a weak/Unknown current lane-role value
can be replaced by the last reliable role measured before entry.  The overlay
shows this source as `진입전` and displays left/right role scores.  The cached
role is kept for up to 180 frames.
```

Auto phase note:

```text
Crosswalk bottom position alone does not switch to IN_INTERSECTION.
The crosswalk must first be near for several frames, then disappear after being
near or sharply shrink. Bottom contact alone keeps the phase in APPROACH.
Small road-edge fragments are ignored for phase estimation. The size threshold
is moderate, but the component must overlap the center driving band.
```

Inside-intersection decision note:

```text
Path-corridor uncertainty is not used as a warning basis once the phase is
IN_INTERSECTION. Pedestrian and visible-crosswalk evidence are the active bases.
```

Multiple risk note:

```text
The final decision level still follows the highest-priority rule, but every
simultaneously satisfied compliance-risk event is kept in SafetyDecision and
shown in the overlay summary.
The console prints [RISK] only when at least one risk event exists.
Event-free caution states stay on the screen overlay and do not print [RISK] none.
```

Signal recognition note:

```text
vehicle_signal and pedestrian_signal default to auto.
YOLO first detects traffic_light_vehicle / traffic_light_pedestrian boxes.
The crop is classified with HSV color ratios into red/yellow/green.
Weak, dark, or ambiguous crops remain UNKNOWN.
Raw signal estimates are smoothed over recent frames. A signal normally needs
3 votes in the latest 5 frames, and short UNKNOWN gaps keep the previous stable
signal for up to 2 frames.
During 자동 `교차로 진입 전`, a recently stable RED is held for up to 24 more
UNKNOWN frames.  GREEN/YELLOW/NONE clears this hold immediately.
The overlay shows held red as `적색(유지)`.
The current model has no dedicated right-turn-signal class, so
right_turn_signal should normally stay manual/none for now.
```

Stop-completion note:

```text
During 자동 `교차로 진입 전` + vehicle RED, v3 estimates red-stop completion
from low frame-to-frame motion in the lower road ROI.
Default: motion score <= 0.42 for 5 consecutive frames -> stop completed.
`--stop-completed-on-red` still forces manual completion.
Use `--no-auto-stop-check` to disable the camera-motion estimate.
```

Pedestrian-motion note:

```text
v3 tracks pedestrian bbox foot points across frames.
It compares each pedestrian with the visible crosswalk in two ways:
1. distance to crosswalk mask
2. pedestrian movement vector vs. pedestrian-to-crosswalk-center vector

distance decreasing or vector aligned to crosswalk -> approaching crosswalk
distance increasing or vector opposite crosswalk   -> leaving crosswalk
small change and weak vector evidence              -> stationary

Defaults:
- distance threshold: 6px
- minimum vector movement: 4px
- approaching cosine threshold: 0.50
- leaving cosine threshold: -0.30
- minimum track hits: 3

This avoids treating small bbox/crosswalk-mask jitter as real pedestrian intent
while still catching a clear pedestrian movement toward the crosswalk.
The overlay shows 접근/이탈/정지/불명 counts when a tracked pedestrian is near
the crosswalk.
```

Server/event logging plan:

```text
Local event folder:
  event_records/<intersection_event_id>/
    risk_events.jsonl
      - frame/change-level risk records
      - selected meaningful CAUTION records:
        pedestrian_near_crosswalk, structure_uncertain, right_turn_not_rightmost_lane
    intersection_event.jsonl
      - one summarized record for that intersection pass
    snapshots/*.jpg
      - representative risky frames with YOLO object boxes drawn

Server upload unit:
  one event_records/<intersection_event_id> folder per intersection pass

Upload payload:
  1. IntersectionPassEvent JSON
  2. representative JPG snapshot(s) for risky moments

Do not stream/upload full video by default on Raspberry Pi.
Use suspected-risk wording, not confirmed legal violation wording.
```

Voice alert note:

```text
Voice alert supports console-only output and pyttsx3 speech.
Run with --voice-alert console to print driver guidance as [VOICE].
Run with --voice-alert pyttsx3 to print the same guidance and speak it through
the OS TTS engine.
Install the optional dependency with `python -m pip install pyttsx3==2.90`.
Active risk factors are merged into one ordered guidance sentence, so red-signal
stop and nearby-pedestrian warnings can be spoken together.  The same combined
alert code is rate-limited by --voice-cooldown-sec.
pyttsx3 speech runs in a background thread so inference does not wait for
runAndWait().  A pre-recorded WAV backend can still be added later.
```

`v3`는 아직 최종 안전 판단기(FSM)가 아니다. 새 segmentation 모델의 인식 품질과 기존 v2_0 후처리 로직의 호환성을 확인하는 실험용 프로그램이다.

## v2_0과 다른 점

| 항목 | v2_0 | v3 |
|---|---|---|
| segmentation 모델 | ResNet18 기반 임시 모델 | 경량 BiSeNetV2 |
| 기본 weight | `road_v4_best.pt` | `road_v4_best_light.pt` |
| 기본 segmentation 입력 | `512x928` | `352x640` |
| 차선 선 표시 | 유지 | 유지 |
| ego lane reboot | 유지 | 유지 |
| crosswalk/path corridor 시각화 | 유지 | 유지 |
| FSM 판단 | 없음 | 없음 |

## 필요한 모델 파일

Colab 학습 결과 중 추론용 파일을 사용한다.

```text
best_light_infer.pt
```

아래 경로로 복사하거나 이름을 바꿔 넣는다.

```text
edge/inference/v3/road_v4_best_light.pt
```

주의:

```text
best_light_infer.pt = 추론용, 사용 권장
best_light_full.pt  = optimizer 포함 학습 재개용, 배포/추론에 사용하지 않음
```

## 실행

영상 저장:

```bash
python edge/inference/v3/intersection_demo_v3.py --source edge/inference/v4.mp4 --save edge/inference/v3/out_v4.mp4 --legend
```

화면 표시:

```bash
python edge/inference/v3/intersection_demo_v3.py --source edge/inference/v4.mp4 --show --legend
```

새 모델만 빠르게 확인:

```bash
python edge/inference/v3/intersection_demo_v3.py --source edge/inference/v4.mp4 --no-yolo --no-ego-lane --show --legend
```

CUDA 사용:

```bash
python edge/inference/v3/intersection_demo_v3.py --source edge/inference/v4.mp4 --device cuda --show --legend
```

라즈베리파이/저성능 환경 빠른 테스트:

```bash
python edge/inference/v3/intersection_demo_v3.py --source edge/inference/v4.mp4 --width 480 --height 270 --seg-input-h 256 --seg-input-w 464 --save edge/inference/v3/out_fast.mp4
```

## road_v4 class

| id | class |
|---:|---|
| 0 | background |
| 1 | lane_white |
| 2 | lane_yellow |
| 3 | lane_blue |
| 4 | crosswalk |
| 5 | stop_line |

## 현재 후처리 흐름

```text
BiSeNet mask
-> lane class와 crosswalk/stop_line class 분리
-> crosswalk/stop_line은 반투명 mask로 표시
-> lane class는 scan-line 중심점 추출
-> 가까운 중심점끼리 track 구성
-> 과도하게 떨어진 조각/fit 오차 큰 track 제거
-> 1~2차 polyline으로 차선 표시
-> ego lane reboot 모듈로 현재 주행 차선 후보 표시
-> path corridor/crosswalk zone 시각화
```

## 확인해야 할 것

```text
1. 새 BiSeNetV2가 차선 색상 class를 안정적으로 분리하는가
2. 점선/끊어진 차선이 과도하게 억지 연결되지 않는가
3. ego lane polygon이 기존 v2_0보다 덜 휘거나 깨지는가
4. crosswalk/stop_line이 검출되는 위치가 실제 도로 구조와 맞는가
5. FPS가 라즈베리파이 목표에 맞게 줄일 수 있는 수준인가
```
