# v3 Implementation Notes

## Safety decision reboot

### Purpose

The v3 visualizer now has a first rule-based safety-decision layer. It does not
replace YOLO/BiSeNet perception. It summarizes perception into
`RoadStructureState`, estimates the intersection phase when `--phase auto` is
used, combines that with `IntersectionContext`, and returns a structured
`SafetyDecision`.

### RoadStructureState

`road_state_reboot.py` contains `RoadStructureState` and
`build_road_structure_state()`.

The state records only road/perception facts:

```text
path availability/source/confidence/age
visible crosswalk existence
pedestrian on or near visible crosswalk
stop-line existence and vertical position
ego-lane leftmost/rightmost role
adjacent vehicle hints
vehicle/pedestrian counts
```

This module never decides STOP, CAUTION, or GO. It intentionally uses the
visible crosswalk mask as the main pedestrian-risk basis. Path corridor is kept
as supporting evidence and visualization, because inside an intersection the
ego-lane/path estimate can become unreliable.

### Intersection phase split

`intersection_phase_reboot.py` estimates phase from crosswalk evidence only.
Stop-line evidence is not used for phase transition because it is often hidden,
worn out, or no longer meaningful after the vehicle starts passing through the
intersection.

The internal FSM is:

```text
UNKNOWN
-> APPROACH when a reliable crosswalk appears in the lower/middle image
-> NEAR_CROSSWALK when the crosswalk bottom approaches the ego area
-> IN_INTERSECTION when the crosswalk reaches the ego area or disappears after being near
-> UNKNOWN after a minimum inside-intersection hold and stable lane/path recovery
```

`NEAR_CROSSWALK` is an internal state only. The safety engine receives it as
`APPROACH`, so red-signal and pre-entry right-turn rules still apply before the
system commits to `IN_INTERSECTION`.

Initial thresholds:

```text
detect_y_ratio = 0.35
near_y_ratio = 0.82
enter_y_ratio = 0.97
min_phase_area_ratio = 0.0012
min_phase_width_ratio = 0.12
center_band_left_ratio = 0.25
center_band_right_ratio = 0.75
min_near_frames_before_enter = 5
bottom_enter_confirm_frames = 3
area_drop_ratio_for_enter = 0.45
min_enter_reference_area_ratio = 0.01
lost_enter_confirm_frames = 2
min_in_intersection_frames = 20
exit_stable_frames = 8
exit_path_confidence = 0.45
```

`IN_INTERSECTION` is intentionally conservative. A single low crosswalk pixel is
not enough. The estimator must first stay in `NEAR_CROSSWALK` for several
frames, then confirm one of these:

```text
near crosswalk disappears for consecutive frames
crosswalk area drops sharply after it was near
```

Crosswalk bottom touching the lower image is not enough to switch to
`IN_INTERSECTION`. The system stays in `APPROACH` until the first large
crosswalk begins to disappear or shrink, because the ego vehicle has not
necessarily crossed the first crosswalk yet.

Small or road-edge crosswalk fragments are ignored for phase estimation. The
size threshold is intentionally moderate, but the component must overlap the
center driving band. This keeps distant/partial valid crosswalks easier to pick
up while still rejecting the v4.mp4 right-edge false positives after frame 548.

`APPROACH` means the vehicle is before entering the intersection. Signal rules
are applied strongly here.

`IN_INTERSECTION` means the vehicle is already inside the intersection. Vehicle
signal red alone is not used as a STOP trigger here. The main basis becomes
visible crosswalk and pedestrian risk.

### Right-turn rules

Approach priority:

```text
right-turn signal red -> STOP
vehicle signal red + no confirmed red stop -> STOP
pedestrian on crosswalk -> STOP
pedestrian near crosswalk -> CAUTION
not estimated as rightmost lane -> CAUTION
visible crosswalk -> CAUTION
visible crosswalk + path unavailable -> CAUTION
otherwise -> GO
```

Inside-intersection priority:

```text
pedestrian on crosswalk -> STOP
pedestrian near crosswalk -> CAUTION
visible crosswalk -> CAUTION
otherwise -> GO
```

Inside the intersection, path-corridor uncertainty is not used as a warning
basis. Lane/path geometry is expected to be unstable during a turn or while
passing through the intersection, so it remains evidence/debug context only.

### Legal/compliance events

`safety_decision_reboot.py` defines structured legal basis and event records.
Current events:

```text
RED_SIGNAL_NO_STOP_SUSPECTED
RIGHT_TURN_SIGNAL_RED_RISK
PEDESTRIAN_PROTECTION_RISK
PEDESTRIAN_NEAR_CROSSWALK_RISK
RIGHT_TURN_LANE_RISK
```

The final decision still uses the highest-priority rule for `level` and
`reason_code`, but all simultaneously satisfied compliance-risk events are
collected into `SafetyDecision.compliance_events`.

Console logging prints `[RISK]` only when at least one compliance-risk event is
present.  Event-free states such as `approaching_crosswalk` or
`visible_crosswalk_in_intersection` are still shown on the overlay, but they do
not create `[RISK] none` console lines.

These are not final legal judgments. They are camera-based suspected-risk
records, so UI and logs should use wording such as suspected violation,
compliance risk, or possible violation.

### Lane-role estimate

The lane-role logic answers whether the ego lane is the leftmost/rightmost lane.
It now uses evidence scoring plus recent-frame voting instead of a single hard
rule:

```text
outside lane boundary visible -> strong negative edge score
adjacent vehicle outside ego  -> negative edge score
visible array edge            -> positive edge score
yellow ego boundary           -> positive edge-support score
stable same evidence          -> small stability bonus
```

If the ego boundary on that side is yellow, outside-boundary and adjacent-vehicle
penalties are reduced.  Yellow can indicate centerline/road-edge context, so an
outside marking beyond a yellow boundary is weaker evidence of a usable
same-direction adjacent lane.

The one-frame score is converted to a tentative tri-state role, then the latest
15 frames are voted.  False needs fewer votes because visible outside lanes or
adjacent vehicles are strong evidence.  True needs more persistence because
missing outside boundaries are weaker evidence.  A stable visible road edge with
score >= 0.40 can become an edge-lane vote, which keeps right-edge roads such as
`v2.mp4` from staying Unknown forever.  `RIGHT_TURN_LANE_RISK` is
emitted only when the voted rightmost estimate is False with confidence >= 0.45.
The overlay shows the voted role and the current left/right scores.

When the automatic phase becomes `APPROACH`, lane markings are often already
unstable because the vehicle is close to the crosswalk/intersection area.  The
demo therefore keeps the last reliable lane-role estimate and reuses it during
APPROACH when the current frame is weak or Unknown.  The overlay marks this as
`진입전` with the cache age in frames, so the operator can distinguish cached
lane role from the current-frame estimate.  The cache is kept for up to 180
frames so the role survives the short lane-loss interval before intersection
entry.

### Screen overlay policy

The normal v3 screen is now a judgment view, not a raw debug view. Keep only:

```text
decision level
intersection phase and maneuver
short reason
legal basis
suspected-risk event
auto-phase hint when --phase auto is active
lane-role estimate: leftmost/rightmost/adjacent vehicle/confidence
compact FPS/model-time line
```

Detailed ego-lane track IDs, lane-role debug strings, crosswalk pixel counts,
and phase reason codes are hidden from the default overlay. They can be restored
later behind an explicit debug flag if needed.

### Signal recognition

`signal_state_reboot.py` estimates signal color from YOLO traffic-light boxes.
The current model detects:

```text
traffic_light_vehicle
traffic_light_pedestrian
```

For each detected box, the crop is slightly shrunk and converted to HSV. Bright,
saturated red/yellow/green pixels are counted. If the evidence is weak or
ambiguous, the signal remains `UNKNOWN`.

The raw color estimate is passed through `SignalStateTracker` before it reaches
the safety engine:

```text
window_frames = 5
min_votes = 3
unknown_hold_frames = 2
approach_red_hold_frames = 24
```

This means RED/GREEN/YELLOW must be seen repeatedly before becoming stable, and
one or two UNKNOWN frames keep the previous stable signal instead of immediately
dropping it.

After the automatic phase is known, an additional approach-only rule is applied.
If vehicle RED was recently stable and the current vehicle signal falls to
UNKNOWN during `APPROACH`, the system keeps RED for up to 24 frames.  This
prevents STOP/CAUTION flicker caused by brief traffic-light crop loss.  A
definite GREEN/YELLOW/NONE clears the RED hold immediately.
When this hold is active, the overlay shows `적색(유지)` instead of plain `적색`,
and event evidence records `vehicle_signal_source=approach_red_hold`.

### Stop-completion estimate

`stop_state_reboot.py` estimates whether the ego vehicle completed a red-light
stop using camera motion only.  It watches the lower road ROI and measures
frame-to-frame optical-flow motion.

```text
active condition: phase=APPROACH and vehicle_signal=RED
default threshold: motion_score <= 0.42
default confirmation: 5 consecutive low-motion frames
manual override: --stop-completed-on-red
disable option: --no-auto-stop-check
```

Once the low-motion condition is confirmed, `context.stop_completed_on_red`
becomes True for the current red-approach episode.  This prevents repeated
`RED_SIGNAL_NO_STOP_SUSPECTED` after a detected stop.  The overlay shows
`정지: 완료(자동 프레임수)` or `정지: 미완료(motion_score)`.

### Pedestrian motion direction

`pedestrian_motion_reboot.py` tracks pedestrian bbox foot points and compares
two crosswalk-relative signals:

1. distance to the visible crosswalk mask
2. alignment between the pedestrian movement vector and the vector from the
   pedestrian foot point to the crosswalk center

```text
distance decreases or vector points to crosswalk  -> approaching
distance increases or vector points away          -> leaving
weak distance/vector evidence                     -> stationary
not enough track history                          -> unknown
```

The default distance threshold is 6 px after at least 3 matched track hits.
Vector evidence requires at least 4 px of pedestrian motion.  `cos >= 0.50`
means the pedestrian is moving toward the crosswalk center, while `cos <= -0.30`
means the pedestrian is moving away.  The 6 px distance threshold was chosen
after checking `v4.mp4`: the median absolute distance jitter was about 3 px, so
the earlier 2 px threshold was too sensitive for a moving-ego camera.

The result is summarized into `RoadStructureState`:

```text
pedestrian_approaching_crosswalk
pedestrian_leaving_crosswalk
approaching_pedestrian_count
leaving_pedestrian_count
stationary_pedestrian_count
unknown_motion_pedestrian_count
nearest_pedestrian_crosswalk_distance_px
```

This is used as explanation evidence and overlay context.  It does not override
the core pedestrian protection rule: a pedestrian on the crosswalk still means
STOP, and a pedestrian near the crosswalk still means CAUTION.

### Event logging and server upload plan

Server upload should use one intersection pass as the main event unit, not one
frame.  The edge device can still keep frame-level JSONL logs for debugging, but
the driving-management app should receive a summarized intersection event.

Recommended local folder layout:

```text
event_records/<intersection_event_id>/
  risk_events.jsonl
    Frame-level or change-level risk records.
    Also includes selected meaningful CAUTION records:
    pedestrian_near_crosswalk, structure_uncertain,
    right_turn_not_rightmost_lane.
    Used for debugging, threshold tuning, and replay analysis.

  intersection_event.jsonl
    One summarized record for that intersection pass.
    Used as the server upload source of truth.

  snapshots/*.jpg
    Representative risky frames.  The saved image includes the safety overlay
    plus YOLO object boxes/labels so the reason can be reviewed without replay.
```

Implemented CLI defaults:

```text
--event-output-dir edge/inference/v3/event_records
--no-event-log
--no-event-snapshot
```

Recommended upload flow:

```text
1. Vehicle approaches an intersection.
2. v3 records the phase/timeline, decisions, risk events, lane role, signal,
   stop-completion state, and pedestrian evidence.
3. When the pass ends, v3 writes one IntersectionPassEvent to
   event_records/<intersection_event_id>/intersection_event.jsonl.
4. v3 sends that IntersectionPassEvent JSON to the server.
5. If a risk event occurred, v3 saves one or more representative JPG snapshots
   and includes their paths/URLs in the event.
```

Do not continuously upload full video from the Raspberry Pi by default.  The
preferred upload payload is:

```text
1st priority: intersection summary JSON
2nd priority: representative JPG snapshot(s) for risky moments
Not default: full video or continuous streaming
```

Suggested `IntersectionPassEvent` shape:

```json
{
  "device_id": "edge-001",
  "vehicle_id": "test-car-01",
  "intersection_event_id": "edge-001-20260524-143012-001",
  "started_at": "2026-05-24T14:30:12+09:00",
  "ended_at": "2026-05-24T14:30:28+09:00",
  "scenario": "RIGHT_TURN",
  "final_decision": "STOP",
  "highest_severity": "VIOLATION_SUSPECTED",
  "phase_sequence": ["APPROACH", "IN_INTERSECTION"],
  "event_codes": [
    "RED_SIGNAL_NO_STOP_SUSPECTED",
    "PEDESTRIAN_NEAR_CROSSWALK_RISK"
  ],
  "evidence_summary": {
    "red_signal_seen": true,
    "stop_completed_on_red": false,
    "pedestrian_on_crosswalk": false,
    "pedestrian_near_crosswalk": true,
    "pedestrian_approaching_crosswalk": true,
    "rightmost_lane": true
  },
  "snapshots": [
    {
      "frame_index": 300,
      "reason": "RED_SIGNAL_NO_STOP_SUSPECTED",
      "path": "event_records/edge-001-20260524-143012-001/snapshots/edge-001_20260524_143012_f300.jpg",
      "detections": [
        {
          "class": "traffic_light_vehicle",
          "confidence": 0.84,
          "box": [515, 42, 548, 80],
          "distance_m": null
        }
      ]
    }
  ],
  "timeline": [
    {
      "frame_index": 300,
      "phase": "APPROACH",
      "decision": "STOP",
      "reason_code": "red_signal_stop_required",
      "events": ["RED_SIGNAL_NO_STOP_SUSPECTED"]
    }
  ]
}
```

Wording rule for server/app:

```text
Use: suspected violation, compliance risk, risk event
Avoid: confirmed violation, illegal, legally determined
```

### Voice alert

`voice_alert_reboot.py` maps safety decisions to short driver guidance strings.
The same alert selector can either print to the console or speak through
pyttsx3:

```text
python -m pip install pyttsx3==2.90
--voice-alert off
--voice-alert console
--voice-alert pyttsx3
--voice-cooldown-sec 5
--voice-rate 185
--voice-volume 1.0
```

Example console output:

```text
[VOICE] red_signal_stop_required: 적색 신호입니다. 일시정지하세요.
[VOICE] pedestrian_approaching_crosswalk: 보행자가 횡단보도로 접근 중입니다. 주의하세요.
[VOICE] red_signal_stop_required+pedestrian_approaching_crosswalk: 적색 신호입니다. 일시정지하세요. 보행자가 횡단보도로 접근 중입니다. 주의하세요.
```

Active risk factors are collected, sorted by priority, and merged into one
driver guidance sentence.  The same combined alert code is rate-limited, and a
new or higher-priority risk combination can interrupt the previous state.
pyttsx3 speech is queued to a background worker because `runAndWait()` blocks.
If messages change faster than the speech engine can speak, stale queued
messages are dropped and the newest guidance is kept.  A WAV playback backend
can later reuse the same `VoiceAlert` selection output.

Default CLI behavior:

```text
--vehicle-signal auto
--pedestrian-signal auto
--right-turn-signal none
```

Dedicated right-turn signal recognition is left open because the current YOLO
class list does not include a separate right-turn-signal class. If such a class
is added later, `signal_state_reboot.py` can map it into
`right_turn_signal`.

### CLI smoke example

```bash
python edge/inference/v3/intersection_demo_v3.py --source edge/inference/v4.mp4 --max-frames 10 --scenario right_turn --phase approach --vehicle-signal red --save edge/inference/v3/smoke_safety.mp4
```

## 설계 의도

`v3`는 새로 학습한 경량 BiSeNetV2 모델을 기존 `v2_0` 후처리 로직에 연결하기 위한 중간 버전이다. `v2_0`에서 이미 만든 차선 선 표시, ego lane reboot, path corridor 시각화 로직을 유지하여 모델 교체 전후의 차이를 직접 비교할 수 있게 했다.

## 파일 구성

| 파일 | 역할 |
|---|---|
| `intersection_demo_v3.py` | 메인 실행 파일 |
| `road_v4_segmentor.py` | 새 경량 BiSeNetV2 checkpoint 로더 |
| `ego_lane_reboot.py` | 현재 주행 차선 후보 추적/시각화 |
| `road_structure_reboot.py` | path corridor, crosswalk zone 구조화 |
| `road_v4_postprocess.py` | road_v4 class 기반 후처리 보조 |
| `visualizer_v2.py` | 기존 시각화 보조 함수 |
| `road_v4_best_light.pt` | 사용자가 넣어야 하는 새 추론용 weight |

## 모델 로딩

`road_v4_segmentor.py`는 `edge/inference/bisenetv2.py`의 `BiSeNetV2` 구조를 import한다. 따라서 새 checkpoint는 다음 조건을 만족해야 한다.

```text
model_type: bisenetv2_light
num_classes: 6
input_size: [352, 640]
state_dict: BiSeNetV2 n_classes=6 aux_mode=eval 호환 weight
```

학습 결과 파일 중 `best_light_infer.pt`를 사용해야 한다. `best_light_full.pt`는 optimizer state가 포함되어 파일이 크고 추론에는 필요하지 않다.

## v2_0 로직 유지 범위

유지한 부분:

```text
YOLO 객체 bbox 표시
차량 거리 표시
lane mask scan-line 추출
점선 gap을 polyline으로 연결
과도한 차선 연결 억제
ego lane 영역 표시
path corridor 표시
crosswalk zone 표시
class pixel count/FPS 표시
```

제외한 부분:

```text
STOP / CAUTION / GO 판단
도로교통법 기반 FSM
신호등 상태 기반 의사결정
라즈베리파이 GPIO 출력
```

## 새 모델 적용 절차

1. Colab 학습 결과에서 `best_light_infer.pt`를 가져온다.
2. 파일명을 `road_v4_best_light.pt`로 바꾼다.
3. 아래 위치에 넣는다.

```text
edge/inference/v3/road_v4_best_light.pt
```

4. 짧은 프레임으로 smoke test를 실행한다.

```bash
python edge/inference/v3/intersection_demo_v3.py --source edge/inference/v4.mp4 --max-frames 30 --save edge/inference/v3/smoke_v3.mp4 --legend
```

## 성능 확인 포인트

새 모델의 test 성능은 다음과 같이 확인되었다.

| class | test IoU |
|---|---:|
| lane_white | 0.622 |
| lane_yellow | 0.636 |
| lane_blue | 0.645 |
| crosswalk | 0.812 |
| stop_line | 0.597 |
| foreground mIoU | 0.662 |
| all mIoU | 0.717 |

따라서 `v3` 테스트에서는 특히 다음을 봐야 한다.

```text
황색 차선이 이전보다 덜 놓치는가
가장자리 차선이 끊겨도 ego lane이 과도하게 휘지 않는가
정지선이 stop_line class로 잡히는가
횡단보도 mask가 path corridor와 겹쳤을 때 zone 표시가 자연스러운가
```

## 주의사항

`v3`는 새 모델을 검증하기 위한 버전이다. 최종 판단 시스템으로 쓰려면 이 결과를 기반으로 SceneContext와 FSM을 다시 연결해야 한다.
