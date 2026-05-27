# 단안 카메라 기반 우회전 교차로 안전 보조 시스템

개발 과정 및 알고리즘 정리 문서  
대상 코드: `edge/inference`

---

## 1. 개발 목표

본 시스템은 단안 카메라 영상만을 이용해 우회전 상황에서 운전자의 판단을 보조하는 교차로 안전 보조장치이다.

핵심 목표는 다음과 같다.

- YOLO로 보행자, 차량, 신호등, 횡단보도 등 주요 객체를 검출한다.
- BiSeNetV2로 차선, 횡단보도, 정지선 등 도로 구조를 segmentation한다.
- 차선 mask를 후처리하여 ego lane, centerline, path corridor를 생성한다.
- 보행자, 차량, 신호, 차선 신뢰도, 교차로 상태를 종합하여 `STOP`, `CAUTION`, `GO`를 판단한다.
- 판단 결과와 근거 변수를 화면에 한글로 표시한다.

주의할 점은 이 시스템이 운전자를 대체하는 자율주행 판단기가 아니라, 우회전 상황에 한정된 판단 보조장치라는 점이다.

---

## 2. 전체 시스템 구조

```text
입력 영상
  ↓
YOLO 객체 검출
  ├─ pedestrian
  ├─ vehicle
  ├─ traffic_light_vehicle
  ├─ traffic_light_pedestrian
  └─ crosswalk 등
  ↓
BiSeNetV2 도로 구조 segmentation
  ├─ lane class
  ├─ crosswalk class
  └─ stop line class
  ↓
차선/도로 구조 후처리
  ├─ 차량 bbox 내부 lane pixel 제거
  ├─ scan line 기반 차선 후보 추출
  ├─ 이전 lane model과 후보 매칭
  ├─ 점선 gap 보간
  ├─ lane width 검증
  └─ centerline / corridor 생성
  ↓
SceneContext 생성
  ├─ 보행자 위치
  ├─ 차량 위치 및 거리
  ├─ 신호 상태
  ├─ 횡단보도/정지선 여부
  └─ 차선 신뢰도
  ↓
FSM 기반 교차로 상태 추정
  ↓
STOP / CAUTION / GO 판단
  ↓
화면 오버레이 및 한글 경고문 표시
```

---

## 3. 주요 파일 역할

| 파일 | 역할 |
|---|---|
| `intersection_demo.py` | 전체 실행 루프, 영상 입력, 옵션 처리, 모델 호출, 결과 저장/표시 |
| `perception.py` | YOLO 객체 검출, BiSeNetV2 segmentation, 차량 거리 추정 |
| `postprocess.py` | lane mask 후처리, ego lane polygon, centerline, corridor 생성 |
| `state_machine.py` | 교차로 상태 FSM, SceneContext 구성, STOP/CAUTION/GO 판단 |
| `visualizer.py` | bbox, 거리, 차선, corridor, 판단 결과, 한글 경고문 시각화 |
| `bisenetv2.py` | inference용 BiSeNetV2 네트워크 구조 |

---

## 4. YOLO 객체 검출

YOLO는 교차로 판단에 필요한 동적 객체와 신호 객체를 검출한다.

검출 대상 예시는 다음과 같다.

- `pedestrian`
- `vehicle`
- `traffic_light_vehicle`
- `traffic_light_pedestrian`
- `crosswalk`
- `left_turn_sign`

검출 결과는 `Detection` 객체로 관리된다.

```python
Detection(
    cls_id,
    cls_name,
    conf,
    box=(x1, y1, x2, y2),
    distance_m
)
```

여기서 `distance_m`은 차량에 대해서만 계산된다.

---

## 5. 차량 거리 추정 알고리즘

차량 거리는 단안 카메라 기반 pinhole camera model을 사용한다.

기본 식은 다음과 같다.

```text
focal_px = (frame_width / 2) / tan(horizontal_FOV / 2)

distance_m = real_object_size_m × focal_px / bbox_size_px
```

이 식의 근거는 pinhole camera model의 투영 관계이다.

```text
이미지상 크기 / 실제 크기 = 초점거리 / 실제 거리

bbox_width_px / real_vehicle_width_m = focal_px / distance_m

distance_m = real_vehicle_width_m × focal_px / bbox_width_px
```

OpenCV의 camera calibration 문서에서도 pinhole camera model을 사용하며, 3D 점이 camera matrix의 `fx`, `fy` focal length를 통해 image plane의 pixel 좌표로 투영된다고 설명한다. 또한 known width, perceived pixel width, focal length를 이용해 거리를 계산하는 방식은 triangle similarity 기반 단안 거리 추정 예제로 널리 사용된다.

현재 구현에서는 차량의 폭과 높이를 모두 사용한다.

```text
거리_폭기반 = vehicle_real_width_m × focal_px / bbox_width_px
거리_높이기반 = vehicle_real_height_m × focal_px / bbox_height_px

최종거리 = 0.7 × 거리_폭기반 + 0.3 × 거리_높이기반
최종거리 = 최종거리 × distance_scale
```

현재 기본값은 다음과 같다.

| 옵션 | 기본값 | 의미 |
|---|---:|---|
| `--camera-hfov` | `70.0` | 카메라 수평 화각 |
| `--distance-focal-px` | `0.0` | 직접 보정한 focal length. 0이면 자동 계산 |
| `--vehicle-real-width` | `1.8` | 차량 평균 실제 폭 |
| `--vehicle-real-height` | `1.5` | 차량 평균 실제 높이 |
| `--distance-scale` | `0.8` | 영상에 맞춘 거리 보정 계수 |

이 방식은 논문에서 흔히 사용하는 단안 카메라 bbox 기반 거리 추정 방식과 같은 계열이다. 다만 실제 차량 크기, 카메라 장착 높이, pitch angle, bbox 흔들림에 따라 오차가 발생할 수 있다.

참고 근거:

- OpenCV Camera Calibration and 3D Reconstruction: pinhole camera model, camera matrix, `fx`, `fy` focal length 설명  
  https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
- PyImageSearch, “Find distance from camera to object/marker using Python and OpenCV”: known width, focal length, perceived width 기반 거리식 `D = (W × F) / P` 예제  
  https://pyimagesearch.com/2015/01/19/find-distance-camera-objectmarker-using-python-opencv/

---

## 6. BiSeNetV2 도로 구조 분할

BiSeNetV2는 도로 영상에서 차선, 횡단보도, 정지선 등 pixel 단위 도로 구조를 분할한다.

이 결과는 바로 판단에 쓰지 않고, 후처리를 거쳐 안정적인 기하 구조로 변환한다.

이유는 다음과 같다.

- segmentation mask는 끊기거나 흔들릴 수 있다.
- 점선 차선은 화면 하단에서 gap이 발생한다.
- 차량 bbox 내부의 윤곽선이나 그림자가 lane class로 오인될 수 있다.
- 교차로 내부에서는 차선이 사라지거나 불확실해진다.

따라서 후처리 단계에서 ego lane과 corridor를 다시 구성한다.

---

## 7. 차선 후처리 알고리즘

현재 적용된 차선 처리 흐름은 다음과 같다.

```text
BiSeNet mask
  ↓
vehicle bbox mask 제거
  ↓
lane class만 추출
  ↓
scan line 후보 추출
  ↓
이전 lane model과 후보 매칭
  ↓
점선 gap 보간
  ↓
lane width 검증
  ↓
centerline / corridor 생성
```

### 7.1 차량 bbox 내부 lane pixel 제거

YOLO에서 검출된 차량 bbox 내부에 lane class pixel이 있으면 제거한다.

목적은 차량의 윤곽, 번호판, 그림자 등이 차선처럼 segmentation되어 ego lane polygon을 뒤틀어버리는 문제를 줄이는 것이다.

대상 클래스는 다음과 같다.

```text
vehicle, car, bus, truck, motorcycle
```

### 7.2 scan line 기반 차선 후보 추출

화면 하단에서 중간 영역까지 여러 개의 수평 scan line을 만든다.

현재 기준은 다음과 같다.

```text
scan_bot_ratio = 0.95
scan_top_ratio = 0.35
num_scan_lines = 20
```

각 scan line에서 lane class pixel의 x 좌표를 모으고, 가까운 pixel끼리 cluster로 묶는다.

그 후 화면 중심 `ego_x = frame_width / 2`를 기준으로 가장 가까운 좌측 후보와 우측 후보를 선택한다.

### 7.3 polynomial fitting

추출한 좌우 차선 점들은 그대로 쓰지 않고 2차 polynomial fitting을 적용한다.

```text
y → x 형태의 2차 곡선 모델
```

또한 median absolute deviation 기반으로 x 좌표가 과도하게 튀는 outlier를 제거한다.

### 7.4 이전 lane model과 후보 매칭

현재 프레임에서 추출된 차선 후보가 이전 프레임의 lane model과 너무 멀리 떨어져 있으면 잘못된 차선 또는 noise로 판단한다.

이 경우 현재 후보를 버리고 이전 모델을 유지한다.

효과:

- 점선 차선 gap에서 차선이 갑자기 다른 후보로 튀는 문제 완화
- 차량이나 segmentation noise 때문에 polygon이 휘는 문제 완화
- 프레임 간 lane centerline 안정화

### 7.5 점선 gap 보간

점선 차선은 scan line에 따라 검출되는 줄과 검출되지 않는 줄이 반복된다.

이를 해결하기 위해 이전 lane model을 y 방향으로 균일하게 resampling한다.

즉, 실제 mask가 끊겨 있어도 lane model은 연속적인 polyline으로 유지된다.

### 7.6 lane width 검증

좌측 차선과 우측 차선을 같은 y 좌표 기준으로 정렬한 뒤, 차선 폭을 검사한다.

```text
lane_width = right_x - left_x
```

차선 폭이 너무 좁거나 음수가 되면 잘못된 polygon으로 보고 제거한다.

이 검증은 ego lane polygon이 뒤집히거나 과도하게 찌그러지는 상황을 막기 위한 장치이다.

---

## 8. 생성되는 기하 구조

차선 후처리 결과로 `LaneGeometry`가 만들어진다.

주요 필드는 다음과 같다.

| 필드 | 의미 |
|---|---|
| `ego_lane_polygon` | 현재 차량이 주행 중인 ego lane 영역 |
| `centerline` | ego lane 중앙선 |
| `left_lane_pts` | 좌측 차선 polyline |
| `right_lane_pts` | 우측 차선 polyline |
| `crosswalk_zone` | 우회전 경로와 겹치는 횡단보도 활성 영역 |
| `path_corridor` | 차량이 우회전하며 지나갈 것으로 예상되는 corridor |
| `lane_confidence` | scan line에서 차선을 안정적으로 찾은 비율 |
| `stop_line_y` | 정지선의 평균 y 좌표 |

`path_corridor`는 보행자와 차량이 우회전 예상 경로 안에 있는지 판단하는 데 사용된다.

---

## 9. 교차로 상태 FSM

교차로 상황은 단일 프레임만 보고 판단하지 않고, FSM으로 상태를 추적한다.

상태는 다음과 같다.

| 상태 | 의미 |
|---|---|
| `LANE_TRACKING` | 일반 차선 추적 상태 |
| `CROSSWALK_APPROACH` | 횡단보도 또는 정지선 접근 상태 |
| `ENTERING_INTERSECTION` | 교차로 진입 가능성이 있는 상태 |
| `INTERSECTION_TRACKING` | 교차로 내부 주행 상태 |
| `RELOCK_LANE` | 교차로 통과 후 차선을 다시 확보하는 상태 |

### 상태 전이 기준

주요 상태 전이 기준은 다음과 같다.

- `crosswalk_zone_exists == True`
- `crosswalk_pixel_count` 증가
- `stop_line_visible == True`
- `stop_line_y_ratio`가 화면 하단에 가까워짐
- `lane_confidence` 급락
- 교차로 이후 `lane_confidence` 회복

즉, 차선이 안 보이는 상황을 단순 실패로 처리하지 않고, 교차로 내부 진입 가능성으로 해석한다.

---

## 10. SceneContext

`SceneContext`는 한 프레임에서 판단에 필요한 모든 정보를 모은 구조체이다.

주요 변수는 다음과 같다.

| 변수 | 의미 |
|---|---|
| `lane_confidence` | 차선 신뢰도 |
| `crosswalk_zone_exists` | 우회전 경로상 횡단보도 영역 존재 여부 |
| `stop_line_visible` | 정지선 검출 여부 |
| `pedestrian_in_corridor` | 보행자가 우회전 예상 경로 안에 있는지 |
| `pedestrian_in_crosswalk` | 보행자가 횡단보도 위에 있는지 |
| `pedestrian_near_crosswalk` | 보행자가 횡단보도 주변에 있는지 |
| `vehicle_in_turn_path` | 차량이 우회전 예상 경로에 있는지 |
| `nearest_turn_path_vehicle_distance_m` | 우회전 경로 내 가장 가까운 차량 거리 |
| `conflicting_vehicle_count` | 교차 차량 가능성이 있는 차량 수 |
| `nearest_conflicting_vehicle_distance_m` | 가장 가까운 교차 차량 거리 |
| `vehicle_signal` | 전방 차량 신호 상태 |
| `pedestrian_signal` | 보행자 신호 상태 |
| `right_turn_signal` | 우회전 전용 신호 상태 |
| `red_light_stop_required` | 적색 신호 일시정지 필요 여부 |
| `stop_completed_on_red` | 적색 신호에서 이미 일시정지했다고 가정할지 여부 |

---

## 11. STOP / CAUTION / GO 판단 규칙

판단은 위험도가 높은 조건부터 순차적으로 검사한다.

### 11.1 STOP 조건

다음 조건 중 하나라도 만족하면 `STOP`이다.

| 조건 | 설명 |
|---|---|
| `right_turn_signal == RED` | 우회전 전용 신호가 적색이면 정지 |
| `red_light_stop_required == True` and `stop_completed_on_red == False` | 전방 차량 신호 적색에서 아직 일시정지하지 않았으면 정지 |
| `pedestrian_in_corridor == True` | 우회전 진행 경로에 보행자가 있으면 정지 |
| `pedestrian_in_crosswalk == True` | 횡단보도 위 보행자가 있으면 정지 |
| `pedestrian_near_crosswalk == True` | 횡단보도 주변 보행자가 있으면 정지 |
| `state == CROSSWALK_APPROACH` and `pedestrian_count > 0` | 횡단보도 접근 중 보행자 감지 시 정지 |
| `vehicle_in_turn_path == True` and `nearest_turn_path_vehicle_distance_m < 7m` | 우회전 경로 차량이 매우 가까우면 정지 |

### 11.2 CAUTION 조건

STOP 조건은 아니지만 위험 또는 불확실성이 있으면 `CAUTION`이다.

| 조건 | 설명 |
|---|---|
| 우회전 경로 차량 거리 불명 | 차량은 있으나 거리 추정 불가 |
| `nearest_turn_path_vehicle_distance_m < 15m` | 우회전 경로 차량이 가까움 |
| `vehicle_signal == YELLOW` | 황색 신호 |
| `right_turn_signal == YELLOW` | 우회전 전용 황색 신호 |
| `red_light_stop_required == True` and `stop_completed_on_red == True` | 적색에서 일시정지 후 양보하며 서행 |
| `conflicting_vehicle_count > 0` and `nearest_conflicting_vehicle_distance_m < 20m` | 교차 차량 가능성 |
| `state == ENTERING_INTERSECTION` | 교차로 진입 가능성 |
| `state == INTERSECTION_TRACKING` | 교차로 내부 주행 |
| `state == CROSSWALK_APPROACH` | 횡단보도 접근 |
| `lane_confidence < 0.30` | 차선 신뢰도 낮음 |
| `state == RELOCK_LANE` | 차선 재확보 중 |

### 11.3 GO 조건

위의 STOP, CAUTION 조건이 모두 없으면 `GO`이다.

단, 이때도 의미는 빠른 진행이 아니라 다음과 같다.

```text
clear: slow right turn
```

즉, 위험 조건이 없으므로 서행 우회전 가능하다는 뜻이다.

---

## 12. 현재 도로교통법 반영 방식

우회전 상황에서 중요한 법규 취지는 다음과 같이 반영했다.

| 법규 취지 | 코드 반영 |
|---|---|
| 적색 신호 시 우선 일시정지 | `red_light_stop_required`와 `stop_completed_on_red`로 판단 |
| 보행자 보호 우선 | 보행자가 corridor, crosswalk, near crosswalk에 있으면 STOP |
| 우회전 전용 신호가 있으면 해당 신호 우선 | `right_turn_signal`이 RED면 STOP |
| 황색 또는 불확실 상황은 보수적으로 판단 | CAUTION 처리 |
| 교차 차량 방해 방지 | `conflicting_vehicle_count`, `nearest_conflicting_vehicle_distance_m`로 CAUTION |

이 구현은 법률 문장을 그대로 기계적으로 판정하는 것이 아니라, 영상 인식 결과로 보조 판단을 제공하는 방식이다.

---

## 13. 화면 표시 방식

시각화 화면에는 다음 정보가 표시된다.

- 객체 bbox
- 객체 confidence
- 차량 거리
- ego lane polygon
- path corridor
- centerline
- crosswalk active zone
- STOP / CAUTION / GO badge
- FPS 및 추론 시간
- lane confidence
- 신호 상태
- 한글 경고문

최근 개선으로 경고문은 변수명과 한글 설명이 함께 표시된다.

예시:

```text
vehicle_in_turn_path=True, nearest_turn_path_vehicle_distance_m=12.4m < 15m
우회전 경로 차량: 주의
```

예시:

```text
lane_confidence=0.24 < 0.30
차선 신뢰도 낮음: 주의
```

---

## 14. v4.mp4 테스트 과정에서 확인한 문제와 개선

### 14.1 차선 영역이 휘는 문제

문제:

- 점선 차선으로 인해 화면 하단 차선이 끊김
- segmentation mask가 순간적으로 다른 차선을 잡음
- 차량 bbox 내부의 윤곽이 lane class로 들어옴
- ego lane polygon이 휘거나 뒤틀림

적용한 해결:

- 차량 bbox 내부 lane pixel 제거
- 이전 lane model과 후보 매칭
- 튄 후보 reject
- 점선 gap 보간
- lane width 검증

### 14.2 황색 차선 인식 문제

관찰:

- v4.mp4에서 황색 중앙선이 약하게 인식되는 구간이 있음
- HSV 기반 황색 차선 보강을 실험함
- 하지만 주유소 간판, 노란 표지판 등도 함께 잡히는 false positive 발생

결론:

- 색상 기반 보강은 롤백
- 황색 차선은 데이터셋 라벨링과 BiSeNet 재학습으로 해결하는 것이 더 안정적
- 추가로 vanishing point, road ROI, 차선 기울기 제약을 함께 쓰면 개선 가능

### 14.3 차량 거리 과대 추정

문제:

- bbox 기반 거리 추정이 실제보다 멀게 나오는 경향

적용:

- `--distance-scale` 옵션 추가
- 기본값을 `0.8`로 설정

---

## 15. 현재 한계

| 한계 | 원인 | 개선 방향 |
|---|---|---|
| 황색 차선 인식 불안정 | 학습 데이터 부족, 마모 차선, 역광 | 황색 차선 class 강화 라벨링 후 재학습 |
| 색상 기반 보강 오검출 | 간판, 표지판, 주유소 색상 혼입 | 색상 단독 사용 금지, ROI/기울기/시간 추적 결합 |
| 거리 추정 오차 | 차량 실제 크기 가정, bbox 흔들림 | 카메라 캘리브레이션, ground-plane 기반 거리식 검토 |
| 신호등 자동 판별 한계 | 작은 crop, 반사광, bbox 흔들림 | 신호등 전용 모델 또는 temporal voting |
| 교차로 구조 다양성 | 도로마다 차선/횡단보도/신호 체계 다름 | 데이터셋 다양화 및 rule parameter 조정 |
| 라즈베리파이 성능 | YOLO와 BiSeNet 동시 추론 비용 | 입력 해상도 축소, interval 조절, ONNX/NCNN 변환 |

---

## 16. 라즈베리파이 실행 옵션

라즈베리파이에서는 속도를 위해 입력 크기와 추론 주기를 낮춰야 한다.

예시:

```powershell
python edge\inference\intersection_demo.py `
  --source 0 `
  --width 480 `
  --height 270 `
  --imgsz 320 `
  --seg-input-h 192 `
  --seg-input-w 320 `
  --yolo-interval 3 `
  --seg-interval 2 `
  --conf 0.4 `
  --vehicle-signal auto `
  --pedestrian-signal auto `
  --right-turn-signal none `
  --camera-hfov 70 `
  --distance-scale 0.8
```

옵션 의미:

| 옵션 | 의미 |
|---|---|
| `--width`, `--height` | 처리할 영상 해상도 |
| `--imgsz` | YOLO 입력 크기 |
| `--seg-input-h`, `--seg-input-w` | BiSeNetV2 입력 크기 |
| `--yolo-interval` | YOLO를 몇 프레임마다 실행할지 |
| `--seg-interval` | segmentation을 몇 프레임마다 실행할지 |
| `--conf` | YOLO confidence threshold |
| `--distance-scale` | 차량 거리 보정 계수 |

---

## 17. 발표 슬라이드 구성안

| 슬라이드 | 제목 | 핵심 내용 |
|---:|---|---|
| 1 | 문제 정의 | 우회전 교차로에서 보행자, 차량, 신호를 동시에 봐야 하는 이유 |
| 2 | 시스템 개요 | YOLO + BiSeNetV2 + FSM 구조 |
| 3 | 객체 검출 | YOLO 검출 대상과 차량 거리 추정 |
| 4 | 도로 구조 분할 | BiSeNetV2 segmentation class와 역할 |
| 5 | 차선 후처리 | scan line, bbox 제거, 이전 모델 매칭, 점선 gap 보간 |
| 6 | 교차로 상태 FSM | LANE_TRACKING → CROSSWALK_APPROACH → INTERSECTION_TRACKING → RELOCK_LANE |
| 7 | 판단 규칙 | STOP / CAUTION / GO 우선순위 |
| 8 | 화면 결과 | bbox, 거리, corridor, 한글 경고문 |
| 9 | 테스트 결과 | v4.mp4에서 확인한 동작 |
| 10 | 한계 및 개선 | 황색 차선, 거리 보정, 신호등 판별, 라즈베리파이 최적화 |

---

## 18. 발표에서 강조할 포인트

1. 단순히 segmentation mask를 칠하는 프로그램이 아니다.
2. YOLO 객체 정보와 BiSeNet 도로 구조 정보를 결합한다.
3. 차량 bbox 내부 lane 제거로 차선 오검출을 줄였다.
4. 이전 lane model을 이용해 점선 차선 gap을 보간한다.
5. 우회전 경로 corridor를 만들어 보행자와 차량의 위험 여부를 판단한다.
6. FSM으로 교차로 접근, 진입, 내부, 탈출 후 차선 재확보 상태를 구분한다.
7. STOP/CAUTION/GO 판단 근거를 변수명과 함께 한글로 표시한다.
