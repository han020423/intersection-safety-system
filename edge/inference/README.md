# Intersection Safety Demo

YOLO11n 객체 인식 + BiSeNetV2 도로 구조 세그멘테이션 + Rule-based FSM을 결합한
교차로 안전 데모 프로그램.

## 파일 구조

```
edge/inference/
├── intersection_demo.py   # 메인 진입점 (입력/출력/루프)
├── perception.py          # YOLO11n + BiSeNetV2 모델 래퍼
├── postprocess.py         # 후처리 (ego lane, centerline, crosswalk zone, path corridor)
├── state_machine.py       # FSM 상태 전이 + STOP/CAUTION/GO 판단
├── visualizer.py          # 데모 화면 오버레이 그리기
├── bisenetv2.py           # BiSeNetV2 모델 아키텍처 (추론 전용)
└── README.md              # 이 문서
```

## 필요한 패키지

```bash
pip install torch torchvision opencv-python numpy ultralytics
```

## Weight 설정

아래 두 파일을 `edge/inference/` 디렉터리에 배치하거나 실행 시 경로를 지정:

| 모델 | 기본 파일명 | 옵션 |
|------|------------|------|
| YOLO11n | `yolo.pt` | `--yolo-weights` |
| BiSeNetV2 | `bisenet_best.pth` | `--seg-weights` |

프로그램은 자동으로 `ai/scripts/road_v1/` 경로도 탐색합니다.

## 실행 방법

```bash
# 비디오 파일 + 화면 표시
python intersection_demo.py --source video.mp4 --show

# 웹캠
python intersection_demo.py --source 0 --show

# 결과 저장 + 디버그 모드
python intersection_demo.py --source video.mp4 --show --save output.mp4 --debug

# 라즈베리파이 최적화
python intersection_demo.py --source 0 --show \
    --width 480 --height 270 \
    --yolo-interval 3 --seg-interval 2 \
    --seg-input-h 192 --seg-input-w 320 \
    --imgsz 320 --conf 0.4

# 우회전 보조 판단: 신호 상태를 수동 지정해서 테스트
python intersection_demo.py --source v1.mp4 --show \
    --vehicle-signal red \
    --right-turn-signal none

# 전방 적색에서 이미 일시정지를 완료했다고 가정한 뒤 서행/양보 판단
python intersection_demo.py --source v1.mp4 --show \
    --vehicle-signal red \
    --assume-stopped-on-red

# 차량 거리 표시 파라미터를 직접 지정
python intersection_demo.py --source v1.mp4 --show \
    --camera-hfov 70 \
    --vehicle-real-width 1.8 \
    --vehicle-real-height 1.5
```

실행 중 `d` 키로 디버그 시각화 토글, `q`/`ESC`로 종료.

## 우회전 보조 판단 기준

이 데모는 대한민국 도로교통법 기준의 우회전 보조장치로 보수적으로 판단한다.

- 전방 차량 신호가 적색이면 정지선/횡단보도/교차로 직전에서 먼저 일시정지한다.
- 우회전 전용 신호등이 있고 적색이면 우회전하지 않는다.
- 횡단보도를 통행 중이거나 통행하려는 보행자가 있으면 일시정지한다.
- 적색 신호에서 일시정지 후에도 다른 차마의 교통을 방해하지 않을 때만 서행 우회전할 수 있다.
- 우회전 예상 경로의 차량은 거리 기반으로 판단한다: 7m 미만은 `STOP`, 7~15m는 `CAUTION`, 15m 초과는 다른 위험 조건이 없으면 진행 가능으로 본다.
- 카메라/모델이 신호나 차량 위험을 확실히 판단하지 못하면 `CAUTION` 쪽으로 기울인다.

관련 옵션:

| 옵션 | 값 | 설명 |
|------|----|------|
| `--vehicle-signal` | `auto/red/yellow/green/unknown` | 전방 차량 신호. `auto`는 YOLO 신호등 crop 색상 추정 |
| `--pedestrian-signal` | `auto/red/green/unknown` | 보행자 신호. 현재 판단은 보행자 검출을 우선 |
| `--right-turn-signal` | `none/auto/red/yellow/green/unknown` | 우회전 전용 신호등. 없으면 `none` |
| `--assume-stopped-on-red` | flag | 전방 적색에서 이미 완전 정지를 완료했다고 가정 |

## 차량 거리 표시

차량 bbox에는 단안 카메라 기반 추정 거리(`m`)가 함께 표시된다. 논문에서 언급된 방식 중, 현재 YOLO 모델에 별도 거리 학습 head가 없으므로 camera calibration/pinhole 모델을 사용한다.

기본식:

```text
distance(m) = real_object_size(m) * focal_length(px) / bbox_size(px)
```

캘리브레이션된 focal length가 있으면 `--distance-focal-px`로 넣고, 없으면 `--camera-hfov`와 현재 프레임 너비로 focal length를 계산한다. 기본 차량 크기 가정은 폭 `1.8m`, 높이 `1.5m`이다.

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--no-distance` | off | 차량 거리 표시 비활성화 |
| `--camera-hfov` | `70.0` | 카메라 수평 화각(deg) |
| `--distance-focal-px` | `0.0` | 캘리브레이션 focal length(px). 0이면 화각 기반 자동 계산 |
| `--vehicle-real-width` | `1.8` | 평균 차량 실제 폭(m) |
| `--vehicle-real-height` | `1.5` | 평균 차량 실제 높이(m) |
| `--distance-scale` | `0.8` | 거리 보정 계수. 추정값이 멀게 나오면 낮추고, 가깝게 나오면 높임 |

단안 거리 추정은 카메라 설치 높이/렌즈 왜곡/차량 종류/부분 가림에 민감하므로, 표시값은 보조 정보로 사용한다.

## 후처리 로직 설명

### 1. Ego Lane Polygon
- 화면 하단~중간까지 수평 scan line을 20줄 그어 차선 픽셀(class 1~6) 탐색
- 각 scan line에서 x좌표 클러스터링 → 화면 중앙(ego) 기준 가장 가까운 좌/우 차선 선택
- 좌/우 polyline을 연결하여 polygon 생성 (한쪽만 있으면 추정 폭으로 보정)

### 2. Lane Centerline
- 좌/우 차선 polyline의 중간점을 연결
- 한쪽만 있으면 고정 offset 적용

### 3. Crosswalk Active Zone
- BiSeNetV2 crosswalk mask(class 7) 중 ego lane polygon과 겹치는 영역만 추출
- ego polygon을 dilate하여 약간의 여유 확보

### 4. Path Corridor
- centerline 좌우로 corridor_half_width만큼 확장한 직사각형 polygon
- 보행자 충돌 판정에 사용

### 5. FSM 상태 전이
```
LANE_TRACKING → CROSSWALK_APPROACH → ENTERING_INTERSECTION → INTERSECTION_TRACKING → RELOCK_LANE → LANE_TRACKING
```
- lane confidence, crosswalk 픽셀 변화, stop line 위치 등으로 전이
- 히스테리시스 카운터로 상태 떨림 방지
- 횡단보도/정지선이 사라지고 `lane_confidence >= 0.30`이 유지되면 교차로 통과 후 차선 재확보로 판단하여 `LANE_TRACKING`으로 복귀

### 6. 판단 규칙
| 판단 | 조건 |
|------|------|
| **STOP** | path corridor 안에 보행자 / crosswalk zone에 보행자 / 횡단보도 접근+보행자 |
| **CAUTION** | 교차로 진입/내부 / 횡단보도 접근(보행자 없음) / lane confidence < 0.3 |
| **GO** | 위 조건 모두 해당 없음 |

## Raspberry Pi 최적화 팁

1. **해상도 낮추기**: `--width 480 --height 270`
2. **YOLO 해상도 낮추기**: `--imgsz 320`
3. **프레임 스킵**: `--yolo-interval 3 --seg-interval 2`
4. **BiSeNetV2 입력 축소**: `--seg-input-h 192 --seg-input-w 320`
5. **ONNX 변환**: PyTorch → ONNX → OpenCV DNN으로 CPU 최적화 가능
6. **confidence 올리기**: `--conf 0.45`로 불필요한 검출 감소
7. **NumPy 연산 최소화**: 후처리에서 불필요한 루프 대신 벡터 연산 사용 (이미 적용됨)
