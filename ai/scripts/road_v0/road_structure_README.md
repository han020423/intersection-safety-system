# Road structure perception for intersection safety assistance

이 프로그램은 이미 만들어 둔 **YOLO 객체검출 모델(best.pt)** 을 그대로 사용하면서,
추가로 **차선 / 도로 경계 / 교차로 진입 상태**를 가볍게 추정하도록 만든 파이프라인입니다.

## 핵심 아이디어

딥러닝으로 모든 것을 한 번에 세그멘테이션하지 않고, 아래처럼 나눕니다.

1. **YOLO**
   - 보행자
   - 차량
   - 차량 신호등
   - 보행자 신호등
   - 횡단보도
   - 좌회전 표지

2. **클래식 비전 기반 도로 구조 인식**
   - 차선 후보: white/yellow 색상 마스크 + Canny + HoughLinesP
   - 도로 영역 후보: 하단 중심 패치의 노면 색/질감 기반 threshold
   - 교차로 대응: 횡단보도 검출 + 차선 신뢰도 저하 시 road-boundary 기반 모드 전환

즉,
- **일반 도로**에서는 차선을 우선 사용
- **교차로/복잡 구간**에서는 도로 영역과 도로 경계를 우선 사용
합니다.

## 출력

각 프레임마다 다음 정보를 얻을 수 있습니다.

- `mode`: `lane` / `road_boundary` / `intersection`
- `center_x`: 현재 주행 중심선 x
- `offset_px`: 영상 중심 대비 편차
- `heading_deg`: 도로 진행 방향 추정
- `lane_confidence`
- `road_confidence`
- `intersection_likely`

## 실행 예시

```bash
python road_structure_assist.py \
  --weights best.pt \
  --source 0 \
  --width 640 --height 360 \
  --show
```

영상 파일로 테스트:

```bash
python road_structure_assist.py \
  --weights best.pt \
  --source test_video.mp4 \
  --show --save output.mp4
```

## Raspberry Pi 권장 세팅

```bash
python road_structure_assist.py \
  --weights best.pt \
  --source 0 \
  --width 640 --height 360 \
  --imgsz 640 \
  --yolo-interval 2 \
  --show
```

더 느리면:
- `--width 512 --height 288`
- `--yolo-interval 3`
- YOLO 입력을 더 작은 크기로 사용

## 설치

```bash
pip install ultralytics opencv-python numpy
```

라즈베리파이에서는 가능하면 `opencv-python-headless` 또는 시스템 OpenCV를 사용하는 편이 가볍습니다.

## 튜닝 포인트

### 1) 차선 색상 범위
현재 기본값은
- white: HLS 기반
- yellow: HSV 기반
으로 잡혀 있습니다.

카메라/노출/도로색에 따라 아래 값을 조정하면 됩니다.
- `white_mask`
- `yellow_mask`

### 2) 차선 검출 강도
다음 파라미터를 조절하세요.
- `cv2.Canny(..., 70, 140)`
- `cv2.HoughLinesP(... threshold, minLineLength, maxLineGap)`

### 3) 도로영역 추정
하단 중앙 패치에서 현재 노면 색을 샘플링하므로,
카메라 장착 위치가 너무 높거나 범퍼가 많이 보이면 seed patch 위치를 조금 위로 조정하면 좋습니다.

## 한계

이 방식은 **가볍고 빠른 대신**, 아래 상황에서는 성능이 흔들릴 수 있습니다.

- 비 오는 야간 반사
- 차선이 거의 지워진 도로
- 아스팔트 색이 크게 달라지는 구간
- 그림자/역광이 매우 강한 상황

## 다음 단계

정확도를 더 올리려면 다음 순서가 좋습니다.

1. 지금 코드로 베이스라인 확보
2. 실패 케이스 수집
3. `road mask`만 경량 세그멘테이션(예: Fast-SCNN/BiSeNet 계열)으로 교체
4. YOLO는 객체 검출 전용으로 유지

이 구조가 Raspberry Pi 실시간성 측면에서는 가장 안전한 편입니다.
