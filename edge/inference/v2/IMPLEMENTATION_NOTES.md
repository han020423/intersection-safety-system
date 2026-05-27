# road_v4 v2 구현 정리

## 만든 이유

기존 판단 프로그램은 BiSeNetV2 세그멘테이션 class id를 다음처럼 해석했다.

```text
1~6: 차선
7: 횡단보도
8: 정지선
```

하지만 `road_v4` 데이터셋은 실선/점선을 구분하지 않고 색상만 구분하도록 다시 만들었다.

```text
0: background
1: lane_white
2: lane_yellow
3: lane_blue
4: crosswalk
5: stop_line
```

따라서 기존 inference 코드를 그대로 쓰면 횡단보도와 정지선을 찾지 못한다.  
그래서 기존 프로그램은 보존하고, `edge/inference/v2`에 road_v4 전용 판단 프로그램을 새로 분리했다.

## 중요한 변경점

### 1. 모델 구조

최종 방향은 `ResNet-18`을 붙인 무거운 모델이 아니라, 기존 경량 `BiSeNetV2` 구조다.

```text
입력 영상
-> 경량 BiSeNetV2
-> 6클래스 도로 구조 마스크
```

이유는 라즈베리파이5에서 실시간성을 확보해야 하기 때문이다.  
`ResNet-18` 버전은 정확도 비교용으로는 쓸 수 있지만, Pi 배포 후보로는 무겁다.

### 2. 후처리 class id

`road_v4_postprocess.py`에서 새 class id를 기준으로 후처리한다.

```text
LANE_CLASS_MIN = 1
LANE_CLASS_MAX = 3
CROSSWALK_CLASS = 4
STOP_LINE_CLASS = 5
```

차선 추출은 기존 scan-line 기반 로직을 재사용한다.  
다만 차선 범위만 `1~3`으로 바꿔 흰색/노란색/파란색 차선을 모두 ego lane 후보로 본다.

### 3. 한 프레임 처리 흐름

`intersection_demo_v2.py`의 처리 순서:

```text
프레임 입력
-> YOLO 객체 검출
-> road_v4 BiSeNetV2 세그멘테이션
-> 차량 bbox 내부 차선 픽셀 제거
-> scan-line 기반 좌/우 차선 추출
-> ego lane / centerline / path corridor 생성
-> crosswalk zone / stop line 추출
-> SceneContext 생성
-> FSM으로 STOP / CAUTION / GO 판단
-> 화면 시각화
```

### 4. 학습 코드

Colab 학습 코드는 아래 파일이다.

```text
ai/scripts/road_v4/colab_train_bisenetv2_road_v4_light.py
```

같은 폴더의 `bisenetv2.py`를 함께 Colab에 올려야 한다.

학습 결과 중 실제 inference에 넣을 파일:

```text
best_light_infer.pt
```

이 파일은 optimizer state를 제외한 추론용 checkpoint라서 기존 full checkpoint보다 작다.

## 실행 방법

학습된 `best_light_infer.pt`를 다음 이름으로 넣는다.

```text
edge/inference/v2/road_v4_best_light.pt
```

영상 테스트:

```bash
python edge/inference/v2/intersection_demo_v2.py --source edge/inference/v4.mp4 --show
```

결과 저장:

```bash
python edge/inference/v2/intersection_demo_v2.py --source edge/inference/v4.mp4 --save edge/inference/v2/out_v4.mp4
```

라즈베리파이용 저부하 실행:

```bash
python edge/inference/v2/intersection_demo_v2.py --source 0 --device cpu --width 480 --height 270 --seg-input-h 256 --seg-input-w 464 --yolo-interval 3 --seg-interval 2 --show
```

## 확인해야 할 항목

테스트 영상에서 우선 확인할 것:

```text
1. 흰색/노란색/파란색 차선이 class 1~3으로 안정적으로 잡히는가
2. 횡단보도 class 4가 ego lane 근처에서 crosswalk zone으로 잡히는가
3. 정지선 class 5가 stop_line_y로 검출되는가
4. lane_confidence가 교차로 진입/이탈 상황에서 과도하게 출렁이지 않는가
5. STOP / CAUTION / GO 판단 reason이 현재 화면과 맞는가
```

## 주의점

`road_v4_best.pt`처럼 ResNet-18 실험에서 나온 큰 checkpoint는 v2 경량 추론기에 맞지 않는다.  
v2에서 사용할 파일은 반드시 경량 BiSeNetV2로 다시 학습한 `best_light_infer.pt` 계열이어야 한다.
