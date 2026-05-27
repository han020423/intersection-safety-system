# v2_0 Recognition Visualizer

`v2_0`은 현재 **판단 로직을 모두 제거한 순수 인지 시각화 버전**이다.

목적은 단순하다.

```text
YOLO가 무엇을 검출하는지
BiSeNet이 어떤 도로 구조 마스크를 내는지
```

이 두 가지를 먼저 눈으로 확인한다.

## 현재 v2_0에서 하는 것

```text
입력 프레임
-> YOLO 객체 검출
-> road_v4 BiSeNet 세그멘테이션
-> YOLO bbox 표시
-> 차선 class는 점선 빈 공간을 연결한 polyline으로 표시
-> 현재 차량이 주행 중인 차선 경계와 주행 차선 영역 표시
-> 횡단보도/정지선 class mask 반투명 표시
-> FPS / 추론 시간 / class 픽셀 수 표시
```

## 현재 v2_0에서 하지 않는 것

```text
판단용 차선 후처리 없음
판단용 ego lane / path corridor 생성 없음
SceneContext 없음
FSM 판단 없음
STOP / CAUTION / GO 없음
```

즉, 이 폴더는 판단 프로그램이 아니라 **모델 인지 결과와 reboot ego lane 후보를 확인하는 뷰어**다.

## 사용 모델

기본 segmentation weight:

```text
edge/inference/v2_0/road_v4_best.pt
```

없으면 자동으로 다음 경로도 찾는다.

```text
edge/inference/v2/road_v4_best.pt
```

YOLO weight:

```text
edge/inference/yolo.pt
```

## road_v4 Segmentation Class

| id | class |
|---:|---|
| 0 | background |
| 1 | lane_white |
| 2 | lane_yellow |
| 3 | lane_blue |
| 4 | crosswalk |
| 5 | stop_line |

## 실행

영상 저장:

```bash
python edge/inference/v2_0/intersection_demo_v2_0.py --source edge/inference/v4.mp4 --save edge/inference/v2_0/out_v4_recognition.mp4 --legend
```

화면 표시:

```bash
python edge/inference/v2_0/intersection_demo_v2_0.py --source edge/inference/v4.mp4 --show --legend
```

처음 100프레임만 테스트:

```bash
python edge/inference/v2_0/intersection_demo_v2_0.py --source edge/inference/v4.mp4 --max-frames 100 --save edge/inference/v2_0/out_100f.mp4 --legend
```

CUDA 사용:

```bash
python edge/inference/v2_0/intersection_demo_v2_0.py --source edge/inference/v4.mp4 --device cuda --show --legend
```

BiSeNet만 보기:

```bash
python edge/inference/v2_0/intersection_demo_v2_0.py --source edge/inference/v4.mp4 --no-yolo --show --legend
```

YOLO만 보기:

```bash
python edge/inference/v2_0/intersection_demo_v2_0.py --source edge/inference/v4.mp4 --no-seg --show
```

저해상도 빠른 테스트:

```bash
python edge/inference/v2_0/intersection_demo_v2_0.py --source edge/inference/v4.mp4 --width 480 --height 270 --seg-input-h 256 --seg-input-w 464 --save edge/inference/v2_0/out_fast.mp4
```

## 다음 단계

이 시각화 결과를 보고 다음을 판단한다.

```text
1. BiSeNet이 차선/횡단보도/정지선을 충분히 잘 잡는가
2. YOLO가 차량/보행자/신호등을 놓치지 않는가
3. 모델 인지 결과가 안정적이면 그 다음에 후처리와 FSM을 다시 얹는다
```

## 차선 표시 방식

차선 class `1~3`은 마스크 면을 그대로 칠하지 않는다.

```text
1. 차선 mask를 여러 y좌표에서 scan-line으로 스캔
2. 각 scan-line에서 차선 픽셀 cluster 중심점 추출
3. 중심점들을 x거리 기준으로 lane track으로 그룹화
4. 각 track을 1차/2차 곡선으로 피팅
5. 점선 사이 빈 공간을 연결한 하나의 연속 선으로 표시
6. 최하단 점선과 화면 바닥 사이 gap은 짧고 안정적일 때만 제한적으로 연장
```

따라서 점선 차선도 화면에서는 끊어진 면이 아니라 하나의 차선 선분처럼 보인다.
다만 화면 바닥까지의 구간은 관측점 밖 외삽이므로, 피팅 오차가 작고 하단 gap이
짧으며 차선 기울기가 과하지 않을 때만 연결한다. 조건을 만족하지 않으면 실제
감지된 최하단 차선 조각까지만 표시한다.

## Reboot Ego Lane 표시

`ego_lane_reboot.py`는 기존 `postprocess.py`나 `v2`의 차선 후처리를 쓰지 않고 새로 만든
현재 주행 차선 표시 모듈이다. 판단/FSM용 로직이 아니라, 세그멘테이션 결과만으로
"내 차량이 지금 밟고 있는 차선이 어디인지"를 눈으로 확인하기 위한 시각화다.

처리 흐름:

```text
road_v4 seg_mask
-> lane_white/lane_yellow/lane_blue만 lane mask로 추출
-> connected component로 작은 잡음 제거
-> 여러 y좌표 scan band에서 차선 cluster 중심 keypoint 추출
-> keypoint를 x/y 연속성 기준으로 lane candidate polyline으로 그룹화
-> 각 candidate를 y->x 곡선으로 피팅
-> 이전 프레임 track과 candidate를 매칭하고 EMA로 temporal smoothing
-> 화면 하단 중앙을 ego 차량 위치로 가정해 좌/우 ego boundary track 선택
-> 한쪽 경계만 보이면 최근 양쪽 차선 폭 캐시 또는 원근 폭 프로파일로 반대편 경계 추정
-> 좌/우 경계를 닫아 ego lane polygon 생성
-> 주행 차선 영역, 좌/우 경계, 중앙선, confidence 표시
-> 주변 차선 track과 비교해 leftmost/rightmost lane role 표시
```

confidence는 차선 candidate 품질, track 안정성, 좌/우 폭의 일관성을 섞어 만든
시각화용 점수다. 정답 라벨 기반 확률은 아니므로, 판단 로직에서는 별도 검증 없이
안전 판단 기준으로 직접 사용하지 않는다.

한쪽 차선만 보이는 프레임에서는 영역이 크게 비틀릴 수 있으므로 다음 안전장치를 둔다.

```text
1. 최근 양쪽 차선이 모두 보였던 프레임의 y별 lane width를 캐시한다.
2. 캐시가 있으면 고정 폭이 아니라 y별 폭 프로파일로 반대편 경계를 만든다.
3. 캐시가 없으면 위쪽으로 갈수록 좁아지는 perspective 폭만 사용한다.
4. 캐시 없이 한쪽만 보일 때는 polygon을 화면 상단까지 길게 외삽하지 않는다.
5. 폭이 너무 넓거나 중심이 ego 위치에서 크게 벗어나면 polygon 표시를 거부한다.
```

표시 색:

| 요소 | 표시 |
|---|---|
| 주행 차선 영역 | 초록 반투명 면 |
| 좌/우 경계 | 노란 선 |
| 중앙선 | 파란 선 |
| 상태/신뢰도 | `ego lane tracked conf:0.xx cand:N trk:N` 텍스트 |
| 차로 위치 | `lane role Lidx:N Ridx:N leftmost:T/F/U rightmost:T/F/U` 텍스트 |

## Lane Role 판단

우회전/비보호좌회전 판단에 쓰기 위해 현재 ego lane이 도로의 왼쪽/오른쪽 끝 차로인지
추정한다. 이 단계는 안전 판단을 직접 내리지 않고, 차로 위치 정보만 만든다.

```text
tracked lane boundaries
-> 화면 하단 ego 위치 근처에서 각 boundary의 x좌표 계산
-> x좌표 기준으로 왼쪽부터 오른쪽까지 정렬
-> ego 차량 중심을 감싸는 좌/우 boundary 쌍을 현재 차로로 정의
-> boundary 목록에서 해당 쌍의 위치를 보고 leftmost/rightmost 여부 계산
-> YOLO 차량 bbox가 ego lane 바깥 좌/우에 있으면 해당 방향에 인접 차로가 있는 신호로 반영
-> ego lane 좌측 경계가 충분히 yellow이면 중앙선 쪽으로 보고 leftmost 신호로 반영
-> 시간적으로 같은 결과가 유지되는 프레임 수를 confidence에 반영
```

출력 의미:

| 값 | 의미 |
|---|---|
| `Lidx` | 현재 차로가 왼쪽에서 몇 번째인지. 0부터 시작 |
| `Ridx` | 현재 차로가 오른쪽에서 몇 번째인지. 0부터 시작 |
| `leftmost` | 왼쪽 끝 차로 여부. `T/F/U` |
| `rightmost` | 오른쪽 끝 차로 여부. `T/F/U` |
| `adjV` | ego lane 좌/우 바깥에서 감지된 차량. `L`, `R`, `LR`, `-` |
| `LY` | 좌측 boundary 주변 yellow lane 비율. 황색 신호가 없으면 `-` |
| `U` | Unknown. 차선이 부족하거나 정렬이 불안정해서 확정하지 않음 |

예를 들어 boundary가 `A B C D` 네 개이고 ego lane이 `C-D` 사이라면
`Lidx:2`, `Ridx:0`, `rightmost:T`, `leftmost:F`가 된다.

옆 차선 boundary가 차량에 가려져 보이지 않아도, YOLO가 ego lane 바깥쪽 차량을
감지하면 그 방향에는 인접 차로가 있는 것으로 보정한다. 예를 들어 차선 기준으로
`rightmost:T`가 나왔더라도 오른쪽 바깥에 차량이 있으면 `rightmost:F`,
`adjV:R`로 표시한다.

한국 도로 환경에서 좌측 경계가 안정적으로 황색이면 중앙선/반대 방향 교통류와의
경계일 가능성이 높다. 따라서 ego lane의 좌측 boundary 주변에서 `lane_yellow`
비율이 충분히 높으면 `leftmost:T`로 보정하고 `LY:0.xx`로 표시한다. 이 규칙은
오검출을 줄이기 위해 좌측 boundary 주변 샘플 수와 yellow 비율을 함께 확인한다.

보이는 boundary가 2개뿐인 경우는 실제 1차로일 수도 있고, 주변 차선을 놓친 것일
수도 있다. 그래서 바로 `leftmost:T rightmost:T`로 확정하지 않고 다음처럼 처리한다.

```text
left boundary가 yellow이면 leftmost:T, 아니면 leftmost:U
오른쪽 바깥 차량이 있으면 rightmost:F
오른쪽 바깥 차량이 없고 left boundary yellow 상태가 여러 프레임 안정적이면 rightmost:T
그 외에는 rightmost:U
```

끄기:

```bash
python edge/inference/v2_0/intersection_demo_v2_0.py --source edge/inference/v4.mp4 --no-ego-lane --show
```

투명도 조절:

```bash
python edge/inference/v2_0/intersection_demo_v2_0.py --source edge/inference/v4.mp4 --ego-lane-alpha 0.35 --show
```

## Crosswalk Zone 판단

`road_structure_reboot.py`는 횡단보도와 보행자를 바로 `CAUTION`으로 바꾸지 않고, 안전 판단 FSM이 사용할 수 있는 구조 정보만 만든다. 이유는 `v2_0`의 역할이 인식/구조화 결과 확인이고, 실제 경고 여부는 우회전/비보호좌회전/신호/속도 같은 조건까지 합쳐서 별도 단계에서 결정해야 하기 때문이다.

정의:

```text
crosswalk_mask = road_v4 seg_mask에서 class 4(crosswalk)만 추출
path_corridor  = ego lane polygon을 약간 확장한 현재 주행 경로 영역

crosswalk_zone(active)
  = crosswalk_mask ∩ path_corridor
  = 현재 차량이 그대로 진행하면 직접 만날 가능성이 큰 횡단보도 영역

crosswalk_near_zone
  = crosswalk_mask ∩ wider_dilated(path_corridor)를 한 번 더 확장한 영역
  = 현재 경로와 바로 겹치지는 않아도 보행자가 active zone으로 들어올 수 있는 주변 횡단보도 영역
```

### Path Corridor

`PathCorridorPipeline`은 임의 회전 곡선이나 긴 예측을 만들지 않는다. 현재 프로젝트에서는 근거가 있는 범위만 사용한다.

```text
1. lane
   - ego lane polygon이 유효하면 polygon을 mask로 채움
   - 화면 하단부도 최소 5px 정도는 확장하고, 상단부로 갈수록 여유 폭을 키움
   - 최대 확장은 lane width의 약 20%, 최소 14px, 최대 약 40px 수준으로 제한
   - 목적은 ego 바로 앞 영역을 과하게 넓히지 않고 먼 영역의 원근/segmentation 오차만 흡수하는 것

2. cached_lane
   - 현재 프레임에서 lane corridor가 사라졌지만 직전 corridor가 있으면 짧게 유지
   - 기본 TTL은 3프레임
   - 프레임이 지날수록 confidence 감소

3. unavailable
   - lane도 없고 short cache도 만료되면 active crosswalk zone을 만들지 않음
```

표시 예:

```text
crosswalk active actP:0 nearP:1 path:lane pconf:0.62 age:0 conf:0.48
crosswalk active actP:0 nearP:1 path:cached_lane pconf:0.39 age:2 conf:0.30
crosswalk no_path_corridor actP:0 nearP:0 path:unavailable pconf:0.00 age:0 conf:0.00
```

화면 표시:

```text
path corridor는 항상 옅은 청록/파랑 underlay로 표시한다.
lane path는 청록 fill과 얇은 외곽선, cached_lane path는 더 진한 파랑 fill과 외곽선으로 표시한다.
crosswalk active/near overlay보다 아래 레이어에 그려 계산 근거만 확인할 수 있게 한다.
```

보행자 판단:

```text
YOLO pedestrian/person bbox의 bottom-center를 발 위치로 본다.
발 위치가 active mask 안이면 actP 증가
발 위치가 near mask 안이면 nearP 증가
```

표시 예:

| 값 | 의미 |
|---|---|
| `crosswalk active` | ego path와 겹치는 횡단보도가 있음 |
| `crosswalk near_only` | 횡단보도는 주변에 있지만 ego path와 직접 겹치는 부분은 약함 |
| `actP:N` | active zone 위 보행자 수 |
| `nearP:N` | near zone 위 보행자 수 |
| `path` | crosswalk zone 계산에 사용한 경로 출처. `lane`, `cached_lane`, `unavailable` |
| `pconf` | path corridor 자체의 신뢰도 |
| `age` | cached path가 몇 프레임 전 corridor인지 |
| `conf` | crosswalk mask 크기와 ego path overlap 기반 시각화 신뢰도 |

중요한 해석:

```text
actP > 0  -> 차량 진행 경로의 횡단보도 위 보행자
nearP > 0 -> 주변 횡단보도 영역의 보행자. 우회전/비보호좌회전에서는 주의 근거가 될 수 있음
```

단, 이 단계에서는 `CAUTION`이라는 단어를 출력하지 않는다. `actP/nearP`는 경고 그 자체가 아니라 경고 판단에 들어갈 근거값이다.

끄기:

```bash
python edge/inference/v2_0/intersection_demo_v2_0.py --source edge/inference/v4.mp4 --no-crosswalk-zone --show
```

short cache 프레임 수 조절:

```bash
python edge/inference/v2_0/intersection_demo_v2_0.py --source edge/inference/v4.mp4 --path-cache-frames 2 --show
```
