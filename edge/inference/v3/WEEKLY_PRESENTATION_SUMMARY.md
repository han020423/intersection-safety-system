# 이번 주차 발표자료 정리

## 슬라이드 1. 이번 주차 목표

```text
목표:
v3 인식 결과를 단순 시각화에서 끝내지 않고,
교차로 상황을 해석하고 운전자에게 설명 가능한 경고를 제공하는 구조로 확장
```

이번 주차 핵심 작업:

```text
1. 교차로 단계 자동 판단
2. 우회전 안전 판단 로직 구조화
3. 보행자 위험 판단 고도화
4. 적색 신호/일시정지 판단 추가
5. 음성 안내 기능 추가
6. 위험 이벤트를 교차로 단위로 기록
7. 발표/보고서용 판단 기준표 문서화
```

## 슬라이드 2. 전체 시스템 흐름

```text
입력 영상
-> YOLO 객체 검출
-> BiSeNetV2 도로 구조 segmentation
-> ego lane / 횡단보도 / 차로 역할 / 보행자 상태 추출
-> RoadStructureState로 요약
-> IntersectionPhase 자동 판단
-> SafetyDecisionEngine 판단
-> 화면 표시 + 음성 안내 + 이벤트 기록
```

핵심 변화:

```text
기존:
인식 결과를 영상 위에 표시하는 디버그 중심

개선:
인식 결과를 판단 가능한 상태값으로 바꾸고,
왜 STOP/CAUTION/GO인지 설명하는 구조
```

## 슬라이드 3. 인식 구조

사용한 인식 정보:

| 인식 대상 | 사용 모델/방법 | 판단에서의 역할 |
|---|---|---|
| 차량, 보행자, 신호등, 횡단보도 객체 | YOLO | 객체 위치, 신뢰도, 거리 추정 |
| 차선, 횡단보도, 정지선 mask | BiSeNetV2 | 도로 구조, 차선, 횡단보도 판단 |
| 현재 주행 차로 | lane mask 후처리 | ego lane, 차로 역할 판단 |
| 차량 신호 색상 | 신호등 bbox crop HSV 분석 | 적색 신호 정지 판단 |
| 보행자 이동 방향 | bbox foot-point tracking | 횡단보도 접근/이탈 판단 |

근거:

```text
BiSeNetV2는 실시간 semantic segmentation을 목적으로 제안된 구조
YOLO 계열 모델은 실시간 객체 검출에 널리 사용
```

참고:
- BiSeNetV2: Bilateral Network with Guided Aggregation for Real-time Semantic Segmentation
- Ultralytics YOLOv8 documentation

## 슬라이드 4. 차선/주행영역 처리

차선 처리 흐름:

```text
lane_white / lane_yellow / lane_blue mask 추출
-> connected component로 작은 노이즈 제거
-> 여러 y좌표 scan band에서 차선 픽셀 cluster 추출
-> cluster 중심점을 keypoint로 변환
-> keypoint를 lane candidate polyline으로 그룹화
-> y -> x 곡선으로 피팅
-> 프레임 간 temporal tracking
-> ego lane 좌/우 경계 선택
-> ego lane polygon 및 관심주행영역 생성
```

개선점:

```text
차선 mask를 그대로 면으로 쓰지 않고,
차선 중심점과 곡선 형태로 변환해서 현재 주행 차로를 안정적으로 추정
```

근거:

```text
차선 검출 연구에서는 segmentation 결과를 후처리해 차선 곡선/좌표로 변환하거나,
row-anchor 방식처럼 여러 y좌표에서 차선 x좌표를 예측하는 방식이 널리 사용됨
```

참고:
- Ultra Fast Structure-aware Deep Lane Detection
- End-to-end Lane Detection through Differentiable Least-Squares Fitting
- 딥러닝 기반 차선 영역 검출 및 차선 정보 인식 알고리즘 연구

## 슬라이드 5. 교차로 단계 판단

교차로 단계:

```text
UNKNOWN
-> APPROACH
-> IN_INTERSECTION
-> UNKNOWN
```

판단 기준:

```text
횡단보도 mask가 하단/중앙 영역에 안정적으로 나타나면 APPROACH
횡단보도에 충분히 가까워지고 이후 사라지거나 면적이 줄어들면 IN_INTERSECTION
정지선은 사용하지 않음
```

정지선을 제외한 이유:

```text
정지선은 마모, 가림, 교차로 진입 이후 시야 이탈이 많아
단계 전환 기준으로 불안정함
```

핵심 설계:

```text
교차로 진입 전(APPROACH)과 교차로 내부(IN_INTERSECTION)는 판단 기준을 다르게 적용
```

## 슬라이드 6. 우회전 안전 판단

APPROACH 단계에서는 신호 판단을 강하게 적용:

| 우선순위 | 조건 | 판단 |
|---:|---|---|
| 1 | 우회전 전용 신호 적색 | STOP |
| 2 | 차량 신호 적색 + 정지 완료 미확인 | STOP |
| 3 | 횡단보도 위 보행자 | STOP |
| 4 | 횡단보도 주변 보행자 | CAUTION |
| 5 | 우측 끝 차로가 아닐 가능성 | CAUTION |
| 6 | 횡단보도는 보이나 주행 구조 불확실 | CAUTION |
| 7 | 횡단보도 접근 중 | CAUTION |
| 8 | 위험 근거 없음 | GO |

IN_INTERSECTION 단계:

```text
이미 교차로 내부에 들어온 상태에서는 차량 신호 적색만으로 STOP 판단하지 않음
보행자/횡단보도 위험을 중심으로 판단
```

법규 근거:

```text
도로교통법 제25조: 교차로 통행방법
도로교통법 제27조: 보행자 보호
도로교통법 제5조/신호 준수 의무: 신호 또는 지시에 따를 의무
```

## 슬라이드 7. 차로 역할 판단

판단 목적:

```text
현재 주행 차로가 우회전에 적절한 우측 끝 차로인지 판단
```

사용 근거:

```text
좌/우 차선 경계 색상
주변 차선 존재 여부
좌우 인접 차량
최근 프레임 투표
교차로 진입 전 안정값 캐시
```

개선한 점:

```text
교차로 근처에서는 차선이 끊기거나 횡단보도와 겹쳐 불안정해짐
따라서 APPROACH 상태에서는 진입 전 안정적으로 측정한 차로 역할을 유지
```

출력 예:

```text
lane_role_rightmost=True
lane_role_confidence=0.60
lane_role_source=cached_before_intersection
```

## 슬라이드 8. 차량 신호 및 정지 완료 판단

신호 인식:

```text
YOLO traffic_light_vehicle bbox 검출
-> crop 영역 HSV 분석
-> RED/YELLOW/GREEN 후보 계산
-> 최근 프레임 smoothing
-> APPROACH 상태에서 RED hold 적용
```

RED hold를 넣은 이유:

```text
신호등 bbox가 순간적으로 사라지면 STOP/CAUTION이 흔들림
따라서 APPROACH에서 최근 RED가 안정적으로 잡힌 경우 UNKNOWN 몇 프레임은 RED로 유지
```

정지 완료 판단:

```text
하단 도로 ROI의 optical flow/프레임 움직임 측정
-> 움직임이 일정 프레임 이하로 작으면 정지 완료
-> 적색 신호에서 정지 완료가 없으면 RED_SIGNAL_NO_STOP_SUSPECTED 기록
```

근거:

```text
Optical flow는 프레임 간 픽셀/특징점 이동을 이용해 움직임을 추정하는 대표적 방법
```

참고:
- OpenCV Lucas-Kanade Optical Flow

## 슬라이드 9. 보행자 위험 판단

보행자 판단 흐름:

```text
YOLO pedestrian bbox 검출
-> bbox 하단 중앙점을 보행자 발 위치로 사용
-> 횡단보도 mask 내부/주변 여부 판단
-> 프레임 간 foot-point tracking
-> 횡단보도 접근/이탈 방향 판단
```

이번 주차 개선:

```text
기존:
횡단보도까지의 거리 변화만 사용

개선:
거리 변화 + 이동 방향 벡터를 함께 사용
```

접근 방향 계산:

```text
보행자 이동 벡터 = 현재 보행자 위치 - 이전 보행자 위치
횡단보도 방향 벡터 = 횡단보도 중심 - 현재 보행자 위치
cos 유사도 >= 0.50이면 횡단보도 방향 접근
cos 유사도 <= -0.30이면 횡단보도에서 이탈
```

근거:

```text
현재 구현은 참고 논문과 동일한 모델을 재현한 것이 아니라,
논문에서 사용하는 특징을 카메라 기반 v3 구조에 맞게 단순화한 방식임.

보행자 의도 예측 연구에서는 보행자 위치, 속도, heading,
횡단보도/도로와의 거리, 보행자 진행방향과 횡단보도 방향의 관계를 주요 특징으로 사용함.

v3 적용:
논문의 heading/trajectory 특징을 YOLO bbox foot-point 이동 벡터로 대체하고,
횡단보도 방향은 segmentation crosswalk mask 중심 방향으로 근사함.
```

참고:
- PIE dataset
- JAAD dataset
- Pedestrian Crossing Intention Forecasting at Unsignalized Intersections

## 슬라이드 10. 음성 안내

목표:

```text
화면을 계속 보지 않아도 운전자가 위험 원인을 알 수 있도록 음성 안내 제공
```

구현:

```text
console 안내
pyttsx3 기반 실제 음성 출력
여러 위험요소를 한 문장으로 합성
cooldown으로 반복 안내 제한
```

예시:

```text
적색 신호입니다. 일시정지하세요.

적색 신호입니다. 일시정지하세요.
보행자가 횡단보도로 접근 중입니다. 주의하세요.
```

구현상 주의:

```text
pyttsx3 runAndWait는 blocking이므로 별도 worker thread에서 실행
```

## 슬라이드 11. 위험 이벤트 기록 구조

기록 단위:

```text
프레임 단위가 아니라 교차로 통과 1회 단위
```

저장 구조:

```text
event_records/<intersection_event_id>/
  risk_events.jsonl
  intersection_event.jsonl
  snapshots/*.jpg
```

파일 역할:

| 파일 | 내용 | 목적 |
|---|---|---|
| `risk_events.jsonl` | 위험/주의 변화 상세 로그 | 디버깅, 임계값 조정 |
| `intersection_event.jsonl` | 교차로 1회 통과 요약 | 서버 전송 기준 |
| `snapshots/*.jpg` | 위험 순간 이미지 | 사후 확인, 앱 표시 |

스냅샷 개선:

```text
운전자용 판단 패널은 제거
도로/횡단보도/객체 박스만 저장
```

## 슬라이드 12. 서버 전송 payload 설계

서버 전송 단위:

```text
IntersectionPassEvent 1개 + 대표 스냅샷 이미지
```

payload 핵심 필드:

```json
{
  "schema_version": "v1",
  "record_type": "intersection_pass_event",
  "device_id": "edge-001",
  "vehicle_id": "test-car-01",
  "intersection_event_id": "...",
  "started_at": "...",
  "ended_at": "...",
  "location": {
    "event_start": {},
    "highest_risk": {},
    "event_end": {}
  },
  "scenario": "RIGHT_TURN",
  "final_decision": "STOP",
  "highest_severity": "VIOLATION_SUSPECTED",
  "event_codes": [],
  "evidence_summary": {},
  "snapshots": [],
  "timeline": []
}
```

위치정보 확장:

```text
event_start: 교차로 접근 시작 위치
highest_risk: 가장 위험도가 높았던 순간의 위치
event_end: 교차로 통과 종료 위치
```

## 슬라이드 13. 위반 의심 이벤트

주의:

```text
카메라 기반 시스템이므로 법적 확정 판단이 아니라
"위반 의심", "법규 리스크", "보호 필요"로 기록
```

| 이벤트 코드 | 의미 | 심각도 |
|---|---|---|
| `RED_SIGNAL_NO_STOP_SUSPECTED` | 적색 신호 일시정지 미확인 | VIOLATION_SUSPECTED |
| `RIGHT_TURN_SIGNAL_RED_RISK` | 우회전 전용 적색 신호 | VIOLATION_SUSPECTED |
| `PEDESTRIAN_PROTECTION_RISK` | 횡단보도 위 보행자 | VIOLATION_SUSPECTED |
| `PEDESTRIAN_NEAR_CROSSWALK_RISK` | 횡단보도 주변 보행자 | WARNING |
| `RIGHT_TURN_LANE_RISK` | 우회전 차로 위치 불확실 | WARNING |

## 슬라이드 14. 이번 주차 결과

완료한 내용:

```text
1. v3 판단 로직 구조화
2. 교차로 단계 자동 판단 개선
3. 차량 신호 smoothing 및 RED hold
4. 정지 완료 판단 추가
5. 보행자 접근/이탈 판단 개선
6. 음성 안내 pyttsx3 연결
7. 이벤트 교차로 단위 저장
8. 스냅샷에 객체 탐지 결과 표시
9. 위험 판단 기준표 문서화
```

결과 의미:

```text
단순 인식 프로그램에서
설명 가능한 교차로 안전 판단 시스템으로 발전
```

## 슬라이드 15. 남은 작업

다음 단계:

```text
1. 차량 위험 판단 추가
   - 앞차 근접
   - 우회전 출구 차량
   - 비보호좌회전 대향 차량

2. 서버 전송 모듈 구현
   - JSON payload 전송
   - snapshot multipart 업로드
   - 실패 시 로컬 재전송 큐

3. 위치정보 연동
   - GPS
   - speed
   - heading
   - event_start / highest_risk / event_end 위치 저장

4. 시연용 UI 정리
   - 영상 오버레이 모드
   - 운전자 요약 UI 모드
```

## 참고자료

### 인식/차선/추적

1. Yu et al., **BiSeNet V2: Bilateral Network with Guided Aggregation for Real-time Semantic Segmentation**  
   https://arxiv.org/abs/2004.02147

2. Ultralytics, **YOLOv8 Documentation**  
   https://docs.ultralytics.com/models/yolov8/

3. Qin et al., **Ultra Fast Structure-aware Deep Lane Detection**  
   https://arxiv.org/abs/2004.11757

4. Van Gansbeke et al., **End-to-end Lane Detection through Differentiable Least-Squares Fitting**  
   https://arxiv.org/abs/1902.00293

5. Bewley et al., **Simple Online and Realtime Tracking (SORT)**  
   https://arxiv.org/abs/1602.00763

6. Wojke et al., **Simple Online and Realtime Tracking with a Deep Association Metric (DeepSORT)**  
   https://arxiv.org/abs/1703.07402

7. OpenCV, **Lucas-Kanade Optical Flow**  
   https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html

8. 한국통신학회, **딥러닝 기반 차선 영역 검출 및 차선 정보 인식 알고리즘 연구**  
   https://conf.kics.or.kr/2024f/media?key=site%2F2024f%2Fabs%2F0037-DREFD.pdf

### 보행자 의도/행동 예측

9. Rasouli et al., **Are They Going to Cross? A Benchmark Dataset and Baseline for Pedestrian Crosswalk Behavior (JAAD)**  
   https://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w3/Rasouli_Are_They_Going_ICCV_2017_paper.pdf

10. Rasouli et al., **PIE: A Large-Scale Dataset and Models for Pedestrian Intention Estimation and Trajectory Prediction**  
    https://openaccess.thecvf.com/content_ICCV_2019/html/Rasouli_PIE_A_Large-Scale_Dataset_and_Models_for_Pedestrian_Intention_Estimation_ICCV_2019_paper.html

11. **Pedestrian Crossing Intention Forecasting at Unsignalized Intersections Using Naturalistic Trajectories**  
    https://pmc.ncbi.nlm.nih.gov/articles/PMC10006956/
    - v3와 가장 가까운 근거: 보행자 위치, 속도, heading, road/crosswalk 거리, 보행자 heading과 zebra 방향 관계를 특징으로 사용
    - 차이점: 해당 연구는 naturalistic trajectory 데이터 기반이고, v3는 단안 카메라 bbox와 segmentation mask로 이를 근사

12. 정보처리학회, **보행자 행동 예측 관련 국내 논문**  
    https://tkips.kips.or.kr/digital-library/manuscript/file/101929/05-SDE-24M-05-062C-%ED%95%A8%EC%A0%9C%EC%84%9D_32-40.pdf

### 법규/판례

13. 국가법령정보센터, **도로교통법 제25조 교차로 통행방법 관련 판례**  
    https://www.law.go.kr/LSW/precInfoP.do?precSeq=162598

14. 국가법령정보센터, **도로교통법 제27조 보행자의 보호**  
    https://law.go.kr/lbook/lbFileDownload.do?flExt=pdf&lbookConflSeq=106103&lbookSeq=106467

15. 국가법령정보센터, **도로교통법 제38조 차의 신호**  
    https://www.law.go.kr/LSW/lsLinkCommonInfo.do?ancYnChk=&chrClsCd=&lsJoLnkSeq=1020801583

16. 국가법령정보센터/관련 자료, **도로교통법 제5조 신호 또는 지시에 따를 의무**  
    https://www.knia.or.kr/file-manager/102952

17. 로톡 법률 해설, **교차로 우회전 총정리, 도로교통법으로 보는 적법한 우회전 방법**  
    https://www.lawtalk.co.kr/posts/153274
