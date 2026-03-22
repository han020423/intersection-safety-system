# API Spec v1

라즈베리파이에서 서버로 위험 이벤트를 전송하기 위한 초기 API 스펙입니다.

## POST /api/events

### request example
```json
{
  "eventTime": "2026-03-22T14:20:31",
  "deviceId": "raspi-01",
  "scenarioType": "RIGHT_TURN",
  "riskLevel": "STOP",
  "reasonCode": "PEDESTRIAN_ON_CROSSWALK",
  "latitude": 36.8000,
  "longitude": 127.1500
}

### request example 2
```json
{
  "eventTime": "2026-03-22T14:21:05",
  "deviceId": "raspi-01",
  "scenarioType": "UNPROTECTED_LEFT",
  "riskLevel": "STOP",
  "reasonCode": "ONCOMING_VEHICLE_APPROACH",
  "latitude": 36.8002,
  "longitude": 127.1503
}
필드 설명
eventTime
이벤트 발생 시각
deviceId
장치 식별자
scenarioType
시나리오 타입
예: RIGHT_TURN, UNPROTECTED_LEFT
riskLevel
위험 수준
예: STOP, CAUTION, GO
reasonCode
위험 판단 원인 코드
예:
PEDESTRIAN_ON_CROSSWALK
PEDESTRIAN_ABOUT_TO_ENTER
RED_LIGHT_STOP_REQUIRED
ONCOMING_VEHICLE_APPROACH
UNPROTECTED_LEFT_SIGN_DETECTED
latitude
위도
longitude
경도
초기 응답 예시
{
  "message": "event saved",
  "status": 200
}