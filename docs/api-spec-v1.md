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