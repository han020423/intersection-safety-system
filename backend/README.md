# Backend

Spring Boot 기반 서버 코드 폴더입니다.

## 목표 기능
- 위험 이벤트 수신 API
- 이벤트 로그 DB 저장
- 지도 시각화용 데이터 조회 API

## 저장 대상 이벤트
- 우회전 위험 이벤트
- 비보호좌회전 위험 이벤트
- STOP / CAUTION / GO 판단 결과
- 위험 원인 코드(reasonCode)
- GPS 기반 위치 정보

## 예정 기능
- `POST /api/events`
- `GET /api/events`
- `GET /api/events/map`

## 사용 예정 기술
- Spring Boot
- MySQL
- JPA
- REST API

## 비고
라즈베리파이에서 위험 판단 결과를 JSON으로 전송하면,
서버가 이를 저장하고 조회할 수 있도록 구성한다.