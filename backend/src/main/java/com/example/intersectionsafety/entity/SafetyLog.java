package com.example.intersectionsafety.entity;

import jakarta.persistence.*;
import lombok.Data;
import java.time.LocalDateTime;
import java.util.List;

@Data
@Entity
public class SafetyLog {
    @Id @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    // 파이썬에서 보내는 데이터 이름표와 100% 일치시킴
    private String record_type;
    private String device_id;
    private String vehicle_id;
    private String intersection_event_id;
    private String timestamp; // 파이썬에서 String으로 오므로 일단 String으로 받음
    private Integer frame_index;
    private String decision;
    private String state;
    private String scenario;
    private String reason;
    private Integer objectCount;

    @ElementCollection
    private List<String> eventCodes;

    private String highestSeverity;

    // 지도 핀 찍기용 GPS (재혁님이 추가해 줄 데이터)
    private Double latitude;
    private Double longitude;

    // 서버 도착 시간 (기존 유지)
    private LocalDateTime serverReceivedTime;
}