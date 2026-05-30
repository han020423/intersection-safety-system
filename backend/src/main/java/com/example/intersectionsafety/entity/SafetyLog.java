package com.example.intersectionsafety.entity;

import jakarta.persistence.*;
import lombok.Getter;
import lombok.Setter;
import java.time.LocalDateTime;

@Entity
@Getter
@Setter
public class SafetyLog {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    private Double latitude;      // 위도 (GPS Y좌표)
    private Double longitude;     // 경도 (GPS X좌표)
    private String dangerType;    // 위험 유형 (예: 보행자_무단횡단)

    private String location;

    private LocalDateTime timestamp; // 위험 발생 시간

    // 컨트롤러 에러 방지용 수동 세터!
    public void setTimestamp(LocalDateTime timestamp) {
        this.timestamp = timestamp;
    }
}