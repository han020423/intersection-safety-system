package com.example.intersectionsafety.entity;

import jakarta.persistence.*;
import lombok.Getter;
import lombok.Setter;
import java.time.LocalDateTime;

@Entity
@Getter @Setter
public class SafetyLog {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    private String location; // 교차로 위치

    private String detectionType; // 위험 감지 종류 (무단횡단 등)

    private LocalDateTime detectedAt; // 감지 시간

    @PrePersist
    public void prePersist() {
        this.detectedAt = LocalDateTime.now();
    }
}