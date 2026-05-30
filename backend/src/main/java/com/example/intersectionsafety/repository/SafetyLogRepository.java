package com.example.intersectionsafety.repository;

import com.example.intersectionsafety.entity.SafetyLog;
import org.springframework.data.jpa.repository.JpaRepository;

public interface SafetyLogRepository extends JpaRepository<SafetyLog, Long> {
}