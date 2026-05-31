package com.example.intersectionsafety.controller;

import com.example.intersectionsafety.entity.SafetyLog;
import com.example.intersectionsafety.repository.SafetyLogRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.time.LocalDateTime;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@CrossOrigin(origins = "*")
@RestController
@RequestMapping("/api/events")
public class SafetyLogController {

    @Autowired
    private SafetyLogRepository safetyLogRepository;

    @PostMapping
    public String receiveDangerEvent(@RequestBody SafetyLog log) {
        log.setTimestamp(LocalDateTime.now());
        safetyLogRepository.save(log);
        return "성공: 위험 데이터가 서버 DB에 저장되었습니다!";
    }

    @GetMapping
    public List<SafetyLog> getAllDangerEvents() {
        return safetyLogRepository.findAll();
    }

    // 🏆 통합된 최종 요약 API (오류 해결 버전)
    // 🏆 공정성 확보된 최종 요약 API
    @GetMapping("/summary")
    public Map<String, Object> getSafetySummary() {
        List<SafetyLog> logs = safetyLogRepository.findAll();

        if (logs == null || logs.isEmpty()) {
            return Map.of("totalScore", 100, "violationCount", 0, "complianceCount", 0, "totalEvents", 0);
        }

        int violationCount = 0;
        int totalEvents = logs.size();

        for (SafetyLog log : logs) {
            if (log.getDangerType() != null && log.getDangerType().contains("위반")) {
                violationCount++;
            }
        }

        // 1. 위반 1건당 10점의 감점치를 가짐
        double totalPenalty = violationCount * 10.0;

        // 2. 전체 주행 이벤트(totalEvents)로 나누어 점수 희석 (공정성 확보)
        // 많이 주행(준수)할수록 위반 1건당 감점 폭이 작아짐
        int finalPenalty = (int) Math.round(totalPenalty / (double) totalEvents);

        int totalScore = 100 - finalPenalty;

        if (totalScore < 0) totalScore = 0;

        Map<String, Object> summaryResult = new HashMap<>();
        summaryResult.put("totalScore", totalScore);
        summaryResult.put("violationCount", violationCount);
        summaryResult.put("complianceCount", totalEvents - violationCount);
        summaryResult.put("totalEvents", totalEvents);

        return summaryResult;
    }
    @GetMapping("/reset")
    public String resetData() {
        safetyLogRepository.deleteAll();
        return "DB 초기화가 완벽하게 완료되었습니다! 데모를 시작하세요!";
    }
}