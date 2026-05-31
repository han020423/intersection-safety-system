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
        log.setServerReceivedTime(LocalDateTime.now());
        safetyLogRepository.save(log);
        return "성공: 위험 데이터가 서버 DB에 저장되었습니다!";
    }

    @GetMapping
    public List<SafetyLog> getAllDangerEvents() {
        return safetyLogRepository.findAll();
    }

    // 🏆 공정성 확보 + 소수점 디테일이 살아있는 최종 요약 API
    @GetMapping("/summary")
    public Map<String, Object> getSafetySummary() {
        List<SafetyLog> logs = safetyLogRepository.findAll();

        if (logs == null || logs.isEmpty()) {
            // 데이터가 없을 때도 100.0(소수점)으로 통일해서 보내주기
            return Map.of("totalScore", 100.0, "violationCount", 0, "complianceCount", 0, "totalEvents", 0);
        }

        int violationCount = 0;
        int totalEvents = logs.size();

        for (SafetyLog log : logs) {
            if ("VIOLATION_SUSPECTED".equals(log.getHighestSeverity())) {
                violationCount++;
            }
        }

        // 1. 위반 1건당 10점의 감점치를 가짐
        double totalPenalty = violationCount * 10.0;

        // 2. 전체 주행 이벤트(totalEvents)로 나누어 점수 희석 (기존의 억지 반올림 제거!)
        double finalPenalty = totalPenalty / (double) totalEvents;

        // 3. 점수를 소수점(double)으로 계산하여 미세한 감점도 놓치지 않음
        double totalScore = 100.0 - finalPenalty;

        if (totalScore < 0) totalScore = 0.0;

        // 무한소수(99.67741...)로 나오는 걸 방지하기 위해 소수점 첫째 자리까지만 깔끔하게 자르기 (예: 99.6)
        double displayScore = Math.floor(totalScore * 10) / 10.0;

        Map<String, Object> summaryResult = new HashMap<>();
        summaryResult.put("totalScore", displayScore); // 이제 화면에 정수 100 대신 디테일한 점수가 날아갑니다!
        summaryResult.put("violationCount", violationCount);
        summaryResult.put("complianceCount", totalEvents - violationCount);
        summaryResult.put("totalEvents", totalEvents);

        return summaryResult;
    }

    // 🏆 데모용 DB 초기화 비밀 스위치
    @GetMapping("/reset")
    public String resetData() {
        safetyLogRepository.deleteAll();
        return "DB 초기화가 완벽하게 완료되었습니다! 데모를 시작하세요!";
    }
}