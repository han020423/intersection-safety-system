"""
Voice-alert selection for v3.

Console and pyttsx3 voice backends share the same alert selection logic.  WAV
playback can be attached later without changing the safety decision layer.
"""

from __future__ import annotations

import queue
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

from safety_decision_reboot import DecisionLevel


@dataclass(frozen=True)
class VoiceAlert:
    code: str
    priority: int
    message: str


class VoiceAlertManager:
    """Build and rate-limit driver voice alerts.

    One frame can contain several risks, such as red-signal no-stop and a
    nearby pedestrian.  The manager therefore collects every active alert and
    speaks them as one ordered message instead of hiding lower-priority causes.
    """

    def __init__(
        self,
        mode: str = "off",
        cooldown_sec: float = 5.0,
        speech_rate: int = 185,
        speech_volume: float = 1.0,
    ):
        self.mode = mode
        self.cooldown_sec = float(cooldown_sec)
        self._last_spoken_at: dict[str, float] = {}
        self._last_priority = 0
        self._speaker = None
        if self.mode == "pyttsx3":
            self._speaker = Pyttsx3Speaker(rate=speech_rate, volume=speech_volume)

    def update(self, decision, road_state=None, context=None) -> VoiceAlert | None:
        if self.mode == "off" or decision is None:
            return None

        alert = self._select_alert(decision, road_state)
        if alert is None:
            self._last_priority = 0
            return None

        now = time.monotonic()
        last_at = self._last_spoken_at.get(alert.code, -1e9)
        higher_priority = alert.priority > self._last_priority
        if not higher_priority and now - last_at < self.cooldown_sec:
            self._last_priority = max(self._last_priority, alert.priority)
            return None

        self._last_spoken_at[alert.code] = now
        self._last_priority = alert.priority
        if self.mode in {"console", "pyttsx3"}:
            print(f"[VOICE] {alert.code}: {alert.message}")
        if self._speaker is not None:
            self._speaker.say(alert.message)
        return alert

    def _select_alert(self, decision, road_state=None) -> VoiceAlert | None:
        alerts = self._collect_alerts(decision, road_state)
        if not alerts:
            return None

        alerts = sorted(alerts, key=lambda item: item.priority, reverse=True)
        codes = "+".join(alert.code for alert in alerts)
        message = " ".join(alert.message for alert in alerts)
        return VoiceAlert(codes, alerts[0].priority, message)

    def _collect_alerts(self, decision, road_state=None) -> list[VoiceAlert]:
        events = list(getattr(decision, "compliance_events", []) or [])
        event_codes = {event.code for event in events}
        reason_code = getattr(decision, "reason_code", "")
        alerts: list[VoiceAlert] = []

        if "PEDESTRIAN_PROTECTION_RISK" in event_codes:
            alerts.append(
                VoiceAlert(
                    "pedestrian_on_crosswalk",
                    100,
                    "횡단보도에 보행자가 있습니다. 정지하세요.",
                )
            )

        if "RIGHT_TURN_SIGNAL_RED_RISK" in event_codes:
            alerts.append(
                VoiceAlert(
                    "right_turn_signal_red",
                    95,
                    "우회전 신호가 적색입니다. 정지하세요.",
                )
            )

        if "RED_SIGNAL_NO_STOP_SUSPECTED" in event_codes:
            alerts.append(
                VoiceAlert(
                    "red_signal_stop_required",
                    90,
                    "적색 신호입니다. 일시정지하세요.",
                )
            )

        if "PEDESTRIAN_NEAR_CROSSWALK_RISK" in event_codes or reason_code in {
            "pedestrian_near_crosswalk",
            "pedestrian_near_crosswalk_in_intersection",
        }:
            approaching = bool(getattr(road_state, "pedestrian_approaching_crosswalk", False))
            if approaching:
                alerts.append(
                    VoiceAlert(
                        "pedestrian_approaching_crosswalk",
                        80,
                        "보행자가 횡단보도로 접근 중입니다. 주의하세요.",
                    )
                )
            else:
                alerts.append(
                    VoiceAlert(
                        "pedestrian_near_crosswalk",
                        75,
                        "횡단보도 주변 보행자에 주의하세요.",
                    )
                )

        if "RIGHT_TURN_LANE_RISK" in event_codes or reason_code == "right_turn_not_rightmost_lane":
            alerts.append(
                VoiceAlert(
                    "right_turn_lane_check",
                    60,
                    "우회전 차로가 아닐 수 있습니다. 차로를 확인하세요.",
                )
            )

        if not alerts and getattr(decision, "level", None) == DecisionLevel.STOP:
            alerts.append(
                VoiceAlert(
                    f"stop_{reason_code}",
                    50,
                    "위험 상황입니다. 정지하세요.",
                )
            )

        return self._dedupe_alerts(alerts)

    @staticmethod
    def _dedupe_alerts(alerts: list[VoiceAlert]) -> list[VoiceAlert]:
        """Keep one message per alert code while preserving first-seen wording."""
        deduped: dict[str, VoiceAlert] = {}
        for alert in alerts:
            if alert.code not in deduped:
                deduped[alert.code] = alert
        return list(deduped.values())


class Pyttsx3Speaker:
    """Small background pyttsx3 wrapper for real-time video loops.

    pyttsx3 `runAndWait()` is blocking.  Running it in the inference loop would
    visibly drop FPS, so speech is queued to a daemon worker thread.  If the
    driver guidance changes quickly, old queued speech is dropped and the newest
    sentence is kept.
    """

    def __init__(self, rate: int = 185, volume: float = 1.0):
        self.rate = int(rate)
        self.volume = max(0.0, min(1.0, float(volume)))
        self._queue: queue.Queue[str | None] = queue.Queue(maxsize=2)
        self._thread = threading.Thread(target=self._run, name="pyttsx3-voice", daemon=True)
        self._thread.start()

    def say(self, message: str) -> None:
        if not message:
            return
        while self._queue.full():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        try:
            self._queue.put_nowait(message)
        except queue.Full:
            pass

    def _run(self) -> None:
        pythoncom = None
        if sys.platform.startswith("win"):
            try:
                import pythoncom as _pythoncom

                pythoncom = _pythoncom
                pythoncom.CoInitialize()
            except Exception as exc:
                print(f"[VOICE] pythoncom init failed: {exc}")

        try:
            self._prepare_windows_comtypes_cache()
            import pyttsx3
        except Exception as exc:
            print(f"[VOICE] pyttsx3 unavailable: {exc}")
            if pythoncom is not None:
                pythoncom.CoUninitialize()
            return

        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", self.rate)
            engine.setProperty("volume", self.volume)
            self._select_korean_voice_if_available(engine)
        except Exception as exc:
            print(f"[VOICE] pyttsx3 init failed: {exc}")
            if pythoncom is not None:
                pythoncom.CoUninitialize()
            return

        try:
            while True:
                message = self._queue.get()
                if message is None:
                    return
                try:
                    engine.say(message)
                    engine.runAndWait()
                except Exception as exc:
                    print(f"[VOICE] pyttsx3 speak failed: {exc}")
        finally:
            if pythoncom is not None:
                pythoncom.CoUninitialize()

    @staticmethod
    def _prepare_windows_comtypes_cache() -> None:
        """Route comtypes generated modules to a workspace-writable folder.

        Windows pyttsx3 uses SAPI through comtypes.  In restricted execution
        environments, comtypes may fail while creating its default AppData cache.
        Pre-registering `comtypes.gen` avoids that by using a local cache folder.
        """
        if not sys.platform.startswith("win"):
            return

        try:
            import comtypes

            cache_dir = Path(__file__).resolve().parent / ".comtypes_cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            gen_module = ModuleType("comtypes.gen")
            gen_module.__path__ = [str(cache_dir)]
            sys.modules["comtypes.gen"] = gen_module
            comtypes.gen = gen_module
        except Exception as exc:
            print(f"[VOICE] comtypes cache setup failed: {exc}")

    @staticmethod
    def _select_korean_voice_if_available(engine) -> None:
        """Prefer a Korean SAPI/eSpeak voice when the OS provides one."""
        try:
            voices = engine.getProperty("voices") or []
        except Exception:
            return

        for voice in voices:
            name = str(getattr(voice, "name", "")).lower()
            voice_id = str(getattr(voice, "id", "")).lower()
            languages = " ".join(str(item).lower() for item in getattr(voice, "languages", []) or [])
            if "korean" in name or "korean" in voice_id or "ko" in languages or "ko-" in voice_id:
                try:
                    engine.setProperty("voice", voice.id)
                except Exception:
                    pass
                return
