#!/usr/bin/env python3
"""
intersection_demo.py — 교차로 안전 데모 프로그램 (메인 진입점)

YOLO11n 객체 인식 + BiSeNetV2 도로 구조 세그멘테이션을 결합하여
교차로 접근/진입 상태를 추정하고 STOP / CAUTION / GO를 판단하는 데모.

사용법:
  # 비디오 파일
  python intersection_demo.py --source video.mp4 --show

  # 웹캠
  python intersection_demo.py --source 0 --show

  # 출력 저장 + 디버그
  python intersection_demo.py --source video.mp4 --show --save output.mp4 --debug

  # 라즈베리파이 최적화 (낮은 해상도, 프레임 스킵)
  python intersection_demo.py --source 0 --show --width 480 --height 270 \\
      --yolo-interval 3 --seg-interval 2 --seg-input-h 192 --seg-input-w 320
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np

# 같은 디렉터리의 모듈 import
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from perception import YoloDetector, RoadSegmentor
from postprocess import compute_lane_geometry, LaneTracker
from state_machine import IntersectionFSM, build_scene_context, Decision
from visualizer import draw_demo_overlay


# ═══════════════════════════ 설정 ═══════════════════════════ #

# 기본 weight 경로 (변수로 분리)
DEFAULT_YOLO_WEIGHTS = os.path.join(SCRIPT_DIR, "yolo.pt")
DEFAULT_SEG_WEIGHTS = os.path.join(SCRIPT_DIR, "bisenet_best.pth")

# 대체 경로 (ai/scripts/road_v1에 weight가 있을 수 있음)
ALT_YOLO_WEIGHTS = os.path.normpath(
    os.path.join(SCRIPT_DIR, "..", "..", "ai", "scripts", "road_v1", "yolo.pt"))
ALT_SEG_WEIGHTS = os.path.normpath(
    os.path.join(SCRIPT_DIR, "..", "..", "ai", "scripts", "road_v1",
                 "bisenet_custom", "weights", "best.pth"))


def find_weights(primary: str, alt: str, name: str) -> str:
    """weight 파일을 찾는다. primary → alt 순서로 탐색."""
    if os.path.isfile(primary):
        return primary
    if os.path.isfile(alt):
        print(f"[{name}] 기본 경로에 없어 대체 경로 사용: {alt}")
        return alt
    raise FileNotFoundError(
        f"[{name}] weight를 찾을 수 없습니다.\n"
        f"  시도한 경로:\n    {primary}\n    {alt}\n"
        f"  --yolo-weights / --seg-weights 옵션으로 직접 지정하세요."
    )


# ═══════════════════════════ 메인 루프 ═══════════════════════════ #


def process_frame(frame, yolo, seg, seg_interval_counter, seg_cache, fsm,
                  corridor_half_width, debug, lane_tracker=None,
                  vehicle_signal="auto", pedestrian_signal="auto",
                  right_turn_signal="none", stop_completed_on_red=False):
    """
    한 프레임을 처리한다.

    Returns:
        vis, seg_cache_updated, detections, lane_geo, ctx
    """
    h, w = frame.shape[:2]

    # YOLO 추론 (interval은 YoloDetector 내부에서 처리)
    detections, yolo_ms = yolo.infer(frame)

    # BiSeNetV2 추론 (외부에서 interval 관리)
    if seg_cache is None or seg_interval_counter <= 0:
        seg_mask, seg_ms = seg.infer(frame)
        seg_overlay = seg.build_color_overlay(seg_mask)
        seg_cache = (seg_mask, seg_overlay, seg_ms)
    else:
        seg_mask, seg_overlay, seg_ms = seg_cache
        seg_ms = 0.0  # 캐시 사용 시 0ms

    # 후처리: ego lane, centerline, crosswalk zone, path corridor
    lane_geo = compute_lane_geometry(
        seg_mask, h, w, corridor_half_width,
        detections=detections,
        suppress_vehicle_lanes=True,
    )

    # 시간적 스무딩 (LaneTracker) — 프레임 간 차선 떨림 방지
    if lane_tracker is not None:
        lane_geo = lane_tracker.smooth(lane_geo, h, w, corridor_half_width)

    # FSM 업데이트
    ctx = build_scene_context(
        detections, lane_geo, seg_mask, h, w,
        frame=frame,
        vehicle_signal_override=vehicle_signal,
        pedestrian_signal_override=pedestrian_signal,
        right_turn_signal_override=right_turn_signal,
        stop_completed_on_red=stop_completed_on_red,
    )
    fsm.update(ctx)

    # 시각화
    total_ms = yolo_ms + seg_cache[2]  # 실제 추론 시간 합산
    vis = draw_demo_overlay(
        frame, detections, seg_mask, seg_overlay, lane_geo,
        fsm.state, fsm.decision, ctx,
        yolo_ms, seg_cache[2], total_ms,
        fps=0.0,  # 나중에 외부에서 갱신
        debug=debug,
    )

    return vis, seg_cache, detections, lane_geo, ctx


def main():
    args = parse_args()

    # weight 경로 확정
    yolo_path = args.yolo_weights or find_weights(DEFAULT_YOLO_WEIGHTS, ALT_YOLO_WEIGHTS, "YOLO")
    seg_path = args.seg_weights or find_weights(DEFAULT_SEG_WEIGHTS, ALT_SEG_WEIGHTS, "BiSeNetV2")

    if args.yolo_weights and not os.path.isfile(args.yolo_weights):
        raise FileNotFoundError(f"YOLO weight 없음: {args.yolo_weights}")
    if args.seg_weights and not os.path.isfile(args.seg_weights):
        raise FileNotFoundError(f"BiSeNetV2 weight 없음: {args.seg_weights}")

    print("=" * 60)
    print("  Intersection Safety Demo")
    print("  YOLO11n + BiSeNetV2 + FSM + STOP/CAUTION/GO")
    print("=" * 60)

    # 모델 로드
    yolo = YoloDetector(
        weights=yolo_path, imgsz=args.imgsz, conf=args.conf,
        iou=args.iou, device=args.device, interval=args.yolo_interval,
        estimate_distance=not args.no_distance,
        camera_hfov_deg=args.camera_hfov,
        distance_focal_px=args.distance_focal_px,
        vehicle_real_width_m=args.vehicle_real_width,
        vehicle_real_height_m=args.vehicle_real_height,
        distance_scale=args.distance_scale,
    )
    seg = RoadSegmentor(
        weights=seg_path,
        input_size=(args.seg_input_h, args.seg_input_w),
        device=args.device,
    )

    # FSM 초기화
    fsm = IntersectionFSM()

    # 차선 시간적 스무딩 트래커 (프레임 간 떨림 방지)
    lane_tracker = LaneTracker(alpha=0.4, max_miss=10)

    # 입력 소스 결정
    source_str = args.source
    if source_str.isdigit():
        source, mode = int(source_str), 'camera'
    elif source_str.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
        source, mode = source_str, 'image'
    else:
        source, mode = source_str, 'video'

    print(f"\n[입력] mode={mode}  source={source}")
    print(f"[해상도] {args.width}x{args.height}  device={args.device}")
    print(f"[YOLO interval] {args.yolo_interval}  [Seg interval] {args.seg_interval}")
    print(f"[Distance] enabled={not args.no_distance}  hfov={args.camera_hfov}  "
          f"focal_px={args.distance_focal_px or 'auto'}  "
          f"vehicle={args.vehicle_real_width}x{args.vehicle_real_height}m  "
          f"scale={args.distance_scale}")
    print(f"[Right turn law mode] vehicle={args.vehicle_signal}  "
          f"pedestrian={args.pedestrian_signal}  right-turn={args.right_turn_signal}  "
          f"stopped_on_red={args.assume_stopped_on_red}")
    print(f"[Debug] {args.debug}\n")

    # ── 단일 이미지 모드 ──
    if mode == 'image':
        frame = cv2.imread(str(source))
        if frame is None:
            raise RuntimeError(f"이미지 로드 실패: {source}")
        frame = cv2.resize(frame, (args.width, args.height))

        vis, _, dets, geo, ctx = process_frame(
            frame, yolo, seg, 0, None, fsm, args.corridor_width, args.debug,
            lane_tracker=lane_tracker,
            vehicle_signal=args.vehicle_signal,
            pedestrian_signal=args.pedestrian_signal,
            right_turn_signal=args.right_turn_signal,
            stop_completed_on_red=args.assume_stopped_on_red)

        print(f"State: {fsm.state.value}  Decision: {fsm.decision.value}")
        print(f"Reason: {ctx.reason}")
        print(f"Objects: {len(dets)}  Lane Conf: {geo.lane_confidence:.2f}")

        if args.save:
            cv2.imwrite(args.save, vis)
            print(f"저장: {args.save}")
        if args.show:
            cv2.imshow('Intersection Demo', vis)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return

    # ── 비디오 / 카메라 모드 ──
    cap = cv2.VideoCapture(source)
    if isinstance(source, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise RuntimeError(f"소스 열기 실패: {source}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if src_fps <= 0 or src_fps > 120:
        src_fps = 30.0

    writer = None
    frame_count = 0
    fps_smooth = 0.0
    seg_cache = None
    seg_counter = 0

    print(f"[원본 FPS] {src_fps:.1f}")
    print("실행 중... (q 또는 ESC로 종료)\n")

    try:
        while True:
            t0 = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.resize(frame, (args.width, args.height))

            # seg interval 관리
            seg_counter -= 1
            if seg_counter <= 0:
                seg_counter = args.seg_interval
                seg_cache_input = None  # 새로 추론
            else:
                seg_cache_input = seg_cache

            vis, seg_cache, dets, geo, ctx = process_frame(
                frame, yolo, seg, seg_counter, seg_cache_input, fsm,
                args.corridor_width, args.debug, lane_tracker=lane_tracker,
                vehicle_signal=args.vehicle_signal,
                pedestrian_signal=args.pedestrian_signal,
                right_turn_signal=args.right_turn_signal,
                stop_completed_on_red=args.assume_stopped_on_red)

            # FPS 계산
            elapsed = time.perf_counter() - t0
            cur_fps = 1.0 / max(1e-6, elapsed)
            fps_smooth = 0.15 * cur_fps + 0.85 * fps_smooth if fps_smooth > 0 else cur_fps

            # FPS를 화면에 덮어쓰기 (좌측 상단)
            cv2.putText(vis, f"FPS: {fps_smooth:.1f}", (16, 88),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1, cv2.LINE_AA)

            frame_count += 1
            if frame_count % 30 == 0:
                print(f"  frame={frame_count:5d}  FPS={fps_smooth:.1f}  "
                      f"state={fsm.state.value:24s}  decision={fsm.decision.value:8s}  "
                      f"objs={len(dets)}  lane_conf={geo.lane_confidence:.2f}  "
                      f"reason={ctx.reason}")

            # 출력 저장
            if args.save:
                if writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    writer = cv2.VideoWriter(args.save, fourcc, src_fps,
                                             (args.width, args.height))
                writer.write(vis)

            # 화면 표시
            if args.show:
                cv2.imshow('Intersection Demo', vis)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord('q')):
                    break
                elif key == ord('d'):  # 'd' 키로 디버그 토글
                    args.debug = not args.debug
                    print(f"  [Debug 토글] debug={args.debug}")

    finally:
        cap.release()
        if writer is not None:
            writer.release()
            print(f"\n결과 저장: {args.save}")
        if args.show:
            cv2.destroyAllWindows()
        print(f"총 {frame_count}프레임 처리 완료.")


# ═══════════════════════════ 인자 파서 ═══════════════════════════ #

def parse_args():
    p = argparse.ArgumentParser(
        description="Intersection Safety Demo — YOLO11n + BiSeNetV2 + FSM"
    )
    # 입력
    p.add_argument('--source', type=str, default='0',
                   help='비디오 파일 경로 또는 웹캠 번호 (기본: 0)')
    p.add_argument('--width', type=int, default=640, help='프레임 리사이즈 너비')
    p.add_argument('--height', type=int, default=360, help='프레임 리사이즈 높이')
    p.add_argument('--show', action='store_true', help='OpenCV 화면 표시')
    p.add_argument('--save', type=str, default='', help='결과 영상 저장 경로')

    # 모델 weight
    p.add_argument('--yolo-weights', type=str, default=None, help='YOLO weight 경로')
    p.add_argument('--seg-weights', type=str, default=None, help='BiSeNetV2 weight 경로')

    # YOLO 설정
    p.add_argument('--imgsz', type=int, default=640, help='YOLO 입력 해상도')
    p.add_argument('--conf', type=float, default=0.35, help='YOLO confidence threshold')
    p.add_argument('--iou', type=float, default=0.45, help='YOLO NMS IoU threshold')

    # BiSeNetV2 설정 (별도 입력 해상도)
    p.add_argument('--seg-input-h', type=int, default=352, help='BiSeNetV2 입력 높이')
    p.add_argument('--seg-input-w', type=int, default=640, help='BiSeNetV2 입력 너비')

    # 최적화
    p.add_argument('--device', type=str, default='cpu', help='추론 디바이스 (cpu/cuda)')
    p.add_argument('--yolo-interval', type=int, default=1,
                   help='YOLO 추론 간격 (1=매 프레임, 3=3프레임마다)')
    p.add_argument('--seg-interval', type=int, default=1,
                   help='BiSeNetV2 추론 간격 (1=매 프레임)')
    p.add_argument('--corridor-width', type=int, default=80,
                   help='Path corridor 반폭 (px)')
    p.add_argument('--no-distance', action='store_true',
                   help='차량 단안 거리 추정 표시 비활성화')
    p.add_argument('--camera-hfov', type=float, default=70.0,
                   help='카메라 수평 화각(deg). focal 미지정 시 거리 추정에 사용')
    p.add_argument('--distance-focal-px', type=float, default=0.0,
                   help='캘리브레이션된 focal length(px). 0이면 화각 기반 자동 계산')
    p.add_argument('--vehicle-real-width', type=float, default=1.8,
                   help='거리 추정에 사용할 평균 차량 실제 폭(m)')
    p.add_argument('--vehicle-real-height', type=float, default=1.5,
                   help='거리 추정에 사용할 평균 차량 실제 높이(m)')
    p.add_argument('--distance-scale', type=float, default=0.8,
                   help='거리 추정 보정 계수. 표시 거리에 곱해짐')
    p.add_argument('--vehicle-signal', type=str, default='auto',
                   choices=['auto', 'red', 'yellow', 'green', 'unknown'],
                   help='전방 차량 신호 상태. auto는 YOLO crop 색상 추정')
    p.add_argument('--pedestrian-signal', type=str, default='auto',
                   choices=['auto', 'red', 'green', 'unknown'],
                   help='보행자 신호 상태. auto는 YOLO crop 색상 추정')
    p.add_argument('--right-turn-signal', type=str, default='none',
                   choices=['none', 'auto', 'red', 'yellow', 'green', 'unknown'],
                   help='우회전 전용 신호등 상태. 없으면 none')
    p.add_argument('--assume-stopped-on-red', action='store_true',
                   help='전방 적색 신호에서 이미 일시정지를 완료했다고 가정')

    # 디버그
    p.add_argument('--debug', action='store_true', help='디버그 시각화 (범례 등)')

    return p.parse_args()


if __name__ == '__main__':
    main()
