#!/usr/bin/env python3
"""
교차로 안전 판단 데모 v2.

이 버전은 road_v4 경량 BiSeNetV2 모델을 위한 새 판단 프로그램이다.
기존 모델과 클래스 번호가 다르기 때문에 v2 폴더에 분리했다.

road_v4 클래스:
  0 background
  1 lane_white
  2 lane_yellow
  3 lane_blue
  4 crosswalk
  5 stop_line

Example:
  python edge/inference/v2/intersection_demo_v2.py ^
      --source edge/inference/v4.mp4 ^
      --seg-weights path/to/best_light_infer.pt ^
      --yolo-weights edge/inference/yolo.pt ^
      --show

Raspberry Pi style:
  python edge/inference/v2/intersection_demo_v2.py --source 0 --device cpu ^
      --width 480 --height 270 --seg-input-h 256 --seg-input-w 464 ^
      --yolo-interval 3 --seg-interval 2
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import cv2

V2_DIR = Path(__file__).resolve().parent
PARENT_DIR = V2_DIR.parent
# 기존 YOLO/FSM/state_machine은 그대로 재사용하되, road_v4 전용 세그멘터와
# 후처리는 v2 폴더 안의 파일을 우선 import한다.
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))
if str(V2_DIR) not in sys.path:
    sys.path.insert(0, str(V2_DIR))

from perception import YoloDetector  # noqa: E402
from road_v4_postprocess import LaneTracker, compute_lane_geometry  # noqa: E402
from road_v4_segmentor import RoadV4Segmentor  # noqa: E402
from state_machine import IntersectionFSM, build_scene_context  # noqa: E402
from visualizer_v2 import draw_overlay  # noqa: E402


DEFAULT_YOLO_WEIGHTS = PARENT_DIR / "yolo.pt"
ALT_YOLO_WEIGHTS = PARENT_DIR.parents[1] / "ai" / "scripts" / "road_v1" / "yolo.pt"
DEFAULT_SEG_WEIGHTS = V2_DIR / "road_v4_best_light.pt"


def find_existing(primary: Path, alt: Path | None, label: str) -> str:
    """기본 weight 경로가 없으면 대체 경로를 찾아 실행 편의성을 높인다."""
    if primary.is_file():
        return str(primary)
    if alt is not None and alt.is_file():
        print(f"[{label}] using alternate path: {alt}")
        return str(alt)
    searched = [str(primary)]
    if alt is not None:
        searched.append(str(alt))
    raise FileNotFoundError(f"{label} weights not found. searched={searched}")


def process_frame(frame, yolo, seg, seg_cache, fsm, lane_tracker, args):
    h, w = frame.shape[:2]

    # 1) YOLO로 차량/보행자/신호등 객체를 검출한다.
    detections, yolo_ms = yolo.infer(frame)

    # 2) BiSeNetV2 세그멘테이션은 seg_interval에 따라 캐시를 재사용할 수 있다.
    # 라즈베리파이에서는 매 프레임 세그멘테이션을 돌리면 느릴 수 있으므로 이 옵션이 중요하다.
    if seg_cache is None or args._seg_countdown <= 0:
        seg_mask, seg_ms = seg.infer(frame)
        seg_overlay = seg.build_color_overlay(seg_mask)
        seg_cache = (seg_mask, seg_overlay, seg_ms)
        args._seg_countdown = max(1, args.seg_interval)
    else:
        seg_mask, seg_overlay, _ = seg_cache
        seg_ms = 0.0
    args._seg_countdown -= 1

    # 3) road_v4 class id 기준으로 차선, 횡단보도, 정지선 기하 정보를 만든다.
    lane_geo = compute_lane_geometry(
        seg_mask,
        h,
        w,
        corridor_half_width=args.corridor_width,
        detections=detections,
        suppress_vehicle_lanes=not args.no_vehicle_lane_suppression,
    )

    if lane_tracker is not None:
        # 프레임별 차선 흔들림을 줄이기 위해 이전 차선 모델과 EMA smoothing을 적용한다.
        lane_geo = lane_tracker.smooth(lane_geo, h, w, args.corridor_width)

    # 4) 객체 검출 결과와 도로 구조를 SceneContext 변수로 묶는다.
    # FSM은 이 변수들만 보고 STOP/CAUTION/GO를 판단한다.
    ctx = build_scene_context(
        detections,
        lane_geo,
        seg_mask,
        h,
        w,
        frame=frame,
        vehicle_signal_override=args.vehicle_signal,
        pedestrian_signal_override=args.pedestrian_signal,
        right_turn_signal_override=args.right_turn_signal,
        stop_completed_on_red=args.assume_stopped_on_red,
    )
    fsm.update(ctx)

    # 5) 최종 판단, 원인, 마스크, 주행 영역을 한 화면에 시각화한다.
    vis = draw_overlay(
        frame,
        detections,
        seg_mask,
        seg_overlay,
        lane_geo,
        fsm.state,
        fsm.decision,
        ctx,
        yolo_ms,
        seg_ms,
        yolo_ms + seg_ms,
        fps=0.0,
        debug=args.debug,
    )
    return vis, seg_cache, detections, lane_geo, ctx


def parse_args():
    p = argparse.ArgumentParser(description="Intersection safety demo v2 for road_v4 segmentation.")
    p.add_argument("--source", default="0", help="video/image path or camera index")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=360)
    p.add_argument("--show", action="store_true")
    p.add_argument("--save", default="", help="output video/image path")

    p.add_argument("--yolo-weights", default=None)
    p.add_argument("--seg-weights", default=None, help="road_v4 lightweight BiSeNetV2 checkpoint")
    p.add_argument("--device", default="cpu", help="cpu/cuda")

    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--iou", type=float, default=0.45)

    p.add_argument("--seg-input-h", type=int, default=352)
    p.add_argument("--seg-input-w", type=int, default=640)
    p.add_argument("--yolo-interval", type=int, default=1)
    p.add_argument("--seg-interval", type=int, default=1)
    p.add_argument("--corridor-width", type=int, default=80)
    p.add_argument("--no-vehicle-lane-suppression", action="store_true")

    p.add_argument("--no-distance", action="store_true")
    p.add_argument("--camera-hfov", type=float, default=70.0)
    p.add_argument("--distance-focal-px", type=float, default=0.0)
    p.add_argument("--vehicle-real-width", type=float, default=1.8)
    p.add_argument("--vehicle-real-height", type=float, default=1.5)
    p.add_argument("--distance-scale", type=float, default=0.8)

    p.add_argument("--vehicle-signal", default="auto", choices=["auto", "red", "yellow", "green", "unknown"])
    p.add_argument("--pedestrian-signal", default="auto", choices=["auto", "red", "green", "unknown"])
    p.add_argument("--right-turn-signal", default="none", choices=["none", "auto", "red", "yellow", "green", "unknown"])
    p.add_argument("--assume-stopped-on-red", action="store_true")
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    args._seg_countdown = 0

    # weight를 명시하지 않으면 v2/road_v4_best_light.pt를 우선 찾는다.
    yolo_path = args.yolo_weights or find_existing(DEFAULT_YOLO_WEIGHTS, ALT_YOLO_WEIGHTS, "YOLO")
    seg_path = args.seg_weights or find_existing(DEFAULT_SEG_WEIGHTS, None, "road_v4 segmentation")

    print("=" * 68)
    print("Intersection Safety Demo v2")
    print("YOLO + road_v4 lightweight BiSeNetV2 + FSM")
    print("=" * 68)
    print(f"[YOLO] {yolo_path}")
    print(f"[SEG ] {seg_path}")
    print(f"[device] {args.device}")

    yolo = YoloDetector(
        weights=yolo_path,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        interval=args.yolo_interval,
        estimate_distance=not args.no_distance,
        camera_hfov_deg=args.camera_hfov,
        distance_focal_px=args.distance_focal_px,
        vehicle_real_width_m=args.vehicle_real_width,
        vehicle_real_height_m=args.vehicle_real_height,
        distance_scale=args.distance_scale,
    )
    seg = RoadV4Segmentor(
        weights=seg_path,
        input_size=(args.seg_input_h, args.seg_input_w),
        device=args.device,
    )
    fsm = IntersectionFSM()
    lane_tracker = LaneTracker(alpha=0.4, max_miss=10)

    # source가 숫자면 웹캠, 이미지 확장자면 단일 이미지, 나머지는 영상으로 처리한다.
    source_arg = args.source
    if source_arg.isdigit():
        source = int(source_arg)
        mode = "camera"
    elif source_arg.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
        source = source_arg
        mode = "image"
    else:
        source = source_arg
        mode = "video"

    if mode == "image":
        # 단일 이미지 테스트는 모델 연결 확인과 시각화 디버깅에 유용하다.
        frame = cv2.imread(str(source))
        if frame is None:
            raise RuntimeError(f"failed to read image: {source}")
        frame = cv2.resize(frame, (args.width, args.height))
        vis, _, dets, geo, ctx = process_frame(frame, yolo, seg, None, fsm, lane_tracker, args)
        print(f"state={fsm.state.value} decision={fsm.decision.value} reason={ctx.reason}")
        if args.save:
            cv2.imwrite(args.save, vis)
            print(f"[SAVE] {args.save}")
        if args.show:
            cv2.imshow("Intersection Demo v2", vis)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return

    cap = cv2.VideoCapture(source)
    if isinstance(source, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise RuntimeError(f"failed to open source: {source}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if src_fps <= 0 or src_fps > 120:
        src_fps = 30.0

    writer = None
    seg_cache = None
    frame_count = 0
    fps_smooth = 0.0

    try:
        while True:
            t0 = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.resize(frame, (args.width, args.height))

            # 한 프레임 처리: YOLO -> BiSeNetV2 -> 후처리 -> FSM -> 시각화
            vis, seg_cache, dets, geo, ctx = process_frame(frame, yolo, seg, seg_cache, fsm, lane_tracker, args)

            elapsed = time.perf_counter() - t0
            cur_fps = 1.0 / max(1e-6, elapsed)
            fps_smooth = 0.15 * cur_fps + 0.85 * fps_smooth if fps_smooth > 0 else cur_fps
            cv2.putText(vis, f"FPS: {fps_smooth:.1f}", (16, 88),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

            frame_count += 1
            if frame_count % 30 == 0:
                print(
                    f"frame={frame_count:5d} FPS={fps_smooth:5.1f} "
                    f"state={fsm.state.value:24s} decision={fsm.decision.value:8s} "
                    f"lane={geo.lane_confidence:.2f} objs={len(dets)} reason={ctx.reason}"
                )

            if args.save:
                if writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(args.save, fourcc, src_fps, (args.width, args.height))
                writer.write(vis)

            if args.show:
                cv2.imshow("Intersection Demo v2", vis)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                if key == ord("d"):
                    args.debug = not args.debug
                    print(f"[debug] {args.debug}")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
            print(f"[SAVE] {args.save}")
        if args.show:
            cv2.destroyAllWindows()
        print(f"[DONE] frames={frame_count}")


if __name__ == "__main__":
    main()
