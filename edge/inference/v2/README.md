# Intersection Demo v2

`v2`는 `road_v4` 경량 BiSeNetV2 모델을 쓰는 새 판단 프로그램이다.
기존 inference 코드는 그대로 두고, 새 class id 체계만 이 폴더에서 따로 처리한다.

자세한 구현 의도와 처리 흐름은 [IMPLEMENTATION_NOTES.md](IMPLEMENTATION_NOTES.md)에 정리했다.

## Segmentation Classes

| id | class |
|---:|---|
| 0 | background |
| 1 | lane_white |
| 2 | lane_yellow |
| 3 | lane_blue |
| 4 | crosswalk |
| 5 | stop_line |

기존 프로그램은 lane `1..6`, crosswalk `7`, stop line `8`을 기대했지만,
`road_v4`는 lane `1..3`, crosswalk `4`, stop line `5`를 사용한다.

## Run

학습된 경량 checkpoint를 다음 위치에 둔다.

```text
edge/inference/v2/road_v4_best_light.pt
```

또는 `--seg-weights`로 직접 지정한다.

```bash
python edge/inference/v2/intersection_demo_v2.py --source edge/inference/v4.mp4 --seg-weights path/to/best_light_infer.pt --show
```

Save output:

```bash
python edge/inference/v2/intersection_demo_v2.py --source edge/inference/v4.mp4 --seg-weights path/to/best_light_infer.pt --save edge/inference/v2/out_v4.mp4
```

라즈베리파이용 저부하 옵션:

```bash
python edge/inference/v2/intersection_demo_v2.py --source 0 --device cpu --width 480 --height 270 --seg-input-h 256 --seg-input-w 464 --yolo-interval 3 --seg-interval 2 --show
```

## Files

- `intersection_demo_v2.py`: main program
- `road_v4_segmentor.py`: lightweight BiSeNetV2 checkpoint loader
- `road_v4_postprocess.py`: road_v4 class id aware lane/crosswalk/stop-line postprocess
- `visualizer_v2.py`: compact overlay renderer
