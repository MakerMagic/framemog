# FrameMog

A small Python project for body frame analysis using MediaPipe Pose and YOLO segmentation.

## What it does

- Uses MediaPipe Pose to detect body landmarks.
- Uses YOLO segmentation (`yolo11n-seg.pt`) to extract the visible body contour.
- Corrects shoulder and hip width measurements by combining landmark Y positions with body mask X edges.
- Calculates a frame ratio using true shoulder width / true hip width.
- Displays a `Framemog Meter` overlay based on the ratio:
  - `David Laid` when ratio >= 1.1
  - `KlaviKular (luxmaxer)` when 1.0 <= ratio < 1.1
  - `Chud` when ratio < 1.0

## Files

- `main.py` — main script.
- `requirements.txt` — Python dependencies.
- `yolo11n-seg.pt` — YOLO segmentation model used for body masking.
- `framemom_klavikular.png`, `framemom_david.png`, `framemom_chud.png` — overlay images for the meter.

## Installation

```bash
python -m pip install -r requirements.txt
```

## Run

```bash
python main.py
```

Press `q` in the OpenCV window to exit.

## Notes

- Place the three PNG files in the same folder as `main.py`.
- If the images are missing, the script will print a warning.
- The meter now uses the shoulder-to-hip ratio, not absolute shoulder width.
