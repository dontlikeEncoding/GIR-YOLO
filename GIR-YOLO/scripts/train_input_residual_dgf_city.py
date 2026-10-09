import sys
from ultralytics import YOLO

yaml_path = sys.argv[1]
run_name = sys.argv[2]

model = YOLO(yaml_path).load("yolov8n.pt")

model.train(
    data="configs/citypersons.yaml",
    epochs=120,
    imgsz=1024,
    batch=24,
    workers=8,
    device=0,
    amp=True,
    optimizer="AdamW",
    lr0=0.001,
    warmup_epochs=1,
    project="runs",
    name=run_name,
    plots=False,
    verbose=False,
)
