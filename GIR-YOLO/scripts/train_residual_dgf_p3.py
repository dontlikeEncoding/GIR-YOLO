from ultralytics import YOLO

model = YOLO("configs/yolov8n_input_residual_dgf.yaml").load(
    "yolov8n.pt"
)

model.train(
    data="configs/llvip.yaml",
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
    name="residual_dgf_p3_llvip_img1024_b24_w8_amp_adamw_lr1e3_e120",
    plots=False,
    verbose=False,
)
