from ultralytics import YOLO

model = YOLO("runs/input_residual_dgf_llvip_img1024_b24_w8_amp_adamw_lr1e3_e120/weights/best.pt")

metrics = model.val(
    data="configs/llvip.yaml",
    imgsz=1024,
    batch=24,
    workers=8,
    device=0,
    project="runs/val",
    name="input_residual_dgf_best_val",
    plots=False,
    verbose=False,
)

print(metrics.results_dict)
