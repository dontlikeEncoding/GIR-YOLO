from ultralytics import YOLO

best = "runs/frb_yola_frontend_llvip_img1024_b24_w8_amp_adamw_lr1e3_e120/weights/best.pt"

model = YOLO(best)
metrics = model.val(
    data="configs/llvip.yaml",
    imgsz=1024,
    batch=24,
    device=0,
    workers=8,
    project="runs/val",
    name="frb_yola_frontend_best_val",
    plots=False,
)
print(metrics.results_dict)
