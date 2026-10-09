from ultralytics import YOLO

tests = [
    ("yolov8n_baseline", "weights/yolov8n_baseline.pt"),
    ("strength_005", "runs/input_residual_dgf_s005_llvip_img1024_b24_w8_amp_adamw_lr1e3_e120/weights/best.pt"),
    ("strength_010_gate_m5", "runs/input_residual_dgf_llvip_img1024_b24_w8_amp_adamw_lr1e3_e120/weights/best.pt"),
    ("strength_020", "runs/input_residual_dgf_s020_llvip_img1024_b24_w8_amp_adamw_lr1e3_e120/weights/best.pt"),
    ("gate_m3", "runs/input_residual_dgf_gate_m3_llvip_img1024_b24_w8_amp_adamw_lr1e3_e120/weights/best.pt"),
    ("gate_m7", "runs/input_residual_dgf_gate_m7_llvip_img1024_b24_w8_amp_adamw_lr1e3_e120/weights/best.pt"),
    ("alpha001", "runs/input_residual_dgf_alpha001_llvip_img1024_b24_w8_amp_adamw_lr1e3_e120/weights/best.pt"),
]

for name, weight in tests:
    print(f"\n===== {name} =====")
    model = YOLO(weight)
    metrics = model.val(
        data="configs/llvip.yaml",
        imgsz=1024,
        batch=1,
        workers=4,
        device=0,
        plots=False,
        verbose=False,
        project="runs/benchmark",
        name=f"{name}_b1",
    )
    print(metrics.results_dict)
