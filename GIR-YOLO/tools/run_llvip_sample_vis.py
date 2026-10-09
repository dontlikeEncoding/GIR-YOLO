import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

cmd = [
    sys.executable,
    str(ROOT / "tools" / "visualize_yolo_predictions.py"),
    "--image_dir", str(ROOT / "data" / "LLVIP_sample" / "images"),
    "--label_dir", str(ROOT / "data" / "LLVIP_sample" / "labels"),
    "--out_dir", str(ROOT / "result" / "llvip_sample_vis"),
    "--model_a", "cspnet_hrnet",
    "--ckpt_a", str(ROOT / "result" / "csp_experiments_results" / "csp_baseline_b16_lr1e3_e150" / "best_model.pth"),
    "--name_a", "CSP-HRNet",
    "--model_b", "lowlight_cspnet_v2",
    "--ckpt_b", str(ROOT / "result" / "csp_experiments_results" / "frbnet_v2_gate_m1_b16_lr1e3_e150" / "best_model.pth"),
    "--name_b", "FRBNet-v2",
    "--limit", "20",
    "--device", "cuda",
    "--score", "0.50",
]

print("Running:")
print(" ".join(cmd))
subprocess.run(cmd, check=True, cwd=str(ROOT))
