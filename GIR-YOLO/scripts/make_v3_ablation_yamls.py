from pathlib import Path

base = Path("configs/yolov8n_input_residual_dgf.yaml").read_text()

variants = {
    "yolov8n_input_residual_dgf_s005.yaml": {
        "strength: 0.1": "strength: 0.05",
    },
    "yolov8n_input_residual_dgf_s020.yaml": {
        "strength: 0.1": "strength: 0.2",
    },
    "yolov8n_input_residual_dgf_gate_m3.yaml": {
        "gate_init: -5.0": "gate_init: -3.0",
    },
    "yolov8n_input_residual_dgf_gate_m7.yaml": {
        "gate_init: -5.0": "gate_init: -7.0",
    },
    "yolov8n_input_residual_dgf_alpha001.yaml": {
        "alpha_init: 0.0": "alpha_init: 0.01",
    },
}

for name, reps in variants.items():
    s = base
    for old, new in reps.items():
        s = s.replace(old, new)
    p = Path("configs") / name
    p.write_text(s)
    print(p)
