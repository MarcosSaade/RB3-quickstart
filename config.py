# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Configuration constants and device detection."""

import os

# ========= Model and File Constants =========
TF_MODEL = "googlenet_quantized.tflite"
PPE_MODEL = "ppe-detection-w8a8.lite"
LABELS = "imagenet_labels.txt"
DELEGATE_PATH = "libQnnTFLiteDelegate.so"

# ========= PPE Detection Configuration =========
PPE_CONFIG = {
    "labels": ["hat", "vest"],
    "input_size": (96, 96),  # Width, Height
    "grid_size": (12, 12),   # Output grid dimensions
    "prob_threshold": 0.3,   # Detection confidence threshold
    "box_scale": 1.2,        # Bounding box size relative to grid cell
}

# ========= Device Detection =========
UNAME = os.uname().nodename
DEVICE_OS = "QualcommLinux" if UNAME == "qcs6490-rb3gen2-vision-kit" else "Ubuntu"

# ========= Display Configuration =========
DISPLAY_CONFIG = {
    "max_width": 800,
    "max_height": 700,  # Reduced from 800 to fit smaller monitors
    "camera_display_size": (320, 240),  # Consistent for both CPU and delegate modes
    "camera_resolution": (640, 480),
}

# ========= Performance Configuration =========
PERFORMANCE_CONFIG = {
    "ui_update_interval_delegate": 0.1,  # 10 FPS for delegate UI
    "ui_update_interval_cpu": 0.05,      # 20 FPS for CPU UI
    "display_update_interval": 0.033,    # 30 FPS display for both modes
    "results_update_interval_delegate": 0.15,  # 150ms for delegate
    "results_update_interval_cpu": 0.1,        # 100ms for CPU
    "inference_delay_delegate": 0.001,   # 1ms delay for delegate
    "inference_delay_cpu": 0.02,        # 20ms delay for CPU
    "status_interval_delegate": 100,     # Print status every 100 inferences
    "status_interval_cpu": 10,          # Print status every 10 inferences
}

# ========= Camera Configuration =========
CAMERA_CONFIG = {
    "pipeline_str": (
        "qtiqmmfsrc camera=0 ! "
        "video/x-raw,format=NV12,width=640,height=480 ! "
        "videoconvert ! "
        "appsink name=appsink emit-signals=true sync=false max-buffers=1 drop=true"
    ),
    "camera_timeout": 10,
    "cleanup_timeout": 5,
    "top_k_predictions": 4,
}
