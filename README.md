# RB3 Quickstart Demo

A minimal inference demo for Qualcomm RB3 platform demonstrating GoogleNet image classification and PPE (Personal Protective Equipment) detection using TensorFlow Lite with optional QNN hardware acceleration.

## Overview

This quickstart guide provides a streamlined demo for running AI inference on the Qualcomm RB3 Gen2 Vision Kit. The demo supports both image classification and PPE detection models, with automatic device detection and optimized performance for both CPU and hardware-accelerated (HTP) inference.

## Features

- **Image Classification**: GoogleNet-based classification with ImageNet labels
- **PPE Detection**: Hard hat and safety vest detection
- **Hardware Acceleration**: Automatic QNN delegate support with HTP backend
- **Live Camera**: Real-time inference using GStreamer camera pipeline
- **Cross-Platform**: Supports both QualcommLinux (RB3) and Ubuntu environments
- **Flexible Display**: GUI mode with camera preview or headless mode for terminal output

## Prerequisites

### Hardware
- Qualcomm RB3 Gen2 Vision Kit (recommended) or Ubuntu system
- Camera module connected to RB3

### Software Dependencies
- Python 3.6+
- TensorFlow Lite runtime
- OpenCV (cv2)
- GStreamer (for camera functionality)
- GTK 3.0 (optional, for GUI display)
- NumPy

### Model Files
The following files should be present in the project directory:
- `googlenet_quantized.tflite` - GoogleNet classification model
- `ppe-detection-w8a8.lite` - PPE detection model
- `imagenet_labels.txt` - ImageNet class labels
- `libQnnTFLiteDelegate.so` - QNN delegate library (for hardware acceleration)

## Project Structure

```
rb3-quickstart-demo/
├── README.md                 # This file
├── classification.py         # Main demo application
├── inference_engine.py       # Core inference engine
├── camera_interface.py       # Camera capture and processing
├── config.py                # Configuration and device detection
├── image_utils.py           # Image preprocessing utilities
├── tflite_setup.py          # TensorFlow Lite setup for different platforms
├── test_camera.py           # Separate camera testing utility
├── googlenet_quantized.tflite    # Classification model (required)
├── ppe-detection-w8a8.lite       # PPE detection model (required)
├── imagenet_labels.txt            # Classification labels (required)
└── libQnnTFLiteDelegate.so        # QNN delegate library (required)
```

## Quick Start

### 1. Image Classification

Classify a single image using GoogleNet:

```bash
# Using hardware acceleration (HTP)
python3 classification.py --image /path/to/image.jpg --htp

# Using CPU only
python3 classification.py --image /path/to/image.jpg --cpu
```

**Example output:**
```
Classifying image: dog.jpg
Device: QualcommLinux
Mode: HTP Hardware Acceleration
--------------------------------------------------
Inference time: 12.3ms

Top predictions:
 1. golden retriever    87.45%
 2. Labrador retriever  8.32%
 3. beagle             2.14%
 4. cocker spaniel     1.89%
 5. Nova Scotia duck   0.20%
```

### 2. Live Camera Classification

Run real-time classification on camera feed:

```bash
# With camera preview window
python3 classification.py --live-camera --htp

# Headless mode (terminal output only)
python3 classification.py --live-camera --htp --headless

# CPU mode with preview
python3 classification.py --live-camera --cpu
```

**Live output example:**
```
Frame #0042 - Top 5 Predictions:
============================================================
  1. coffee mug                   92.34% [█████████░]
  2. cup                          4.23%  [░░░░░░░░░░]
  3. espresso                     2.11%  [░░░░░░░░░░]
  4. pitcher                      0.89%  [░░░░░░░░░░]
  5. tea cup                      0.43%  [░░░░░░░░░░]
============================================================
Device: QualcommLinux | Mode: HTP
Display: With Camera Preview
Press Ctrl+C to stop
```

## Command Line Options

### Input Modes (Required - Choose One)
- `--image PATH` - Classify a single image file
- `--live-camera` - Use live camera feed for continuous classification

### Processing Modes (Required - Choose One)
- `--htp` - Use HTP hardware acceleration (recommended for RB3)
- `--cpu` - Use CPU-only processing

### Display Options
- `--headless` - Run without GUI camera preview (terminal output only)

## Performance Notes

### Hardware Acceleration (HTP Mode)
- **Inference Time**: ~10-15ms per frame
- **Recommended For**: Real-time applications, battery efficiency
- **Requirements**: QNN delegate library, compatible hardware

### CPU Mode
- **Inference Time**: ~100-200ms per frame
- **Recommended For**: Development, testing, compatibility
- **Requirements**: Standard TensorFlow Lite runtime

## Camera Testing

A separate camera testing utility is provided for verifying camera functionality:

```bash
python3 test_camera.py
```

**Purpose**: `test_camera.py` is a standalone script for testing basic camera connectivity and GStreamer pipeline functionality without running inference. Use this to verify your camera setup before running the main demo.

## Device Detection

The demo automatically detects the platform:
- **QualcommLinux**: Detected on RB3 Gen2 Vision Kit (`qcs6490-rb3gen2-vision-kit`)
- **Ubuntu**: Detected on standard Ubuntu systems

Platform-specific optimizations are applied automatically, including:
- TensorFlow Lite API selection (C API for QualcommLinux, Python API for Ubuntu)
- Camera pipeline configuration
- Performance tuning parameters

## Troubleshooting

### Camera Issues
1. **Camera not detected**: Run `test_camera.py` to verify camera connectivity
2. **GStreamer errors**: Ensure GStreamer and camera drivers are properly installed
3. **Permission errors**: Check camera device permissions (`/dev/video*`)

### Inference Issues
1. **QNN delegate fails**: Demo will fallback to CPU mode automatically
2. **Model files missing**: Ensure all `.tflite` and `.txt` files are in the project directory
3. **GTK not available**: Demo will run in headless mode automatically

### Performance Issues
1. **Slow inference**: Try HTP mode (`--htp`) instead of CPU mode
2. **High CPU usage**: Use headless mode (`--headless`) to reduce display overhead
3. **Memory issues**: Close other applications to free system resources

## Configuration

Key settings can be modified in `config.py`:

```python
# Camera settings
CAMERA_CONFIG = {
    "camera_timeout": 10,
    "cleanup_timeout": 5,
    "top_k_predictions": 4,
}

# Display settings
DISPLAY_CONFIG = {
    "max_width": 800,
    "max_height": 700,
    "camera_display_size": (320, 240),
}

# Performance tuning
PERFORMANCE_CONFIG = {
    "ui_update_interval_delegate": 0.1,  # 10 FPS for HTP
    "ui_update_interval_cpu": 0.05,      # 20 FPS for CPU
}
```

## Model Support

### Classification Model
- **Model**: GoogleNet quantized for mobile deployment
- **Input**: 224x224 RGB images
- **Output**: 1000 ImageNet class probabilities
- **Labels**: Standard ImageNet class names

### PPE Detection Model (Future Support)
- **Model**: Custom YOLO-based PPE detection
- **Input**: 96x96 RGB images
- **Output**: Bounding boxes for hard hats and safety vests
- **Classes**: `["hat", "vest"]`

## Development

### Core Components

1. **`inference_engine.py`**: Main inference engine with platform-specific optimizations
2. **`camera_interface.py`**: GStreamer-based camera capture with frame processing
3. **`image_utils.py`**: Image preprocessing, postprocessing, and utility functions
4. **`config.py`**: Centralized configuration and automatic device detection
5. **`tflite_setup.py`**: Platform-specific TensorFlow Lite module loading

### Adding New Models

To add support for new models:
1. Add model configuration to `config.py`
2. Implement preprocessing in `image_utils.py`
3. Add output processing in `inference_engine.py`
4. Update model type handling in `classification.py`

## License

Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
SPDX-License-Identifier: BSD-3-Clause

## Support

For technical support and questions:
- Check the troubleshooting section above
- Verify all prerequisites are installed
- Test camera functionality with `test_camera.py`
- Ensure all model files are present and accessible