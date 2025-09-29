# RB3 GoogleNet Classification Quickstart

Minimal image classification demo for Qualcomm RB3 platform using GoogleNet model.

## Quick Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download Required Files

#### GoogleNet Model
Download the **GoogleNet Quantized INT8 (W8A8)** model from Qualcomm AI Hub:
- Visit: https://aihub.qualcomm.com/iot/models/googlenet
- Download the **INT8 quantized** version (w8a8)
- Save as `googlenet_quantized.tflite` in the project directory

#### ImageNet Labels
Download ImageNet class labels:
```bash
wget https://raw.githubusercontent.com/tensorflow/models/master/research/slim/datasets/imagenet_2012_challenge_label_map_proto.pbtxt -O imagenet_labels.txt
```
Or download from any ImageNet labels source and save as `imagenet_labels.txt`

### 3. Run Classification

#### Image Classification
```bash
# Using HTP hardware acceleration
python classification.py --image path/to/your/image.jpg --htp

# Using CPU only
python classification.py --image path/to/your/image.jpg --cpu
```

#### Live Camera Classification
```bash
# Using HTP hardware acceleration with camera preview
python classification.py --live-camera --htp

# Using CPU only with camera preview
python classification.py --live-camera --cpu

# Headless mode (no preview window, for systems without display)
python classification.py --live-camera --htp --headless
```

## Arguments

### Required Arguments (choose one):
- `--image PATH`: Classify a static image file
- `--live-camera`: Use live camera feed for classification

### Required Processing Mode (choose one):
- `--htp`: Use HTP hardware acceleration (faster, recommended for RB3)
- `--cpu`: Use CPU-only processing (slower, for debugging)

### Optional Arguments:
- `--headless`: Run without camera preview window (only for live camera mode)

## Examples

```bash
# Classify a dog image using HTP acceleration
python classification.py --image dog.jpg --htp

# Live camera with CPU processing
python classification.py --live-camera --cpu

# Headless live camera with HTP (no GUI)
python classification.py --live-camera --htp --headless
```

## File Structure

### Core Files

- **`classification.py`** - Main entry point script that provides command-line interface for both image and live camera classification using GoogleNet model
- **`inference_engine.py`** - TensorFlow Lite inference engine with QNN delegate support for hardware acceleration on RB3 platform
- **`camera_interface.py`** - GStreamer-based camera interface for live video capture and real-time processing with optional GUI preview
- **`config.py`** - Configuration constants including model paths, device detection, and performance settings
- **`image_utils.py`** - Image processing utilities for preprocessing, format conversion, and display operations
- **`tflite_setup.py`** - TensorFlow Lite setup and C API bindings for different device types (RB3 vs desktop)

### Supporting Files

- **`test_camera.py`** - Standalone camera testing utility to verify GStreamer pipeline functionality on RB3
- **`requirements.txt`** - Python package dependencies for the project
- **`README.md`** - This documentation file with setup instructions and usage examples

## Requirements

- Qualcomm RB3 platform (or compatible Linux system)
- Python 3.6+
- Camera (for live classification mode)
- Downloaded GoogleNet model and ImageNet labels (see setup above)
