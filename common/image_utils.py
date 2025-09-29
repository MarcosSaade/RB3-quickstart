# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Image processing and preprocessing utilities."""

import cv2
import numpy as np
import gi
gi.require_version('GdkPixbuf', '2.0')
from gi.repository import GdkPixbuf
from .config import DISPLAY_CONFIG, PPE_CONFIG


def stable_softmax(logits):
    """Apply stable softmax to prevent numerical overflow."""
    # Convert logits to float32 for precision
    logits = np.array(logits, dtype=np.float32)
    
    # Ensure we have a 1D array
    if logits.ndim > 1:
        logits = logits.flatten()
    
    # Subtract the maximum logit to prevent overflow
    shifted_logits = logits - np.max(logits)
    
    # Clip the shifted logits to a safe range to prevent overflow in exp
    shifted_logits = np.clip(shifted_logits, -500, 500)
    
    # Calculate the exponentials and normalize
    exp_scores = np.exp(shifted_logits)
    probabilities = exp_scores / np.sum(exp_scores)
    
    return probabilities


def load_labels(label_path):
    """Load classification labels from file."""
    try:
        with open(label_path, 'r') as f:
            return [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        print(f"Warning: Label file {label_path} not found")
        return [f"Class_{i}" for i in range(1000)]  # Fallback labels


def resize_image_for_display(pixbuf):
    """Resize image pixbuf for display while preserving aspect ratio."""
    original_width = pixbuf.get_width()
    original_height = pixbuf.get_height()

    # Target display size from config
    max_width = DISPLAY_CONFIG["max_width"]
    max_height = DISPLAY_CONFIG["max_height"]

    # Calculate new size preserving aspect ratio
    scale = min(max_width / original_width, max_height / original_height)
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)

    return new_width, new_height


def preprocess_image(image_path, input_shape, input_dtype):
    """Load and preprocess input image for inference."""
    # Read the image using OpenCV
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image at {image_path}")
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize the image to the desired input shape
    img = cv2.resize(img, (input_shape[2], input_shape[1]))
    
    # Convert to the desired data type
    img = img.astype(input_dtype)
    
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    
    return img


def preprocess_frame(frame, input_shape, input_dtype):
    """Preprocess camera frame for inference."""
    # Frame is already in RGB format from camera
    # Resize the frame to the desired input shape
    img = cv2.resize(frame, (input_shape[2], input_shape[1]))
    
    # Convert to the desired data type
    img = img.astype(input_dtype)
    
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    
    return img


def convert_nv12_to_rgb(frame_data, width, height):
    """Convert NV12 format frame to RGB."""
    # NV12 format: Y plane followed by interleaved UV plane
    y_size = width * height
    uv_size = width * height // 2
    
    if len(frame_data) < y_size + uv_size:
        raise ValueError(f"Unexpected NV12 frame size: {len(frame_data)} bytes")
    
    # Extract Y and UV planes
    y_plane = frame_data[:y_size].reshape((height, width))
    uv_plane = frame_data[y_size:y_size + uv_size].reshape((height // 2, width))
    
    # Convert NV12 to RGB using OpenCV
    # First reshape to the format OpenCV expects
    nv12_frame = np.zeros((height + height // 2, width), dtype=np.uint8)
    nv12_frame[:height, :] = y_plane
    nv12_frame[height:, :] = uv_plane
    
    # Convert NV12 to RGB
    rgb_frame = cv2.cvtColor(nv12_frame, cv2.COLOR_YUV2RGB_NV12)
    
    return rgb_frame


def create_pixbuf_from_frame(display_frame):
    """Create GdkPixbuf from numpy array for display."""
    height, width, channels = display_frame.shape
    
    # Convert to the format GdkPixbuf expects (RGB, 8-bit)
    if display_frame.dtype != np.uint8:
        display_frame = display_frame.astype(np.uint8)
    
    # Create GdkPixbuf from numpy array
    pixbuf = GdkPixbuf.Pixbuf.new_from_data(
        display_frame.tobytes(),
        GdkPixbuf.Colorspace.RGB,
        False,  # No alpha channel
        8,      # Bits per sample
        width,
        height,
        width * channels  # Row stride
    )
    
    return pixbuf


# ========= PPE Detection Preprocessing Functions =========

def letterbox_bgr(img_bgr, new_w, new_h):
    """
    Resize image while preserving aspect ratio using letterboxing.
    Returns the letterboxed image and scaling information for coordinate conversion.
    """
    h, w = img_bgr.shape[:2]
    scale = min(new_w / w, new_h / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((new_h, new_w, 3), dtype=img_bgr.dtype)
    x0 = (new_w - nw) // 2
    y0 = (new_h - nh) // 2
    canvas[y0:y0+nh, x0:x0+nw] = resized
    return canvas, scale, x0, y0, w, h


def quantize_input(img_rgb, input_detail):
    """
    Quantize input image based on model's quantization parameters.
    Supports float32, uint8, and int8 input types.
    """
    dtype = input_detail['dtype']
    scale, zp = input_detail['quantization']
    
    # Normalize image to [0, 1] range first
    img_norm = img_rgb.astype(np.float32) / 255.0
    
    if dtype == np.float32:
        return np.expand_dims(img_norm, 0).astype(np.float32)
    elif dtype == np.uint8:
        # Quantize to uint8 range [0, 255]
        q = np.round(img_norm / max(scale, 1e-12) + zp).clip(0, 255).astype(np.uint8)
        return np.expand_dims(q, 0)
    elif dtype == np.int8:
        # Quantize to int8 range [-128, 127]
        # The model expects: quantized_value = (float_value / scale) + zero_point
        q = np.round(img_norm / max(scale, 1e-12) + zp).clip(-128, 127).astype(np.int8)
        return np.expand_dims(q, 0)
    else:
        raise RuntimeError(f"Unhandled input dtype: {dtype}")


def dequantize(arr, detail):
    """
    Dequantize model output back to float values.
    """
    scale, zp = detail.get('quantization', (0.0, 0))
    if scale == 0 or arr.dtype in (np.float32, np.float64):
        return arr.astype(np.float32)
    # Properly dequantize: float_value = (quantized_value - zero_point) * scale
    return ((arr.astype(np.float32) - zp) * scale).astype(np.float32)


def preprocess_ppe_image(image_path):
    """
    Preprocess image for PPE detection model.
    Returns letterboxed RGB image and scaling information.
    """
    # Load image
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    
    input_w, input_h = PPE_CONFIG["input_size"]
    lb_bgr, scale, x0, y0, orig_w, orig_h = letterbox_bgr(bgr, input_w, input_h)
    rgb = cv2.cvtColor(lb_bgr, cv2.COLOR_BGR2RGB)
    
    return rgb, scale, x0, y0, orig_w, orig_h


def preprocess_ppe_frame(frame):
    """
    Preprocess camera frame for PPE detection model.
    Frame is assumed to be already in RGB format.
    Returns letterboxed frame and scaling information.
    """
    input_w, input_h = PPE_CONFIG["input_size"]
    
    # Convert RGB to BGR for letterbox function
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    
    lb_bgr, scale, x0, y0, orig_w, orig_h = letterbox_bgr(bgr, input_w, input_h)
    rgb = cv2.cvtColor(lb_bgr, cv2.COLOR_BGR2RGB)
    
    return rgb, scale, x0, y0, orig_w, orig_h
