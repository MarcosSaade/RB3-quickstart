# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""TensorFlow Lite inference engine with support for QNN delegate."""

import time
import numpy as np
import ctypes
import cv2
from typing import List, Tuple, Optional

from .config import DEVICE_OS, TF_MODEL, PPE_MODEL, LABELS, PPE_CONFIG, DELEGATE_PATH, CAMERA_CONFIG
from .tflite_setup import get_tflite_module, get_delegate_options_class
from .image_utils import (preprocess_image, preprocess_frame, load_labels, stable_softmax,
                        preprocess_ppe_image, preprocess_ppe_frame, quantize_input, dequantize)


class InferenceEngine:
    """TensorFlow Lite inference engine with optional QNN delegate support."""
    
    def __init__(self):
        """Initialize the inference engine."""
        self.tflite = get_tflite_module()
        self.delegate_options_class = get_delegate_options_class()
        self.labels = load_labels(LABELS)
        self.ppe_labels = PPE_CONFIG["labels"]
        
    def _create_qualcomm_interpreter(self, use_delegate: bool, model_path: str = None, model_type: str = "classification"):
        """Create interpreter for QualcommLinux using C API."""
        # Load the TFLite model
        if model_path is None:
            model_path = TF_MODEL if model_type == "classification" else PPE_MODEL
        model = self.tflite.TfLiteModelCreateFromFile(model_path.encode("utf-8"))
        if not model:
            raise RuntimeError("Failed to load model.")
            
        options = self.tflite.TfLiteInterpreterOptionsCreate()
        
        if use_delegate:
            try:
                print("Attempting to load QNN delegate")
                delegate_options = self.tflite.TfLiteExternalDelegateOptionsDefault(
                    DELEGATE_PATH.encode("utf-8")
                )
                if not delegate_options:
                    raise RuntimeError("Failed to create delegate options")
                
                # Insert key-value option
                status = self.tflite.TfLiteExternalDelegateOptionsInsert(
                    ctypes.byref(delegate_options), b"backend_type", b"htp"
                )
                if status != 0:
                    raise RuntimeError("Failed to insert delegate option")

                delegate = self.tflite.TfLiteExternalDelegateCreate(
                    ctypes.byref(delegate_options)
                )
                if not delegate:
                    raise RuntimeError("Delegate creation failed")
         
                self.tflite.TfLiteInterpreterOptionsAddDelegate(options, delegate)
                print("QNN delegate loaded successfully")

            except Exception as e:
                print(f"WARNING: Failed to load QNN delegate: {e}")
                print("INFO: Continuing without QNN delegate")
        else:
            self.tflite.TfLiteInterpreterOptionsSetUseNNAPI(options, True)
                   
        interpreter = self.tflite.TfLiteInterpreterCreate(model, options)
        if interpreter is None:
            raise RuntimeError("Failed to create interpreter")
        
        self.tflite.TfLiteInterpreterAllocateTensors(interpreter)
        
        return interpreter, model, options
    
    def _create_ubuntu_interpreter(self, use_delegate: bool, model_path: str = None, model_type: str = "classification"):
        """Create interpreter for Ubuntu using Python API."""
        if model_path is None:
            model_path = TF_MODEL if model_type == "classification" else PPE_MODEL
        if use_delegate:
            try:
                # Load the QNN delegate library
                delegate_options = {'backend_type': 'htp'}
                delegate = self.tflite.load_delegate(DELEGATE_PATH, delegate_options)
                
                # Load the TFLite model
                interpreter = self.tflite.Interpreter(
                    model_path=model_path, 
                    experimental_delegates=[delegate]
                )
                print("INFO: Loaded QNN delegate with HTP backend")
            except Exception as e:
                print(f"WARNING: Failed to load QNN delegate: {e}")
                print("INFO: Continuing without QNN delegate")
                interpreter = self.tflite.Interpreter(model_path=model_path)
        else:
            interpreter = self.tflite.Interpreter(model_path=model_path)
        
        interpreter.allocate_tensors()
        return interpreter
    
    def _get_qualcomm_tensor_info(self, interpreter):
        """Get tensor information for QualcommLinux."""
        input_tensor = self.tflite.TfLiteInterpreterGetInputTensor(interpreter, 0)
        input_dims = self.tflite.TfLiteTensorNumDims(input_tensor)
        input_shape = tuple(
            self.tflite.TfLiteTensorDim(input_tensor, i) for i in range(input_dims)
        )
        tensor_type = self.tflite.TfLiteTensorType(input_tensor)

        # Map TFLite tensor type to NumPy dtype
        type_mapping = {
            1: np.float32,  # kTfLiteFloat32
            2: np.int32,    # kTfLiteInt32
            3: np.uint8,    # kTfLiteUInt8
        }
        
        if tensor_type not in type_mapping:
            raise ValueError(f"Unsupported tensor type: {tensor_type}")
            
        input_dtype = type_mapping[tensor_type]
        return input_tensor, input_shape, input_dtype
    
    def _get_ubuntu_tensor_info(self, interpreter):
        """Get tensor information for Ubuntu."""
        input_details = interpreter.get_input_details()
        input_shape = input_details[0]['shape']
        input_dtype = input_details[0]['dtype']
        return input_details, input_shape, input_dtype
    
    def _run_qualcomm_inference(self, interpreter, model, options, image_input, 
                              input_tensor, input_shape, input_dtype, is_frame, model_type="classification"):
        """Run inference on QualcommLinux."""
        # Process input based on model type
        if model_type == "ppe":
            if is_frame:
                rgb, scale, x0, y0, orig_w, orig_h = preprocess_ppe_frame(image_input)
            else:
                rgb, scale, x0, y0, orig_w, orig_h = preprocess_ppe_image(image_input)
            
            # Get input details for quantization (simulate what Ubuntu would provide)
            input_detail = {
                'dtype': input_dtype,
                'quantization': (0.007874016, 0)  # Default quantization params, should be read from model
            }
            input_data = quantize_input(rgb, input_detail)
        else:
            # Classification preprocessing
            if is_frame:
                input_data = preprocess_frame(image_input, input_shape, input_dtype)
            else:
                input_data = preprocess_image(image_input, input_shape, input_dtype)

        # Load input data to input tensor
        try:
            self.tflite.TfLiteTensorCopyFromBuffer(
                input_tensor, input_data.ctypes.data, input_data.nbytes
            )
        except Exception as e:
            print(f"Error copying input data to tensor: {e}")
            return [], 0

        # Run inference
        start_time = time.time()
        status = self.tflite.TfLiteInterpreterInvoke(interpreter)
        
        if status != 0:
            raise RuntimeError("TfLiteInterpreterInvoke failed!")
        
        end_time = time.time()
        inference_time = end_time - start_time

        # Get output tensor
        output_tensor = self.tflite.TfLiteInterpreterGetOutputTensor(interpreter, 0)
        output_dims = self.tflite.TfLiteTensorNumDims(output_tensor)
        output_shape = tuple(
            self.tflite.TfLiteTensorDim(output_tensor, i) for i in range(output_dims)
        )
        output_data = np.zeros(output_shape, dtype=np.uint8)
        
        # Load output data from output tensor
        self.tflite.TfLiteTensorCopyToBuffer(
            output_tensor, output_data.ctypes.data, output_data.nbytes
        )

        # Store scaling info for PPE detection coordinate conversion
        if model_type == "ppe":
            scaling_info = (scale, x0, y0, orig_w, orig_h) if 'scale' in locals() else None
            return self._process_output(output_data, model_type, scaling_info), inference_time
        else:
            return self._process_output(output_data, model_type), inference_time
    
    def _run_ubuntu_inference(self, interpreter, image_input, input_details, 
                            input_shape, input_dtype, is_frame, model_type="classification"):
        """Run inference on Ubuntu."""
        # Process input based on model type
        if model_type == "ppe":
            if is_frame:
                rgb, scale, x0, y0, orig_w, orig_h = preprocess_ppe_frame(image_input)
            else:
                rgb, scale, x0, y0, orig_w, orig_h = preprocess_ppe_image(image_input)
            
            input_data = quantize_input(rgb, input_details[0])
        else:
            # Classification preprocessing
            if is_frame:
                input_data = preprocess_frame(image_input, input_shape, input_dtype)
            else:
                input_data = preprocess_image(image_input, input_shape, input_dtype)
        
        # Load input data to input tensor
        interpreter.set_tensor(input_details[0]['index'], input_data)
        
        # Run inference
        try:
            start_time = time.time()
            interpreter.invoke()
            end_time = time.time()
        except Exception as e:
            print(f"Error during model invocation: {e}")
            return [], 0

        inference_time = end_time - start_time

        # Get output data
        output_details = interpreter.get_output_details()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        # Store scaling info for PPE detection coordinate conversion
        if model_type == "ppe":
            scaling_info = (scale, x0, y0, orig_w, orig_h) if 'scale' in locals() else None
            return self._process_output(output_data, model_type, scaling_info, output_details[0]), inference_time
        else:
            return self._process_output(output_data, model_type), inference_time
    
    def _process_output(self, output_data, model_type="classification", scaling_info=None, output_detail=None):
        """Process model output to get predictions."""
        if model_type == "ppe":
            return self._process_ppe_output(output_data, scaling_info, output_detail)
        else:
            return self._process_classification_output(output_data)
    
    def _process_classification_output(self, output_data) -> List[Tuple[str, float]]:
        """Process classification model output."""
        predicted_index = np.argmax(output_data)
        predicted_label = self.labels[predicted_index]
        
        # Apply softmax function
        logits = output_data[0]
        probabilities = stable_softmax(logits)

        # Get top K predictions
        top_k = CAMERA_CONFIG["top_k_predictions"]
        top_indices = np.argsort(probabilities)[::-1][:top_k]
        
        results = []
        for i in top_indices:
            if i < len(self.labels):  # Safety check
                result = (self.labels[i], probabilities[i] * 100)
                results.append(result)
            
        return results
    
    def _process_ppe_output(self, output_data, scaling_info=None, output_detail=None):
        """Process PPE detection model output."""
        # Dequantize output if needed
        if output_detail is not None:
            heat = dequantize(output_data, output_detail)[0]  # (12,12,3)
        else:
            # For Qualcomm path, assume already float or handle quantization
            heat = output_data.astype(np.float32)[0] if output_data.ndim > 3 else output_data.astype(np.float32)
        
        # Apply softmax
        exp_heat = np.exp(heat - np.max(heat, axis=2, keepdims=True))
        softmax_probs = exp_heat / np.sum(exp_heat, axis=2, keepdims=True)
        
        # Extract object probabilities (skip background channel 0)
        obj_probs = softmax_probs[:, :, 1:]  # (12,12,2) for hat,vest
        
        # Detection parameters
        prob_thresh = PPE_CONFIG["prob_threshold"]
        grid_size = PPE_CONFIG["grid_size"]
        input_size = PPE_CONFIG["input_size"]
        labels = PPE_CONFIG["labels"]
        box_scale = PPE_CONFIG["box_scale"]
        
        detections = []
        cell_w = input_size[0] / grid_size[0]  # 96 / 12 = 8
        cell_h = input_size[1] / grid_size[1]  # 96 / 12 = 8
        
        for cls_id, label in enumerate(labels):
            obj_channel = obj_probs[:, :, cls_id]
            
            # Find cells above threshold
            mask = obj_channel >= prob_thresh
            
            if np.any(mask):
                # Apply 3x3 non-maximum suppression
                prob8 = (obj_channel * 255).astype(np.uint8)
                kernel = np.ones((3,3), np.uint8)
                local_max = (prob8 == cv2.dilate(prob8, kernel)) & (prob8 > 0)
                
                # Combine threshold mask with local maxima
                final_mask = mask & local_max
                
                ys, xs = np.where(final_mask)
                
                for y, x in zip(ys, xs):
                    confidence = float(obj_channel[y, x])
                    
                    # Convert grid cell to image coordinates
                    cx_l = (x + 0.5) * cell_w
                    cy_l = (y + 0.5) * cell_h
                    
                    # Box size
                    bw_l = cell_w * box_scale
                    bh_l = cell_h * box_scale
                    
                    x1_l = cx_l - bw_l/2
                    y1_l = cy_l - bh_l/2
                    x2_l = cx_l + bw_l/2
                    y2_l = cy_l + bh_l/2
                    
                    if scaling_info is not None:
                        # Convert letterbox coordinates back to original image
                        scale, x0, y0, orig_w, orig_h = scaling_info
                        x1 = int((x1_l - x0) / scale)
                        y1 = int((y1_l - y0) / scale)
                        x2 = int((x2_l - x0) / scale)
                        y2 = int((y2_l - y0) / scale)
                        
                        # Clamp to image bounds
                        x1 = max(0, min(orig_w-1, x1))
                        y1 = max(0, min(orig_h-1, y1))
                        x2 = max(0, min(orig_w-1, x2))
                        y2 = max(0, min(orig_h-1, y2))
                    else:
                        # Use letterbox coordinates directly
                        x1, y1, x2, y2 = int(x1_l), int(y1_l), int(x2_l), int(y2_l)
                    
                    # Return format: (label, confidence, x1, y1, x2, y2)
                    detections.append((label, confidence * 100, x1, y1, x2, y2))
        
        return detections
    
    def _cleanup_qualcomm_resources(self, interpreter, model, options):
        """Clean up QualcommLinux resources."""
        self.tflite.TfLiteInterpreterDelete(interpreter)
        self.tflite.TfLiteModelDelete(model)
        self.tflite.TfLiteInterpreterOptionsDelete(options)
    
    def run_inference(self, image_input, use_delegate: bool, 
                     is_frame: bool = False, model_type: str = "classification") -> Tuple[List[Tuple[str, float]], float]:
        """
        Run inference on input image or frame.
        
        Args:
            image_input: Image file path or numpy array frame
            use_delegate: Whether to use QNN delegate
            is_frame: Whether input is a camera frame (True) or image path (False)
            model_type: Type of model - only "classification" is supported
            
        Returns:
            Tuple of (results, inference_time) where results is list of (label, confidence)
        """
        print(f"Running on {DEVICE_OS} using Delegate: {use_delegate}, Model: {model_type}")
        
        try:
            if DEVICE_OS == "QualcommLinux":
                return self._run_qualcomm_inference_complete(image_input, use_delegate, is_frame, model_type)
            else:
                return self._run_ubuntu_inference_complete(image_input, use_delegate, is_frame, model_type)
        except Exception as e:
            print(f"Error during inference: {e}")
            return [], 0
    
    def _run_qualcomm_inference_complete(self, image_input, use_delegate, is_frame, model_type="classification"):
        """Complete inference pipeline for QualcommLinux."""
        interpreter, model, options = self._create_qualcomm_interpreter(use_delegate, model_type=model_type)
        
        try:
            input_tensor, input_shape, input_dtype = self._get_qualcomm_tensor_info(interpreter)
            results, inference_time = self._run_qualcomm_inference(
                interpreter, model, options, image_input, 
                input_tensor, input_shape, input_dtype, is_frame, model_type
            )
            return results, inference_time
        finally:
            self._cleanup_qualcomm_resources(interpreter, model, options)
    
    def _run_ubuntu_inference_complete(self, image_input, use_delegate, is_frame, model_type="classification"):
        """Complete inference pipeline for Ubuntu."""
        interpreter = self._create_ubuntu_interpreter(use_delegate, model_type=model_type)
        input_details, input_shape, input_dtype = self._get_ubuntu_tensor_info(interpreter)
        
        results, inference_time = self._run_ubuntu_inference(
            interpreter, image_input, input_details, 
            input_shape, input_dtype, is_frame, model_type
        )
        return results, inference_time


# Global inference engine instance
_inference_engine = None


def get_inference_engine() -> InferenceEngine:
    """Get singleton inference engine instance."""
    global _inference_engine
    if _inference_engine is None:
        _inference_engine = InferenceEngine()
    return _inference_engine


def run_inference(image_input, use_delegate: bool, 
                 is_frame: bool = False, model_type: str = "classification") -> Tuple[List[Tuple[str, float]], float]:
    """
    Convenience function to run inference using the global engine.
    
    Args:
        image_input: Image file path or numpy array frame
        use_delegate: Whether to use QNN delegate
        is_frame: Whether input is a camera frame (True) or image path (False)
        model_type: Type of model - only "classification" is supported
        
    Returns:
        Tuple of (results, inference_time) where results is list of (label, confidence)
    """
    engine = get_inference_engine()
    return engine.run_inference(image_input, use_delegate, is_frame, model_type)
