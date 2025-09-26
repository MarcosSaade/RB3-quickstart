# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""TensorFlow Lite setup and imports based on device type."""

import ctypes
from config import DEVICE_OS, DELEGATE_PATH

if DEVICE_OS == "QualcommLinux":
    # Load TensorFlow Lite C library for QualcommLinux
    tflite = ctypes.CDLL('libtensorflowlite_c.so')
    ctypes.CDLL("libQnnTFLiteDelegate.so")
    
    class TfLiteExternalDelegateOptions(ctypes.Structure):
        """Structure for TensorFlow Lite external delegate options."""
        _fields_ = [
            ("lib_path", ctypes.c_char_p),
            ("count", ctypes.c_int),
            ("keys", ctypes.c_char_p * 256),
            ("values", ctypes.c_char_p * 256),
            ("insert", ctypes.c_void_p),
        ]

    # ========= TFLite Function Signatures =========
    def setup_tflite_function_signatures():
        """Setup TensorFlow Lite C API function signatures."""
        # Model functions
        tflite.TfLiteModelCreateFromFile.restype = ctypes.c_void_p
        tflite.TfLiteModelCreateFromFile.argtypes = [ctypes.c_char_p]
        tflite.TfLiteModelDelete.argtypes = [ctypes.c_void_p]
        
        # Interpreter functions
        tflite.TfLiteInterpreterOptionsCreate.restype = ctypes.c_void_p
        tflite.TfLiteInterpreterCreate.restype = ctypes.c_void_p
        tflite.TfLiteInterpreterCreate.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        tflite.TfLiteInterpreterDelete.argtypes = [ctypes.c_void_p]
        tflite.TfLiteInterpreterOptionsDelete.argtypes = [ctypes.c_void_p]
        tflite.TfLiteInterpreterAllocateTensors.argtypes = [ctypes.c_void_p]
        tflite.TfLiteInterpreterInvoke.argtypes = [ctypes.c_void_p]
        
        # Tensor functions
        tflite.TfLiteInterpreterGetInputTensor.restype = ctypes.c_void_p
        tflite.TfLiteInterpreterGetInputTensor.argtypes = [ctypes.c_void_p, ctypes.c_int]
        tflite.TfLiteInterpreterGetOutputTensor.restype = ctypes.c_void_p
        tflite.TfLiteInterpreterGetOutputTensor.argtypes = [ctypes.c_void_p, ctypes.c_int]
        tflite.TfLiteInterpreterGetOutputTensorCount.restype = ctypes.c_int
        tflite.TfLiteInterpreterGetOutputTensorCount.argtypes = [ctypes.c_void_p]
        
        # Tensor properties
        tflite.TfLiteTensorNumDims.restype = ctypes.c_int
        tflite.TfLiteTensorNumDims.argtypes = [ctypes.c_void_p]
        tflite.TfLiteTensorDim.restype = ctypes.c_int
        tflite.TfLiteTensorDim.argtypes = [ctypes.c_void_p, ctypes.c_int]
        tflite.TfLiteTensorType.restype = ctypes.c_int
        tflite.TfLiteTensorType.argtypes = [ctypes.c_void_p]
        
        # Tensor data operations
        tflite.TfLiteTensorCopyFromBuffer.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
        tflite.TfLiteTensorCopyToBuffer.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
        
        # NNAPI options
        tflite.TfLiteInterpreterOptionsSetUseNNAPI.restype = ctypes.c_void_p
        tflite.TfLiteInterpreterOptionsSetUseNNAPI.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        
        # Delegate functions
        tflite.TfLiteDelegateCreate.restype = ctypes.c_void_p
        tflite.TfLiteDelegateCreate.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        tflite.TfLiteInterpreterModifyGraphWithDelegate.restype = ctypes.c_void_p
        tflite.TfLiteInterpreterModifyGraphWithDelegate.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        
        # External delegate functions
        tflite.TfLiteExternalDelegateOptionsInsert.restype = ctypes.c_int
        tflite.TfLiteExternalDelegateOptionsInsert.argtypes = [
            ctypes.POINTER(TfLiteExternalDelegateOptions), 
            ctypes.c_char_p, 
            ctypes.c_char_p
        ]
        tflite.TfLiteExternalDelegateOptionsDefault.restype = TfLiteExternalDelegateOptions
        tflite.TfLiteExternalDelegateOptionsDefault.argtypes = [ctypes.c_void_p]
        tflite.TfLiteExternalDelegateCreate.argtypes = [ctypes.POINTER(TfLiteExternalDelegateOptions)]
        tflite.TfLiteExternalDelegateCreate.restype = ctypes.c_void_p
        tflite.TfLiteInterpreterOptionsAddDelegate.restype = None
        tflite.TfLiteInterpreterOptionsAddDelegate.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        tflite.TfLiteExternalDelegateDelete.argtypes = [ctypes.c_void_p]
        tflite.TfLiteExternalDelegateDelete.restype = None
    
    # Setup function signatures
    setup_tflite_function_signatures()

else:
    # Use TensorFlow Lite runtime for Ubuntu
    import tflite_runtime.interpreter as tflite
    # Placeholder for consistency
    TfLiteExternalDelegateOptions = None


def get_tflite_module():
    """Get the appropriate TensorFlow Lite module for the current platform."""
    return tflite


def get_delegate_options_class():
    """Get the delegate options class for the current platform."""
    return TfLiteExternalDelegateOptions
