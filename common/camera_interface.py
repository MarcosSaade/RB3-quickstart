# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Camera interface using GStreamer for live video capture and processing."""

import time
import threading
import numpy as np
import cv2
import gi

gi.require_version('Gst', '1.0')
gi.require_version('GstVideo', '1.0')
from gi.repository import Gst, GLib

from .config import CAMERA_CONFIG, PERFORMANCE_CONFIG
from .image_utils import convert_nv12_to_rgb
from .inference_engine import run_inference


def draw_ppe_detections(frame, detections, original_size=None, display_size=None):
    """Draw PPE detection bounding boxes on frame.
    
    Args:
        frame: Image frame to draw on
        detections: List of (label, confidence, x1, y1, x2, y2) tuples
        original_size: Original frame size (width, height) - for coordinate scaling
        display_size: Display frame size (width, height) - for coordinate scaling
    
    Returns:
        Frame with bounding boxes drawn
    """
    if not detections:
        return frame
    
    annotated_frame = frame.copy()
    
    # Calculate scaling factors if sizes are provided
    scale_x = scale_y = 1.0
    if original_size and display_size:
        scale_x = display_size[0] / original_size[0]
        scale_y = display_size[1] / original_size[1]
    
    for detection in detections:
        if len(detection) >= 6:  # (label, confidence, x1, y1, x2, y2)
            label, confidence, x1, y1, x2, y2 = detection[:6]
            
            # Scale coordinates to display size
            x1_scaled = int(x1 * scale_x)
            y1_scaled = int(y1 * scale_y)
            x2_scaled = int(x2 * scale_x)
            y2_scaled = int(y2 * scale_y)
            
            # Choose color based on label
            color = (0, 255, 0) if label == "vest" else (255, 0, 0)  # Green for vest, red for hat
            
            # Draw rectangle
            cv2.rectangle(annotated_frame, (x1_scaled, y1_scaled), (x2_scaled, y2_scaled), color, 2)
            
            # Draw label with confidence
            label_text = f"{label} {confidence:.1f}%"
            label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            
            # Background rectangle for text
            cv2.rectangle(annotated_frame, 
                         (x1_scaled, max(0, y1_scaled - label_size[1] - 10)),
                         (x1_scaled + label_size[0], y1_scaled),
                         color, -1)
            
            # Text
            cv2.putText(annotated_frame, label_text, 
                       (x1_scaled, max(label_size[1], y1_scaled - 5)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return annotated_frame


class CameraInference:
    """Camera inference class for live video processing with GStreamer."""
    
    def __init__(self, callback=None, model_type="classification", 
                 use_delegate=True, detection_callback=None, image_callback=None):
        """
        Initialize camera inference.
        
        Args:
            callback: Callback function for inference results and display frames (legacy)
            model_type: Type of model to use - "classification", "ppe", or "yolo"
            use_delegate: Whether to use hardware acceleration
            detection_callback: Callback function for detection results only
            image_callback: Callback function for camera frames only
        """
        # Initialize GStreamer
        Gst.init(None)
        
        self.pipeline = None
        self.appsink = None
        self.callback = callback
        self.detection_callback = detection_callback
        self.image_callback = image_callback
        self.model_type = model_type
        self.use_delegate = use_delegate
        self.is_running = False
        self.is_stopping = False
        self.inference_thread = None
        self.current_frame = None
        
        # Threading locks
        self.frame_lock = threading.Lock()
        self.stop_lock = threading.Lock()
        
        # Frame counting for debugging
        self.frame_count = 0
        
    def check_camera_availability(self) -> bool:
        """Check if camera is available before creating pipeline."""
        try:
            # Try to create a simple test pipeline
            test_pipeline = Gst.parse_launch("qtiqmmfsrc camera=0 ! fakesink")
            if not test_pipeline:
                print("✗ Cannot create test pipeline - camera may be in use")
                return False
            
            # Try to set it to READY state
            ret = test_pipeline.set_state(Gst.State.READY)
            if ret == Gst.StateChangeReturn.FAILURE:
                print("✗ Camera is not available or in use by another application")
                test_pipeline.set_state(Gst.State.NULL)
                return False
            
            # Clean up test pipeline
            test_pipeline.set_state(Gst.State.NULL)
            test_pipeline.get_state(timeout=2000000000)  # 2 seconds
            print("✓ Camera availability confirmed")
            return True
            
        except Exception as e:
            print(f"✗ Camera availability check failed: {e}")
            return False
    
    def create_pipeline(self) -> bool:
        """Create the GStreamer pipeline for camera capture."""
        # First check if camera is available
        if not self.check_camera_availability():
            print("Camera not available - may be in use by another application")
            print("Try closing other camera applications and wait a moment")
            return False
        
        try:
            pipeline_str = CAMERA_CONFIG["pipeline_str"]
            print("Creating camera pipeline...")
            
            self.pipeline = Gst.parse_launch(pipeline_str)
            print("✓ Camera pipeline created successfully")
            
            # Get the appsink element
            self.appsink = self.pipeline.get_by_name("appsink")
            if self.appsink:
                self.appsink.connect("new-sample", self.on_new_sample)
                print("✓ Connected to appsink")
            
            # Connect to bus for messages
            bus = self.pipeline.get_bus()
            bus.add_signal_watch()
            bus.connect("message", self.on_message)
            
            return True
                
        except Exception as e:
            print(f"✗ Pipeline creation failed: {e}")
            return False
    
    def on_new_sample(self, appsink):
        """Handle new frame from camera."""
        sample = appsink.emit("pull-sample")
        if sample:
            buffer = sample.get_buffer()
            caps = sample.get_caps()
            
            # Get buffer info
            success, map_info = buffer.map(Gst.MapFlags.READ)
            if success:
                try:
                    self._process_camera_frame(map_info, caps)
                except Exception as e:
                    print(f"Error processing camera frame: {e}")
                finally:
                    buffer.unmap(map_info)
        
        return Gst.FlowReturn.OK
    
    def _process_camera_frame(self, map_info, caps):
        """Process individual camera frame."""
        # Convert to numpy array
        caps_struct = caps.get_structure(0)
        width = caps_struct.get_int("width")[1]
        height = caps_struct.get_int("height")[1]
        format_str = caps_struct.get_string("format")
        
        # Debug: Print frame info for first few frames
        self.frame_count += 1
        if self.frame_count <= 5:
            print(f"Frame {self.frame_count}: {width}x{height}, format: {format_str}")
        
        # Create numpy array from buffer
        frame_data = np.frombuffer(map_info.data, dtype=np.uint8)
        
        # Convert frame based on format
        rgb_frame = self._convert_frame_to_rgb(frame_data, width, height, format_str)
        
        if rgb_frame is not None:
            # Store frame for inference
            with self.frame_lock:
                self.current_frame = rgb_frame.copy()
            
            # Print frame processing status periodically
            if self.frame_count % 30 == 0:
                print(f"Processed {self.frame_count} camera frames")
    
    def _convert_frame_to_rgb(self, frame_data, width, height, format_str):
        """Convert frame data to RGB format based on input format."""
        try:
            if format_str == "NV12":
                return convert_nv12_to_rgb(frame_data, width, height)
            
            elif format_str == "RGB":
                return frame_data.reshape((height, width, 3))
            
            elif format_str == "BGR":
                frame = frame_data.reshape((height, width, 3))
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            else:
                print(f"Unsupported format: {format_str}")
                return None
                
        except Exception as e:
            print(f"Error converting frame format {format_str}: {e}")
            return None
    
    def on_message(self, bus, message):
        """Handle GStreamer bus messages."""
        if message.type == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"Camera Error: {err}, {debug}")
            
            # Only stop on critical errors
            if "Failed to delete stream" not in str(err) and "Failed to stop stream" not in str(err):
                print("Critical camera error - stopping")
                GLib.idle_add(self.stop)
            else:
                print("Non-critical camera error - continuing")
            
        elif message.type == Gst.MessageType.WARNING:
            warn, debug = message.parse_warning()
            print(f"Camera Warning: {warn}, {debug}")
            
        elif message.type == Gst.MessageType.EOS:
            print("Camera stream ended")
            GLib.idle_add(self.stop)
    
    def wait_for_camera_ready(self, timeout=None) -> bool:
        """Wait for camera to be ready and producing frames."""
        if timeout is None:
            timeout = CAMERA_CONFIG["camera_timeout"]
            
        print("Waiting for camera to be ready...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            with self.frame_lock:
                if self.current_frame is not None:
                    print("✓ Camera is ready and producing frames")
                    return True
            time.sleep(0.1)
        
        print("✗ Camera did not produce frames within timeout")
        return False
    
    def start_inference(self, use_delegate: bool) -> bool:
        """Start camera stream and inference."""
        # Reset stopping flag
        with self.stop_lock:
            self.is_stopping = False
        
        if not self.pipeline:
            if not self.create_pipeline():
                return False
        
        # Start pipeline
        ret = self.pipeline.set_state(Gst.State.PLAYING)
        
        if ret == Gst.StateChangeReturn.FAILURE:
            print("Failed to start camera pipeline")
            return False
            
        print("Camera pipeline started, waiting for frames...")
        
        # Wait for camera to be ready before starting inference
        if not self.wait_for_camera_ready():
            print("Failed to get camera frames, stopping pipeline")
            self.stop()
            return False
            
        print("Camera inference started")
        self.is_running = True
        
        # Start inference thread
        self.inference_thread = threading.Thread(
            target=self.inference_loop, 
            args=(use_delegate,),
            daemon=True
        )
        self.inference_thread.start()
        
        return True
    
    def inference_loop(self, use_delegate: bool):
        """Main inference loop running in separate thread."""
        inference_count = 0
        last_ui_update = 0
        last_display_update = 0
        
        # Get performance configuration
        perf_config = PERFORMANCE_CONFIG
        ui_update_interval = (perf_config["ui_update_interval_delegate"] if use_delegate 
                             else perf_config["ui_update_interval_cpu"])
        display_update_interval = perf_config["display_update_interval"]
        inference_delay = (perf_config["inference_delay_delegate"] if use_delegate 
                          else perf_config["inference_delay_cpu"])
        status_interval = (perf_config["status_interval_delegate"] if use_delegate 
                          else perf_config["status_interval_cpu"])
        
        while self.is_running:
            try:
                # Get current frame
                frame = self._get_current_frame()
                if frame is None:
                    if inference_count == 0:
                        print("Waiting for first frame...")
                    time.sleep(0.1)
                    continue
                
                inference_count += 1
                
                # Run inference on frame
                if inference_count <= 3:
                    print(f"Inference {inference_count}: Processing frame {frame.shape}")
                
                if self.model_type == "yolo":
                    # Use Smart Shop YOLO inference
                    try:
                        from smart_shop_inference import SmartShopInference
                        if not hasattr(self, '_yolo_inference'):
                            self._yolo_inference = SmartShopInference(use_delegate=use_delegate)
                        
                        start_time = time.time()
                        detections = self._yolo_inference._run_inference_on_frame(frame)
                        inference_time = time.time() - start_time
                        
                        # Convert to legacy format for compatibility
                        results = []
                        for det in detections:
                            bbox = det['bbox']
                            results.append((
                                det['class_name'], det['score'], 
                                bbox[0], bbox[1], bbox[2], bbox[3]
                            ))
                    except Exception as e:
                        print(f"YOLO inference error: {e}")
                        results, inference_time = [], 0.001
                else:
                    # Use existing inference engine
                    results, inference_time = run_inference(frame, use_delegate, is_frame=True, model_type=self.model_type)
                
                if inference_count <= 3:
                    print(f"Inference {inference_count}: Got {len(results)} results in {inference_time:.3f}s")
                
                # Calculate FPS
                fps = 1.0 / inference_time if inference_time > 0 else 0
                current_time = time.time()
                
                # Send results and display frames based on timing
                self._send_inference_results(
                    results, inference_time, fps, frame, current_time,
                    last_ui_update, last_display_update, 
                    ui_update_interval, display_update_interval
                )
                
                # Update timing
                if current_time - last_ui_update >= ui_update_interval:
                    last_ui_update = current_time
                if current_time - last_display_update >= display_update_interval:
                    last_display_update = current_time
                
                # Print status periodically
                if inference_count % status_interval == 0:
                    print(f"Completed {inference_count} inferences, FPS: {fps:.1f}")
                
                # Adaptive delay based on performance
                time.sleep(inference_delay)
                
            except Exception as e:
                print(f"Error in inference loop: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.5)
    
    def _get_current_frame(self):
        """Get current frame safely."""
        with self.frame_lock:
            if self.current_frame is not None:
                return self.current_frame.copy()
        return None
    
    def _send_inference_results(self, results, inference_time, fps, frame, current_time,
                               last_ui_update, last_display_update, 
                               ui_update_interval, display_update_interval):
        """Send inference results and display frames to callbacks."""
        
        # Send to legacy callback for backward compatibility
        if self.callback:
            # Throttle UI updates to prevent overwhelming the interface
            if current_time - last_ui_update >= ui_update_interval:
                if results:
                    GLib.idle_add(self.callback, results, inference_time, fps, None)
            
            # Send display frames at consistent rate
            if current_time - last_display_update >= display_update_interval:
                # Use consistent display resolution for both modes
                from config import DISPLAY_CONFIG
                display_size = DISPLAY_CONFIG["camera_display_size"]
                display_frame = cv2.resize(frame, display_size)
                
                # For PPE detection, draw bounding boxes on the display frame
                if self.model_type == "ppe" and results:
                    original_size = (frame.shape[1], frame.shape[0])  # (width, height)
                    display_frame = draw_ppe_detections(display_frame, results, original_size, display_size)
                
                GLib.idle_add(self.callback, None, 0, 0, display_frame)
        
        # Send to new separate callbacks
        if self.detection_callback and results:
            # Convert results to smart shop format if using YOLO
            if self.model_type == "yolo":
                # Convert to smart shop detection format
                detections = []
                for result in results:
                    if len(result) >= 6:  # (label, confidence, x1, y1, x2, y2)
                        label, confidence, x1, y1, x2, y2 = result[:6]
                        detections.append({
                            'class_name': label,
                            'score': confidence,
                            'bbox': [int(x1), int(y1), int(x2), int(y2)]
                        })
                GLib.idle_add(self.detection_callback, detections)
            else:
                GLib.idle_add(self.detection_callback, results)
        
        if self.image_callback:
            # Send camera frame for display
            if current_time - last_display_update >= display_update_interval:
                GLib.idle_add(self.image_callback, frame)
    
    def stop(self):
        """Stop camera stream and inference."""
        # Use lock to prevent multiple simultaneous stop calls
        with self.stop_lock:
            if self.is_stopping or not self.is_running:
                print("Stop already in progress or camera not running")
                return
            
            print("Stopping camera inference...")
            self.is_stopping = True
            self.is_running = False
        
        # Stop inference thread first
        if self.inference_thread and self.inference_thread.is_alive():
            print("Waiting for inference thread to stop...")
            self.inference_thread.join(timeout=3.0)
            if self.inference_thread.is_alive():
                print("Warning: Inference thread did not stop cleanly")
        
        # Stop pipeline
        self._stop_pipeline()
        
        # Clear current frame
        with self.frame_lock:
            self.current_frame = None
            
        # Reset frame counter
        self.frame_count = 0
        
        # Reset flags
        with self.stop_lock:
            self.is_stopping = False
        
        print("✓ Camera stop complete")
    
    def _stop_pipeline(self):
        """Stop the GStreamer pipeline safely."""
        if not self.pipeline:
            return
            
        print("Stopping GStreamer pipeline...")
        try:
            # Get current state
            state_ret, current_state, pending_state = self.pipeline.get_state(
                timeout=1000000000  # 1 second
            )
            
            if current_state != Gst.State.NULL:
                # First pause, then stop to ensure clean shutdown
                self.pipeline.set_state(Gst.State.PAUSED)
                
                # Wait for state change
                state_ret, state, pending = self.pipeline.get_state(
                    timeout=2000000000  # 2 seconds
                )
                
                if state_ret in [Gst.StateChangeReturn.SUCCESS, Gst.StateChangeReturn.ASYNC]:
                    # Now set to NULL
                    self.pipeline.set_state(Gst.State.NULL)
                    
                    # Wait for final state change
                    state_ret, state, pending = self.pipeline.get_state(
                        timeout=3000000000  # 3 seconds
                    )
                    
                    if state_ret == Gst.StateChangeReturn.SUCCESS:
                        print("✓ Camera pipeline stopped cleanly")
                    else:
                        print("Warning: Pipeline did not reach NULL state cleanly")
                else:
                    print("Warning: Pipeline did not pause cleanly, forcing to NULL")
                    self.pipeline.set_state(Gst.State.NULL)
            else:
                print("Pipeline already in NULL state")
                
        except Exception as e:
            print(f"Error stopping pipeline: {e}")
            # Force pipeline to NULL state
            try:
                self.pipeline.set_state(Gst.State.NULL)
            except:
                pass
    
    def cleanup_pipeline(self):
        """Completely clean up and destroy the pipeline."""
        if self.pipeline:
            print("Cleaning up pipeline...")
            
            # Remove bus watch
            bus = self.pipeline.get_bus()
            if bus:
                bus.remove_signal_watch()
            
            # Ensure pipeline is in NULL state and wait
            self.pipeline.set_state(Gst.State.NULL)
            state_ret, state, pending = self.pipeline.get_state(
                timeout=CAMERA_CONFIG["cleanup_timeout"] * 1000000000  # Convert to nanoseconds
            )
            
            if state_ret == Gst.StateChangeReturn.SUCCESS:
                print("✓ Pipeline reached NULL state")
            else:
                print("Warning: Pipeline may not have reached NULL state cleanly")
            
            # Clear references
            self.pipeline = None
            self.appsink = None
            
            # Force garbage collection to release resources
            import gc
            gc.collect()
            
            print("✓ Pipeline cleaned up")
            
            # Add a small delay to ensure camera resources are fully released
            time.sleep(0.5)
