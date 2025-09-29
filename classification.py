#!/usr/bin/env python3
# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
Minimal RB3 inference demo for Qualcomm RB3 platform.
Demonstrates GoogleNet image classification on image files and live camera feed.
"""

import argparse
import sys
import time
import signal
import threading
import cv2
import numpy as np
from typing import Optional, List, Tuple

# Import existing inference components
from common.inference_engine import run_inference
from common.camera_interface import CameraInference
from common.config import DEVICE_OS
from common.image_utils import create_pixbuf_from_frame

 # Attempt to import GTK for optional camera display
try:
    import gi
    gi.require_version('Gtk', '3.0')
    gi.require_version('GdkPixbuf', '2.0')
    from gi.repository import Gtk, GLib, GdkPixbuf
    GTK_AVAILABLE = True
except ImportError:
    GTK_AVAILABLE = False


class MinimalDemo:
    """
    Minimal demo class for RB3 inference.
    Provides image classification using GoogleNet on static images or live camera feed.
    """
    
    def __init__(self, use_delegate: bool = True, headless: bool = False):
        """
        Initialize the demo.
        Args:
            use_delegate: Enable hardware acceleration if True.
            headless: Run without GUI display if True.
        """
        self.use_delegate = use_delegate
        self.headless = headless
        self.camera_inference = None
        self.running = False
        self.classification_count = 0
        self.last_display_time = 0
        # More frequent updates in headless mode for better terminal experience
        self.display_interval = 0.3 if headless else 0.5  # 300ms for headless, 500ms for GUI
        
        # GTK window for camera display (if not headless)
        self.window = None
        self.image_widget = None
        
        self.setup_signal_handlers()
    
    def setup_signal_handlers(self):
        """
        Setup signal handlers for graceful shutdown.
        """
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """
        Handle shutdown signals for graceful exit.
        """
        print("\nShutting down gracefully...")
        self.stop()
        if self.window and GTK_AVAILABLE:
            Gtk.main_quit()
        sys.exit(0)
    
    def classify_image(self, image_path: str) -> None:
        """
        Classify a single image using GoogleNet.
        Args:
            image_path: Path to the image file.
        """
        print(f"Classifying image: {image_path}")
        print(f"Device: {DEVICE_OS}")
        print(f"Mode: {'HTP Hardware Acceleration' if self.use_delegate else 'CPU Mode'}")
        print("-" * 50)
        
        try:
            # Run inference
            results, inference_time = run_inference(
                image_input=image_path,
                use_delegate=self.use_delegate,
                is_frame=False,
                model_type="classification"
            )
            
            # Display results
            print(f"Inference time: {inference_time*1000:.1f}ms")
            print("\nTop predictions:")
            for i, (label, confidence) in enumerate(results[:5], 1):
                print(f"{i:2d}. {label:<20} {confidence:6.2f}%")
                
        except Exception as e:
            print(f"Error during inference: {e}")
    
    def run_live_camera(self) -> None:
        """
        Run live camera classification using GoogleNet.
        """
        print("Starting live camera classification...")
        print(f"Device: {DEVICE_OS}")
        print(f"Mode: {'HTP Hardware Acceleration' if self.use_delegate else 'CPU Mode'}")
        print(f"Display: {'Headless' if self.headless else 'With Camera Preview'}")
        print("Press Ctrl+C to stop")
        print("-" * 50)
        
        # Initialize GTK if not headless and GTK is available
        if not self.headless and GTK_AVAILABLE:
            self._setup_camera_display()
        elif not self.headless and not GTK_AVAILABLE:
            print("Warning: GTK not available, running in headless mode")
            self.headless = True
        
        try:
            # Create camera inference
            self.camera_inference = CameraInference(
                model_type="classification",
                use_delegate=self.use_delegate,
                detection_callback=self._on_classification_results,
                image_callback=self._on_camera_frame if not self.headless else None
            )
            
            # In headless mode, directly call the callback since GLib.idle_add requires GTK main loop
            if self.headless:
                # Override the detection callback to call directly
                original_send_method = self.camera_inference._send_inference_results
                def send_results_headless(results, inference_time, fps, frame, current_time,
                                       last_ui_update, last_display_update, 
                                       ui_update_interval, display_update_interval):
                    # Call original method
                    original_send_method(results, inference_time, fps, frame, current_time,
                                       last_ui_update, last_display_update, 
                                       ui_update_interval, display_update_interval)
                    
                    # Also call detection callback directly for headless mode
                    if self.camera_inference.detection_callback and results:
                        if self.camera_inference.model_type == "yolo":
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
                            self.camera_inference.detection_callback(detections)
                        else:
                            self.camera_inference.detection_callback(results)
                
                # Replace the method
                self.camera_inference._send_inference_results = send_results_headless
            
            # Start camera
            success = self.camera_inference.start_inference(self.use_delegate)
            if not success:
                print("Failed to start camera")
                return
            
            self.running = True
            print("Camera started successfully. Classifying live feed...")
            if self.headless:
                print("\nRunning in HEADLESS mode - predictions will appear below:")
            print("\nLive Classification Results:")
            print("=" * 60)
            
            # Run GTK main loop if display is enabled
            if not self.headless and GTK_AVAILABLE and self.window:
                Gtk.main()
            else:
                # Keep running until stopped (headless mode)
                print("Waiting for camera frames...")
                while self.running:
                    time.sleep(0.1)
                
        except Exception as e:
            print(f"Error during live camera: {e}")
        finally:
            self.stop()
    
    def _setup_camera_display(self) -> None:
        """
        Setup GTK window for camera display.
        """
        if not GTK_AVAILABLE:
            return
            
        Gtk.init(None)
        
        # Create main window
        self.window = Gtk.Window()
        self.window.set_title("RB3 Live Classification")
        self.window.set_default_size(640, 480)
        self.window.connect("destroy", self._on_window_destroy)
        
        # Create image widget
        self.image_widget = Gtk.Image()
        self.image_widget.set_size_request(640, 480)
        
        # Add to window
        self.window.add(self.image_widget)
        self.window.show_all()
    
    def _on_window_destroy(self, widget):
        """
        Handle window close event.
        """
        self.stop()
        Gtk.main_quit()
    
    def _on_camera_frame(self, frame):
        """
        Handle new camera frame for display.
        Args:
            frame: Camera frame as numpy array.
        """
        if self.headless or not self.image_widget:
            return
            
        try:
            # Convert frame to pixbuf and display
            pixbuf = create_pixbuf_from_frame(frame)
            if pixbuf:
                # Scale to fit window
                scaled_pixbuf = pixbuf.scale_simple(
                    640, 480, GdkPixbuf.InterpType.BILINEAR
                )
                # Update image widget on main thread
                GLib.idle_add(self.image_widget.set_from_pixbuf, scaled_pixbuf)
        except Exception as e:
            print(f"Error displaying frame: {e}")
    
    def _on_classification_results(self, results: List[Tuple[str, float]]) -> None:
        """
        Handle classification results from live camera.
        Args:
            results: List of (label, confidence) tuples.
        """
        current_time = time.time()
        
        # Throttle display updates
        if current_time - self.last_display_time < self.display_interval:
            return
            
        self.last_display_time = current_time
        self.classification_count += 1
        
        if results:
            # Clear screen for both headless and non-headless modes for clean display
            print("\033[2J\033[H", end="")  # Clear screen and move cursor to top
            
            # Show frame counter and top 5 predictions
            print(f"Frame #{self.classification_count:04d} - Top 5 Predictions:")
            print("=" * 60)
            
            for i, (label, confidence) in enumerate(results[:5], 1):
                # Create visual confidence bar
                confidence_bar = "█" * int(confidence / 10) + "░" * (10 - int(confidence / 10))
                print(f"  {i}. {label:<30} {confidence:6.2f}% [{confidence_bar}]")
            
            print("=" * 60)
            print(f"Device: {DEVICE_OS} | Mode: {'HTP' if self.use_delegate else 'CPU'}")
            print(f"Display: {'Headless' if self.headless else 'With Camera Preview'}")
            print("Press Ctrl+C to stop")
            
            # Flush output to ensure immediate display
            sys.stdout.flush()
        else:
            # Clear screen and show no results message
            print("\033[2J\033[H", end="")  # Clear screen and move cursor to top
            print(f"Frame #{self.classification_count:04d} - No classification results")
            print("=" * 60)
            print(f"Device: {DEVICE_OS} | Mode: {'HTP' if self.use_delegate else 'CPU'}")
            print(f"Display: {'Headless' if self.headless else 'With Camera Preview'}")
            print("Press Ctrl+C to stop")
            sys.stdout.flush()
    
    def stop(self) -> None:
        """
        Stop the demo and release resources.
        """
        self.running = False
        if self.camera_inference:
            self.camera_inference.stop()
            self.camera_inference = None


def main():
    """
    Main function with argument parsing.
    Parses command-line arguments and runs the appropriate demo mode.
    """
    parser = argparse.ArgumentParser(
        description="Minimal RB3 inference demo for GoogleNet image classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python minimal.py --image dog.jpg --htp
  python minimal.py --live-camera --cpu
  python minimal.py --live-camera --htp --headless
  python minimal.py --image bear.jpg --cpu
        """
    )
    
    # Input mode (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--image", 
        type=str, 
        help="Path to image file for classification"
    )
    input_group.add_argument(
        "--live-camera", 
        action="store_true", 
        help="Use live camera for classification"
    )
    
    # Processing mode (mutually exclusive)
    proc_group = parser.add_mutually_exclusive_group(required=True)
    proc_group.add_argument(
        "--htp", 
        action="store_true", 
        help="Use HTP hardware acceleration"
    )
    proc_group.add_argument(
        "--cpu", 
        action="store_true", 
        help="Use CPU-only processing"
    )
    
    # Display options
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode (no camera preview window)"
    )
    
    args = parser.parse_args()
    
    # Determine processing mode
    use_delegate = args.htp
    
    # Create demo instance
    demo = MinimalDemo(use_delegate=use_delegate, headless=args.headless)
    
    try:
        if args.image:
            # Image classification mode
            demo.classify_image(args.image)
        elif args.live_camera:
            # Live camera mode
            demo.run_live_camera()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()