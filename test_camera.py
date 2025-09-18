#!/usr/bin/env python3

import gi
import sys
import signal

# Import GStreamer
gi.require_version('Gst', '1.0')
gi.require_version('GstVideo', '1.0')
from gi.repository import Gst, GLib, GstVideo

class RB3Camera:
    def __init__(self):
        # Initialize GStreamer
        Gst.init(None)
        
        # Create the pipeline
        self.pipeline = None
        self.loop = None
        
    def create_pipeline(self):
        """Create the GStreamer pipeline for RB3 Gen2 camera"""
        
        # Create pipeline string - matches your working gst-launch command
        pipeline_str = (
            "qtiqmmfsrc camera=0 name=camsrc "
            "video_0::type=preview ! "
            "video/x-raw(memory:GBM),format=NV12,width=1280,height=720,framerate=30/1,compression=ubwc ! "
            "waylandsink fullscreen=true async=true sync=false"
        )
        
        try:
            # Parse the pipeline
            self.pipeline = Gst.parse_launch(pipeline_str)
            print("Pipeline created successfully")
            
            # Connect to bus for messages
            bus = self.pipeline.get_bus()
            bus.add_signal_watch()
            bus.connect("message", self.on_message)
            
        except Exception as e:
            print(f"Error creating pipeline: {e}")
            return False
            
        return True
    
    def on_message(self, bus, message):
        """Handle GStreamer bus messages"""
        
        if message.type == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"Error: {err}, {debug}")
            self.stop()
            
        elif message.type == Gst.MessageType.WARNING:
            warn, debug = message.parse_warning()
            print(f"Warning: {warn}, {debug}")
            
        elif message.type == Gst.MessageType.EOS:
            print("End of stream")
            self.stop()
            
        elif message.type == Gst.MessageType.STATE_CHANGED:
            if message.src == self.pipeline:
                old_state, new_state, pending = message.parse_state_changed()
                print(f"Pipeline state changed from {old_state.value_name} to {new_state.value_name}")
    
    def start(self):
        """Start the camera stream"""
        
        if not self.pipeline:
            if not self.create_pipeline():
                return False
        
        # Start playing
        ret = self.pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            print("Failed to start pipeline")
            return False
            
        print("Camera stream started. Press Ctrl+C to stop.")
        
        # Create main loop
        self.loop = GLib.MainLoop()
        
        try:
            self.loop.run()
        except KeyboardInterrupt:
            print("\nStopping camera...")
            self.stop()
            
        return True
    
    def stop(self):
        """Stop the camera stream"""
        
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)
            print("Pipeline stopped")
            
        if self.loop and self.loop.is_running():
            self.loop.quit()

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print("\nReceived interrupt signal, stopping...")
    if 'camera' in globals():
        camera.stop()
    sys.exit(0)

def main():
    """Main function"""
    
    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    print("RB3 Gen2 Camera Stream")
    print("=====================")
    
    # Create camera instance
    global camera
    camera = RB3Camera()
    
    # Start the camera
    success = camera.start()
    
    if not success:
        print("Failed to start camera")
        sys.exit(1)

if __name__ == "__main__":
    main()
