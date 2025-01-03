import argparse
from pathlib import Path
import time
import subprocess
import sys
import signal
import logging
from datetime import datetime, timedelta
import shutil
import json
import gc

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('timelapse.log')
    ]
)

class TimelapseCapture:
    def __init__(self, duration, interval, device_name):
        self.duration = duration
        self.interval = interval
        self.device_name = device_name
        self.running = True
        self.current_frame = 1
        self.num_frames = int(duration // interval)
        
        # Create timestamped directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = Path(f"timelapse_{timestamp}")
        self.temp_dir = self.output_dir / "temp_shots"
        
        # State management
        self.state_file = Path("timelapse_state.json")
        self.total_pause_duration = 0
        self.pause_start_time = None
        self.failed_frames = 0
        self.last_camera_check = 0
        self.camera_check_interval = 300  # Check camera every 5 minutes
        
        # Create directories
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.temp_dir.mkdir(exist_ok=True, parents=True)
        
        # Save capture settings
        self.save_capture_info()
        
        # Initialize state file
        self.update_state({"paused": False})
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self.handle_shutdown)
        signal.signal(signal.SIGTERM, self.handle_shutdown)

    def save_capture_info(self):
        """Save capture settings for reference"""
        info = {
            "start_time": datetime.now().isoformat(),
            "duration_hours": self.duration / 3600,
            "interval_seconds": self.interval,
            "total_frames": self.num_frames,
            "device": self.device_name
        }
        info_file = self.output_dir / "capture_info.json"
        with info_file.open('w') as f:
            json.dump(info, f, indent=2)

    def update_state(self, state):
        """Update the state file with current status"""
        try:
            with self.state_file.open('w') as f:
                json.dump(state, f)
        except Exception as e:
            logging.error(f"Error updating state file: {e}")

    def read_state(self):
        """Read current state from file"""
        try:
            if self.state_file.exists():
                with self.state_file.open('r') as f:
                    return json.load(f)
            return {"paused": False}
        except Exception as e:
            logging.error(f"Error reading state file: {e}")
            return {"paused": False}

    def handle_shutdown(self, signum, frame):
        """Handle graceful shutdown on SIGINT or SIGTERM"""
        logging.info("Shutdown signal received. Cleaning up...")
        self.running = False
        if self.state_file.exists():
            self.state_file.unlink()

    def cleanup_temp_dir(self):
        """Remove temporary directory and its contents"""
        try:
            shutil.rmtree(self.temp_dir)
            logging.info("Cleaned up temporary files")
        except Exception as e:
            logging.error(f"Error cleaning up temporary files: {e}")

    def verify_camera(self):
        """Verify camera is still accessible"""
        try:
            result = subprocess.run(
                ["imagesnap", "-l"],
                capture_output=True,
                text=True,
                check=True
            )
            if self.device_name not in result.stdout:
                logging.error("Camera not found! Please check connection.")
                return False
            return True
        except subprocess.CalledProcessError:
            logging.error("Failed to check camera status")
            return False

    def initialize_camera(self):
        """Take initial test shots to stabilize camera"""
        logging.info("Initializing camera with test shots...")
        if not self.verify_camera():
            raise Exception("Camera not accessible")
            
        for i in range(3):
            try:
                subprocess.run([
                    "imagesnap",
                    "-d", self.device_name,
                    "-w", "7.0",
                    str(self.temp_dir / f"test_frame_{i}.jpg")
                ], check=True, capture_output=True)
                time.sleep(3)
            except subprocess.CalledProcessError as e:
                logging.error(f"Failed to initialize camera: {e}")
                raise

    def capture_frame(self, frame_number):
        """Capture a single frame with error handling"""
        current_time = time.time()
        
        # Periodically verify camera is still accessible
        if current_time - self.last_camera_check >= self.camera_check_interval:
            if not self.verify_camera():
                return False
            self.last_camera_check = current_time
        
        try:
            warmup_file = self.temp_dir / f"warmup_{frame_number}.jpg"
            subprocess.run([
                "imagesnap",
                "-d", self.device_name,
                "-w", "5.0",
                str(warmup_file)
            ], check=True, capture_output=True)
            
            time.sleep(4)
            
            filename = self.output_dir / f"frame_{frame_number:04d}.jpg"
            result = subprocess.run([
                "imagesnap",
                "-d", self.device_name,
                "-w", "3.0",
                str(filename)
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                self.failed_frames += 1
                logging.warning(f"Frame {frame_number} capture failed: {result.stderr}")
                return False
            
            if not filename.exists() or filename.stat().st_size == 0:
                self.failed_frames += 1
                logging.warning(f"Frame {frame_number} may be corrupt or empty")
                return False
                
            warmup_file.unlink(missing_ok=True)
            
            logging.info(f"Captured frame {frame_number}/{self.num_frames}")
            return True
            
        except subprocess.CalledProcessError as e:
            self.failed_frames += 1
            logging.error(f"Error capturing frame: {e}")
            return False

    def format_time(self, seconds):
        """Format time in HH:MM:SS"""
        return str(timedelta(seconds=int(seconds)))

    def run_timelapse(self):
        """Run the main timelapse capture loop"""
        logging.info(f"\nStarting timelapse capture:")
        logging.info(f"Camera: {self.device_name}")
        logging.info(f"Duration: {self.duration/3600:.1f} hours")
        logging.info(f"Interval: {self.interval} seconds")
        logging.info(f"Total frames: {self.num_frames}")
        logging.info(f"Output directory: {self.output_dir}")
        logging.info(f"State file: {self.state_file}")
        
        try:
            self.initialize_camera()
            
            start_time = time.time()
            last_progress_log = start_time
            
            while self.running and self.current_frame <= self.num_frames:
                state = self.read_state()
                
                if state.get("paused", False):
                    if self.pause_start_time is None:
                        self.pause_start_time = time.time()
                        logging.info("Capture paused")
                    time.sleep(0.1)
                    continue
                elif self.pause_start_time is not None:
                    pause_duration = time.time() - self.pause_start_time
                    self.total_pause_duration += pause_duration
                    logging.info(f"Capture resumed. Was paused for {self.format_time(pause_duration)}")
                    self.pause_start_time = None
                    
                frame_start = time.time()
                
                success = self.capture_frame(self.current_frame)
                if success:
                    self.current_frame += 1
                
                if self.running and self.current_frame <= self.num_frames:
                    elapsed = time.time() - frame_start
                    wait_time = max(0, self.interval - elapsed)
                    time.sleep(wait_time)
                
                # Log progress every 15 minutes
                current_time = time.time()
                active_time = current_time - start_time - self.total_pause_duration
                if current_time - last_progress_log >= 900:  # 15 minutes
                    elapsed_time = self.format_time(active_time)
                    remaining_time = self.format_time(self.duration - active_time)
                    success_rate = ((self.current_frame - self.failed_frames) / self.current_frame) * 100
                    
                    logging.info(
                        f"Progress: {elapsed_time} completed, {remaining_time} remaining\n"
                        f"Frames: {self.current_frame}/{self.num_frames} "
                        f"(Success rate: {success_rate:.1f}%)"
                    )
                    
                    last_progress_log = current_time
                    gc.collect()  # Clean up memory periodically
                    
        except Exception as e:
            logging.error(f"Unexpected error during capture: {e}")
        finally:
            if self.state_file.exists():
                self.state_file.unlink()
            self.cleanup_temp_dir()
            
            # Log final statistics
            success_rate = ((self.current_frame - self.failed_frames) / self.current_frame) * 100
            logging.info(f"\nCapture completed:")
            logging.info(f"Total frames captured: {self.current_frame-1}/{self.num_frames}")
            logging.info(f"Failed frames: {self.failed_frames}")
            logging.info(f"Success rate: {success_rate:.1f}%")
            logging.info(f"Total pause time: {self.format_time(self.total_pause_duration)}")
            logging.info(f"Output directory: {self.output_dir}")

def list_cameras():
    """List available camera devices"""
    try:
        subprocess.run(["imagesnap", "-l"], check=True)
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to list cameras: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="Capture timelapse images using specified camera"
    )
    parser.add_argument(
        "--hours",
        "-H",
        type=float,
        required=True,
        help="Duration of timelapse in hours"
    )
    parser.add_argument(
        "--interval",
        "-i",
        type=int,
        required=True,
        help="Interval between frames in seconds"
    )
    parser.add_argument(
        "--device",
        "-d",
        required=True,
        help="Camera device name (e.g., 'iPhone (3) Camera')"
    )
    parser.add_argument(
        "--list-cameras",
        "-l",
        action="store_true",
        help="List available cameras and exit"
    )
    
    args = parser.parse_args()
    
    if args.list_cameras:
        list_cameras()
        return
        
    duration = 3600 * args.hours
    
    timelapse = TimelapseCapture(
        duration=duration,
        interval=args.interval,
        device_name=args.device
    )
    
    timelapse.run_timelapse()

if __name__ == "__main__":
    main()