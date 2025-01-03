import argparse
from dataclasses import dataclass
from pathlib import Path
import platform
import time
import subprocess
from datetime import datetime
import cv2
import numpy as np
import sys
import logging
import shutil

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('progress_photos.log')
    ]
)

@dataclass
class Resolution:
    width: int
    height: int

def play_sound(sound_name="Ping"):
    """Play a system sound"""
    try:
        subprocess.run(["afplay", f"/System/Library/Sounds/{sound_name}.aiff"])
    except Exception as e:
        logging.error(f"Failed to play sound: {e}")

def verify_camera(device_name):
    """Verify if specified camera is available"""
    try:
        result = subprocess.run(
            ["imagesnap", "-l"],
            capture_output=True,
            text=True,
            check=True
        )
        if device_name not in result.stdout:
            logging.error(f"Camera '{device_name}' not found! Available cameras:")
            logging.info(result.stdout)
            return False
        return True
    except subprocess.CalledProcessError:
        logging.error("Failed to check camera status")
        return False

def create_progress_bar():
    """Create a compact progress bar"""
    # Base dimensions
    bar_width = 400
    bar_height = 100
    
    # Create base image with dark background
    bar_img = np.zeros((bar_height, bar_width, 3), dtype=np.uint8)
    bar_img[:] = (26, 20, 20)  # Dark background in BGR
    
    # Colors
    title_color = (255, 255, 255)  # White
    bar_bg_color = (40, 30, 30)    # Slightly lighter than background
    progress_color = (255, 165, 0)  # Blue in BGR
    
    # Title
    font = cv2.FONT_HERSHEY_SIMPLEX
    year_text = "2025"
    font_scale = 1.0
    thickness = 2
    text_size = cv2.getTextSize(year_text, font, font_scale, thickness)[0]
    text_x = (bar_width - text_size[0]) // 2
    text_y = 30
    cv2.putText(bar_img, year_text, (text_x, text_y), font, font_scale, title_color, thickness)

    # Calculate progress
    today = datetime.now()
    year_start = datetime(today.year, 1, 1)
    year_end = datetime(today.year, 12, 31)
    days_in_year = (year_end - year_start).days + 1
    days_completed = (today - year_start).days + 1
    year_percentage = (days_completed / days_in_year) * 100

    # Progress bar
    bar_height = 20
    bar_y = 45
    
    # Background bar
    cv2.rectangle(bar_img,
                 (20, bar_y),
                 (bar_width - 20, bar_y + bar_height),
                 bar_bg_color,
                 -1)
    
    # Progress fill
    progress_width = int((bar_width - 40) * (year_percentage / 100))
    if progress_width > 0:
        cv2.rectangle(bar_img,
                     (20, bar_y),
                     (20 + progress_width, bar_y + bar_height),
                     progress_color,
                     -1)
    
    # Percentage
    percentage_text = f"{year_percentage:.1f}%"
    text_size = cv2.getTextSize(percentage_text, font, font_scale, thickness)[0]
    text_x = (bar_width - text_size[0]) // 2
    text_y = bar_y + bar_height + 30
    cv2.putText(bar_img, percentage_text, (text_x, text_y), font, font_scale, title_color, thickness)
    
    return bar_img

def add_date_and_progress_overlay(frame):
    """Add date overlay and progress bar with compact boxes"""
    height, width = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    font_thickness = 2
    color = (255, 255, 255)  # White text
    padding = 30
    
    # Add date with dark background
    now = datetime.now()
    day = now.day
    suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(day % 20, 'th')
    if 11 <= day <= 13:
        suffix = 'th'
    date_text = now.strftime(f"%-d{suffix} %b %Y %-I:%M %p")
    
    # Date text measurements and position
    (text_width, text_height), _ = cv2.getTextSize(date_text, font, font_scale, font_thickness)
    date_x = width - text_width - padding - 20
    date_y = height - 200  # Position for date
    
    # Create compact dark background for date
    date_bg_height = text_height + 20
    date_bg_width = text_width + 40
    date_bg_x = width - date_bg_width - padding
    date_bg_y = date_y - text_height - 10
    
    date_overlay = frame.copy()
    cv2.rectangle(date_overlay,
                 (date_bg_x, date_bg_y),
                 (date_bg_x + date_bg_width, date_bg_y + date_bg_height),
                 (26, 20, 20),
                 -1)
    frame = cv2.addWeighted(date_overlay, 0.85, frame, 0.15, 0)
    
    # Add date text
    cv2.putText(frame, date_text, (date_x, date_y), font, font_scale, color, font_thickness)
    
    # Add progress bar
    progress_bar = create_progress_bar()
    bar_height, bar_width = progress_bar.shape[:2]
    x_offset = width - bar_width - padding
    y_offset = height - bar_height - padding
    
    # Create compact background for progress bar
    progress_bg_height = bar_height + 20
    progress_bg_width = bar_width + 20
    progress_bg_x = x_offset - 10
    progress_bg_y = y_offset - 10
    
    progress_overlay = frame.copy()
    cv2.rectangle(progress_overlay,
                 (progress_bg_x, progress_bg_y),
                 (progress_bg_x + progress_bg_width, progress_bg_y + progress_bg_height),
                 (26, 20, 20),
                 -1)
    frame = cv2.addWeighted(progress_overlay, 0.85, frame, 0.15, 0)
    
    # Add progress bar over background
    roi = frame[y_offset:y_offset+bar_height, x_offset:x_offset+bar_width]
    mask = np.any(progress_bar != [26, 20, 20], axis=2)
    roi[mask] = progress_bar[mask]
    
    return frame

def get_next_frame_number(output_base_dir: Path) -> int:
    """Find the highest frame number across all pose directories and return next number"""
    highest_frame = 0  # Start at 0 so first frame will be 0001
    for pattern in ["*clothed", "*shirtless"]:
        for pose_dir in output_base_dir.glob(pattern):
            if pose_dir.is_dir():
                frames = list(pose_dir.glob("frame_*.jpg"))
                for frame in frames:
                    try:
                        # Extract number from frame_XXXX.jpg format
                        frame_num = int(frame.stem.split('_')[1])
                        highest_frame = max(highest_frame, frame_num)
                    except (IndexError, ValueError):
                        continue
    
    next_frame = highest_frame + 1
    if next_frame >= 9999:
        raise Exception("Maximum frame limit (9999) reached!")
        
    return next_frame

def show_year_progress():
    """Display year progress based on current date"""
    today = datetime.now()
    year_start = datetime(today.year, 1, 1)
    year_end = datetime(today.year, 12, 31)
    
    days_in_year = (year_end - year_start).days + 1
    days_completed = (today - year_start).days + 1
    days_remaining = days_in_year - days_completed
    year_percentage = (days_completed / days_in_year) * 100
    
    print(f"\nYear Progress for {today.year}:")
    print(f"Day {days_completed}/{days_in_year} ({year_percentage:.1f}% completed)")
    print(f"Days remaining: {days_remaining}")

def list_cameras():
    """List available camera devices"""
    try:
        result = subprocess.run(["imagesnap", "-l"], capture_output=True, text=True, check=True)
        print("\nAvailable cameras:")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to list cameras: {e}")
        sys.exit(1)

def capture_progress_photos(output_base_dir: Path, device_name: str, resolution: Resolution = None):
    """Capture progress photos for different poses"""
    pose_sequence = [
        ("with_clothes", [
            ("front_clothed", "Front view with clothes"),
            ("back_clothed", "Back view with clothes")
        ]),
        ("shirtless", [
            ("front_shirtless", "Front view shirtless"),
            ("back_shirtless", "Back view shirtless")
        ])
    ]

    # Create directories
    output_base_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = output_base_dir / "temp"
    temp_dir.mkdir(exist_ok=True)
    
    pose_dirs = {}
    for _, poses in pose_sequence:
        for pose_name, _ in poses:
            pose_dir = output_base_dir / pose_name
            pose_dir.mkdir(parents=True, exist_ok=True)
            pose_dirs[pose_name] = pose_dir

    try:
        # Verify camera at start
        if not verify_camera(device_name):
            raise Exception("Camera not accessible")

        frame_number = get_next_frame_number(output_base_dir)
        if frame_number >= 9999:
            raise Exception("Maximum frame limit (9999) reached!")
            
        print(f"\nStarting session with frame number: {frame_number:04d}")
        show_year_progress()

        for section_name, poses in pose_sequence:
            print(f"\n=== Starting {section_name} poses ===")
            
            for i, (pose_name, pose_description) in enumerate(poses):
                print(f"\nPreparing to capture: {pose_description}")
                input("Press Enter when ready...")

                # Start camera capture process with 5-second warmup
                temp_file = temp_dir / "temp_frame.jpg"
                capture_process = subprocess.Popen([
                    "imagesnap",
                    "-d", device_name,
                    "-w", "5.0",  # 5 second warmup
                    str(temp_file)
                ])

                # Countdown with beeps
                for i in range(5, 0, -1):
                    print(f"{i}...")
                    play_sound("Ping")
                    time.sleep(1)

                # Wait for capture to complete
                capture_process.wait()

                # Process the captured frame
                frame = cv2.imread(str(temp_file))
                if frame is None:
                    print(f"Error capturing {pose_description}")
                    continue

                # Add date overlay and progress bar
                frame = add_date_and_progress_overlay(frame)
                
                # Save with proper frame number format (always 4 digits)
                filename = pose_dirs[pose_name] / f"frame_{frame_number:04d}.jpg"
                cv2.imwrite(str(filename), frame)
                print(f"Captured {pose_description} as frame_{frame_number:04d}.jpg")
                
                # Clean up temp file
                temp_file.unlink(missing_ok=True)

                # Delay between front and back poses
                if i == 0:
                    print("\nTaking a 30-second break before next pose...")
                    play_sound("Submarine")
                    time.sleep(30)

            # Prompt for shirt removal between sections
            if section_name == "with_clothes":
                print("\n=== Ready for shirtless poses ===")
                play_sound("Glass")
                input("\nPlease remove shirt and press Enter when ready...")

    except KeyboardInterrupt:
        print("\nCapture interrupted by user")
    finally:
        # Clean up temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)
        print(f"\nAll photos captured successfully with frame number {frame_number:04d}!")
        play_sound("Glass")

def main():
    parser = argparse.ArgumentParser(
        description="Capture progress photos organized for future timelapse creation"
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="progress_photos",
        help="Base directory for saving photos"
    )
    parser.add_argument(
        "--device",
        "-d",
        help="Camera device name (e.g., 'iPhone Camera')"
    )
    parser.add_argument(
        "--list-cameras",
        "-l",
        action="store_true",
        help="List available cameras and exit"
    )
    parser.add_argument(
        "--width",
        type=int,
        help="Custom width for capture"
    )
    parser.add_argument(
        "--height",
        type=int,
        help="Custom height for capture"
    )

    args = parser.parse_args()
    
    if args.list_cameras:
        list_cameras()
        return

    if not args.device:
        list_cameras()
        sys.exit("Please specify a camera device using --device")

    resolution = None
    if args.width and args.height:
        resolution = Resolution(args.width, args.height)

    output_dir = Path(args.output_dir)
    capture_progress_photos(output_dir, args.device, resolution)

if __name__ == "__main__":
    main()