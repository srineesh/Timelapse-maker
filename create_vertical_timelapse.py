import os
import subprocess
import re

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]

def create_vertical_timelapse(image_folder, output_video, width=1080, height=1920):
    """
    Creates a vertical timelapse video suitable for Instagram Reels.
    Default resolution is 1080x1920 (9:16 aspect ratio).
    """
    # Get all jpg files and sort them naturally
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
    images.sort(key=natural_sort_key)

    # Create a text file with the correct order of frames
    with open('frames.txt', 'w') as f:
        for img in images:
            f.write(f"file '{os.path.join(image_folder, img)}'\n")

    # FFmpeg command to create vertical video
    # -vf "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black"
    # This will:
    # 1. Scale the video to fit within 1080x1920 while maintaining aspect ratio
    # 2. Pad the remaining space with black bars to achieve 9:16 ratio
    subprocess.run([
        "ffmpeg",
        "-f", "concat",
        "-safe", "0",
        "-i", "frames.txt",
        "-framerate", "30",
        "-vf", f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "ultrafast",
        "-metadata:s:v:0", "rotate=0",  # Ensure correct orientation
        output_video
    ])

    # Clean up the temporary file
    os.remove('frames.txt')
    print(f"Vertical timelapse video saved as {output_video}")

if __name__ == "__main__":
    # Default to Instagram Reels dimensions (9:16 aspect ratio)
    create_vertical_timelapse(
        "./timelapse_images", 
        "vertical_timelapse.mp4",
        width=1080,
        height=1920
    )