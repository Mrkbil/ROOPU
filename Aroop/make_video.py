import os
import subprocess
from typing import List

def run_ffmpeg(args: List[str]) -> bool:
    commands = ['ffmpeg', '-hide_banner', '-loglevel', 'info']
    commands.extend(args)
    print(commands)
    # try:
    #     subprocess.check_output(commands, stderr=subprocess.STDOUT)
    #     return True
    # except Exception:
    #     pass
    # return False

def create_video(fps: float = 25.0) -> bool:
    temp_output_path = 'output/tmp.mp4'
    temp_directory_path = 'output/'
    output_video_quality = (20 + 1) * 51 // 100
    commands = ['-hwaccel', 'auto', '-r', str(fps), '-i', os.path.join(temp_directory_path, '%04d.' + 'png'), '-c:v', 'libx264']
    # if roop.globals.output_video_encoder in ['libx264', 'libx265', 'libvpx']:
    #     commands.extend(['-crf', str(output_video_quality)])
    # if roop.globals.output_video_encoder in ['h264_nvenc', 'hevc_nvenc']:
    #     commands.extend(['-cq', str(output_video_quality)])
    commands.extend(['-pix_fmt', 'yuv420p', '-vf', 'colorspace=bt709:iall=bt601-6-625:fast=1', '-y', temp_output_path])
    return run_ffmpeg(commands)

create_video()

# ffmpeg -hide_banner -loglevel info -hwaccel auto -r 25.0 -i output/%04d.png -c:v libx264 -pix_fmt yuv420p -vf colorspace=bt709:iall=bt601-6-625:fast=1 -y output/tmp.mp4