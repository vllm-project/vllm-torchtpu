"""Write the synthetic conditioning media used by run_all_pipelines.py.

Creates three 256x256 PNGs, a 4 s 24 fps H.264 clip and a 4 s stereo 32 kHz
WAV in /mnt/workspace/h3_tpu/inputs, then prints ffmpeg's view of the clip
and the WAV.
"""

import os
import subprocess
import wave

import imageio_ffmpeg
import numpy as np
from PIL import Image

inputs_dir = "/mnt/workspace/h3_tpu/inputs"
os.makedirs(inputs_dir, exist_ok=True)
width, height = 256, 256
fps = 24
duration = 4.0
frames_count = int(fps * duration)

# Images: vertical RGB color bars, an x/y gradient and a 32-pixel checkerboard.
bars = np.zeros((height, width, 3), dtype=np.uint8)
bar_width = width // 3
bars[:, :bar_width] = [255, 0, 0]
bars[:, bar_width : bar_width * 2] = [0, 255, 0]
bars[:, bar_width * 2 :] = [0, 0, 255]
Image.fromarray(bars).save(f"{inputs_dir}/image_color_bars.png")

x = np.linspace(0, 255, width, dtype=np.uint8)
y = np.linspace(0, 255, height, dtype=np.uint8)
xv, yv = np.meshgrid(x, y)
grad = np.stack([xv, yv, np.zeros_like(xv)], axis=-1)
Image.fromarray(grad).save(f"{inputs_dir}/image_gradient.png")

checker = np.indices((height // 32, width // 32)).sum(axis=0) % 2
checker = np.kron(checker, np.ones((32, 32))).astype(np.uint8) * 255
checker_rgb = np.stack([checker] * 3, axis=-1)
Image.fromarray(checker_rgb).save(f"{inputs_dir}/image_checkerboard.png")

# Video: the gradient scrolling horizontally, piped to ffmpeg as raw RGB.
video_path = f"{inputs_dir}/video_moving_gradient.mp4"
exe = imageio_ffmpeg.get_ffmpeg_exe()
cmd = [
    exe,
    "-y",
    "-f",
    "rawvideo",
    "-vcodec",
    "rawvideo",
    "-s",
    f"{width}x{height}",
    "-pix_fmt",
    "rgb24",
    "-r",
    str(fps),
    "-i",
    "-",
    "-an",
    "-vcodec",
    "libx264",
    "-pix_fmt",
    "yuv420p",
    video_path,
]
proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
for i in range(frames_count):
    shift = int((i / frames_count) * width)
    frame = np.roll(grad, shift, axis=1)
    proc.stdin.write(frame.tobytes())
proc.stdin.close()
proc.wait()

# Audio: 16-bit stereo sine sweep, 100-2000 Hz left and 1.5x that right.
wav_path = f"{inputs_dir}/audio_sine_sweep.wav"
sample_rate = 32000
num_samples = int(sample_rate * duration)
t = np.linspace(0, duration, num_samples, endpoint=False)
f_start, f_end = 100, 2000
freqs = np.linspace(f_start, f_end, num_samples)
left_channel = np.sin(2 * np.pi * freqs * t)
right_channel = np.sin(2 * np.pi * (freqs * 1.5) * t)
audio_data = np.stack([left_channel, right_channel], axis=-1)
audio_data = (audio_data * 32767).astype(np.int16)

with wave.open(wav_path, "wb") as wf:
    wf.setnchannels(2)
    wf.setsampwidth(2)
    wf.setframerate(sample_rate)
    wf.writeframes(audio_data.tobytes())

print("Video validation:")
subprocess.run([exe, "-i", video_path])

print("\nAudio validation:")
subprocess.run([exe, "-i", wav_path])
