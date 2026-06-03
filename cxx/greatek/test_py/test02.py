from moviepy.editor import VideoFileClip, concatenate
import numpy as np
import tqdm


from moviepy.editor import VideoFileClip, concatenate_videoclips
import numpy as np


def slice_video(video: str, out_mp4: str, start_sec: int = 1800, end_sec: int = 3600, interval: int = 10):
    clip = VideoFileClip(video).subclip(start_sec, end_sec)
    times = np.arange(0, clip.duration, interval)
    target_fps = 1 / interval

    # 每 1 帧持续 *interval* 秒 → 视频总长度 = 原段长度
    frames = [clip.to_ImageClip(t=t, duration=interval) for t in times]  # 关键：duration=interval

    video_clip = concatenate_videoclips(frames, method="compose")
    video_clip.write_videofile(
        out_mp4, codec="libx264", fps=target_fps, ffmpeg_params=["-color_range", "pc", "-colorspace", "bt709"]
    )
    clip.close()


# 直接跑
if __name__ == "__main__":
    slice_video(
        r"E:\demo\py\test01\西南油气元坝A炉炉膛火焰_20250915112114-20250916115114_1.mp4",
        "out.mp4",
        start_sec=1800,
        end_sec=3600,
        interval=10,
    )
