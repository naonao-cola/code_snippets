import argparse
import asyncio
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
import time
import cv2
from tqdm import tqdm


def _parse_ts(ts_str: str) -> float:
    """把 01:23:45 或 123.4 转成秒"""
    ts_str = ts_str.strip().replace(".", ":")  # 关键一行
    try:
        return float(ts_str)
    except ValueError:
        x = datetime.strptime(ts_str, "%H:%M:%S")
        return timedelta(hours=x.hour, minutes=x.minute, seconds=x.second).total_seconds()



def _slice_once(in_path, out_path, start_sec, end_sec, interval_sec):
    cap = cv2.VideoCapture(in_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)  # 让它自动转 BGR
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    totalFrames  = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print(f"总帧数 {totalFrames}, fps {fps}, 宽高 {w}x{h}")

    sum =0
    while cap.isOpened() and sum < totalFrames:
        cap.grab()
        sum = sum + 1
        if (sum % (interval_sec * fps)) == 0:
            success, im = cap.retrieve()
            print(f"sum {sum}")
            if success:
                out.write(im)
        time.sleep(0.0)
    cap.release()
    out.release()


async def _async_slice(loop, executor, task_dict):
    """包装成 async"""
    await loop.run_in_executor(
        executor,
        _slice_once,
        task_dict["input"],
        task_dict["output"],
        task_dict["ss"],
        task_dict["to"],
        task_dict["interval"],
    )


async def main(jobs, max_workers: int = 1):
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        await asyncio.gather(*[_async_slice(loop, pool, j) for j in jobs])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="异步长视频抽帧裁剪")
    ap.add_argument(
        "-i",
        "--input",
        type=str,
        default=r"./西南油气元坝A炉炉膛火焰_20250915112114-20250916115114_1.mp4",
        help="输入视频路径（可多组）",
    )
    ap.add_argument(
        "-o", "--output", action="append", type=str, default="5.mp4",help="输出视频路径"
    )
    ap.add_argument(
        "-ss",
        "--start",
        type=str,
        default="00:30:00",
        help="开始时间 hh:mm:ss 或秒",
    )
    ap.add_argument("-to", "--end", type=str, default="1:00:00",  help="结束时间")
    ap.add_argument("-interval", "--interval", type=int, default=10,  help="抽帧间隔（秒）")
    ap.add_argument("-j", "--jobs", type=int, default=1, help="并发线程数")
    args = ap.parse_args()

    tasks = [{
    "input": args.input,
    "output": args.output,
    "ss": _parse_ts(args.start),
    "to": _parse_ts(args.end),
    "interval": args.interval,
}]

    s = time.time()
    asyncio.run(main(tasks, max_workers=args.jobs))
    print("全部完成，耗时 {:.1f}s".format(time.time() - s))
