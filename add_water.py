# -*- coding: utf-8 -*-
import numpy as np

np.float = float
np.int = int
np.object = object
import os
import cv2
import skvideo.io
import subprocess  # 导入 subprocess


def process_video(
        input_video_path: str,
        output_video_path: str,
        ffmpeg_path: str = "ffmpeg.exe",
        crf: int = 23,
):
    os.environ["FFMPEG_BINARY"] = ffmpeg_path

    # 使用 FFmpegReader 读取主视频元数据
    reader = skvideo.io.FFmpegReader(input_video_path)
    meta = skvideo.io.ffprobe(input_video_path)

    # 检查原始视频是否有音频流
    has_audio = "audio" in meta

    video_meta = meta["video"]
    width = int(video_meta["@width"])
    height = int(video_meta["@height"])
    watermark_video_path = "resources/sora-heng_x264.mp4" if width > height else "resources/sora-shu_x264.mp4"
    input_fps = float(video_meta["@avg_frame_rate"].split("/")[0]) / float(video_meta["@avg_frame_rate"].split("/")[1])

    print(f"Input video: {input_video_path}, size={width}x{height}, input_fps={input_fps:.2f}")

    # 加载水印视频并获取其元数据
    print(f"Loading watermark video from {watermark_video_path} ...")
    wm_meta = skvideo.io.ffprobe(watermark_video_path)
    wm_video_meta = wm_meta["video"]
    wm_width = int(wm_video_meta["@width"])
    wm_height = int(wm_video_meta["@height"])

    # 将水印视频帧读入内存
    watermark_frames = []
    wm_reader = skvideo.io.FFmpegReader(watermark_video_path)
    for wm_frame in wm_reader.nextFrame():
        watermark_frames.append(wm_frame)
    wm_reader.close()
    num_wm_frames = len(watermark_frames)
    if num_wm_frames == 0:
        print("Error: Watermark video is empty or could not be read.")
        return
    print(f"Watermark video loaded: {num_wm_frames} frames, size={wm_width}x{wm_height}")

    # 水印预处理
    # 1. 先计算一次缩放和padding的参数
    print("Pre-processing watermark frames...")
    target_ratio = width / height
    mask_ratio = wm_width / wm_height

    pad_top, pad_bottom, pad_left, pad_right = 0, 0, 0, 0

    if mask_ratio > target_ratio:
        # 水印比目标“更宽”，以宽度为准缩放，上下padding
        new_wm_width = width
        new_wm_height = int(new_wm_width / mask_ratio)
        pad_top = (height - new_wm_height) // 2
        pad_bottom = height - new_wm_height - pad_top
    else:
        # 水印比目标“更高”或比例相同，以高度为准缩放，左右padding
        new_wm_height = height
        new_wm_width = int(new_wm_height * mask_ratio)
        pad_left = (width - new_wm_width) // 2
        pad_right = width - new_wm_width - pad_left

    # 2. 循环处理所有水印帧，生成处理好的mask列表
    for idx, wm_frame in enumerate(watermark_frames):
        mask_gray = cv2.cvtColor(wm_frame, cv2.COLOR_RGB2GRAY)
        if mask_ratio > target_ratio:
            resized_mask = cv2.resize(mask_gray, (new_wm_width, new_wm_height), interpolation=cv2.INTER_LINEAR)
        else:
            resized_mask = cv2.resize(mask_gray, (new_wm_width, new_wm_height), interpolation=cv2.INTER_LINEAR)
        final_mask = cv2.copyMakeBorder(resized_mask, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
        mask_normalized = np.expand_dims(final_mask.astype(np.float32) / 255.0, axis=-1)
        watermark_frames[idx] = mask_normalized * 0.667  # mix alpha is 0.667
    print(f"Finished pre-processing {len(watermark_frames)} watermark masks.")
    # 水印预处理结束

    # --- 修改部分：输出到临时文件 ---
    # 生成一个临时文件名
    temp_output_path = output_video_path + ".temp.mp4"

    # 设置输出
    outputdict = {
        "-vcodec": "libx264",
        "-crf": str(crf),
        "-pix_fmt": "yuv420p",
    }
    # 写入到临时文件
    writer = skvideo.io.FFmpegWriter(temp_output_path, inputdict={'-framerate': str(input_fps)}, outputdict=outputdict)

    frame_index = 0
    # 对主视频循环逐帧加水印
    for frame in reader.nextFrame():
        if frame_index % 100 == 0:
            print(f"Processed {frame_index} frames …")

        input_frame = np.array(frame)

        mask_normalized = watermark_frames[frame_index % num_wm_frames]
        input_float = input_frame.astype(np.float32)
        output_float = input_float * (1.0 - mask_normalized) + 255.0 * mask_normalized
        output = np.clip(output_float, 0, 255).astype(np.uint8)

        writer.writeFrame(output)

        frame_index += 1

    writer.close()
    reader.close()
    print(f"Video-only processing complete. Wrote to {temp_output_path}")

    # --- 新增部分：使用 FFmpeg 合并音频 ---

    if not has_audio:
        print("Original video has no audio stream. Renaming temp file to final output.")
        # 如果原始视频没有音频，直接重命名临时文件
        os.rename(temp_output_path, output_video_path)
        print(f"Output video (no audio) saved to {output_video_path}")
        return

    print("Starting audio muxing...")

    # 优先级 1: 尝试直接复制音频流 (-c:a copy)
    print("Attempt 1: Copying audio stream...")
    ffmpeg_command_copy = [
        ffmpeg_path,
        "-y",  # 覆盖输出文件
        "-i", temp_output_path,  # 输入 0 (带水印的视频)
        "-i", input_video_path,  # 输入 1 (原始视频，用于音频)
        "-map", "0:v:0",  # 映射: 从输入0选择视频流0
        "-map", "1:a:0?",  # 映射: 从输入1选择音频流0 ('?'表示如果不存在不报错)
        "-c:v", "copy",  # 复制视频流 (已经编码)
        "-c:a", "copy",  # [优先级1] 复制音频流
        "-shortest",  # 以最短的输入流为准结束
        output_video_path
    ]

    result_copy = subprocess.run(ffmpeg_command_copy, capture_output=True, text=True, encoding='utf-8')

    if result_copy.returncode == 0:
        print("Audio stream copied successfully.")
        os.remove(temp_output_path)  # 清理临时文件
        print(f"Output video saved to {output_video_path}")
        return

    # 优先级 2: 如果复制失败，尝试转码为 AAC (-c:a aac)
    print(f"Audio copy failed. Retrying by re-encoding to AAC...")
    print(f"FFmpeg stderr (copy attempt): {result_copy.stderr}")

    ffmpeg_command_aac = [
        ffmpeg_path,
        "-y",
        "-i", temp_output_path,
        "-i", input_video_path,
        "-map", "0:v:0",
        "-map", "1:a:0?",
        "-c:v", "copy",
        "-c:a", "aac",  # [优先级2] 转码为 AAC
        "-b:a", "192k",  # 设置一个合理的AAC比特率
        "-shortest",
        output_video_path
    ]

    result_aac = subprocess.run(ffmpeg_command_aac, capture_output=True, text=True, encoding='utf-8')

    if result_aac.returncode == 0:
        print("Audio re-encoded to AAC successfully.")
        os.remove(temp_output_path)  # 清理临时文件
    else:
        # 优先级 3: 实在不行，保留只有视频的文件
        print(f"Audio re-encoding also failed. Saving video-only file as fallback.")
        print(f"FFmpeg stderr (aac attempt): {result_aac.stderr}")
        os.rename(temp_output_path, output_video_path)  # 重命名临时文件
        print(f"Output video (video only) saved to {output_video_path}")
        # 注意：这里不删除 temp_output_path，因为它被重命名了

    print(f"Processing finished.")


if __name__ == "__main__":
    # input_video = "xxx"
    # output_video = "xxx"
    input_video = os.environ["input"]
    output_video = os.environ["output"]

    ffmpeg_exe = os.environ.get("ffmpeg_exe_cmd", "ffmpeg.exe")
    process_video(
        input_video_path=input_video,
        output_video_path=output_video,
        ffmpeg_path=ffmpeg_exe,
        crf=19,
    )