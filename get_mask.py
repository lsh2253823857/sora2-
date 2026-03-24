# -*- coding: utf-8 -*-
import pdb
import numpy as np
np.float = float
# 如果还遇到 np.int、np.object alias 问题
np.int = int
np.object = object
import os
os.environ["TORCH_HOME"]="resources"
import sys
from PIL import Image, ImageDraw
import skvideo.io
import torch

mouth_use_original=os.environ["mouth_use_original"]
topk_mouth=int(os.environ["topk_mouth"])
scene_cut_detect_result=eval(os.environ["scene_cut_detect_result"])
extra_white_set = set()##得到需要额外保留的场景切换区间的所有帧
interval = 8
for n in set(scene_cut_detect_result):
    n_shift = n + 0.5
    interval_start = int(n_shift // interval) * interval
    interval_frames = set(range(interval_start+1, interval_start + interval))
    extra_white_set.update(interval_frames)
print("切镜导致要维持原样的帧集合：%s"%extra_white_set)
if mouth_use_original=="二次元人嘴":
    from models.experimental import attempt_load
    from utils.datasets import LoadStreams, LoadImages
    from utils.general import (
        check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, plot_one_box, strip_optimizer)
    from utils.torch_utils import select_device, load_classifier, time_synchronized
    device = "cuda"
    half = True
    model = attempt_load("resources/yolov5x_anime.pt", map_location=device)
    model.half()  # to FP16
elif mouth_use_original=="三次元人嘴":
    import face_alignment
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device='cuda')

def process_video(
    input_video_path: str,
    output_video_path: str,
    ffmpeg_path: str,
    crf: int = 19,
):
    # 设置环境变量，让 skvideo 使用指定 ffmpeg
    os.environ["FFMPEG_BINARY"] = ffmpeg_path

    # 使用 FFmpegReader 读取视频元数据
    reader = skvideo.io.FFmpegReader(input_video_path)
    meta = skvideo.io.ffprobe(input_video_path)
    video_meta = meta["video"]
    width = int(video_meta["@width"])
    height = int(video_meta["@height"])
    video_fps = float(video_meta["@avg_frame_rate"].split("/")[0]) / float(video_meta["@avg_frame_rate"].split("/")[1])
    print(f"Input video: {input_video_path}, size={width}x{height}, video_fps={video_fps:.2f}")
    outputdict = {
        "-vcodec": "libx264",
        "-crf": str(crf),
        "-pix_fmt": "yuv420p",
    }
    writer = skvideo.io.FFmpegWriter(output_video_path ,inputdict={'-framerate': str(video_fps)}, outputdict=outputdict)
    writer_mouth = skvideo.io.FFmpegWriter(os.environ.get("mouth_video_path",output_video_path+".mouth_result.mp4") ,inputdict={'-framerate': str(video_fps)}, outputdict=outputdict)

    frame_index=0
    for frame in reader.nextFrame():
        if frame_index % 100 == 0:
            print(f"Processed {frame_index} frames …")
        if frame_index%8==0 or frame_index in extra_white_set:
            img1=out_frame = np.array(np.ones_like(frame)*255)
        else:
            img1 = Image.fromarray(frame)
            img01 = np.zeros_like(img1)
            img1np = np.array(frame)
            if mouth_use_original == "二次元人嘴":
                if width < height:
                    scale = 576 / width
                    new_w = 576
                    new_h = int(height * scale)
                else:
                    scale = 576 / height
                    new_h = 576
                    new_w = int(width * scale)
                new_w = (new_w // 32) * 32
                new_h = (new_h // 32) * 32
                img2 = img1.resize((new_w, new_h))#, Image.Resampling.LANCZOS
                draw = ImageDraw.Draw(img1)

                ratio_w = new_w / width
                ratio_h = new_h / height

                with torch.no_grad():
                    img = torch.from_numpy(np.array(img2)).to(device)
                    img = img.half() if half else img.float()  # uint8 to fp16/32
                    img /= 255.0  # 0 - 255 to 0.0 - 1.0
                    img = img.unsqueeze(0).permute(0,3,1,2)
                    # print(img.shape)#torch.Size([1, 720, 1280, 3])
                    pred = model(img, augment=False)[0]
                    pred = non_max_suppression(pred, 0.4, 0.5, classes=[0], agnostic=False)

                # result.results[0] 包含 .boxes.xyxy 结构
                for i1,res in enumerate(pred):
                    if type(res)!=type(None):
                        res[:, [0,2]]=(res[:, [0,2]]/ratio_w)
                        res[:, [1,3]]=(res[:, [1,3]]/ratio_h)
                        for i2,box in enumerate(res[:topk_mouth]):
                            x1, y1, x2, y2,confidence = box[:5]
                            x1,y1,x2,y2=x1.item(),y1.item(),x2.item(),y2.item()
                            y1=(y1+y2)/2
                            if abs((x2-x1)*(y2-y1)/width/height)>0.004:
                                x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
                                img01[y1:y2,x1:x2]+=255
                                draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
                                draw.text((x1, y1 - 10), f"{confidence:.2f}" + "+%s-%s" % (i1, i2), fill="red")

            elif mouth_use_original == "三次元人嘴":
                img_h, img_w, _ = img1np.shape
                landmark_list = fa.get_landmarks(img1np)
                if landmark_list:
                    idx = 0
                    draw = ImageDraw.Draw(img1)
                    for landmarks in landmark_list:

                        ### 1. 提取嘴部关键点
                        # 嘴部索引为 48 到 67 (在python切片中是 48:68)
                        mouth_indices = range(48, 68)
                        mouth_points = landmarks[mouth_indices, :]

                        ### 2. 计算嘴部的外接矩形 (BBox)
                        # np.min(axis=0) 找到所有点中最小的 (x, y)
                        # np.max(axis=0) 找到所有点中最大的 (x, y)
                        min_coords = np.min(mouth_points, axis=0)
                        max_coords = np.max(mouth_points, axis=0)

                        x_min, y_min = min_coords
                        x_max, y_max = max_coords

                        ### 3. 计算扩展 5% 的 BBox
                        # 3.1 计算原始宽高和中心点
                        width1 = x_max - x_min
                        height1 = y_max - y_min
                        center_x = x_min + width1 / 2
                        center_y = y_min + height1 / 2

                        # 3.2 定义扩展比例
                        expand_ratio = 2

                        # 3.3 计算新的宽高
                        new_width = width1 * expand_ratio
                        new_height = height1 * expand_ratio

                        # 3.4 计算新的坐标 (从中心点扩展)
                        new_x_min = center_x - new_width / 2
                        new_y_min = center_y - new_height / 2
                        new_x_max = center_x + new_width / 2
                        new_y_max = center_y + new_height / 2

                        # 3.5 确保坐标不会超出图像边界 (Clamping)
                        # 并转换为整数以便绘制
                        x1 = int(max(0, new_x_min))
                        y1 = int(max(0, new_y_min))
                        x2 = int(min(img_w, new_x_max))
                        y2 = int(min(img_h, new_y_max))
                        #####过滤过小的框
                        if abs((x2 - x1) * (y2 - y1) / width / height) > 0.007:
                            img01[y1:y2, x1:x2] += 255
                            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

                        idx += 1
                        if idx > topk_mouth-1:  # 1 is top2
                            break  # 只拿topk最大的人嘴框

            out_frame = np.array(img01)
        bbox_frame = np.array(img1)
        writer.writeFrame(out_frame)
        writer_mouth.writeFrame(bbox_frame)
        frame_index+=1

    writer.close()
    writer_mouth.close()
    print(f"Output video saved to {output_video_path}")

if __name__ == "__main__":
    input_video  =   os.environ["input"]
    output_video =   os.environ["output"]
    ffmpeg_exe_cmd   =   os.environ["ffmpeg_exe_cmd"]
    process_video(
        input_video_path=input_video,
        output_video_path=output_video,
        ffmpeg_path=ffmpeg_exe_cmd,
        crf=19,
    )
