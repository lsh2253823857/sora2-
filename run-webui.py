# -*- coding: utf-8 -*-
import os
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "OFF"
os.environ["CRYPTOGRAPHY_OPENSSL_NO_LEGACY"]="0"#
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=SyntaxWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.filterwarnings("ignore")
try:os.link("resources/ffmpeg.exe","ffmpeg.exe")
except:pass
try:os.link("resources/ffprobe.exe","ffprobe.exe")
except:pass
import argparse
import traceback
now_dir = os.getcwd()
tmp = "%s/output"%now_dir
os.makedirs(tmp, exist_ok=True)
os.environ["TEMP"] = tmp

import sys
import datetime
import imageio
import numpy as np
import torch
import gradio as gr
from PIL import Image
sys.path.insert(0, os.path.sep.join(os.path.realpath(__file__).split(os.path.sep)[:-3]))
import logging
import math
from fastapi.responses import PlainTextResponse
from PIL import Image
from huggingface_hub.utils.tqdm import progress_bar_states
from numpy import ndarray
import random
import threading
import time
import cv2
import tempfile
from datetime import datetime, timedelta
import asyncio

os.makedirs("./output", exist_ok=True)
import psutil,os,platform,subprocess
from subprocess import Popen
def kill_proc_tree(pid, including_parent=True):
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        # Process already terminated
        return
    children = parent.children(recursive=True)
    for child in children:
        try:
            os.kill(child.pid, signal.SIGTERM)  # or signal.SIGKILL
        except OSError:
            pass
    if including_parent:
        try:
            os.kill(parent.pid, signal.SIGTERM)  # or signal.SIGKILL
        except OSError:
            pass
system = platform.system()
def kill_process(pid, process_name=""):
    if system == "Windows":
        cmd = "taskkill /t /f /pid %s" % pid
        # os.system(cmd)
        subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        kill_proc_tree(pid)
    print(process_name + "进程已终止")


from multiprocessing import freeze_support
str2size={
    "720P":"1280*720",
    "540P":"960*544",
    "480P":"832*480",
    "360P":"640*360",
}
os.environ["ffmpeg_exe_cmd"]=ffmpeg_exe_cmd="ffmpeg"
python_exe_cmd=r"runtime\python"

if os.name == "nt":
    # Proactor loop on Windows may emit noisy connection_lost tracebacks
    # when browsers disconnect abruptly; selector loop is usually quieter.
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass
def extract_frame(input_path,output_path):
    cmd='%s -i "%s" -vf "fps=16" -c:v libx264 -crf 18 -an "%s" -y'%(ffmpeg_exe_cmd,input_path,output_path)
    print(cmd)
    os.system(cmd)

p_remove_water=None
def change_remove_water(input_path):
    global p_remove_water
    if p_remove_water is None:
        output_path = "%s-去水印.mp4" % (os.path.splitext(input_path)[0])
        os.environ["input"]=input_path
        os.environ["output"]=output_path
        cmd="%s dewater.py"%python_exe_cmd
        yield (
            {"__type__": "update", "visible": False},
            {"__type__": "update", "visible": True},
            {"__type__": "update", "value": None,"label":"去水印中..."},
        )
        print(cmd)
        p_remove_water = Popen(cmd, shell=True)
        p_remove_water.wait()
        p_remove_water=None
        yield (
            {"__type__": "update", "visible": True},
            {"__type__": "update", "visible": False},
            {"__type__": "update", "value": output_path,"label":"去水印进程执行完毕."},
        )
    else:
        kill_process(p_remove_water.pid, "去水印")
        p_remove_water = None
        yield (
            {"__type__": "update", "visible": True},
            {"__type__": "update", "visible": False},
            {"__type__": "update", "value": None,"label":"去水印进程已关闭."},
        )
p_add_water=None
def change_add_water(input_path):
    global p_add_water
    if p_add_water is None:
        output_path = "%s-加水印.mp4" % (os.path.splitext(input_path)[0])
        os.environ["input"]=input_path
        os.environ["output"]=output_path
        cmd="%s add_water.py"%python_exe_cmd
        yield (
            {"__type__": "update", "visible": False},
            {"__type__": "update", "visible": True},
            {"__type__": "update", "value": None,"label":"加水印中..."},
        )
        print(cmd)
        p_add_water = Popen(cmd, shell=True)
        p_add_water.wait()
        p_add_water=None
        yield (
            {"__type__": "update", "visible": True},
            {"__type__": "update", "visible": False},
            {"__type__": "update", "value": output_path,"label":"加水印进程执行完毕."},
        )
    else:
        kill_process(p_add_water.pid, "加水印")
        p_add_water = None
        yield (
            {"__type__": "update", "visible": True},
            {"__type__": "update", "visible": False},
            {"__type__": "update", "value": None,"label":"加水印进程已关闭."},
        )



if __name__ == "__main__":
    freeze_support()
    from time import time as ttime
    with gr.Blocks() as demo:
        gr.Markdown("""
               <div style="text-align: center; font-size: 32px; font-weight: bold; margin-bottom: 20px;">
                   Sora2后处理工具箱
               </div>
               """)
        with gr.Row():
            with gr.Column():
                video_input = gr.Video(label="输入视频")
                with gr.Row():
                    open_add_water_button = gr.Button("开启加水印",visible=True)
                    close_add_water_button = gr.Button("关闭加水印",visible=False)
                    open_dewater_button = gr.Button("开启去水印",visible=True)
                    close_dewater_button = gr.Button("关闭去水印",visible=False)

            with gr.Column():
                video_output = gr.Video(label="生成结果")

        open_add_water_button.click(change_add_water,inputs=[video_input],outputs=[open_add_water_button,close_add_water_button,video_output],)
        close_add_water_button.click(change_add_water,inputs=[video_input],outputs=[open_add_water_button,close_add_water_button,video_output],)
        open_dewater_button.click(change_remove_water,inputs=[video_input],outputs=[open_dewater_button,close_dewater_button,video_output],)
        close_dewater_button.click(change_remove_water,inputs=[video_input],outputs=[open_dewater_button,close_dewater_button,video_output],)


    demo.queue(default_concurrency_limit=2, max_size=2).launch(  # concurrency_count=511, max_size=1022
        server_name="0.0.0.0",
        inbrowser=True,
        # share=True,
        server_port=14321#, allowed_paths = [tmp]
        # quiet=True,
    )