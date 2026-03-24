# coding=utf-8
from pathlib import Path
import os
from sorawm.core import SoraWM

if __name__ == "__main__":
    input_video_path = Path(os.environ["input"])
    output_video_path = Path(os.environ["output"])
    sora_wm = SoraWM()
    sora_wm.run(input_video_path, output_video_path)
