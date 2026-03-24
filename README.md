# SoraPostProcess 优化说明（基于花儿不哭整合包）

本次优化针对 B 站 UP 主 **花儿不哭** 发布的整合包进行稳定性与兼容性增强，主要用于降低去水印流程中的内存崩溃概率，并减少运行时告警噪声。

- 参考来源（原始整合包介绍）：
  - [发布Sora2后处理工具，包括加水印、去水印和画质增强等功能，永久免费。](https://www.bilibili.com/video/BV1QpyZBZE3v/?spm_id_from=333.1387.homepage.video_card.click&vd_source=8eb4a1b3b0aa63969242e9b2bab1bae1)

## 优化内容

### 1) 修复 NumPy 内存分配异常（去水印核心路径）

- 文件：`sorawm/iopaint/model/base.py`
- 问题：掩码混合过程会触发较大的临时数组分配，内存紧张时可能报错：
  - `numpy._core._exceptions._ArrayMemoryError`
- 优化：
  - 将混合计算统一到 `float32`，避免隐式 `float64` 放大；
  - 最终结果安全转回 `uint8`。

### 2) 降低整体内存占用（核心流程重构为流式两遍处理）

- 文件：`sorawm/core.py`
- 问题：旧流程在检测阶段缓存全部视频帧，长视频或高分辨率时内存快速攀升。
- 优化：
  - 第一遍仅检测并记录每帧 `bbox`（不缓存帧图像）；
  - 第二遍重新读取视频并执行去水印、写出结果；
  - 显著降低峰值内存，提升长视频稳定性。

### 3) 修复 Windows 路径转义告警

- 文件：`run-webui.py`
- 问题：
  - `python_exe_cmd="runtime\python"` 会触发 `SyntaxWarning: invalid escape sequence '\p'`
- 优化：
  - 改为原始字符串：`python_exe_cmd = r"runtime\python"`。

### 4) 减少 Windows 下 asyncio 连接噪声日志

- 文件：`run-webui.py`
- 现象：偶发 `ConnectionResetError [WinError 10054]` 回调堆栈刷屏（通常不影响主流程）。
- 优化：
  - 在 Windows 下优先使用 `Selector` 事件循环策略，降低断连噪声。

### 5) 兼容新版 PyTorch AMP 用法

- 文件：`sorawm/iopaint/model/ldm.py`
- 问题：旧写法 `torch.cuda.amp.autocast()` 存在弃用告警。
- 优化：
  - 迁移为 `torch.amp.autocast(device_type="cuda", enabled=...)`。

## 预期效果

- 去水印任务的稳定性提升，尤其是 CPU 环境和中长视频场景；
- 运行日志更干净，排障更直观；
- 兼容性更好（路径转义与 AMP API）。

## 说明与致谢

本优化基于原整合包进行工程层面的稳定性改进，不改变项目原始功能定位。  
感谢原作者与社区分享。

