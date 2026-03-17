
# -*- coding: utf-8 -*-
"""
视频处理器 - 均匀采样版本
功能：按固定时间间隔提取视频帧
"""
import os
import cv2
import json
import subprocess
import time
from typing import List, Dict
from tqdm import tqdm

from config import TEMP_DIR

try:
    import imageio_ffmpeg
    FFMPEG_PATH = imageio_ffmpeg.get_ffmpeg_exe()
    FFPROBE_PATH = FFMPEG_PATH.replace('ffmpeg.exe', 'ffprobe.exe').replace('ffmpeg', 'ffprobe')
except:
    FFMPEG_PATH = 'ffmpeg'
    FFPROBE_PATH = 'ffprobe'


class VideoProcessor:
    """视频处理器 - 均匀采样提取关键帧"""
    
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap = None
        self._init_video_info()
        
    def _init_video_info(self):
        """初始化视频信息"""
        self.cap = cv2.VideoCapture(self.video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frames / self.fps if self.fps > 0 else 0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if self.fps <= 0 or self.total_frames <= 0:
            print("OpenCV 无法读取，使用 ffprobe...")
            self._get_info_from_ffprobe()
    
    def _get_info_from_ffprobe(self):
        """使用 ffprobe 获取视频信息"""
        try:
            cmd = [FFPROBE_PATH, '-v', 'quiet', '-print_format', 'json',
                  '-show_format', '-show_streams', self.video_path]
            result = subprocess.run(cmd, capture_output=True, text=True)
            info = json.loads(result.stdout)
            
            for stream in info.get('streams', []):
                if stream.get('codec_type') == 'video':
                    self.width = int(stream.get('width', 0))
                    self.height = int(stream.get('height', 0))
                    fps_str = stream.get('r_frame_rate', '0/1')
                    if '/' in fps_str:
                        num, den = fps_str.split('/')
                        self.fps = float(num) / float(den) if float(den) > 0 else 25
                    self.duration = float(stream.get('duration', info.get('format', {}).get('duration', 0)))
                    self.total_frames = int(self.duration * self.fps)
                    print(f"ffprobe: {self.duration:.1f}s, {self.fps:.1f}fps, {self.width}x{self.height}")
                    break
        except Exception as e:
            print(f"ffprobe 失败：{e}")
    
    def get_video_info(self) -> Dict:
        """获取视频信息"""
        return {
            "path": self.video_path,
            "fps": round(self.fps, 2),
            "total_frames": self.total_frames,
            "duration": round(self.duration, 2),
            "width": self.width,
            "height": self.height,
            "duration_formatted": self._format_duration(self.duration)
        }
    
    def _format_duration(self, seconds: float) -> str:
        """格式化时长为 HH:MM:SS 或 MM:SS"""
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h:02d}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"
    
    def close(self):
        """关闭视频文件"""
        if self.cap and self.cap.isOpened():
            self.cap.release()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()
