import os
import subprocess
import tempfile
from typing import List, Dict
import datetime

from config import OUTPUT_DIR

try:
    import imageio_ffmpeg
    FFMPEG_PATH = imageio_ffmpeg.get_ffmpeg_exe()
except:
    FFMPEG_PATH = 'ffmpeg'


class VideoEditor:
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.video_duration = None
        
    def _get_video_duration(self) -> float:
        if self.video_duration is None:
            cmd = [
                FFMPEG_PATH, '-i', self.video_path,
                '-hide_banner', '-f', 'null', '-'
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='ignore')
            for line in result.stderr.split('\n'):
                if 'Duration:' in line:
                    time_str = line.split('Duration:')[1].split(',')[0].strip()
                    h, m, s = time_str.split(':')
                    self.video_duration = int(h) * 3600 + int(m) * 60 + float(s)
                    break
        return self.video_duration or 0
    
    def create_compilation(self, segments: List[Dict], fade_duration: float = 0.5, min_relevance: int = 0, progress_callback=None) -> str:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        video_duration = self._get_video_duration()
        if video_duration == 0:
            print("无法获取视频时长")
            return None
        
        filtered_segments = [
            seg for seg in segments 
            if seg.get("relevance_score", 0) >= min_relevance
        ]
        
        if not filtered_segments:
            print("没有符合条件的片段")
            return None
        
        filtered_segments.sort(key=lambda x: x.get("start_time", 0))
        
        total_duration = sum(
            min(video_duration, float(seg.get("end_time", 0))) - max(0, float(seg.get("start_time", 0)))
            for seg in filtered_segments
            if min(video_duration, float(seg.get("end_time", 0))) - max(0, float(seg.get("start_time", 0))) >= 0.5
        )
        
        print(f"正在剪辑 {len(filtered_segments)} 个片段，总时长: {total_duration:.1f}秒")
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            concat_list_path = f.name
            for i, seg in enumerate(filtered_segments):
                start = max(0, float(seg.get("start_time", 0)))
                end = min(video_duration, float(seg.get("end_time", 0)))
                
                if end - start < 0.5:
                    print(f"片段 {i+1} 时长太短，跳过")
                    continue
                
                f.write(f"file '{os.path.abspath(self.video_path)}'\n")
                f.write(f"inpoint {start}\n")
                f.write(f"outpoint {end}\n")
                print(f"片段 {i+1}: {start:.2f}s - {end:.2f}s")
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"ai_cut_{timestamp}.mp4"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        print(f"正在导出视频: {output_path}")
        
        cmd = [
            FFMPEG_PATH,
            '-f', 'concat',
            '-safe', '0',
            '-i', concat_list_path,
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '23',
            '-c:a', 'aac',
            '-b:a', '128k',
            '-movflags', '+faststart',
            '-stats',
            '-y',
            output_path
        ]
        
        print(f"执行命令: {' '.join(cmd[:10])}...")
        print(f"总时长: {total_duration:.1f}秒，开始导出...")
        
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8',
                errors='ignore'
            )
            
            import re
            time_pattern = re.compile(r'time=(\d+):(\d+):(\d+\.?\d*)')
            
            while True:
                line = process.stderr.readline()
                if not line and process.poll() is not None:
                    break
                
                match = time_pattern.search(line)
                if match:
                    h, m, s = int(match.group(1)), int(match.group(2)), float(match.group(3))
                    current_time = h * 3600 + m * 60 + s
                    if total_duration > 0:
                        progress_pct = min(current_time / total_duration, 1.0)
                        print(f"\r  导出进度: {progress_pct*100:.1f}% ({current_time:.1f}s / {total_duration:.1f}s)", end='', flush=True)
                        if progress_callback:
                            progress_callback(progress_pct)
            
            print()
            process.wait()
            
            if process.returncode != 0:
                print(f"\nFFmpeg 错误: 返回码 {process.returncode}")
                return None
            
            if os.path.exists(output_path):
                size_mb = os.path.getsize(output_path) / 1024 / 1024
                print(f"导出完成: {output_path} ({size_mb:.2f} MB)")
            else:
                print("文件不存在!")
                return None
                
        except Exception as e:
            print(f"导出失败: {e}")
            return None
        finally:
            try:
                os.unlink(concat_list_path)
            except:
                pass
        
        return output_path
    
    def create_individual_clips(self, segments: List[Dict], fade_duration: float = 0.5) -> List[str]:
        """创建独立的片段文件"""
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        video_duration = self._get_video_duration()
        if video_duration == 0:
            print("无法获取视频时长")
            return []
        
        output_paths = []
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for i, seg in enumerate(segments):
            start = max(0, float(seg.get("start_time", 0)))
            end = min(video_duration, float(seg.get("end_time", 0)))
            
            if end - start < 0.5:
                print(f"片段 {i+1} 时长太短，跳过")
                continue
            
            output_filename = f"ai_clip_{timestamp}_{i+1:03d}.mp4"
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            
            cmd = [
                FFMPEG_PATH,
                '-ss', str(start),
                '-i', self.video_path,
                '-t', str(end - start),
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-crf', '23',
                '-c:a', 'aac',
                '-b:a', '128k',
                '-movflags', '+faststart',
                '-y',
                output_path
            ]
            
            print(f"正在导出片段 {i+1}: {start:.2f}s - {end:.2f}s")
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='ignore')
                
                if result.returncode != 0:
                    print(f"FFmpeg 错误: {result.stderr[-500:]}")
                    continue
                
                if os.path.exists(output_path):
                    output_paths.append(output_path)
                    print(f"片段 {i+1} 导出完成: {output_path}")
                    
            except Exception as e:
                print(f"片段 {i+1} 导出失败: {e}")
        
        print(f"共导出 {len(output_paths)} 个片段")
        return output_paths
