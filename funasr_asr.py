"""
FunASR 语音识别封装（多进程并行版）
GitHub: https://github.com/alibaba-damo-academy/FunASR
完全开源免费，阿里巴巴达摩院开发
"""
import os
import sys
import time
import subprocess
from typing import List, Dict, Optional
import numpy as np


class FunASRRecognizer:
    """FunASR 语音识别器"""
    
    def __init__(self, model_name: str = "paraformer-zh"):
        """
        初始化 FunASR 识别器
        
        Args:
            model_name: 模型名称
                - "paraformer-zh": 中文语音识别（推荐）
        """
        self.model_name = model_name
        self.model = None
        self.device = None
        self._load_model()
    
    def _load_model(self):
        """加载模型（自动检测 GPU）"""
        try:
            from funasr import AutoModel
            import torch
            
            # 检测是否有 GPU（支持 CUDA 和 MPS）
            if torch.cuda.is_available():
                self.device = "cuda"
                print(f"检测到 CUDA GPU: {torch.cuda.get_device_name(0)}")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = "mps"
                print("检测到 Apple Silicon GPU (MPS)")
            else:
                self.device = "cpu"
                print("使用 CPU 进行识别")
            
            print(f"正在加载 FunASR 模型: {self.model_name}...")
            start_time = time.time()
            
            # 加载模型
            if self.model_name == "paraformer-zh":
                self.model = AutoModel(
                    model="paraformer-zh",
                    model_revision="v2.0.4",
                    vad_model="fsmn-vad",
                    vad_model_revision="v2.0.4",
                    punc_model="ct-punc",
                    punc_model_revision="v2.0.4",
                    device=self.device,
                )
            else:
                self.model = AutoModel(
                    model=self.model_name,
                    device=self.device
                )
            
            elapsed = time.time() - start_time
            print(f"FunASR 模型加载完成 (耗时 {elapsed:.1f}s)")
            print(f"使用设备: {self.device}")
            
        except Exception as e:
            print(f"加载 FunASR 模型失败: {e}")
            raise
    
    def recognize(self, audio_path: str) -> Optional[List[Dict]]:
        """
        识别音频文件（单进程模式，适合短音频）
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            识别结果列表，每个元素包含 start/end/text
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")
        
        if self.model is None:
            raise RuntimeError("模型未加载")
        
        print(f"正在识别音频: {audio_path}")
        print(f"音频大小: {os.path.getsize(audio_path) / 1024 / 1024:.1f} MB")
        start_time = time.time()
        
        # 转换为 wav 格式
        temp_wav = None
        try:
            if not audio_path.lower().endswith('.wav'):
                temp_wav = audio_path.rsplit('.', 1)[0] + '_temp.wav'
                self._convert_to_wav(audio_path, temp_wav)
                wav_path = temp_wav
            else:
                wav_path = audio_path
            
            # 加载音频为 numpy 数组
            audio_data, sample_rate = self._load_audio(wav_path)
            
            # 执行识别（传入 numpy 数组而不是文件路径，绕过 torchaudio）
            result = self.model.generate(
                input=audio_data,  # numpy 数组
                batch_size_s=300,
                hotword='',
            )
            
            elapsed = time.time() - start_time
            
            # 解析结果
            segments = self._parse_result(result)
            
            # 计算实时率
            total_duration = len(audio_data) / sample_rate
            rtf = elapsed / total_duration if total_duration > 0 else 0
            print(f"识别完成: {len(segments)} 个片段")
            print(f"音频时长: {total_duration:.1f}s, 处理耗时: {elapsed:.1f}s, RTF: {rtf:.3f}")
            
            return segments
            
        except Exception as e:
            print(f"识别失败: {e}")
            import traceback
            traceback.print_exc()
            raise
        finally:
            if temp_wav and os.path.exists(temp_wav):
                try:
                    os.remove(temp_wav)
                except:
                    pass
    
    def recognize_fast(self, audio_path: str, num_processes: int = None) -> Optional[List[Dict]]:
        """
        快速识别长音频（使用多进程并行）
        
        Args:
            audio_path: 音频文件路径
            num_processes: 并行进程数，默认为 CPU 核心数
            
        Returns:
            识别结果列表
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")
        
        import multiprocessing as mp
        
        if num_processes is None:
            num_processes = min(mp.cpu_count(), 4)  # 默认最多4个进程
        
        print(f"使用多进程模式识别长音频: {audio_path}")
        print(f"并行进程数: {num_processes}")
        print("注意：每个进程需要单独加载模型，启动可能需要一些时间...")
        
        # 先转换为 wav
        temp_wav = audio_path.rsplit('.', 1)[0] + '_temp.wav'
        if not audio_path.lower().endswith('.wav'):
            self._convert_to_wav(audio_path, temp_wav)
            wav_path = temp_wav
        else:
            wav_path = audio_path
        
        try:
            # 加载完整音频
            audio_data, sample_rate = self._load_audio(wav_path)
            total_duration = len(audio_data) / sample_rate
            
            print(f"音频总时长: {total_duration:.1f}s")
            
            # 计算每段长度（根据进程数均分）
            chunk_samples = len(audio_data) // num_processes
            chunks = []
            for i in range(num_processes):
                start_sample = i * chunk_samples
                if i == num_processes - 1:
                    chunk = audio_data[start_sample:]
                else:
                    chunk = audio_data[start_sample:start_sample + chunk_samples]
                start_time = start_sample / sample_rate
                chunks.append((chunk, start_time, sample_rate, i+1, num_processes))
            
            print(f"分成 {len(chunks)} 段并行处理...")
            
            # 使用多进程并行识别
            start_time_total = time.time()
            
            with mp.Pool(processes=num_processes) as pool:
                results = pool.map(_recognize_chunk_worker, chunks)
            
            # 合并结果
            all_segments = []
            for segments in results:
                if segments:
                    all_segments.extend(segments)
            
            # 按时间排序
            all_segments.sort(key=lambda x: x['start'])
            
            elapsed = time.time() - start_time_total
            rtf = elapsed / total_duration if total_duration > 0 else 0
            
            print(f"并行识别完成: {len(all_segments)} 个片段")
            print(f"音频时长: {total_duration:.1f}s, 处理耗时: {elapsed:.1f}s, RTF: {rtf:.3f}")
            
            return all_segments
            
        finally:
            if temp_wav != audio_path and os.path.exists(temp_wav):
                try:
                    os.remove(temp_wav)
                except:
                    pass
    
    def _convert_to_wav(self, input_path: str, output_path: str):
        """使用 ffmpeg 将音频转换为 wav 格式"""
        ffmpeg_path = 'ffmpeg'
        try:
            import imageio_ffmpeg
            ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        except:
            pass
        
        cmd = [
            ffmpeg_path,
            '-y',
            '-i', input_path,
            '-ar', '16000',
            '-ac', '1',
            '-c:a', 'pcm_s16le',
            output_path
        ]
        
        print(f"转换音频格式: {input_path} -> {output_path}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"音频转换失败: {result.stderr}")
    
    def _load_audio(self, wav_path: str):
        """加载 wav 音频为 numpy 数组，使用 soundfile 替代 torchaudio"""
        import soundfile as sf
        
        audio, samplerate = sf.read(wav_path, dtype='float32')
        
        # 确保是单声道
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        
        return audio, samplerate
    
    def _parse_result(self, result) -> List[Dict]:
        """解析 FunASR 返回结果"""
        segments = []
        
        if not result:
            return segments
        
        for item in result:
            if 'sentence_info' in item and item['sentence_info']:
                for sent in item['sentence_info']:
                    segments.append({
                        'start': sent.get('start', 0) / 1000,
                        'end': sent.get('end', 0) / 1000,
                        'text': sent.get('text', '').strip()
                    })
            elif 'timestamp' in item and item['timestamp']:
                text = item.get('text', '')
                timestamps = item['timestamp']
                
                words = list(text)
                current_segment = {'text': '', 'start': None, 'end': None}
                
                for i, (word, ts) in enumerate(zip(words, timestamps)):
                    if current_segment['start'] is None:
                        current_segment['start'] = ts[0]
                    current_segment['text'] += word
                    current_segment['end'] = ts[1]
                    
                    if word in '。！？，；：' or len(current_segment['text']) >= 15:
                        segments.append({
                            'start': current_segment['start'] / 1000,
                            'end': current_segment['end'] / 1000,
                            'text': current_segment['text'].strip()
                        })
                        current_segment = {'text': '', 'start': None, 'end': None}
                
                if current_segment['text']:
                    segments.append({
                        'start': current_segment['start'] / 1000,
                        'end': current_segment['end'] / 1000,
                        'text': current_segment['text'].strip()
                    })
            elif 'text' in item:
                segments.append({
                    'start': 0.0,
                    'end': 0.0,
                    'text': item['text'].strip()
                })
        
        return segments


def _recognize_chunk_worker(args):
    """
    多进程工作函数（必须在模块级别定义才能被 pickle）
    
    Args:
        args: (audio_chunk, start_time, sample_rate, chunk_id, total_chunks)
    """
    audio_chunk, start_time, sample_rate, chunk_id, total_chunks = args
    
    # 重定向 stdout/stderr 到 devnull，避免子进程打印混乱
    import os
    import sys
    
    # 保存原始输出
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    
    # 重定向到空设备
    with open(os.devnull, 'w') as devnull:
        sys.stdout = devnull
        sys.stderr = devnull
        
        try:
            from funasr import AutoModel
            
            # 每个进程加载自己的模型实例
            model = AutoModel(
                model="paraformer-zh",
                model_revision="v2.0.4",
                vad_model="fsmn-vad",
                vad_model_revision="v2.0.4",
                punc_model="ct-punc",
                punc_model_revision="v2.0.4",
                device="cpu",  # 多进程只能用 CPU
            )
            
            result = model.generate(
                input=audio_chunk,
                batch_size_s=60,
                hotword='',
            )
            
            # 恢复输出以打印进度
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            print(f"  进程 {chunk_id}/{total_chunks} 完成")
            
            # 解析结果
            segments = []
            for item in result:
                if 'sentence_info' in item and item['sentence_info']:
                    for sent in item['sentence_info']:
                        segments.append({
                            'start': sent.get('start', 0) / 1000 + start_time,
                            'end': sent.get('end', 0) / 1000 + start_time,
                            'text': sent.get('text', '').strip()
                        })
                elif 'timestamp' in item and item['timestamp']:
                    text = item.get('text', '')
                    timestamps = item['timestamp']
                    
                    words = list(text)
                    current_segment = {'text': '', 'start': None, 'end': None}
                    
                    for i, (word, ts) in enumerate(zip(words, timestamps)):
                        if current_segment['start'] is None:
                            current_segment['start'] = ts[0]
                        current_segment['text'] += word
                        current_segment['end'] = ts[1]
                        
                        if word in '。！？，；：' or len(current_segment['text']) >= 15:
                            segments.append({
                                'start': current_segment['start'] / 1000 + start_time,
                                'end': current_segment['end'] / 1000 + start_time,
                                'text': current_segment['text'].strip()
                            })
                            current_segment = {'text': '', 'start': None, 'end': None}
                    
                    if current_segment['text']:
                        segments.append({
                            'start': current_segment['start'] / 1000 + start_time,
                            'end': current_segment['end'] / 1000 + start_time,
                            'text': current_segment['text'].strip()
                        })
                elif 'text' in item:
                    segments.append({
                        'start': start_time,
                        'end': start_time,
                        'text': item['text'].strip()
                    })
            
            return segments
            
        except Exception as e:
            # 恢复输出以打印错误
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            print(f"  进程 {chunk_id}/{total_chunks} 失败: {e}")
            return []
        finally:
            # 确保恢复输出
            sys.stdout = old_stdout
            sys.stderr = old_stderr


# 全局模型实例（懒加载）
_FUNASR_MODEL = None

def get_funasr_model() -> Optional[FunASRRecognizer]:
    """获取 FunASR 模型实例（单例模式）"""
    global _FUNASR_MODEL
    if _FUNASR_MODEL is None:
        try:
            _FUNASR_MODEL = FunASRRecognizer()
        except Exception as e:
            print(f"初始化 FunASR 失败: {e}")
            return None
    return _FUNASR_MODEL


def recognize_audio(audio_path: str, parallel: bool = True, num_processes: int = None) -> Optional[List[Dict]]:
    """便捷的识别函数
    
    Args:
        audio_path: 音频文件路径
        parallel: 是否使用多进程并行模式（适合长音频）
        num_processes: 并行进程数，默认为 CPU 核心数
    """
    model = get_funasr_model()
    if model is None:
        return None
    
    if parallel:
        return model.recognize_fast(audio_path, num_processes)
    else:
        return model.recognize(audio_path)
