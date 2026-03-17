import os
import json
import subprocess
import base64
import sys
import time
from typing import List, Dict, Optional
from openai import OpenAI
from tqdm import tqdm

from config import (
    OPENAI_API_KEY, OPENAI_BASE_URL, MODEL_NAME, TEMP_DIR
)

try:
    import imageio_ffmpeg
    FFMPEG_PATH = imageio_ffmpeg.get_ffmpeg_exe()
except:
    FFMPEG_PATH = 'ffmpeg'

FUNASR_MODEL = None


def get_funasr_model():
    global FUNASR_MODEL
    if FUNASR_MODEL is None:
        try:
            from funasr_asr import FunASRRecognizer
            print("正在加载 FunASR 模型...")
            FUNASR_MODEL = FunASRRecognizer()
            print("FunASR 模型加载完成")
        except ImportError:
            print("未安装 FunASR，请运行: pip install funasr modelscope")
            return None
        except Exception as e:
            print(f"加载 FunASR 模型失败: {e}")
            return None
    return FUNASR_MODEL


class AIAnalyzer:
    def __init__(self):
        self.client = OpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL
        )
    
    def encode_image_to_base64(self, image_path: str) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')
    
    def _format_time(self, seconds: float) -> str:
        seconds = int(seconds)
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        if h > 0:
            return f"{h:02d}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"
    
    def _get_audio_duration(self, audio_path: str) -> float:
        cmd = [
            FFMPEG_PATH.replace('ffmpeg', 'ffprobe'),
            '-v', 'quiet',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            audio_path
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            return float(result.stdout.strip())
        except:
            return 0
    
    def transcribe_audio_with_timestamps(self, video_path: str) -> Optional[List[Dict]]:
        audio_path = os.path.join(TEMP_DIR, "audio.mp3")
        
        print("\n" + "="*50)
        print("【步骤1】提取音频")
        print("="*50)
        
        total_duration = self._get_audio_duration(video_path)
        if total_duration > 0:
            print(f"视频时长: {self._format_time(total_duration)}")
        
        cmd = [
            FFMPEG_PATH, '-y', '-i', video_path,
            '-vn', '-acodec', 'libmp3lame', '-q:a', '9', '-ar', '16000', '-ac', '1',
            audio_path
        ]
        
        try:
            print("正在提取音频（监控文件大小）...")
            
            with open(os.devnull, 'w') as devnull:
                process = subprocess.Popen(
                    cmd, 
                    stdout=devnull,
                    stderr=devnull
                )
            
            start_time = time.time()
            
            while process.poll() is None:
                time.sleep(2)
                
                if os.path.exists(audio_path):
                    current_size = os.path.getsize(audio_path)
                    size_mb = current_size / 1024 / 1024
                    elapsed = time.time() - start_time
                    print(f"  已生成: {size_mb:.1f} MB, 运行: {int(elapsed)}秒", end='\r')
            
            print()
            
            if not os.path.exists(audio_path):
                print("音频提取失败")
                return None
            
            file_size = os.path.getsize(audio_path) / 1024 / 1024
            elapsed = time.time() - start_time
            print(f"音频提取完成: {file_size:.2f} MB (耗时 {elapsed:.1f} 秒)")
            
        except Exception as e:
            print(f"音频提取错误: {e}")
            return None
        
        print("\n" + "="*50)
        print("【步骤2】语音识别转录 (FunASR)")
        print("="*50)
        
        audio_size = os.path.getsize(audio_path) / 1024 / 1024
        print(f"音频大小: {audio_size:.2f} MB")
        
        all_segments = []
        start_time = time.time()
        
        model = get_funasr_model()
        if model is None:
            print("无法加载 FunASR 模型")
            return None
        
        try:
            segments = model.recognize(audio_path)
            
            if segments:
                all_segments = segments
                elapsed = time.time() - start_time
                print(f"转录完成: {len(all_segments)} 个片段 (耗时 {int(elapsed)}秒)")
            else:
                print("FunASR 返回空结果")
                return None
            
        except Exception as e:
            print(f"FunASR 转录失败: {e}")
            return None
        
        if all_segments:
            all_segments.sort(key=lambda x: x['start'])
            total_duration = all_segments[-1]['end'] if all_segments else 0
            print(f"总时长 {total_duration:.1f}s")
            
            srt_path = os.path.join(TEMP_DIR, "transcript.srt")
            self._save_srt(all_segments, srt_path)
            print(f"字幕已保存: {srt_path}")
            
            txt_path = os.path.join(TEMP_DIR, "transcript.txt")
            self._save_transcript_text(all_segments, txt_path)
            print(f"转录文本已保存: {txt_path}")
            
            print("="*50 + "\n")
        
        return all_segments if all_segments else None
    
    def _format_srt_time(self, seconds: float) -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds - int(seconds)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
    
    def _save_srt(self, segments: List[Dict], output_path: str):
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, seg in enumerate(segments, 1):
                f.write(f"{i}\n")
                f.write(f"{self._format_srt_time(seg['start'])} --> {self._format_srt_time(seg['end'])}\n")
                f.write(f"{seg['text'].strip()}\n")
                f.write("\n")
    
    def _save_transcript_text(self, segments: List[Dict], output_path: str):
        with open(output_path, 'w', encoding='utf-8') as f:
            for seg in segments:
                f.write(f"[{self._format_srt_time(seg['start'])}] {seg['text'].strip()}\n")
    
    def analyze_audio_first(self, audio_segments: List[Dict], user_request: str) -> List[Dict]:
        if not audio_segments:
            return []
        
        transcript_text = "\n".join([
            f"[{seg['start']:.1f}s - {seg['end']:.1f}s] {seg['text']}"
            for seg in audio_segments
        ])
        
        prompt = f"""
你是一位专业的视频内容分析助手。根据用户需求，从语音转录中找出【真正相关】的时间段。

【核心原则】：
1. 理解用户的具体需求意图
2. 扫描转录内容，找出【直接对应用户需求】的段落
3. 只返回【确实相关】的时间段，不返回"可能有用"的段落
4. 如果找不到相关内容，返回空列表

用户剪辑需求：{user_request}

语音转录内容：
{transcript_text}

请以JSON格式返回：
{{
    "suggested_segments": [
        {{"start": 开始时间, "end": 结束时间, "reason": "这段为什么符合用户需求"}}
    ]
}}

如果找不到相关内容，返回 {{"suggested_segments": []}}
"""

        
        try:
            response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "你是一个专业的视频内容分析助手，擅长从语音转录中识别关键内容和剪辑点。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2000
            )
            
            content = response.choices[0].message.content
            
            try:
                json_str = content
                if '```json' in content:
                    json_str = content.split('```json')[1].split('```')[0]
                elif '```' in content:
                    json_str = content.split('```')[1].split('```')[0]
                
                result = json.loads(json_str.strip())
                return result.get('suggested_segments', [])
            except json.JSONDecodeError:
                print("AI 返回格式错误，尝试手动解析...")
                return []
                
        except Exception as e:
            print(f"AI 分析失败: {e}")
            return []
    
    def _parse_json_response(self, content: str) -> Optional[Dict]:
        """
        尝试多种策略解析 AI 返回的 JSON 响应
        处理 AI 可能返回的各种格式问题（如未转义的引号、被截断的 JSON 等）
        """
        json_str = content.strip()
        
        # 策略1：从 markdown 代码块中提取 JSON（可能被截断）
        if '```json' in content:
            json_str = content.split('```json')[1]
            # 移除可能存在的结束标记或保留截断的内容
            if '```' in json_str:
                json_str = json_str.split('```')[0]
            json_str = json_str.strip()
        elif '```' in content:
            json_str = content.split('```')[1]
            if '```' in json_str:
                json_str = json_str.split('```')[0]
            json_str = json_str.strip()
        else:
            # 策略2：查找 JSON 对象边界
            start = content.find('{')
            end = content.rfind('}') + 1
            if start != -1 and end > start:
                json_str = content[start:end].strip()
        
        # 策略3：直接解析
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
        
        # 策略4：尝试修复常见的 JSON 问题
        try:
            import re
            
            # 移除可能的 BOM 标记
            fixed_str = json_str.encode().decode('utf-8-sig')
            
            # 第一次尝试：直接解析
            try:
                return json.loads(fixed_str)
            except json.JSONDecodeError:
                pass
            
            # 处理中文引号被转义的问题（""替换为""）
            fixed_str = fixed_str.replace('\\u201c', '"').replace('\\u201d', '"')
            
            try:
                return json.loads(fixed_str)
            except json.JSONDecodeError:
                pass
            
            # 移除可能的控制字符
            fixed_str = ''.join(c if ord(c) >= 32 or c in '\n\r\t' else '' for c in fixed_str)
            
            try:
                return json.loads(fixed_str)
            except json.JSONDecodeError:
                pass
            
        except Exception as e:
            pass
        
        # 策略5：处理被截断的 JSON - 尝试修复不完整的结构
        try:
            # 统计括号匹配
            open_braces = json_str.count('{')
            close_braces = json_str.count('}')
            open_brackets = json_str.count('[')
            close_brackets = json_str.count(']')
            
            fixed_str = json_str
            
            # 如果有未闭合的括号，尝试添加
            if open_braces > close_braces:
                fixed_str += '}' * (open_braces - close_braces)
            if open_brackets > close_brackets:
                fixed_str += ']' * (open_brackets - close_brackets)
            
            # 尝试解析修复后的 JSON
            result = json.loads(fixed_str)
            print(f"✓ 通过补完括号成功解析 JSON")
            return result
            
        except Exception as e:
            pass
        
        # 策略6：处理被截断的值 - 查找最后一个完整的对象/数组
        try:
            # 方法1: 找最后一个完整的匹配 "}]" 或 "}}"
            for end_pattern in ['}]', '}}', ']', '}']:
                if end_pattern in json_str:
                    last_idx = json_str.rfind(end_pattern)
                    if last_idx != -1:
                        # 从后向前查找开始位置
                        test_str = json_str[:last_idx+len(end_pattern)]
                        
                        # 如果是 }]，试图找 [ 或 {
                        if end_pattern == '}]':
                            # 这表示数组中的对象
                            bracket_pos = test_str.rfind('[')
                            if bracket_pos != -1:
                                test_str = test_str[:bracket_pos+1] + test_str[bracket_pos+1:]
                        
                        try:
                            result = json.loads(test_str)
                            print(f"✓ 通过截断到最后完整对象成功解析 JSON")
                            return result
                        except:
                            pass
        except Exception as e:
            pass
        
        # 策略7: 处理被截断的 matching_segments - 找最后完整的 }
        try:
            segments_pos = json_str.rfind('"matching_segments"')
            if segments_pos != -1:
                # 在这之后寻找 [
                bracket_start = json_str.find('[', segments_pos)
                if bracket_start != -1:
                    # 从后往前找最后一个完整的 }
                    brace_level = 0
                    last_complete_brace = -1
                    
                    for i in range(len(json_str) - 1, bracket_start - 1, -1):
                        if json_str[i] == '}':
                            brace_level += 1
                        elif json_str[i] == '{':
                            brace_level -= 1
                            if brace_level == 0:
                                last_complete_brace = i
                                break
                    
                    if last_complete_brace > bracket_start:
                        # 找到了最后一个完整的对象
                        # 现在构造一个有效的 JSON
                        try:
                            partial_json = (
                                '{'
                                '"analysis": "被截断的分析",'
                                '"matching_segments": ' + json_str[bracket_start:last_complete_brace+1] + '],'
                                '"recommendations": ""'
                                '}'
                            )
                            result = json.loads(partial_json)
                            print(f"✓ 通过截断处理成功解析 JSON")
                            return result
                        except Exception as e2:
                            pass
        except Exception as e:
            pass
        
        # 策略8: 处理截断的字符串值（最常见的截断点）
        try:
            # 找最后一个没有闭合的双引号
            last_quote_pos = json_str.rfind('"')
            if last_quote_pos != -1:
                # 检查这个引号是否被转义
                escaped = False
                for i in range(last_quote_pos - 1, -1, -1):
                    if json_str[i] == '\\':
                        escaped = not escaped
                    else:
                        break
                
                # 如果最后的引号没有被转义，说明可能是被截断的字符串的开始引号
                if not escaped and last_quote_pos == len(json_str.strip()) - 1:
                    # 尝试找最后一个完整的 }
                    for i in range(len(json_str) - 1, -1, -1):
                        if json_str[i] == '}':
                            # 找到了，尝试从这之前闭合字符串
                            test_str = json_str[:i+1]
                            
                            # 补全所有缺失的引号、括号
                            quote_count = 0
                            for c in test_str:
                                if c == '"' and (test_str.index(c) == 0 or test_str[test_str.index(c)-1] != '\\'):
                                    quote_count += 1
                            
                            if quote_count % 2 == 1:
                                test_str += '"'
                            
                            # 补全括号
                            open_braces = test_str.count('{')
                            close_braces = test_str.count('}')
                            open_brackets = test_str.count('[')
                            close_brackets = test_str.count(']')
                            
                            if open_braces > close_braces:
                                test_str += '}' * (open_braces - close_braces)
                            if open_brackets > close_brackets:
                                test_str += ']' * (open_brackets - close_brackets)
                            
                            try:
                                result = json.loads(test_str)
                                print(f"✓ 通过截断字符串修复成功解析 JSON")
                                return result
                            except:
                                pass
                            break
        except Exception as e:
            pass
        
        # 策略9: 最后尝试 - 移除最后的不完整行
        try:
            lines = json_str.split('\n')
            # 从后往前找，移除不完整的行
            while lines and not lines[-1].strip().endswith((',', ']', '}')):
                lines.pop()
            
            # 确保最后一行不以逗号结尾
            if lines and lines[-1].strip().endswith(','):
                lines[-1] = lines[-1].rstrip(',').rstrip()
            
            # 补全括号
            fixed_str = '\n'.join(lines)
            open_braces = fixed_str.count('{')
            close_braces = fixed_str.count('}')
            open_brackets = fixed_str.count('[')
            close_brackets = fixed_str.count(']')
            
            if open_braces > close_braces:
                fixed_str += '}' * (open_braces - close_braces)
            if open_brackets > close_brackets:
                fixed_str += ']' * (open_brackets - close_brackets)
            
            result = json.loads(fixed_str)
            print(f"✓ 通过行删除和补全成功解析 JSON")
            return result
        except Exception as e:
            pass
        
        print(f"JSON 解析失败（可能是响应被截断）")
        return None
    
    def analyze_content(self, frames: List[Dict], user_request: str, transcript_text: Optional[str] = None, subtitles: Optional[List[Dict]] = None) -> Dict:
        """
        综合分析视频内容（帧 + 音频转录 + 字幕），返回匹配片段和建议。
        这是 main.py 调用的主分析入口。
        
        【改进】只提供帧索引表给 AI，让 AI 通过 Agent 工具自己决定要查看哪些帧
        """
        if not frames:
            return {"analysis": "无可分析的帧", "matching_segments": [], "recommendations": ""}

        print(f"\n开始综合分析，共 {len(frames)} 帧...")

        # 生成帧的索引表
        frame_index_table = []
        
        for i, f in enumerate(frames):
            ts = f.get('timestamp', 0)
            path = f.get('path')
            
            # 提取文件名
            if path:
                filename = os.path.basename(str(path))
                # 添加到索引表
                frame_index_table.append({
                    "index": i,
                    "filename": filename,
                    "timestamp": ts,
                    "time_display": self._format_time(ts)
                })

        print(f"  [DEBUG] 生成帧索引表: {len(frame_index_table)} 帧", flush=True)
        print(f"  [DEBUG] 第一帧: {frame_index_table[0] if frame_index_table else 'N/A'}", flush=True)
        print(f"  [DEBUG] 最后一帧: {frame_index_table[-1] if frame_index_table else 'N/A'}", flush=True)
        
        # 计算帧间隔
        if len(frame_index_table) > 1:
            avg_interval = (frame_index_table[-1]['timestamp'] - frame_index_table[0]['timestamp']) / (len(frame_index_table) - 1)
            print(f"  [DEBUG] 帧间隔: {avg_interval:.1f}s", flush=True)
        else:
            avg_interval = 0

        print(f"  [DEBUG] 不预先编码帧，将由 Agent 通过工具决定查看哪些帧", flush=True)
        
        # 保存帧索引表和原始帧路径，供后续使用
        self._frame_index_table = frame_index_table
        self._frame_paths = {f.get('index', i): f.get('path') for i, f in enumerate(frames)}

        # 构建帧索引列表说明（告诉 Agent 所有可用的帧及其时间）
        frame_index_str = "【完整的视频帧索引表】\n"
        frame_index_str += "序号 | 时间(HH:MM:SS) | 秒数\n"
        frame_index_str += "-" * 45 + "\n"
        
        for i, fi in enumerate(frame_index_table[:80]):  # 显示前80帧
            frame_index_str += f"{fi['index']:3d}  | {fi['time_display']:13s} | {fi['timestamp']:8.1f}s\n"
        
        if len(frame_index_table) > 80:
            frame_index_str += f"... (共 {len(frame_index_table)} 帧，每帧间隔约 {avg_interval:.1f}s)\n"
        
        frame_index_str += f"\n【指导】\n"
        frame_index_str += f"1. 根据上表，你可以看到所有可用的帧及其对应的时间\n"
        frame_index_str += f"2. 当你需要查看某些帧来做决定时，使用工具调用来获取它们\n"
        frame_index_str += f"3. 帧间隔: 约 {avg_interval:.1f}s\n"
        frame_index_str += f"\n【重要】最终返回时间范围必须使用秒数格式：\n"
        frame_index_str += f"  - 帧序号 0 → 0.0 秒\n"
        frame_index_str += f"  - 帧序号 105 → {frame_index_table[min(105, len(frame_index_table)-1)]['timestamp']:.1f} 秒\n"

        # 构建 prompt - 仅包含帧索引表和文本上下文，不包含图像
        context_parts = []
        if transcript_text:
            context_parts.append(f"【语音转录内容】\n{transcript_text}")
        if subtitles:
            sub_text = "\n".join([f"[{s.get('start_time',0):.1f}s] {s.get('text','')}" for s in subtitles])
            context_parts.append(f"【字幕内容】\n{sub_text}")

        context_str = "\n\n".join(context_parts) if context_parts else "（无音频/字幕信息）"

        prompt = f"""你是一位专业的视频内容分析师。你的任务是根据帧索引表和用户需求，分析视频内容。

【用户需求】：{user_request}

【帧索引表 - 所有可用帧的列表】
{frame_index_str}

{context_str}

【你的任务】：
1. 基于帧索引表和音频/字幕信息，推断视频内容
2. 识别与用户需求相关的时间段
3. 返回匹配片段列表

【重要说明】：
- 根据帧索引表，你可以看到整个视频的时间结构和采样帧
- 需要查看具体帧画面时，你可以通过工具调用请求查看某些帧
- 当前只提供了帧索引表，具体帧图像由工具按需提供

【返回格式】（JSON）：
{{
    "analysis": "基于索引表分析的内容概述",
    "matching_segments": [
        {{
            "start_time": 秒数,
            "end_time": 秒数,
            "description": "片段描述",
            "relevance_score": 0-100
        }}
    ],
    "recommendations": "建议"
}}"""

        try:
            response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "你是一个专业的视频内容分析助手。你可以根据帧索引表、音频文本和字幕来分析视频内容。当需要查看具体帧时，你可以通过工具调用来获取它们。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=3000
            )

            content = response.choices[0].message.content.strip()
            
            # 使用新的 JSON 解析方法
            result = self._parse_json_response(content)
            
            if result is None:
                print(f"AI 返回 JSON 解析失败")
                print(f"原始返回: {content}")
                return {"analysis": "AI返回格式错误", "matching_segments": [], "recommendations": ""}

            # 确保必要字段存在
            if 'matching_segments' not in result:
                result['matching_segments'] = []
            if 'analysis' not in result:
                result['analysis'] = '分析完成'
            if 'recommendations' not in result:
                result['recommendations'] = ''

            print(f"AI 分析完成，找到 {len(result['matching_segments'])} 个匹配片段")
            return result

        except Exception as e:
            print(f"AI 分析失败: {e}")
            return {"analysis": f"分析失败: {e}", "matching_segments": [], "recommendations": ""}

    def analyze_frames_directly(self, frames: List[Dict], user_request: str, transcript_text: Optional[str] = None) -> Dict:
        """
        直接分析已有帧信息的函数（针对 Agent 工具调用）
        frames 包含完整的路径信息，可以直接编码和分析
        【关键】不生成帧索引表，直接使用帧的时间戳标注
        """
        if not frames:
            return {"analysis": "无可分析的帧", "matching_segments": [], "recommendations": ""}
        
        print(f"  [DEBUG] 直接分析 {len(frames)} 帧（已有完整路径，不重新生成索引）", flush=True)
        
        # 编码帧图像 - 使用原始的帧信息，不改变索引
        encoded_frames = {}
        for idx, f in enumerate(frames):
            path = f.get('path')
            if path and os.path.exists(path):
                try:
                    b64 = self.encode_image_to_base64(path)
                    # 使用帧的原始时间戳，不使用位置作为索引
                    timestamp = f.get('timestamp', 0)
                    encoded_frames[idx] = {
                        "base64": b64,
                        "timestamp": timestamp,
                        "time_display": self._format_time(timestamp),
                        "filename": os.path.basename(path)
                    }
                    print(f"  [DEBUG] 编码帧 {idx} ({timestamp:.1f}s, {os.path.basename(path)}) 成功", flush=True)
                except Exception as e:
                    print(f"  [DEBUG] 编码帧 {idx} 失败: {e}", flush=True)
        
        if not encoded_frames:
            print(f"  [DEBUG] 没有成功编码任何帧", flush=True)
            return {"analysis": "无可编码的帧", "matching_segments": [], "recommendations": ""}
        
        print(f"  [DEBUG] 成功编码 {len(encoded_frames)} 张帧，准备发送给 AI", flush=True)
        
        # 构建 prompt
        context_str = ""
        if transcript_text:
            context_str = f"【语音转录内容】\n{transcript_text}\n\n"
        
        prompt = f"""你是一位专业的视频内容分析师。你的任务是分析下面的视频帧，找出符合用户需求的片段。

【用户需求】：{user_request}

{context_str}

【分析指导】：
1. 仅基于【画面中能看到】的内容做出判断
2. 查找与用户需求【直接相关】的画面内容
3. 对于连续的相关帧，合并为一个长片段
4. 不要猜测或假设，只基于实际看到的内容

【返回JSON格式】：
{{
    "analysis": "基于画面分析：发现了哪些与需求相关的内容",
    "matching_segments": [
        {{
            "start_time": 秒数,
            "end_time": 秒数,
            "description": "具体描述看到了什么",
            "relevance_score": 0-100的整数
        }}
    ],
    "recommendations": "剪辑建议"
}}"""
        
        try:
            # 构建消息内容：文本 + 帧图像
            message_content = [
                {"type": "text", "text": prompt}
            ]
            
            # 按顺序添加帧图像
            for idx in sorted(encoded_frames.keys()):
                frame_info = encoded_frames[idx]
                image_label = f"\n[帧 {idx} | 时间: {frame_info['time_display']} ({frame_info['timestamp']:.1f}s)]\n"
                
                message_content.append({
                    "type": "text",
                    "text": image_label
                })
                message_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{frame_info['base64']}"
                    }
                })
            
            print(f"  [DEBUG] 发送 {len(encoded_frames)} 张帧给 AI 进行分析...", flush=True)
            
            response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "你是一个专业的视频内容分析助手，擅长从实际的视频帧画面中识别关键内容。只返回合法的JSON格式结果。"},
                    {"role": "user", "content": message_content}
                ],
                temperature=0.5,
                max_tokens=2000
            )
            
            content = response.choices[0].message.content.strip()
            print(f"  [DEBUG] AI 返回内容长度: {len(content)} 字符", flush=True)
            
            # 解析 JSON
            result = self._parse_json_response(content)
            
            if result is None:
                print(f"  [DEBUG] JSON 解析失败，原始返回: {content[:300]}", flush=True)
                return {"analysis": "分析完成但解析失败", "matching_segments": [], "recommendations": ""}
            
            # 确保字段存在
            if 'matching_segments' not in result:
                result['matching_segments'] = []
            if 'analysis' not in result:
                result['analysis'] = '分析完成'
            if 'recommendations' not in result:
                result['recommendations'] = ''
            
            print(f"  [DEBUG] 分析完成，找到 {len(result['matching_segments'])} 个匹配片段", flush=True)
            return result
            
        except Exception as e:
            print(f"  [DEBUG] 分析失败: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return {"analysis": f"分析失败: {e}", "matching_segments": [], "recommendations": ""}

    def analyze_video_frames(self, frame_paths: List[str], audio_context: str = "", user_request: str = "") -> List[Dict]:
        if not frame_paths:
            return []
        
        print(f"\n分析 {len(frame_paths)} 帧画面...")
        
        base64_images = []
        for path in frame_paths[:10]:
            try:
                base64_images.append(self.encode_image_to_base64(path))
            except:
                continue
        
        if not base64_images:
            return []
        
        prompt = f"""
你是一位专业的视频内容分析助手。请分析这些视频帧画面，结合语音内容给出剪辑建议。

用户剪辑需求：{user_request}

语音内容摘要：{audio_context}

请分析画面内容，识别：
1. 视觉重点（人物表情、动作、场景变化）
2. 与语音内容的匹配度
3. 建议保留或删除的画面

以JSON格式返回：
{{
    "visual_analysis": "画面内容描述",
    "editing_suggestions": [
        {{"type": "keep/remove", "reason": "原因", "confidence": 0.8}}
    ]
}}
"""
        
        try:
            messages = [
                {"role": "system", "content": "你是专业的视频内容分析助手。"},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    *[{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}} for img in base64_images]
                ]}
            ]
            
            response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.7,
                max_tokens=1500
            )
            
            return []
            
        except Exception as e:
            print(f"画面分析失败: {e}")
            return []
