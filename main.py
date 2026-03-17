import os
import json
import shutil
import subprocess
from pathlib import Path
from typing import Tuple, List, Optional, Dict
import gradio as gr
import cv2

from config import OPENAI_API_KEY, TEMP_DIR, OUTPUT_DIR, OPENAI_BASE_URL, MODEL_NAME
from video_processor import VideoProcessor
from ai_analyzer import AIAnalyzer
from video_editor import VideoEditor
from subtitle_parser import parse_srt, get_subtitle_at_time
from agent_analyzer import AgentAnalyzer

try:
    import imageio_ffmpeg
    FFMPEG_PATH = imageio_ffmpeg.get_ffmpeg_exe()
except:
    FFMPEG_PATH = 'ffmpeg'


SESSION_FILE = os.path.join("temp", "session.json")


def save_session(data: dict):
    """保存会话状态到 temp/session.json"""
    os.makedirs("temp", exist_ok=True)
    try:
        with open(SESSION_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"保存会话失败: {e}")


def load_session() -> dict:
    """从 temp/session.json 加载会话状态"""
    if not os.path.exists(SESSION_FILE):
        return {}
    try:
        with open(SESSION_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"加载会话失败: {e}")
        return {}


def check_session() -> Tuple[bool, str]:
    """检查是否有可恢复的会话，返回 (是否可恢复, 描述信息)"""
    session = load_session()
    if not session:
        return False, ""

    video_path = session.get("video_path", "")
    
    # 如果原路径不存在，尝试使用 temp/input_video.mp4
    if not video_path or not os.path.exists(video_path):
        temp_video = os.path.join(TEMP_DIR, "input_video.mp4")
        if os.path.exists(temp_video):
            video_path = temp_video
        else:
            return False, ""

    frames = session.get("extracted_frames", [])
    valid_frames = [f for f in frames if os.path.exists(f.get("path", ""))]
    audio_segments = session.get("audio_segments")
    video_info = session.get("video_info", {})

    # 即使帧文件丢失，也可以恢复（会重新提取）
    video_name = os.path.basename(video_path)
    duration = video_info.get("duration_formatted", "未知时长")
    audio_info = f"，音频 {len(audio_segments)} 段" if audio_segments else ""
    
    if not valid_frames:
        msg = f"发现上次会话：{video_name}（{duration}）{audio_info}，但帧文件已丢失，需要重新提取"
    else:
        msg = f"发现上次会话：{video_name}（{duration}），{len(valid_frames)} 帧{audio_info}"
    
    return True, msg


class AiCutApp:
    def __init__(self):
        self.video_processor = None
        self.ai_analyzer = None
        self.analysis_result = None
        self.video_path = None
        self.extracted_frames = None
        self.subtitles = None
        self.audio_segments = None
        
    def check_api_key(self) -> Tuple[bool, str]:
        if not OPENAI_API_KEY or OPENAI_API_KEY == "your_api_key_here":
            return False, "请先在 .env 文件中配置 OPENAI_API_KEY"
        return True, "API密钥已配置"
    
    def process_video(self, video_file, subtitle_file, video_description: str = "", progress=gr.Progress()) -> Tuple[str, str]:
        if not video_file:
            return "请上传视频文件", ""
        
        self.last_video_description = video_description
        
        progress(0.1, desc="正在初始化...")
        
        api_check, msg = self.check_api_key()
        if not api_check:
            return msg, ""
        
        self.cleanup_temp()
        
        progress(0.15, desc="正在处理视频文件...")
        
        if hasattr(video_file, 'name'):
            video_path = video_file.name
        else:
            video_path = video_file
        
        video_ext = os.path.splitext(video_path)[1]
        if not video_ext:
            video_ext = ".mp4"
        
        safe_video_path = os.path.join(TEMP_DIR, f"input_video{video_ext}")
        
        print(f"原始路径: {video_path}")
        print(f"复制到: {safe_video_path}")
        
        try:
            shutil.copy2(video_path, safe_video_path)
            print("文件复制成功")
        except Exception as e:
            print(f"文件复制失败: {e}")
            return f"文件复制失败: {e}", ""
        
        if subtitle_file:
            try:
                if hasattr(subtitle_file, 'name'):
                    srt_path = subtitle_file.name
                else:
                    srt_path = subtitle_file
                self.subtitles = parse_srt(srt_path)
                print(f"加载了 {len(self.subtitles)} 条字幕")
            except Exception as e:
                print(f"字幕加载失败：{e}")
                self.subtitles = None
        
        # 清理旧的帧分析结果
        frame_analysis_file = os.path.join(TEMP_DIR, "frame_analysis.json")
        if os.path.exists(frame_analysis_file):
            try:
                os.remove(frame_analysis_file)
                print(f"已删除旧的帧分析文件：{frame_analysis_file}")
            except Exception as e:
                print(f"删除旧文件失败：{e}")
        
        # 清理旧的帧图片
        print("正在清理旧的帧图片...")
        for filename in os.listdir(TEMP_DIR):
            if filename.endswith('.jpg'):
                try:
                    os.remove(os.path.join(TEMP_DIR, filename))
                except:
                    pass
        print("旧帧图片清理完成")
        
        progress(0.2, desc="正在分析视频信息...")
        
        self.video_path = safe_video_path
        self.video_processor = VideoProcessor(safe_video_path)
        video_info = self.video_processor.get_video_info()
        
        info_text = f"""视频信息:
- 时长: {video_info.get('duration_formatted', 'N/A')}
- 帧率: {video_info.get('fps', 0)} FPS
- 分辨率: {video_info.get('width', 0)} x {video_info.get('height', 0)}
- 总帧数: {video_info.get('total_frames', 0)}
"""
        if self.subtitles:
            info_text += f"\n- 字幕条数: {len(self.subtitles)}"
        
        if video_info.get('duration', 0) <= 0:
            return "无法读取视频信息，请检查视频文件是否有效", ""
        
        def update_progress(p, desc):
            progress(0.25 + p * 0.50, desc=desc)
        
        # 提取帧并分析内容 - 使用 pHash 智能帧分析器
        progress(0.25, desc="正在智能提取视频帧并分析...")
        print("\n" + "="*70)
        print("【开始智能帧分析：pHash 逐级细化 + 流式处理】")
        print("="*70)
        
        from smart_frame_analyzer_v2 import SmartFrameAnalyzer
        
        smart_analyzer = SmartFrameAnalyzer(
            safe_video_path, 
            api_key=OPENAI_API_KEY if OPENAI_API_KEY and OPENAI_API_KEY != "your_api_key_here" else "",
            output_dir=TEMP_DIR
        )
        
        # 流式处理：边提取帧边分析
        self.frame_analysis = []
        self.extracted_frames = []
        
        try:
            for result in smart_analyzer.analyze_streaming(
                video_description=video_description,
                similarity_threshold=0.85
            ):
                if result.get('success'):
                    self.frame_analysis.append(result)
                    self.extracted_frames.append({
                        "path": result['image_path'],
                        "timestamp": result['timestamp'],
                        "time_str": result['timestamp']
                    })
        finally:
            smart_analyzer.close()
        
        # 更新进度
        progress(0.80, desc="帧分析完成")
        
        
        # 收集提取的帧信息（从分析结果中提取）
        self.extracted_frames = []
        for analysis in self.frame_analysis:
            if analysis.get('success'):
                self.extracted_frames.append({
                    "path": analysis['image_path'],
                    "timestamp": analysis['timestamp'],
                    "time_str": analysis['timestamp']
                })
        
        self.video_processor.close()
        self.video_processor = None
        
        success_count = sum(1 for f in self.frame_analysis if f.get('success'))
        info_text += f"\n已提取并分析 {success_count}/{len(self.frame_analysis)} 帧内容"
        print(f"\n帧分析完成：{success_count}/{len(self.frame_analysis)} 帧成功")
        
        if not self.subtitles:
            progress(0.85, desc="正在提取音频并转录...")
            self.ai_analyzer = AIAnalyzer()
            self.audio_segments = self.ai_analyzer.transcribe_audio_with_timestamps(self.video_path)
            if self.audio_segments:
                info_text += f"\n已转录 {len(self.audio_segments)} 个音频片段"
            else:
                info_text += f"\n音频转录失败或无音频"
        
        # 保存会话状态
        save_session({
            "video_path": self.video_path,
            "extracted_frames": self.extracted_frames,
            "frame_analysis": self.frame_analysis,
            "audio_segments": self.audio_segments,
            "video_info": video_info,
            "user_request": getattr(self, 'last_user_request', ''),  # 保存用户请求
            "video_description": getattr(self, 'last_video_description', ''),  # 保存视频简介
        })

        progress(1.0, desc="处理完成")
        return info_text, f"素材已准备完成，可以开始分析"
    
    def restore_session(self, video_description: str = "", progress=gr.Progress()) -> Tuple[str, str, str, str, str, str, list, str]:
        """从 temp 目录恢复上次会话，返回 (video_info, status, video_description, user_request, analysis_result_text, segments_json, gallery_data, segments_json_edit)"""
        session = load_session()
        if not session:
            return "❌ 没有找到可恢复的会话", "无会话", "", "", "", "", [], ""

        video_path = session.get("video_path", "")
        if not video_path or not os.path.exists(video_path):
            # 尝试使用 temp/input_video.mp4
            temp_video = os.path.join(TEMP_DIR, "input_video.mp4")
            if os.path.exists(temp_video):
                video_path = temp_video
                print(f"使用备份视频：{video_path}")
            else:
                return "❌ 上次会话的视频文件已不存在，无法恢复", "视频文件丢失", "", "", "", "", [], ""

        # 恢复用户输入
        saved_user_request = session.get("user_request", "")
        saved_video_description = session.get("video_description", "")
        
        # 如果用户没有输入视频简介，使用保存的
        if not video_description and saved_video_description:
            video_description = saved_video_description
            print(f"📝 已恢复视频简介：{video_description[:50]}...")

        frames = session.get("extracted_frames", [])
        valid_frames = [f for f in frames if os.path.exists(f.get("path", ""))]
        
        # 检查是否需要重新提取帧
        need_reextract = False
        if not valid_frames:
            need_reextract = True
            print("帧文件丢失，需要重新提取")
        
        # 检查 frame_analysis.json 是否存在且完整
        json_file = os.path.join(TEMP_DIR, "frame_analysis.json")
        need_reanalyze = False
        
        if os.path.exists(json_file):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    analysis_data = json.load(f)
                
                frames_in_json = analysis_data.get('frames', [])
                total_frames = analysis_data.get('total_frames', 0)
                
                print(f"\n检测到 frame_analysis.json: {len(frames_in_json)} 帧")
                
                # 检查是否完整
                if len(frames_in_json) < len(valid_frames):
                    need_reanalyze = True
                    print(f"⚠️ 帧分析不完整：JSON 中有 {len(frames_in_json)} 帧，但已提取 {len(valid_frames)} 帧")
                else:
                    # 完整，直接加载
                    self.frame_analysis = frames_in_json
                    print(f"✅ 帧分析已加载（{len(frames_in_json)} 帧）")
            except Exception as e:
                print(f"⚠️ 加载 frame_analysis.json 失败：{e}")
                need_reanalyze = True
        else:
            print("\n未检测到 frame_analysis.json，需要重新分析")
            need_reanalyze = True
        
        # 恢复状态
        self.video_path = video_path
        video_info = session.get("video_info", {})
        
        if need_reextract:
            # 帧文件丢失，需要重新提取 - 使用 pHash 智能提取（但不分析）
            progress(0.2, desc="正在智能提取视频帧...")
            print("\n【智能提取帧（使用 pHash 过滤，暂不分析）】")
            
            # 重要：先备份 frame_analysis.json（如果存在）
            if os.path.exists(json_file):
                backup_file = json_file + ".backup"
                try:
                    shutil.copy2(json_file, backup_file)
                    print(f"✅ 已备份帧分析文件：{backup_file}")
                except Exception as e:
                    print(f"⚠️ 备份失败：{e}")
            
            # 使用智能帧分析器提取帧（但不分析）
            from smart_frame_analyzer_v2 import SmartFrameAnalyzer
            
            # 存储结果
            self.extracted_frames = []
            self.frame_analysis = []  # 先空着，等点击"分析视频"时再分析
            self.audio_segments = session.get("audio_segments")
            
            smart_analyzer = SmartFrameAnalyzer(
                video_path,
                api_key="",  # 不需要 API Key，因为不分析
                output_dir=TEMP_DIR
            )
            
            # 只提取帧，不分析（模拟分析过程但不调用 AI）
            print(f"\n开始 pHash 智能帧提取...")
            print(f"视频时长：{smart_analyzer.duration:.1f}秒")
            print(f"相似度阈值：85%")
            print("="*80)
            
            # 逐级细化的间隔（秒）
            levels = [
                (300, 0.6),   # Level 1: 5 分钟，阈值 60%
                (60, 0.5),    # Level 2: 1 分钟，阈值 50%
                (5, 0.4),     # Level 3: 5 秒，阈值 40%
                (1, 0.3),     # Level 4: 1 秒，阈值 30%
            ]
            
            # 存储已分析的帧
            analyzed_frames = []  # [(timestamp, hash)]
            changes_detected = []  # [(timestamp, similarity)]
            
            total_samples = 0
            skip_count = 0
            
            # Level 1: 5 分钟一帧
            print("\n【Level 1: 5 分钟一帧 - 全视频扫描】")
            print("-"*80)
            
            interval = 300  # 5 分钟
            timestamp = 0
            
            while timestamp < smart_analyzer.duration:
                frame_idx = int(timestamp * smart_analyzer.fps)
                smart_analyzer.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = smart_analyzer.cap.read()
                
                if not ret:
                    break
                
                total_samples += 1
                timestamp_str = f"{int(timestamp)//60:02d}:{int(timestamp)%60:02d}"
                
                # 计算 pHash
                current_hash = smart_analyzer.compute_phash(frame)
                
                # 判断是否保留
                should_keep = True
                
                if analyzed_frames:
                    # 与上一帧比较
                    last_timestamp, last_hash = analyzed_frames[-1]
                    similarity = smart_analyzer.calculate_similarity(last_hash, current_hash)
                    
                    if similarity > 0.85:
                        # 相似，跳过
                        should_keep = False
                        skip_count += 1
                
                # 保存帧
                if should_keep:
                    # 保存帧文件
                    frame_filename = os.path.join(TEMP_DIR, f"frame_{int(timestamp)}s.jpg")
                    cv2.imwrite(str(frame_filename), frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
                    
                    # 记录
                    analyzed_frames.append((timestamp, current_hash))
                    
                    if len(analyzed_frames) <= 20:
                        print(f"📸 帧 {len(analyzed_frames):3d} [{timestamp_str}] - 已保存")
                
                timestamp += interval
            
            # 检查是否有变化
            if len(analyzed_frames) > 1:
                for i in range(1, len(analyzed_frames)):
                    sim = smart_analyzer.calculate_similarity(analyzed_frames[i-1][1], analyzed_frames[i][1])
                    if sim < 0.6:
                        changes_detected.append((analyzed_frames[i][0], sim))
            
            # Level 2: 如果有变化，细化分析
            if changes_detected:
                print(f"\n【Level 2: 1 分钟一帧 - 细化 {len(changes_detected)} 个变化区域】")
                print("-"*80)
                
                # 记录已保存的时间戳，避免重复
                saved_timestamps = set(ts for ts, _ in analyzed_frames)
                
                for change_ts, change_sim in changes_detected:
                    start_sec = max(0, change_ts - 300)
                    end_sec = min(smart_analyzer.duration, change_ts + 300)
                    
                    timestamp = start_sec
                    prev_hash = None
                    
                    while timestamp <= end_sec:
                        # 如果这个时间点已经保存过，跳过
                        if int(timestamp) in saved_timestamps:
                            timestamp += 60
                            continue
                        
                        frame_idx = int(timestamp * smart_analyzer.fps)
                        smart_analyzer.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                        ret, frame = smart_analyzer.cap.read()
                        
                        if not ret:
                            break
                        
                        total_samples += 1
                        current_hash = smart_analyzer.compute_phash(frame)
                        
                        if prev_hash is None:
                            # 第一帧
                            frame_filename = os.path.join(TEMP_DIR, f"frame_{int(timestamp)}s.jpg")
                            cv2.imwrite(str(frame_filename), frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
                            analyzed_frames.append((timestamp, current_hash))
                            saved_timestamps.add(int(timestamp))
                            print(f"  📸 帧 {len(analyzed_frames)} [{int(timestamp)//60:02d}:{int(timestamp)%60:02d}] - 变化点分析")
                        else:
                            similarity = smart_analyzer.calculate_similarity(prev_hash, current_hash)
                            if similarity < 0.5:
                                # 有变化，保存
                                frame_filename = os.path.join(TEMP_DIR, f"frame_{int(timestamp)}s.jpg")
                                cv2.imwrite(str(frame_filename), frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
                                analyzed_frames.append((timestamp, current_hash))
                                saved_timestamps.add(int(timestamp))
                                print(f"  📸 帧 {len(analyzed_frames)} [{int(timestamp)//60:02d}:{int(timestamp)%60:02d}] - 变化点分析")
                        
                        prev_hash = current_hash
                        timestamp += 60
            
            smart_analyzer.close()
            
            # 构建 extracted_frames 列表
            for timestamp, _ in analyzed_frames:
                timestamp_str = f"{int(timestamp)//60:02d}:{int(timestamp)%60:02d}"
                frame_filename = os.path.join(TEMP_DIR, f"frame_{int(timestamp)}s.jpg")
                self.extracted_frames.append({
                    "path": str(frame_filename),
                    "timestamp": timestamp_str,
                    "time_str": timestamp_str
                })
            
            # 更新进度
            progress(0.80, desc="帧提取完成")
            
            # 检查是否需要重新分析
            if need_reanalyze:
                self.frame_analysis = []  # 需要重新分析
                info_text = f"""✅ 会话已恢复（帧已重新提取）！

视频信息:
- 文件：{os.path.basename(video_path)}
- 时长：{video_info.get('duration_formatted', 'N/A')}
- 帧率：{video_info.get('fps', 0)} FPS
- 分辨率：{video_info.get('width', 0)} x {video_info.get('height', 0)}
- 已提取：{len(self.extracted_frames)} 帧（使用 pHash 智能过滤）
- 视频简介：{video_description[:50] if video_description else '无'}...

⚠️ 帧分析不完整，请点击"🔍 分析视频"重新分析！"""
            else:
                info_text = f"""✅ 会话已恢复（帧已重新提取）！

视频信息:
- 文件：{os.path.basename(video_path)}
- 时长：{video_info.get('duration_formatted', 'N/A')}
- 帧率：{video_info.get('fps', 0)} FPS
- 分辨率：{video_info.get('width', 0)} x {video_info.get('height', 0)}
- 已提取：{len(self.extracted_frames)} 帧（使用 pHash 智能过滤）
- 视频简介：{video_description[:50] if video_description else '无'}...

请点击"🔍 分析视频"开始分析！"""
            
            print(f"\n【阶段 1 完成】")
            print(f"  总采样：{total_samples} 帧")
            print(f"  保留：{len(analyzed_frames)} 帧 ({len(analyzed_frames)/total_samples*100:.1f}%)")
            print(f"  跳过：{skip_count} 帧 ({skip_count/total_samples*100:.1f}%)")
            
            # 恢复备份的 frame_analysis.json（如果存在）
            backup_file = json_file + ".backup"
            if os.path.exists(backup_file):
                try:
                    shutil.copy2(backup_file, json_file)
                    os.remove(backup_file)
                    
                    # 加载备份的分析结果
                    with open(json_file, 'r', encoding='utf-8') as f:
                        analysis_data = json.load(f)
                    self.frame_analysis = analysis_data.get('frames', [])
                    
                    print(f"✅ 已恢复帧分析文件：{len(self.frame_analysis)} 帧")
                    
                    # 重新检查完整性
                    if len(self.frame_analysis) >= len(analyzed_frames):
                        need_reanalyze = False
                        print(f"✅ 帧分析完整，不需要重新分析")
                    else:
                        need_reanalyze = True
                        print(f"⚠️ 帧分析不完整：{len(self.frame_analysis)}/{len(analyzed_frames)} 帧")
                except Exception as e:
                    print(f"⚠️ 恢复备份失败：{e}")
                    need_reanalyze = True
            else:
                # 没有备份，需要重新分析
                need_reanalyze = True
            
            # 恢复 AI 分析结果（如果有）
            saved_analysis = session.get("analysis_result")
            saved_analysis_text = session.get("analysis_result_text", "")
            saved_segments_json = ""
            if saved_analysis:
                self.analysis_result = saved_analysis
                saved_segments_json = json.dumps(saved_analysis.get('matching_segments', []), ensure_ascii=False, indent=2)
                print(f"✅ 已恢复 AI 分析结果：{len(saved_analysis.get('matching_segments', []))} 个片段")
                info_text += "\n\n✅ **已恢复上次 AI 分析结果，可直接开始剪辑！**"

            print(f"\n会话已恢复（帧已重新提取）：{video_path}, {len(self.extracted_frames)} 帧")
            
            saved_gallery = []
            if saved_analysis:
                saved_gallery = self._build_gallery_from_session(saved_analysis)
            
            return info_text, "✅ 帧已提取，请点击分析视频", video_description, saved_user_request, saved_analysis_text, saved_segments_json, saved_gallery, saved_segments_json
        else:
            # 帧文件存在，直接恢复
            self.extracted_frames = valid_frames
            self.audio_segments = session.get("audio_segments")
            
            # 检查 frame_analysis.json 是否存在且完整
            json_file = os.path.join(TEMP_DIR, "frame_analysis.json")
            need_reanalyze = False
            
            if os.path.exists(json_file):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        analysis_data = json.load(f)
                    
                    frames_in_json = analysis_data.get('frames', [])
                    total_frames = analysis_data.get('total_frames', 0)
                    
                    print(f"\n检测到 frame_analysis.json: {len(frames_in_json)} 帧")
                    
                    # 检查是否完整
                    if len(frames_in_json) < len(valid_frames):
                        need_reanalyze = True
                        print(f"⚠️ 帧分析不完整：JSON 中有 {len(frames_in_json)} 帧，但已提取 {len(valid_frames)} 帧")
                        # 加载已有的分析结果
                        self.frame_analysis = frames_in_json
                        print(f"✅ 已加载 {len(self.frame_analysis)} 帧的分析结果（将补充分析缺失的帧）")
                    else:
                        # 完整，直接加载
                        self.frame_analysis = frames_in_json
                        print(f"✅ 帧分析已加载（{len(frames_in_json)} 帧）")
                except Exception as e:
                    print(f"⚠️ 加载 frame_analysis.json 失败：{e}")
                    need_reanalyze = True
                    self.frame_analysis = []
            else:
                print("\n未检测到 frame_analysis.json，需要重新分析")
                need_reanalyze = True
                self.frame_analysis = []

            audio_info = f"\n已恢复 {len(self.audio_segments)} 个音频片段" if self.audio_segments else "\n无音频转录数据"
            
            # 根据是否需要重新分析，显示不同提示
            if need_reanalyze:
                info_text = f"""✅ 已恢复上次会话！

视频信息:
- 文件：{os.path.basename(video_path)}
- 时长：{video_info.get('duration_formatted', 'N/A')}
- 帧率：{video_info.get('fps', 0)} FPS
- 分辨率：{video_info.get('width', 0)} x {video_info.get('height', 0)}
- 已恢复帧数：{len(valid_frames)} 帧（共 {len(frames)} 帧）{audio_info}
- 视频简介：{video_description[:50] if video_description else '无'}...

⚠️ 帧分析不完整，请点击"🔍 分析视频"重新分析！"""
            else:
                info_text = f"""✅ 已恢复上次会话！

视频信息:
- 文件：{os.path.basename(video_path)}
- 时长：{video_info.get('duration_formatted', 'N/A')}
- 帧率：{video_info.get('fps', 0)} FPS
- 分辨率：{video_info.get('width', 0)} x {video_info.get('height', 0)}
- 已恢复帧数：{len(valid_frames)} 帧（共 {len(frames)} 帧）{audio_info}
- 视频简介：{video_description[:50] if video_description else '无'}...

请点击"🔍 分析视频"开始分析！"""

            # 恢复 AI 分析结果（如果有）
            saved_analysis = session.get("analysis_result")
            saved_analysis_text = session.get("analysis_result_text", "")
            saved_segments_json = ""
            if saved_analysis:
                self.analysis_result = saved_analysis
                saved_segments_json = json.dumps(saved_analysis.get('matching_segments', []), ensure_ascii=False, indent=2)
                print(f"✅ 已恢复 AI 分析结果：{len(saved_analysis.get('matching_segments', []))} 个片段")
                info_text += "\n\n✅ **已恢复上次 AI 分析结果，可直接开始剪辑！**"

            print(f"会话已恢复：{video_path}, {len(valid_frames)} 帧")
            status = "✅ 会话恢复成功（含 AI 分析结果），可直接开始剪辑" if saved_analysis else "✅ 会话恢复成功，请点击分析视频"
            
            saved_gallery = []
            if saved_analysis:
                saved_gallery = self._build_gallery_from_session(saved_analysis)
            
            return info_text, status, video_description, saved_user_request, saved_analysis_text, saved_segments_json, saved_gallery, saved_segments_json

    def analyze_video(self, user_request: str, video_description: str = "", use_uniform: bool = True, use_audio: bool = True, use_subtitle: bool = True, progress=gr.Progress()) -> Tuple[str, str, str, list, str]:
        # 保存用户输入以便下次恢复
        self.last_user_request = user_request
        self.last_video_description = video_description
        
        # 立即保存到 session.json
        try:
            session = load_session()
            if session:
                session['user_request'] = user_request
                session['video_description'] = video_description
                save_session(session)
                print(f"📝 已保存用户输入到 session.json")
        except Exception as e:
            print(f"⚠️ 保存用户输入失败：{e}")
        
        if not self.video_path:
            return "请先上传并处理视频", "", "", [], ""
        
        if not user_request.strip():
            return "请输入您希望剪辑的内容描述", "", "", [], ""
        
        progress(0.1, desc="正在初始化 AI 分析器...")
        
        api_check, msg = self.check_api_key()
        if not api_check:
            return msg, "", "", [], ""
        
        if not self.ai_analyzer:
            self.ai_analyzer = AIAnalyzer()
        
        # 使用 Agent 架构进行分析
        print("\n" + "="*70)
        print("【启用 Agent 分析模式】")
        print("="*70)
        
        # 配置 Agent 的优先级
        agent_config = {
            "analysis_priority": [],
            "frame_sample_size": 10,
            "video_description": video_description.strip() if video_description else "",  # 添加视频简介
        }
        
        if use_audio:
            agent_config["analysis_priority"].append("audio")
        if use_subtitle:
            agent_config["analysis_priority"].append("subtitle")
        if use_uniform:
            agent_config["analysis_priority"].append("uniform")
        
        if not agent_config["analysis_priority"]:
            agent_config["analysis_priority"] = ["uniform"]
        
        print(f"Agent 分析优先级：{agent_config['analysis_priority']}")
        video_description = agent_config.get("video_description", "")
        if video_description:
            print(f"📝 用户提供的视频简介：{video_description[:100]}...")
        
        # 检查是否需要分析帧（完全没分析 或 部分缺失）
        need_analyze_frames = False
        existing_analysis_count = len(self.frame_analysis) if self.frame_analysis else 0
        total_frames = len(self.extracted_frames)
        
        if existing_analysis_count < total_frames:
            need_analyze_frames = True
            print(f"\n⚠️ 需要分析帧：已有 {existing_analysis_count}/{total_frames} 帧")
        
        # 如果需要分析帧，分析缺失的部分
        if need_analyze_frames and self.extracted_frames:
            progress(0.2, desc="正在分析视频帧...")
            print("\n" + "="*70)
            print("【分析缺失的帧：使用用户提供的视频简介】")
            print("="*70)
            
            # 使用 SmartFrameAnalyzer 分析已提取的帧（不重新提取，只分析）
            from smart_frame_analyzer_v2 import SmartFrameAnalyzer
            
            # 创建一个临时 analyzer 用于 AI 分析
            smart_analyzer = SmartFrameAnalyzer(
                self.video_path,
                api_key=OPENAI_API_KEY,
                output_dir=TEMP_DIR
            )
            
            # 如果已经有部分分析结果，先加载
            if self.frame_analysis:
                print(f"✅ 已加载 {len(self.frame_analysis)} 帧的分析结果")
                # 构建已分析帧的字典 {timestamp: analysis}
                analyzed_dict = {}
                for frame in self.frame_analysis:
                    ts = frame.get('timestamp')
                    if ts:
                        analyzed_dict[ts] = frame
                print(f"📋 已分析的时间戳：{list(analyzed_dict.keys())[:10]}...")
            else:
                analyzed_dict = {}
                self.frame_analysis = []
            
            # 只分析缺失的帧
            new_frame_analysis = []
            missing_count = 0
            
            print(f"\n开始检查 {total_frames} 帧...")
            
            for idx, frame_data in enumerate(self.extracted_frames, 1):
                frame_path = frame_data['path']
                timestamp = frame_data.get('timestamp', '')
                
                # 检查是否已经分析过
                if timestamp in analyzed_dict:
                    print(f"  ✓ 帧 {idx}/{total_frames} ({timestamp}) - 已分析，跳过")
                    continue
                
                missing_count += 1
                
                if not os.path.exists(frame_path):
                    print(f"  ⚠️ 帧文件不存在：{frame_path}")
                    continue
                
                print(f"\n📸 分析第 {idx}/{total_frames} 帧 ({timestamp})... [缺失]")
                
                # 读取帧图片
                frame = cv2.imread(frame_path)
                if frame is None:
                    print(f"  ❌ 无法读取帧：{frame_path}")
                    continue
                
                # 调用 AI 分析
                result = smart_analyzer.analyze_frame_with_ai(
                    frame, 
                    timestamp, 
                    video_description
                )
                
                if result.get('success'):
                    result['image_path'] = frame_path
                    # 转换为 Agent 期望的嵌套格式
                    agent_format_result = {
                        "timestamp": timestamp,
                        "image_path": frame_path,
                        "analysis": {
                            "success": True,
                            "analysis": result.get('analysis', {})
                        }
                    }
                    new_frame_analysis.append(agent_format_result)
                    analyzed_dict[timestamp] = agent_format_result  # 记录已分析
                    print(f"  ✅ 分析成功")
                else:
                    print(f"  ❌ 分析失败：{result.get('error', '未知错误')}")
                
                # 每分析一帧就立即保存
                json_file = os.path.join(TEMP_DIR, "frame_analysis.json")
                with open(json_file, 'w', encoding='utf-8') as f:
                    output_data = {
                        "frames": list(analyzed_dict.values()),
                        "total_frames": len(analyzed_dict.values())
                    }
                    json.dump(output_data, f, ensure_ascii=False, indent=2)
            
            smart_analyzer.close()
            
            # 合并结果
            self.frame_analysis = list(analyzed_dict.values())
            
            success_count = len(self.frame_analysis)
            print(f"\n帧分析完成：{success_count}/{total_frames} 帧成功 (新增 {missing_count} 帧)")
            print(f"✅ 帧分析已保存到：{json_file}")
        elif video_description and self.frame_analysis:
            # 已经有分析结果且有简介，检查是否需要重新分析
            print("\n✅ 帧已分析过，使用已有结果")
        
        # 初始化 Agent
        print("[DEBUG] 初始化 Agent 分析器...", flush=True)
        agent = AgentAnalyzer(ai_analyzer=self.ai_analyzer)
        print("[DEBUG] 设置 Agent 配置...", flush=True)
        agent.set_config(agent_config)
        print("[DEBUG] Agent 初始化完成", flush=True)
        
        progress(0.4, desc="Agent 正在分析视频（显示详细工具调用过程）...")
        
        print("\n" + "="*70)
        print("【Agent 实时日志输出】")
        print("="*70 + "\n")
        print("[DEBUG] 即将调用 agent.analyze_video()...", flush=True)
        print(f"[DEBUG] 参数: user_request={user_request[:50]}...", flush=True)
        print(f"[DEBUG] audio_segments={len(self.audio_segments or [])}, subtitles={len(self.subtitles or [])}, frames={len(self.extracted_frames or [])}", flush=True)
        
        # 执行 Agent 分析（不捕获输出，直接实时显示）
        try:
            print("[DEBUG] 开始执行 agent.analyze_video()...", flush=True)
            self.analysis_result = agent.analyze_video(
                user_request=user_request,
                audio_segments=self.audio_segments,
                subtitles=self.subtitles,
                frames=self.extracted_frames,
                video_path=self.video_path,
                video_description=agent_config.get("video_description", "")
            )
            print("[DEBUG] agent.analyze_video() 执行完成", flush=True)
        except Exception as e:
            print(f"[DEBUG] agent.analyze_video() 出错: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return f"Agent 分析失败: {e}", "", "", [], ""
        
        print("\n" + "="*70)
        print("【Agent 分析完成】")
        print("="*70)
        
        progress(0.9, desc="正在生成报告...")
        
        # 构建结果文本
        result_text = f"""## 🤖 Agent 分析结果

### 整体分析
{self.analysis_result.get('analysis', '无')}

### 匹配的片段 ({len(self.analysis_result.get('matching_segments', []))} 个)
"""
        
        for i, seg in enumerate(self.analysis_result.get('matching_segments', []), 1):
            result_text += f"""
**片段 {i}**:
- 时间: {seg.get('start_time', 0):.2f}s - {seg.get('end_time', 0):.2f}s
- 时长: {seg.get('end_time', 0) - seg.get('start_time', 0):.2f}秒
- 描述: {seg.get('description', '无')}
- 相关性评分: {seg.get('relevance_score', 0)}%
"""
        
        result_text += f"""
### 剪辑建议
{self.analysis_result.get('recommendations', '无')}
"""
        
        segments_json = json.dumps(self.analysis_result.get('matching_segments', []), ensure_ascii=False, indent=2)
        
        # 为每个片段找到对应的帧
        segment_frames = self._get_frames_for_segments(self.analysis_result.get('matching_segments', []))
        
        # 构建 Gallery 显示数据
        gallery_data = []
        for sf in segment_frames:
            seg = sf.get('segment', {})
            frames = sf.get('frames', [])
            for fr in frames:
                img_path = fr.get('path', '')
                if img_path:
                    img_path = img_path.replace('\\', '/')
                caption = f"{seg.get('reason', '')}\nAI描述: {fr.get('description', '')[:80]}"
                gallery_data.append((img_path, caption))
        
        print(f"[DEBUG] analyze_video gallery_data: {len(gallery_data)} 张图片")
        
        # 保存分析结果到 session.json，防止刷新丢失
        try:
            session = load_session()
            if session:
                session['analysis_result'] = self.analysis_result
                session['analysis_result_text'] = result_text
                session['segment_frames'] = segment_frames
                save_session(session)
                print("📝 已保存 AI 分析结果到 session.json")
        except Exception as e:
            print(f"⚠️ 保存分析结果失败：{e}")
        
        progress(1.0, desc="Agent 分析完成")
        
        return result_text, segments_json, "✅ Agent 分析完成，可以开始剪辑", gallery_data, segments_json
    
    def re_analyze(self, user_request: str, video_description: str = "", use_uniform: bool = True, use_audio: bool = True, use_subtitle: bool = True, progress=gr.Progress()) -> Tuple[str, str, str, list, str]:
        """重新分析视频（使用已提取的素材）"""
        return self.analyze_video(user_request, video_description, use_uniform, use_audio, use_subtitle, progress)
    
    def _get_frames_for_segments(self, segments: List[Dict]) -> List[Dict]:
        """为每个片段找到对应的帧图片和描述"""
        if not segments:
            print("[DEBUG] _get_frames_for_segments: segments 为空")
            return []
        
        # 从 temp 目录扫描所有帧文件
        temp_dir = Path(TEMP_DIR)
        all_frames = []
        for f in temp_dir.glob("frame_*.jpg"):
            try:
                name = f.stem.replace('frame_', '').replace('s', '')
                ts = float(name)
                all_frames.append({
                    'path': str(f).replace('\\', '/'),
                    'timestamp': ts,
                    'description': ''
                })
            except:
                pass
        
        # 也尝试从 session 加载已有的分析描述
        session = load_session()
        frame_analysis = session.get('frame_analysis', [])
        
        # 将 session 中的描述合并到帧数据中
        for fa in frame_analysis:
            try:
                ts_str = fa.get('timestamp', '0')
                if ':' in ts_str:
                    parts = ts_str.split(':')
                    ts = int(parts[0]) * 60 + int(parts[1])
                else:
                    ts = float(ts_str)
                
                for frame in all_frames:
                    if abs(frame['timestamp'] - ts) < 1:
                        analysis = fa.get('analysis', {})
                        frame['description'] = analysis.get('description', '')[:100]
                        break
            except:
                pass
        
        print(f"[DEBUG] _get_frames_for_segments: 总帧数 = {len(all_frames)}")
        
        segment_frames = []
        for seg in segments:
            start_time = seg.get('start_time', 0)
            end_time = seg.get('end_time', 0)
            reason = seg.get('reason', '')
            print(f"[DEBUG] 匹配片段: {start_time}s - {end_time}s ({reason})")
            
            matched_frames = []
            for frame in all_frames:
                ts = frame.get('timestamp', 0)
                if start_time - 5 <= ts <= end_time + 5:
                    matched_frames.append(frame)
                    print(f"[DEBUG]   匹配帧: {frame.get('path', '')} at {ts}s")
            
            if matched_frames:
                matched_frames.sort(key=lambda x: x['timestamp'])
                segment_frames.append({
                    'segment': seg,
                    'frames': matched_frames[:3],
                    'reason': reason
                })
        
        print(f"[DEBUG] _get_frames_for_segments 返回: {len(segment_frames)} 个片段")
        return segment_frames
    
    def _build_gallery_from_session(self, analysis_result: Dict) -> list:
        """从 session 中的分析结果构建 Gallery 数据"""
        print("[DEBUG] _build_gallery_from_session 被调用")
        segment_frames = self._get_frames_for_segments(analysis_result.get('matching_segments', []))
        
        gallery_data = []
        for sf in segment_frames:
            seg = sf.get('segment', {})
            frames = sf.get('frames', [])
            for fr in frames:
                img_path = fr.get('path', '')
                if img_path:
                    img_path = img_path.replace('\\', '/')
                caption = f"{seg.get('reason', '')}\nAI描述: {fr.get('description', '')[:80]}"
                gallery_data.append((img_path, caption))
        
        print(f"[DEBUG] gallery_data 构建完成: {len(gallery_data)} 张图片")
        return gallery_data
    
    def _extract_frames_for_ranges(self, time_ranges: List[Dict]) -> List[Dict]:
        """根据时间范围提取帧"""
        frames = []
        os.makedirs(TEMP_DIR, exist_ok=True)
        
        frame_idx = len([f for f in os.listdir(TEMP_DIR) if f.endswith('.jpg')])
        for tr in time_ranges:
            start = tr.get('start_time', 0)
            end = tr.get('end_time', 0)
            
            duration = end - start
            num_frames = max(3, min(10, int(duration / 2)))
            
            for i in range(num_frames):
                ts = start + (duration * i / num_frames)
                output_path = os.path.join(TEMP_DIR, f"frame_{frame_idx:05d}.jpg")
                
                cmd = [
                    FFMPEG_PATH, '-y', '-ss', str(ts),
                    '-i', self.video_path,
                    '-vframes', '1', '-q:v', '2',
                    output_path
                ]
                
                try:
                    subprocess.run(cmd, capture_output=True, timeout=10)
                    if os.path.exists(output_path):
                        frames.append({
                            "path": output_path,
                            "timestamp": round(ts, 2),
                            "frame_number": int(ts * 25)
                        })
                        frame_idx += 1
                except:
                    pass
        
        print(f"为 {len(time_ranges)} 个时间范围提取了 {len(frames)} 帧")
        return frames
    
    def _parse_json_response(self, content: str) -> Optional[Dict]:
        """
        尝试多种策略解析 AI 返回的 JSON 响应
        处理 AI 可能返回的各种格式问题（如未转义的引号等）
        """
        json_str = content.strip()
        
        # 策略1：从 markdown 代码块中提取 JSON
        if '```json' in content:
            json_str = content.split('```json')[1].split('```')[0].strip()
        elif '```' in content:
            json_str = content.split('```')[1].split('```')[0].strip()
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
            
            # 最后尝试：移除可能的控制字符
            fixed_str = ''.join(c if ord(c) >= 32 or c in '\n\r\t' else '' for c in fixed_str)
            return json.loads(fixed_str)
            
        except Exception as e:
            print(f"JSON 修复失败: {e}")
            return None
    
    def _analyze_subtitles(self, user_request: str) -> List[Dict]:
        """分析字幕内容，找出相关时间段"""
        if not self.subtitles:
            return []
        
        print("正在分析字幕内容...")
        
        subtitle_text = ""
        for sub in self.subtitles:
            subtitle_text += f"[{sub['start_time']:.1f}s] {sub['text']}\n"
        
        prompt = f"""你是一个专业的视频内容分析师。用户希望从视频中找到符合特定主题的片段。

用户需求: {user_request}

以下是视频的字幕内容（带时间戳）：
{subtitle_text}

请找出所有符合用户需求的时间段。返回JSON格式：
{{
    "relevant_segments": [
        {{
            "start_time": 开始时间(秒),
            "end_time": 结束时间(秒),
            "reason": "为什么这个片段相关",
            "confidence": 置信度(0-100)
        }}
    ]
}}

只返回JSON，不要其他文字。"""
        
        try:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
            
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
                temperature=0.3
            )
            
            result_text = response.choices[0].message.content
            result = self._parse_json_response(result_text)
            
            if result is None:
                print(f"字幕分析 JSON 解析失败")
                return []
            
            print(f"字幕分析完成，找到 {len(result.get('relevant_segments', []))} 个相关片段")
            return result.get('relevant_segments', [])
                
        except Exception as e:
            print(f"字幕分析错误: {e}")
        
        return []
    
    def edit_video(self, merge_mode: bool, min_relevance: int, fade_duration: float, progress=gr.Progress()) -> Tuple[str, Optional[str]]:
        if not self.video_path:
            return "请先上传视频", None
        
        if not self.analysis_result or not self.analysis_result.get('matching_segments'):
            return "请先进行视频分析", None
        
        progress(0.1, desc="正在初始化编辑器...")
        
        segments = self.analysis_result.get('matching_segments', [])
        
        filtered_segments = [
            seg for seg in segments 
            if seg.get("relevance_score", 0) >= min_relevance
        ]
        
        if not filtered_segments:
            return f"没有符合相关性评分>={min_relevance}的片段。当前片段评分: {[s.get('relevance_score', 0) for s in segments]}", None
        
        progress(0.2, desc=f"准备剪辑 {len(filtered_segments)} 个片段...")
        
        import traceback
        error_log = []
        
        def update_progress(p):
            progress(0.3 + p * 0.6, desc=f"正在导出视频... {p*100:.0f}%")
        
        try:
            editor = VideoEditor(self.video_path)
            
            if merge_mode:
                progress(0.3, desc="正在创建合集...")
                output_path = editor.create_compilation(
                    segments,
                    fade_duration=fade_duration,
                    min_relevance=min_relevance,
                    progress_callback=update_progress
                )
            else:
                progress(0.3, desc="正在创建独立片段...")
                output_paths = editor.create_individual_clips(segments, fade_duration)
                output_path = output_paths[0] if output_paths else None
            
        except Exception as e:
            error_msg = f"剪辑出错: {str(e)}\n\n详细信息:\n" + "\n".join(error_log)
            print(error_msg)
            traceback.print_exc()
            return error_msg, None
        
        progress(1.0, desc="剪辑完成")
        
        if output_path and os.path.exists(output_path):
            return f"视频已保存到: {output_path}", output_path
        else:
            return f"剪辑失败。日志:\n" + "\n".join(error_log), None
    
    def re_edit(self, merge_mode: bool, min_relevance: int, fade_duration: float, progress=gr.Progress()) -> Tuple[str, Optional[str]]:
        """重新剪辑"""
        return self.edit_video(merge_mode, min_relevance, fade_duration, progress)
    
    def update_segments(self, segments_json: str) -> Tuple[str, str]:
        """更新片段数据（用户编辑后）"""
        if not segments_json:
            return "没有片段数据", ""
        
        try:
            new_segments = json.loads(segments_json)
            if not isinstance(new_segments, list):
                return "片段数据格式错误，应为JSON数组", ""
            
            # 验证片段数据
            valid_segments = []
            for seg in new_segments:
                if isinstance(seg, dict) and 'start_time' in seg and 'end_time' in seg:
                    valid_segments.append(seg)
            
            if not valid_segments:
                return "没有有效的片段数据", ""
            
            # 更新 analysis_result
            if not self.analysis_result:
                self.analysis_result = {}
            self.analysis_result['matching_segments'] = valid_segments
            
            # 保存到 session
            session = load_session()
            if session:
                session['analysis_result'] = self.analysis_result
                save_session(session)
            
            return f"已更新 {len(valid_segments)} 个片段", json.dumps(valid_segments, ensure_ascii=False, indent=2)
        except json.JSONDecodeError as e:
            return f"JSON解析错误: {e}", ""
    
    def get_segments_table(self, min_relevance: int = 0) -> Tuple[list, str]:
        """获取片段列表"""
        if not self.analysis_result or not self.analysis_result.get('matching_segments'):
            return [], "没有可预览的片段，请先分析视频"
        
        segments = [s for s in self.analysis_result.get('matching_segments', []) 
                    if s.get('relevance_score', 0) >= min_relevance]
        
        if not segments:
            return [], f"没有相关性评分 >= {min_relevance} 的片段"
        
        rows = []
        for i, seg in enumerate(segments):
            start = seg.get('start_time', 0)
            end = seg.get('end_time', 0)
            dur = end - start
            score = seg.get('relevance_score', 0)
            reason = seg.get('reason', '')
            
            def fmt(s):
                return f"{int(s)//60:02d}:{int(s)%60:02d}"
            
            rows.append([
                i + 1,
                fmt(start),
                fmt(end),
                f"{dur:.1f}s",
                f"{score}%",
                reason
            ])
        
        return rows, f"共 {len(segments)} 个片段，点击行可跳转播放"
    
    def load_full_video(self) -> Tuple[Optional[str], str]:
        """加载原始完整视频到播放器"""
        if not self.video_path or not os.path.exists(self.video_path):
            return None, "视频文件不存在，请先上传或恢复会话"
        return self.video_path, f"✅ 已加载原视频：{os.path.basename(self.video_path)}"
    
    def preview_segment_by_row(self, evt: gr.SelectData, min_relevance: int = 0) -> Tuple[str, str, float, float, int, str]:
        """点击 Dataframe 行时，返回跳转时间信息（JS 跳转由前端 js 参数执行）"""
        if not self.analysis_result or not self.analysis_result.get('matching_segments'):
            return "", "没有片段数据", 0.0, 0.0, -1, ""
        
        row_idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
        
        segments = [s for s in self.analysis_result.get('matching_segments', []) 
                    if s.get('relevance_score', 0) >= min_relevance]
        
        if row_idx < 0 or row_idx >= len(segments):
            return "", f"无效的片段索引: {row_idx}", 0.0, 0.0, -1, ""
        
        seg = segments[row_idx]
        start_time = float(seg.get('start_time', 0))
        end_time = float(seg.get('end_time', 0))
        duration = end_time - start_time
        reason = seg.get('reason', '')
        
        def fmt(s):
            return f"{int(s)//60:02d}:{int(s)%60:02d}"
        
        info = f"▶ 片段 {row_idx+1}: {fmt(start_time)} - {fmt(end_time)}（{duration:.1f}秒）| {reason}"
        
        # jump_js 输出一个带时间戳的标记，前端 JS 监听这个变化来执行跳转
        # 用时间戳确保每次都不同，触发 Gradio 更新
        import time as _time
        marker = f"{start_time}|{end_time}|{_time.time()}"
        
        return marker, info, start_time, end_time, row_idx, reason
    
    def delete_segment(self, row_idx: int, min_relevance: int = 0) -> Tuple[str, list, str]:
        """删除当前选中的片段"""
        if not self.analysis_result or not self.analysis_result.get('matching_segments'):
            return "没有片段数据", [], ""
        
        if row_idx < 0:
            return "请先点击一个片段", [], ""
        
        all_segments = self.analysis_result.get('matching_segments', [])
        filtered = [(i, s) for i, s in enumerate(all_segments) if s.get('relevance_score', 0) >= min_relevance]
        
        if row_idx >= len(filtered):
            return f"片段索引无效: {row_idx}", [], ""
        
        orig_idx, seg = filtered[row_idx]
        
        def fmt(s):
            return f"{int(s)//60:02d}:{int(s)%60:02d}"
        
        start_time = seg.get('start_time', 0)
        end_time = seg.get('end_time', 0)
        
        # 从列表中删除
        all_segments.pop(orig_idx)
        self.analysis_result['matching_segments'] = all_segments
        
        # 保存到 session
        try:
            session = load_session()
            if session:
                session['analysis_result'] = self.analysis_result
                save_session(session)
        except:
            pass
        
        # 重新生成表格
        rows, _ = self.get_segments_table(min_relevance)
        segments_json_str = json.dumps(all_segments, ensure_ascii=False, indent=2)
        
        return f"🗑️ 已删除片段: {fmt(start_time)} - {fmt(end_time)}，剩余 {len(rows)} 个片段", rows, segments_json_str
    
    def save_segment_edit(self, row_idx: int, new_start: float, new_end: float, min_relevance: int = 0) -> Tuple[str, list, str]:
        """保存用户对单个片段起止时间的修改"""
        if not self.analysis_result or not self.analysis_result.get('matching_segments'):
            return "没有片段数据", [], ""
        
        if row_idx < 0:
            return "请先点击一个片段", [], ""
        
        segments = [s for s in self.analysis_result.get('matching_segments', []) 
                    if s.get('relevance_score', 0) >= min_relevance]
        
        if row_idx >= len(segments):
            return f"片段索引无效: {row_idx}", [], ""
        
        if new_start >= new_end:
            return "起始时间必须小于结束时间", [], ""
        
        # 找到原始片段在 analysis_result 中的位置并更新
        all_segments = self.analysis_result.get('matching_segments', [])
        # 找到对应的片段（按过滤后的索引映射回原始列表）
        filtered = [(i, s) for i, s in enumerate(all_segments) if s.get('relevance_score', 0) >= min_relevance]
        if row_idx < len(filtered):
            orig_idx, _ = filtered[row_idx]
            all_segments[orig_idx]['start_time'] = new_start
            all_segments[orig_idx]['end_time'] = new_end
            self.analysis_result['matching_segments'] = all_segments
            
            # 保存到 session
            try:
                session = load_session()
                if session:
                    session['analysis_result'] = self.analysis_result
                    save_session(session)
            except:
                pass
            
            # 重新生成表格
            rows, _ = self.get_segments_table(min_relevance)
            segments_json_str = json.dumps(all_segments, ensure_ascii=False, indent=2)
            
            def fmt(s):
                return f"{int(s)//60:02d}:{int(s)%60:02d}"
            return f"✅ 片段 {row_idx+1} 已更新: {fmt(new_start)} - {fmt(new_end)}", rows, segments_json_str
        
        return "更新失败", [], ""
    
    def generate_preview_all(self, min_relevance: int = 0, progress=gr.Progress()) -> Tuple[Optional[str], str]:
        """生成所有片段的合并预览视频"""
        if not self.video_path or not os.path.exists(self.video_path):
            return None, "视频文件不存在"
        
        if not self.analysis_result or not self.analysis_result.get('matching_segments'):
            return None, "没有可预览的片段"
        
        segments = [s for s in self.analysis_result.get('matching_segments', []) 
                    if s.get('relevance_score', 0) >= min_relevance]
        
        if not segments:
            return None, f"没有相关性评分 >= {min_relevance} 的片段"
        
        progress(0.05, desc="正在生成合并预览...")
        
        concat_file = os.path.join(TEMP_DIR, "preview_concat.txt")
        segment_files = []
        
        # 清空 concat 文件
        with open(concat_file, 'w', encoding='utf-8') as f:
            pass
        
        for i, seg in enumerate(segments):
            start_time = seg.get('start_time', 0)
            end_time = seg.get('end_time', 0)
            duration = end_time - start_time
            
            seg_output = os.path.join(TEMP_DIR, f"preview_clip_{i}.mp4")
            cmd = [
                FFMPEG_PATH, '-y',
                '-ss', str(start_time),
                '-i', self.video_path,
                '-t', str(duration),
                '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '28',
                '-c:a', 'aac', '-b:a', '96k',
                seg_output
            ]
            subprocess.run(cmd, capture_output=True, timeout=120)
            
            if os.path.exists(seg_output) and os.path.getsize(seg_output) > 0:
                segment_files.append(seg_output)
                with open(concat_file, 'a', encoding='utf-8') as f:
                    f.write(f"file '{seg_output}'\n")
            
            progress(0.05 + 0.8 * (i + 1) / len(segments), desc=f"处理片段 {i+1}/{len(segments)}")
        
        if not segment_files:
            return None, "所有片段截取失败"
        
        output_path = os.path.join(TEMP_DIR, "preview_all.mp4")
        concat_cmd = [
            FFMPEG_PATH, '-y',
            '-f', 'concat', '-safe', '0', '-i', concat_file,
            '-c', 'copy',
            output_path
        ]
        
        try:
            subprocess.run(concat_cmd, capture_output=True, timeout=300)
            progress(1.0, desc="完成")
            
            for f in segment_files:
                try: os.remove(f)
                except: pass
            try: os.remove(concat_file)
            except: pass
            
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                total_dur = sum(s.get('end_time', 0) - s.get('start_time', 0) for s in segments)
                return output_path, f"✅ 合并预览已生成，共 {len(segments)} 个片段，总时长约 {total_dur:.1f} 秒"
            else:
                return None, "合并失败"
        except Exception as e:
            return None, f"合并错误: {e}"
    
    def preview_segment(self, segment_index: int, progress=gr.Progress()) -> Tuple[Optional[str], str]:
        """预览单个片段"""
        if not self.video_path or not os.path.exists(self.video_path):
            return None, "视频文件不存在"
        
        if not self.analysis_result or not self.analysis_result.get('matching_segments'):
            return None, "没有可预览的片段"
        
        segments = self.analysis_result.get('matching_segments', [])
        if segment_index < 0 or segment_index >= len(segments):
            return None, f"片段索引无效: {segment_index}"
        
        seg = segments[segment_index]
        start_time = seg.get('start_time', 0)
        end_time = seg.get('end_time', 0)
        
        # 生成预览视频片段
        output_path = os.path.join(TEMP_DIR, f"preview_segment_{segment_index}.mp4")
        
        cmd = [
            FFMPEG_PATH, '-y',
            '-ss', str(start_time),
            '-i', self.video_path,
            '-t', str(end_time - start_time),
            '-c:v', 'libx264', '-preset', 'fast',
            '-c:a', 'aac', '-b:a', '128k',
            output_path
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=60)
            if os.path.exists(output_path):
                return output_path, f"片段 {segment_index + 1}: {start_time:.1f}s - {end_time:.1f}s"
            else:
                return None, f"预览生成失败"
        except Exception as e:
            return None, f"预览错误: {e}"
    
    def generate_preview_video(self, min_relevance: int = 0, progress=gr.Progress()) -> Tuple[Optional[str], str]:
        """生成成片预览视频（连续播放所有片段）"""
        if not self.video_path or not os.path.exists(self.video_path):
            return None, "视频文件不存在"
        
        if not self.analysis_result or not self.analysis_result.get('matching_segments'):
            return None, "没有可预览的片段"
        
        segments = [s for s in self.analysis_result.get('matching_segments', []) 
                    if s.get('relevance_score', 0) >= min_relevance]
        
        if not segments:
            return None, f"没有相关性评分 >= {min_relevance} 的片段"
        
        progress(0.1, desc="正在生成预览...")
        
        # 创建临时 concat 文件
        concat_file = os.path.join(TEMP_DIR, "preview_concat.txt")
        segment_files = []
        
        for i, seg in enumerate(segments):
            start_time = seg.get('start_time', 0)
            end_time = seg.get('end_time', 0)
            duration = end_time - start_time
            
            # 提取片段
            seg_output = os.path.join(TEMP_DIR, f"preview_clip_{i}.mp4")
            cmd = [
                FFMPEG_PATH, '-y',
                '-ss', str(start_time),
                '-i', self.video_path,
                '-t', str(duration),
                '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
                '-c:a', 'aac', '-b:a', '128k',
                seg_output
            ]
            subprocess.run(cmd, capture_output=True, timeout=60)
            
            if os.path.exists(seg_output):
                segment_files.append(seg_output)
                with open(concat_file, 'a', encoding='utf-8') as f:
                    f.write(f"file '{seg_output}'\n")
            
            progress(0.1 + 0.8 * (i + 1) / len(segments), desc=f"处理片段 {i+1}/{len(segments)}")
        
        if not segment_files:
            return None, "片段提取失败"
        
        # 合并所有片段
        output_path = os.path.join(TEMP_DIR, "preview_compilation.mp4")
        concat_cmd = [
            FFMPEG_PATH, '-y',
            '-f', 'concat', '-safe', '0', '-i', concat_file,
            '-c:v', 'libx264', '-preset', 'fast',
            '-c:a', 'aac', '-b:a', '192k',
            output_path
        ]
        
        try:
            subprocess.run(concat_cmd, capture_output=True, timeout=120)
            progress(1.0, desc="预览生成完成")
            
            # 清理临时文件
            for f in segment_files:
                try:
                    os.remove(f)
                except:
                    pass
            try:
                os.remove(concat_file)
            except:
                pass
            
            if os.path.exists(output_path):
                return output_path, f"预览已生成，包含 {len(segments)} 个片段"
            else:
                return None, "预览生成失败"
        except Exception as e:
            return None, f"预览生成错误: {e}"
    
    def open_smart_player(self) -> Tuple[str, str]:
        """打开智能播放器HTML文件"""
        html_file = os.path.join(TEMP_DIR, "smart_player.html")
        if not os.path.exists(html_file):
            return "❌ 智能播放器文件不存在，请先生成", ""
        
        import webbrowser
        try:
            webbrowser.open(f"file:///{html_file.replace(os.sep, '/')}")
            return f"✅ 已在浏览器中打开智能播放器\n\n📁 文件位置: {html_file}", "播放器已打开"
        except Exception as e:
            return f"❌ 打开失败: {e}\n\n请手动打开: {html_file}", ""
    
    def cleanup_temp(self):
        import gc
        
        if self.video_processor:
            try:
                self.video_processor.close()
            except:
                pass
            self.video_processor = None
        
        self.audio_segments = None
        
        gc.collect()
        
        os.makedirs(TEMP_DIR, exist_ok=True)
        
        if os.path.exists(TEMP_DIR):
            for filename in os.listdir(TEMP_DIR):
                # 保留 session.json 和视频文件，只清理帧图片和音频
                if filename == 'session.json':
                    continue
                if filename.startswith('input_video'):
                    continue
                if filename.endswith('.jpg') or filename.endswith('.mp3') or filename.endswith('.srt') or filename.endswith('.txt'):
                    filepath = os.path.join(TEMP_DIR, filename)
                    try:
                        os.unlink(filepath)
                    except:
                        pass


def create_ui():
    app = AiCutApp()

    # 启动时检查是否有可恢复的会话
    has_session, session_msg = check_session()

    with gr.Blocks(title="AiCut - AI智能视频剪辑") as demo:
        gr.Markdown("""
        # 🎬 AiCut - AI智能视频剪辑工具
        
        上传视频，描述您想要剪辑的内容，AI将自动分析并剪辑出符合主题的视频片段。
        
        **支持字幕分析**：可上传SRT字幕文件，AI将结合画面和字幕内容进行分析。
        
        **使用步骤**:
        1. 上传视频文件（可选：上传字幕文件）
        2. 点击"处理视频"按钮（或点击"🔥 恢复上次会话"跳过此步骤）
        3. 输入您希望剪辑的内容描述
        4. 点击"🔍 分析视频"按钮
        5. 查看分析结果后，点击"开始剪辑"
        """)

        # 热启动区域：始终显示，点击时检查是否有会话
        with gr.Row():
            with gr.Column():
                session_hint = gr.Markdown(
                    value=f"### 🔥 检测到上次会话\n> {session_msg}" if has_session else "### 💡 热启动\n> 如果上次处理过视频，可点击下方按钮直接恢复，无需重新上传",
                )
                restore_btn = gr.Button(
                    "🔥 恢复上次会话（跳过上传和提取步骤）",
                    variant="primary" if has_session else "secondary",
                    size="lg"
                )

        with gr.Row():
            with gr.Column(scale=1):
                video_input = gr.File(label="上传视频", file_types=[".mp4", ".avi", ".mov", ".mkv", ".webm"])
                subtitle_input = gr.File(label="上传字幕（可选）", file_types=[".srt"])
                process_btn = gr.Button("处理视频", variant="secondary")
                video_info = gr.Textbox(label="视频信息", lines=8, interactive=False)
                
            with gr.Column(scale=1):
                user_request = gr.Textbox(
                    label="剪辑内容描述",
                    placeholder="例如：找出所有有猫出现的片段\n或者：提取所有人物对话的场景\n或者：找出所有户外运动的镜头",
                    lines=3
                )
                video_description = gr.Textbox(
                    label="视频简介（可选）",
                    placeholder="简单描述这个视频的内容，帮助 AI 更好地理解视频...\n例如：这是一个游戏直播视频，主播在玩某某游戏\n或者：这是一个聊天杂谈视频，主要讨论某某话题",
                    lines=3
                )
                with gr.Row():
                    use_uniform = gr.Checkbox(label="智能取帧", value=True, info="使用 pHash 智能提取关键帧")
                    use_audio = gr.Checkbox(label="音频转文字", value=True, info="从音频中提取文字转录")
                    use_subtitle = gr.Checkbox(label="读取字幕", value=True, info="读取上传的字幕文件")
                with gr.Row():
                    analyze_btn = gr.Button("🔍 分析视频", variant="primary")
                    re_analyze_btn = gr.Button("🔄 重新分析", variant="secondary")
                status_text = gr.Textbox(label="状态", interactive=False)
        
        with gr.Row():
            analysis_result = gr.Markdown(label="分析结果")
            segments_json = gr.Code(label="片段数据 (JSON)", language="json", lines=10)
        
        with gr.Row():
            frame_gallery = gr.Gallery(
                label="关键帧预览",
                columns=3,
                rows=2,
                object_fit="contain",
                height=400
            )
        
        # 片段预览区域
        gr.Markdown("""
        ---
        ### 🎞️ 片段预览与编辑
        
        点击"加载片段列表"后，点击表格中任意一行即可预览该片段；点击"▶️ 合并预览全部"可连续播放所有片段。
        如需调整时间，在 JSON 编辑框中修改后点击"保存编辑"。
        """)
        
        # 隐藏状态：记录当前选中的片段行索引
        current_row_idx = gr.State(-1)
        
        with gr.Row():
            with gr.Column(scale=2):
                with gr.Row():
                    load_segments_btn = gr.Button("📋 加载片段列表", variant="primary")
                    load_video_btn = gr.Button("🎬 加载原视频", variant="secondary")
                    preview_all_btn = gr.Button("▶️ 合并预览全部", variant="secondary")
                segments_table = gr.Dataframe(
                    headers=["#", "开始", "结束", "时长", "评分", "原因"],
                    datatype=["number", "str", "str", "str", "str", "str"],
                    label="片段列表（点击行预览）",
                    interactive=False,
                    wrap=True,
                )
                player_info = gr.Textbox(label="状态", interactive=False)
                # 用于注入 JS 跳转指令的隐藏 HTML 组件
                jump_js = gr.HTML(value="", visible=True, elem_id="aicut_jump_js")
                preview_video = gr.Video(label="原视频播放器（点击片段行自动跳转）", height=400, elem_id="aicut_preview_video")
            
            with gr.Column(scale=1):
                gr.Markdown("#### ✏️ 编辑当前片段时间")
                seg_start_input = gr.Number(label="起始时间（秒）", value=0, precision=1)
                seg_end_input = gr.Number(label="结束时间（秒）", value=0, precision=1)
                seg_reason_display = gr.Textbox(label="片段描述", interactive=False)
                with gr.Row():
                    save_seg_btn = gr.Button("💾 保存时间", variant="primary")
                    delete_seg_btn = gr.Button("🗑️ 删除此片段", variant="stop")
                save_seg_status = gr.Textbox(label="保存状态", interactive=False)
                gr.Markdown("---")
                update_segments_btn = gr.Button("💾 批量保存（JSON）", variant="secondary")
                segments_json_edit = gr.Code(label="编辑片段数据 (JSON)", language="json", lines=10, interactive=True)
                edit_status = gr.Textbox(label="编辑状态", interactive=False)
        
        with gr.Row():
            with gr.Column(scale=1):
                merge_mode = gr.Checkbox(label="合并为一个视频", value=True, info="勾选则将所有片段合并，否则输出多个独立片段")
                min_relevance = gr.Slider(0, 100, value=50, step=5, label="最低相关性评分", info="只剪辑相关性评分高于此值的片段")
                fade_duration = gr.Slider(0, 2, value=0.5, step=0.1, label="淡入淡出时长(秒)")
            
            with gr.Column(scale=1):
                with gr.Row():
                    edit_btn = gr.Button("✂️ 开始剪辑", variant="primary")
                    re_edit_btn = gr.Button("🔄 重新剪辑", variant="secondary")
                edit_result = gr.Textbox(label="剪辑结果", interactive=False)
                output_video = gr.Video(label="输出视频")
        
        process_btn.click(
            fn=app.process_video,
            inputs=[video_input, subtitle_input, video_description],
            outputs=[video_info, status_text]
        )
        
        analyze_btn.click(
            fn=app.analyze_video,
            inputs=[user_request, video_description, use_uniform, use_audio, use_subtitle],
            outputs=[analysis_result, segments_json, status_text, frame_gallery, segments_json_edit]
        )
        
        re_analyze_btn.click(
            fn=app.re_analyze,
            inputs=[user_request, video_description, use_uniform, use_audio, use_subtitle],
            outputs=[analysis_result, segments_json, status_text, frame_gallery, segments_json_edit]
        )
        
        edit_btn.click(
            fn=app.edit_video,
            inputs=[merge_mode, min_relevance, fade_duration],
            outputs=[edit_result, output_video]
        )
        
        restore_btn.click(
            fn=app.restore_session,
            inputs=[video_description],
            outputs=[video_info, status_text, video_description, user_request, analysis_result, segments_json, frame_gallery, segments_json_edit]
        )

        re_edit_btn.click(
            fn=app.re_edit,
            inputs=[merge_mode, min_relevance, fade_duration],
            outputs=[edit_result, output_video]
        )
        
        # 片段预览相关事件
        # 加载片段列表（只更新表格和状态）
        load_segments_btn.click(
            fn=app.get_segments_table,
            inputs=[min_relevance],
            outputs=[segments_table, player_info]
        )
        
        # 加载原视频到播放器（只加载一次）
        load_video_btn.click(
            fn=app.load_full_video,
            inputs=[],
            outputs=[preview_video, player_info]
        )
        
        preview_all_btn.click(
            fn=app.generate_preview_all,
            inputs=[min_relevance],
            outputs=[preview_video, player_info]
        )
        
        # 点击行时：注入 JS 跳转（输出到 jump_js），同时填充起止时间编辑框
        segments_table.select(
            fn=app.preview_segment_by_row,
            inputs=[min_relevance],
            outputs=[jump_js, player_info, seg_start_input, seg_end_input, current_row_idx, seg_reason_display]
        )
        
        # 保存单个片段时间修改
        save_seg_btn.click(
            fn=app.save_segment_edit,
            inputs=[current_row_idx, seg_start_input, seg_end_input, min_relevance],
            outputs=[save_seg_status, segments_table, segments_json_edit]
        )
        
        # 删除当前选中的片段
        delete_seg_btn.click(
            fn=app.delete_segment,
            inputs=[current_row_idx, min_relevance],
            outputs=[save_seg_status, segments_table, segments_json_edit]
        )
        
        update_segments_btn.click(
            fn=app.update_segments,
            inputs=[segments_json_edit],
            outputs=[edit_status, segments_json_edit]
        )
        
        # 注入全局 JS：监听 jump_js 组件内容变化，自动跳转视频
        demo.load(
            fn=None,
            js="""
() => {
  // 等待页面完全加载后再注入监听器
  function setupJumpListener() {
    var jumpEl = document.querySelector('#aicut_jump_js');
    if (!jumpEl) {
      setTimeout(setupJumpListener, 500);
      return;
    }
    
    var observer = new MutationObserver(function(mutations) {
      mutations.forEach(function(mutation) {
        var text = jumpEl.innerText || jumpEl.textContent || '';
        text = text.trim();
        if (!text) return;
        
        // 解析 marker 格式: "start_time|end_time|nonce"
        var parts = text.split('|');
        if (parts.length < 2) return;
        
        var startTime = parseFloat(parts[0]);
        var endTime = parseFloat(parts[1]);
        if (isNaN(startTime) || isNaN(endTime)) return;
        
        // 找到预览视频播放器
        var container = document.querySelector('#aicut_preview_video');
        var vid = container ? container.querySelector('video') : null;
        if (!vid) {
          var vids = document.querySelectorAll('video');
          vid = vids.length > 0 ? vids[0] : null;
        }
        
        if (vid) {
          console.log('[AiCut] 跳转到', startTime, '-', endTime);
          
          // 先暂停，等 seek 完成后再播放，避免闪烁第0帧
          vid.pause();
          
          // 移除旧的 timeupdate 监听器
          if (vid._aicutEndListener) {
            vid.removeEventListener('timeupdate', vid._aicutEndListener);
            vid._aicutEndListener = null;
          }
          // 移除旧的 seeked 监听器
          if (vid._aicutSeekedListener) {
            vid.removeEventListener('seeked', vid._aicutSeekedListener);
            vid._aicutSeekedListener = null;
          }
          
          // 等 seek 完成后再播放
          vid._aicutSeekedListener = function() {
            vid.removeEventListener('seeked', vid._aicutSeekedListener);
            vid._aicutSeekedListener = null;
            vid.play().catch(function(){});
            
            // 到结束时间自动暂停
            vid._aicutEndListener = function() {
              if (vid.currentTime >= endTime) {
                vid.pause();
                vid.removeEventListener('timeupdate', vid._aicutEndListener);
                vid._aicutEndListener = null;
              }
            };
            vid.addEventListener('timeupdate', vid._aicutEndListener);
          };
          vid.addEventListener('seeked', vid._aicutSeekedListener);
          
          // 设置跳转时间（触发 seeked 事件）
          vid.currentTime = startTime;
        } else {
          console.warn('[AiCut] 未找到视频元素，请先点击"加载原视频"');
        }
      });
    });
    
    observer.observe(jumpEl, { childList: true, subtree: true, characterData: true });
    console.log('[AiCut] 视频跳转监听器已就绪');
  }
  
  setupJumpListener();
}
"""
        )
        
        gr.Markdown("""
        ---
        ### ⚙️ 配置说明
        
        在使用前，请复制 `.env.example` 为 `.env` 并配置您的 API 密钥。
        
        ### 📝 使用提示
        
        - 描述越具体，AI分析越准确
        - 相关性评分越高，匹配越精准
        - 支持的视频格式: MP4, AVI, MOV, MKV 等
        - 支持的字幕格式: SRT
        """)
    
    return demo


if __name__ == "__main__":
    # 修复 Windows 上 asyncio ProactorEventLoop 的 ConnectionResetError 警告
    import asyncio
    import sys
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    demo = create_ui()
    import socket
    import webbrowser
    import time
    import threading
    
    # 确保 temp 目录存在
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    # 找一个可用端口
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(('127.0.0.1', 0))
    port = sock.getsockname()[1]
    sock.close()
    
    # 延迟打开浏览器
    def open_browser():
        time.sleep(1.5)  # 等待服务器启动
        webbrowser.open(f"http://127.0.0.1:{port}")
    
    # 在后台线程中打开浏览器
    threading.Thread(target=open_browser, daemon=True).start()
    
    print(f"\n🚀 正在启动 Web 界面...")
    print(f"📱 浏览器将自动打开 http://127.0.0.1:{port}")
    print(f"💡 如果没有自动打开，请手动访问上述地址\n")
    
    temp_abs = os.path.abspath(TEMP_DIR)
    demo.launch(
        server_name="127.0.0.1",
        server_port=port,
        share=False,
        show_error=True,
        max_file_size="10gb",
        allowed_paths=[
            temp_abs,
            temp_abs.replace("\\", "/"),   # 正斜杠版本
            TEMP_DIR,                       # 相对路径
        ]
    )
