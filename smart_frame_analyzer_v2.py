"""
智能帧分析器 - pHash 逐级细化 + 流式分析
功能：
1. 使用 pHash 逐级细化算法检测场景变化
2. 边提取帧边分析（流式处理）
3. 只分析变化明显的帧，跳过重复内容
4. 保存原分辨率帧图片（按秒命名，如 7200s.jpg）
5. 实时返回分析结果

使用示例：
    analyzer = SmartFrameAnalyzer(video_path, api_key, output_dir="temp")
    for result in analyzer.analyze_streaming(video_description="视频简介"):
        print(f"分析帧：{result['timestamp']} - {result.get('analysis', {})}")
"""

import cv2
import numpy as np
from PIL import Image
import io
import requests
import os
import json
import re
import time
from pathlib import Path
from typing import List, Dict, Optional, Generator
import base64
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import OPENAI_API_KEY, OPENAI_BASE_URL, MODEL_NAME


class SmartFrameAnalyzer:
    """智能帧分析器 - pHash 逐级细化"""
    
    def __init__(self, video_path: str, api_key: str = "", output_dir: str = "temp"):
        """
        Args:
            video_path: 视频文件路径
            api_key: SiliconFlow API Key
            output_dir: 帧图片输出目录
        """
        self.video_path = video_path
        self.api_key = OPENAI_API_KEY
        self.api_url = f"{OPENAI_BASE_URL}/chat/completions" if OPENAI_BASE_URL else "https://api.openai.com/v1/chat/completions"
        self.model = MODEL_NAME
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 打开视频获取信息
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"无法打开视频：{video_path}")
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frames / self.fps
        
        print(f"📹 视频已加载：{self.duration:.1f}秒 ({self.total_frames}帧，{self.fps:.1f}FPS)")
        print(f"🔑 使用 API: SiliconFlow ({self.model})")
    
    def close(self):
        """关闭视频"""
        if self.cap:
            self.cap.release()
    
    def compute_phash(self, frame: np.ndarray, hash_size: int = 8) -> np.ndarray:
        """
        计算帧的感知哈希（使用 48x27 缩小加速）
        
        Args:
            frame: OpenCV 帧 (BGR 格式)
            hash_size: 哈希尺寸
        
        Returns:
            二进制数组
        """
        # 缩小到 48x27 加速处理
        small = cv2.resize(frame, (48, 27), interpolation=cv2.INTER_AREA)
        # 转灰度
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        # 缩小到 8x8 并二值化
        resized = cv2.resize(gray, (hash_size, hash_size), interpolation=cv2.INTER_AREA)
        avg = resized.mean()
        return (resized > avg).astype(int).flatten()
    
    def calculate_similarity(self, hash1: np.ndarray, hash2: np.ndarray) -> float:
        """计算两个哈希的相似度"""
        hash1 = hash1.flatten()
        hash2 = hash2.flatten()
        same_bits = np.sum(hash1 == hash2)
        return same_bits / len(hash1)
    
    def encode_frame(self, frame: np.ndarray, quality: int = 85) -> str:
        """
        编码帧为 base64
        
        Args:
            frame: OpenCV 帧
            quality: JPEG 质量 (1-100)
        
        Returns:
            base64 字符串
        """
        # BGR 转 RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        # 压缩
        buffered = io.BytesIO()
        img.save(buffered, format='JPEG', quality=quality)
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    def analyze_frame_with_ai(self, frame: np.ndarray, timestamp: str, 
                             video_description: str = "",
                             model: str = None,
                             max_retries: int = 3) -> Dict:
        """
        使用 AI 分析单帧
        
        Args:
            frame: OpenCV 帧
            timestamp: 时间戳字符串
            video_description: 视频简介
            model: 模型名称
            max_retries: 最大重试次数
        
        Returns:
            分析结果字典
        """
        last_error = None
        
        for attempt in range(max_retries):
            try:
                # 压缩并编码
                base64_image = self.encode_frame(frame)
                
                # 构建提示词
                prompt = """请分析这个视频帧的内容，返回 JSON 格式：
{
    "scene_type": "场景类型（游戏/聊天杂谈/舞蹈/其他）",
    "main_objects": ["主要物体/元素列表"],
    "text_visible": "画面中可见的文字（如有，没有则填'无'）",
    "action": "正在发生的动作或活动",
    "description": "详细描述（仅对画面中的事实做客观描述，不要做任何推断和联想，100 字以内）"
}
只返回 JSON，不要其他内容。"""
                
                # 如果有视频简介，添加到提示词
                if video_description:
                    prompt = f"""视频背景信息：
{video_description}

请结合以上视频背景，分析这个视频帧的内容，返回 JSON 格式：
{{
    "scene_type": "场景类型（游戏/聊天杂谈/舞蹈/其他）",
    "main_objects": ["主要物体/元素列表"],
    "text_visible": "画面中可见的文字（如有，没有则填'无'）",
    "action": "正在发生的动作或活动",
    "description": "详细描述（结合视频背景进行描述，100 字以内）"
}}
只返回 JSON，不要其他内容。"""
                
                # 调用 API（使用 SiliconFlow 格式）
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                
                payload = {
                    "model": model if model else self.model,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image[:50]}..." 
                                    }
                                },
                                {
                                    "type": "text",
                                    "text": prompt
                                }
                            ]
                        }
                    ],
                    "max_tokens": 1500
                }
                
                print(f"\n{'='*60}")
                print(f"[LLM 请求] 时间戳: {timestamp} (尝试 {attempt + 1}/{max_retries})")
                print(f"[LLM 请求] Model: {model if model else self.model}")
                print(f"[LLM 请求] Prompt:\n{prompt}")
                print(f"[LLM 请求] 图片大小: {len(base64_image)} bytes (base64)")
                
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    json={
                        "model": model if model else self.model,
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/jpeg;base64,{base64_image}"
                                        }
                                    },
                                    {
                                        "type": "text",
                                        "text": prompt
                                    }
                                ]
                            }
                        ],
                        "max_tokens": 1500
                    },
                    timeout=60
                )
                
                print(f"[LLM 响应] 状态码: {response.status_code}")
                
                if response.status_code != 200:
                    print(f"[LLM 响应] 错误内容: {response.text}")
                
                if response.status_code == 200:
                    result = response.json()
                    content = result['choices'][0]['message']['content']
                    
                    print(f"[LLM 响应] 完整内容:\n{content}")
                    print(f"{'='*60}\n")
                    
                    content = content.replace('```json', '').replace('```', '').strip()
                    content = re.sub(r'<\|[^|]+\|>', '', content).strip()
                    
                    try:
                        analysis = json.loads(content)
                        return {
                            'success': True,
                            'timestamp': timestamp,
                            'analysis': analysis
                        }
                    except json.JSONDecodeError as e:
                        print(f"[DEBUG] JSON 解析失败，尝试修复...")
                        
                        fixed_analysis = self._try_fix_json(content)
                        if fixed_analysis:
                            return {
                                'success': True,
                                'timestamp': timestamp,
                                'analysis': fixed_analysis
                            }
                        
                        return {
                            'success': True,
                            'timestamp': timestamp,
                            'analysis': {
                                'scene_type': '未知',
                                'main_objects': [],
                                'text_visible': '无',
                                'action': '',
                                'description': content if content else '分析失败'
                            }
                        }
                else:
                    last_error = f'API 错误：{response.status_code}'
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt
                        print(f"[DEBUG] 等待 {wait_time} 秒后重试...")
                        time.sleep(wait_time)
                    continue
                    
            except Exception as e:
                last_error = str(e)
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"[DEBUG] 异常: {e}，等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                continue
        
        return {
            'success': False,
            'error': last_error or '未知错误',
            'timestamp': timestamp
        }
    
    def _try_fix_json(self, content: str) -> Optional[Dict]:
        """尝试修复损坏的 JSON"""
        import re
        
        if not content:
            return None
        
        content = content.strip()
        print(f"[DEBUG] 尝试修复 JSON，内容长度: {len(content)}")
        
        if not content.startswith('{'):
            match = re.search(r'\{', content)
            if match:
                content = content[match.start():]
        
        result = {}
        
        scene_match = re.search(r'"scene_type"\s*:\s*"([^"]*)"', content)
        if scene_match:
            result['scene_type'] = scene_match.group(1)
        
        objects_match = re.search(r'"main_objects"\s*:\s*\[([^\]]*)', content)
        if objects_match:
            objects_str = objects_match.group(1)
            objects = re.findall(r'"([^"]*)"', objects_str)
            result['main_objects'] = objects if objects else []
        
        text_match = re.search(r'"text_visible"\s*:\s*"([^"]*)"', content)
        if text_match:
            result['text_visible'] = text_match.group(1)
        
        action_match = re.search(r'"action"\s*:\s*"([^"]*)"', content)
        if action_match:
            result['action'] = action_match.group(1)
        
        desc_match = re.search(r'"description"\s*:\s*"([^"]*)"', content)
        if desc_match:
            result['description'] = desc_match.group(1)
        
        if result:
            print(f"[DEBUG] 通过正则提取到字段: {list(result.keys())}")
            return result
        
        if not content.endswith('}'):
            content += '}'
        
        try:
            return json.loads(content)
        except:
            pass
        
        try:
            fixed = content
            fixed = re.sub(r'"([^"]+)":\s*$', r'"\1": ""', fixed)
            fixed = re.sub(r':\s*([^"\[\{][^,}\]]*),', r': "\1",', fixed)
            fixed = re.sub(r':\s*([^"\[\{][^}\]]*)$', r': "\1"', fixed)
            
            open_brackets = fixed.count('[') - fixed.count(']')
            if open_brackets > 0:
                fixed += ']' * open_brackets
            
            open_braces = fixed.count('{') - fixed.count('}')
            if open_braces > 0:
                fixed += '}' * open_braces
            
            return json.loads(fixed)
        except:
            pass
        
        return None
    
    def analyze_streaming(self, 
                         video_description: str = "",
                         similarity_threshold: float = 0.85,
                         model: str = None) -> Generator[Dict, None, None]:
        """
        分两阶段分析：
        1. 先提取所有不重复的帧（使用 pHash 过滤）
        2. 然后批量分析这些帧
        """
        print("\n" + "="*80)
        print("【智能帧分析 - pHash 逐级细化 + 两阶段处理】")
        print("="*80)
        print(f"[DEBUG] API Key: {'*' * 10}...{'*' * 5}")
        print(f"[DEBUG] 视频路径: {self.video_path}")
        print(f"[DEBUG] 视频时长: {self.duration:.1f}秒")
        print(f"[DEBUG] FPS: {self.fps}")
        print(f"[DEBUG] 总帧数: {self.total_frames}")
        print(f"[DEBUG] 相似度阈值: {similarity_threshold:.0%}")
        if video_description:
            print(f"[DEBUG] 视频简介: {video_description[:50]}...")
        print("="*80)
        
        if self.duration < 1:
            print("⚠️ 视频时长太短，无法分析")
            return
        
        # 逐级细化的间隔（秒）
        levels = [
            (300, 0.6),   # Level 1: 5 分钟，阈值 60%
            (60, 0.5),    # Level 2: 1 分钟，阈值 50%
            (5, 0.4),     # Level 3: 5 秒，阈值 40%
            (1, 0.3),     # Level 4: 1 秒，阈值 30%
        ]
        
        # 存储已分析的帧
        analyzed_frames = []  # [(timestamp, hash, frame_image)]
        changes_detected = []  # [(timestamp, similarity)]
        
        total_samples = 0
        skip_count = 0
        
        # ===== 第一阶段：提取帧 =====
        print("\n【阶段 1: 提取帧（使用 pHash 过滤重复）】")
        print("-"*80)
        
        # Level 1: 5 分钟一帧
        print("\n【Level 1: 5 分钟一帧 - 全视频扫描】")
        print("-"*80)
        print(f"[DEBUG] 开始 Level 1，interval=300 秒，duration={self.duration}秒")
        print(f"[DEBUG] 预计采样次数: {int(self.duration // 300) + 1}")
        
        interval = 300  # 5 分钟
        timestamp = 0
        loop_count = 0
        
        while timestamp < self.duration:
            loop_count += 1
            print(f"[DEBUG] Level 1 循环 {loop_count}: timestamp={timestamp:.1f}s, frame_idx={int(timestamp * self.fps)}")
            
            frame_idx = int(timestamp * self.fps)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self.cap.read()
            
            if not ret:
                print(f"[DEBUG] ❌ 读取帧失败：timestamp={timestamp}, frame_idx={frame_idx}")
                break
            
            total_samples += 1
            timestamp_str = f"{int(timestamp)//60:02d}:{int(timestamp)%60:02d}"
            
            # 计算 pHash
            current_hash = self.compute_phash(frame)
            
            # 判断是否保留
            should_keep = True
            similarity = 0
            
            if analyzed_frames:
                # 与上一帧比较
                last_timestamp, last_hash, _ = analyzed_frames[-1]
                similarity = self.calculate_similarity(last_hash, current_hash)
                
                print(f"[DEBUG]   与上一帧相似度: {similarity:.2%}, 阈值: {similarity_threshold:.0%}")
                
                if similarity > similarity_threshold:
                    # 相似，跳过
                    should_keep = False
                    skip_count += 1
                    print(f"[DEBUG]   ⏭️ 跳过（相似度 {similarity:.2%} > {similarity_threshold:.0%}）")
            
            # 保存帧
            if should_keep:
                # 保存帧文件
                frame_filename = self.output_dir / f"frame_{int(timestamp)}s.jpg"
                cv2.imwrite(str(frame_filename), frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
                
                # 记录
                analyzed_frames.append((timestamp, current_hash, frame))
                print(f"[DEBUG]   📸 帧 {len(analyzed_frames)} [{timestamp_str}] - 已保存 (相似度: {similarity:.2%})")
            
            timestamp += interval
        
        print(f"\n[DEBUG] Level 1 完成: 循环 {loop_count} 次，保留 {len(analyzed_frames)} 帧")
        
        # 检查是否有变化
        if len(analyzed_frames) > 1:
            for i in range(1, len(analyzed_frames)):
                sim = self.calculate_similarity(analyzed_frames[i-1][1], analyzed_frames[i][1])
                if sim < 0.6:
                    changes_detected.append((analyzed_frames[i][0], sim))
        
        # Level 2: 如果有变化，细化分析
        if changes_detected:
            print(f"\n【Level 2: 1 分钟一帧 - 细化 {len(changes_detected)} 个变化区域】")
            print("-"*80)
            
            # 记录已保存的时间戳，避免重复
            saved_timestamps = set(ts for ts, _, _ in analyzed_frames)
            
            for change_ts, change_sim in changes_detected:
                start_sec = max(0, change_ts - 300)
                end_sec = min(self.duration, change_ts + 300)
                
                timestamp = start_sec
                prev_hash = None
                
                while timestamp <= end_sec:
                    # 如果这个时间点已经保存过，跳过
                    if int(timestamp) in saved_timestamps:
                        timestamp += 60
                        continue
                    
                    frame_idx = int(timestamp * self.fps)
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = self.cap.read()
                    
                    if not ret:
                        break
                    
                    total_samples += 1
                    current_hash = self.compute_phash(frame)
                    
                    if prev_hash is None:
                        # 第一帧
                        frame_filename = self.output_dir / f"frame_{int(timestamp)}s.jpg"
                        cv2.imwrite(str(frame_filename), frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
                        analyzed_frames.append((timestamp, current_hash, frame))
                        saved_timestamps.add(int(timestamp))
                        print(f"  📸 帧 {len(analyzed_frames)} [{int(timestamp)//60:02d}:{int(timestamp)%60:02d}] - 变化点分析")
                    else:
                        similarity = self.calculate_similarity(prev_hash, current_hash)
                        if similarity < 0.5:
                            # 有变化，保存
                            frame_filename = self.output_dir / f"frame_{int(timestamp)}s.jpg"
                            cv2.imwrite(str(frame_filename), frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
                            analyzed_frames.append((timestamp, current_hash, frame))
                            saved_timestamps.add(int(timestamp))
                            print(f"  📸 帧 {len(analyzed_frames)} [{int(timestamp)//60:02d}:{int(timestamp)%60:02d}] - 变化点分析")
                    
                    prev_hash = current_hash
                    timestamp += 60
        
        print(f"\n【阶段 1 完成】")
        print(f"[DEBUG] 总采样：{total_samples} 帧")
        print(f"[DEBUG] 保留：{len(analyzed_frames)} 帧 ({len(analyzed_frames)/total_samples*100:.1f}%)")
        print(f"[DEBUG] 跳过：{skip_count} 帧 ({skip_count/total_samples*100:.1f}%)")
        
        if len(analyzed_frames) == 0:
            print("⚠️ 没有提取到任何帧，无法分析")
            return
        
        # ===== 第二阶段：并行批量分析 =====
        # 并发数：根据帧数自动调整，最多 8 线程，避免 API 限流
        max_workers = min(8, max(1, len(analyzed_frames)))
        print(f"\n【阶段 2: 并行分析 {len(analyzed_frames)} 帧（{max_workers} 线程并发）】")
        print("-"*80)
        print(f"[DEBUG] API URL: {self.api_url}")
        print(f"[DEBUG] Model: {model if model else self.model}")
        print(f"[DEBUG] 并发线程数: {max_workers}")
        
        json_file = self.output_dir / "frame_analysis.json"
        print(f"[DEBUG] 输出文件: {json_file}")
        # 用于保存结果的有序列表（按原始顺序）
        analysis_results = [None] * len(analyzed_frames)
        save_lock = threading.Lock()
        completed_count = [0]

        def analyze_one(idx_ts_hash_frame):
            """单帧分析任务（线程安全）"""
            i, timestamp, frame_hash, frame = idx_ts_hash_frame
            timestamp_str = f"{int(timestamp)//60:02d}:{int(timestamp)%60:02d}"
            frame_filename = self.output_dir / f"frame_{int(timestamp)}s.jpg"

            print(f"  🔍 [{i+1}/{len(analyzed_frames)}] 开始分析 [{timestamp_str}]...")

            result = self.analyze_frame_with_ai(frame, timestamp_str, video_description, model)
            result['image_path'] = str(frame_filename)

            agent_format_result = {
                "timestamp": timestamp_str,
                "image_path": str(frame_filename),
                "analysis": {
                    "success": result.get('success', False),
                    "analysis": result.get('analysis', {}) if result.get('success') else {}
                }
            }

            # 写入有序槽位
            analysis_results[i] = agent_format_result

            with save_lock:
                completed_count[0] += 1
                done = completed_count[0]
                status = "✅" if result.get('success') else "❌"
                print(f"  {status} [{done}/{len(analyzed_frames)}] 完成 [{timestamp_str}]"
                      + (f"：{result.get('error','')}" if not result.get('success') else ""))

                # 每完成 3 帧或全部完成时保存一次
                if done % 3 == 0 or done == len(analyzed_frames):
                    valid = [r for r in analysis_results if r is not None]
                    with open(json_file, 'w', encoding='utf-8') as f:
                        json.dump({"frames": valid, "total_frames": len(valid)},
                                  f, ensure_ascii=False, indent=2)
                    print(f"  💾 已保存 {len(valid)} 帧结果")

            return result

        # 构建任务列表
        tasks = [(i, ts, h, fr) for i, (ts, h, fr) in enumerate(analyzed_frames)]

        if self.api_key and tasks:
            # 用于 yield 结果的队列（保持流式输出）
            result_queue = queue.Queue()

            def worker_wrapper(task):
                res = analyze_one(task)
                result_queue.put(res)
                return res

            # 手动管理线程池生命周期，避免 generator yield 导致死锁
            executor = ThreadPoolExecutor(max_workers=max_workers)
            try:
                # 提交所有任务
                futures = [executor.submit(worker_wrapper, t) for t in tasks]

                # 流式 yield 结果（不阻塞线程池）
                yielded = 0
                total = len(tasks)
                while yielded < total:
                    try:
                        res = result_queue.get(timeout=120)
                        yield res
                        yielded += 1
                    except queue.Empty:
                        # 检查是否所有 future 都已完成
                        if all(f.done() for f in futures):
                            break
            finally:
                # 等待所有任务完成后关闭线程池
                executor.shutdown(wait=True)

        # 最终保存（确保完整）
        valid = [r for r in analysis_results if r is not None]
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump({"frames": valid, "total_frames": len(valid)},
                      f, ensure_ascii=False, indent=2)
        print(f"\n✅ 帧分析结果已保存到：{json_file}（共 {len(valid)} 帧）")
        
        # 总结
        print("\n" + "="*80)
        print("【分析总结】")
        print("="*80)
        print(f"总采样帧数：{total_samples}")
        print(f"分析帧数：{len(analyzed_frames)} ({len(analyzed_frames)/total_samples*100:.1f}%)")
        print(f"效率提升：{skip_count/total_samples*100:.1f}%")
        print("="*80)
        print("✅ 分析完成！")


if __name__ == "__main__":
    from config import TEMP_DIR
    
    # 测试
    video_path = os.path.join(TEMP_DIR, "input_video.mp4")
    
    if not os.path.exists(video_path):
        print(f"❌ 视频不存在：{video_path}")
    else:
        # 测试（不提供 API Key，只演示帧提取）
        analyzer = SmartFrameAnalyzer(video_path, output_dir="test_frames")
        
        print("\n开始测试帧提取...")
        for result in analyzer.analyze_streaming(video_description="测试视频"):
            pass
        
        analyzer.close()
        print(f"\n✅ 测试完成！帧图片已保存到：{analyzer.output_dir}")
