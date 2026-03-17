import os
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Dict, Optional, Any
from openai import OpenAI
from config import (
    OPENAI_API_KEY, OPENAI_BASE_URL, MODEL_NAME, TEMP_DIR,
    AGENT_NUM_ITERATIONS, AGENT_TOOLS_PER_ITERATION,
    AGENT_LLM_TEMPERATURE, AGENT_LLM_TIMEOUT,
    ANALYSIS_LLM_TEMPERATURE, ANALYSIS_LLM_MAX_TOKENS
)

try:
    import imageio_ffmpeg
    FFMPEG_PATH = imageio_ffmpeg.get_ffmpeg_exe()
except:
    FFMPEG_PATH = 'ffmpeg'


class AgentAnalyzer:
    """
    智能 Agent 分析器 - 基于四种工具进行视频分析
    
    四个工具：
    1. analyze_audio_for_ranges - 分析音频找时间范围
    2. analyze_subtitle_for_ranges - 分析字幕找时间范围
    3. analyze_frame_analysis_for_ranges - 读取已分析的帧 JSON 找时间范围
    4. extract_extra_frames - 从原视频提取额外帧并用 Qwen VL 分析
    """
    
    def __init__(self, ai_analyzer=None):
        self.ai_analyzer = ai_analyzer
        self.client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
        self.model = MODEL_NAME
        
        # 帧分析数据（从 JSON 加载）
        self.frame_analysis_data = None
        
        # 输出目录
        self.output_dir = "temp"
        
        # Agent 配置
        self.config = {
            "analysis_priority": ["audio", "subtitle", "frame_analysis"],
            "min_time_range": 5,
        }
        
        # 当前资源
        self.current_resources = {}
        
        # 定义工具
        self.tools = self._define_tools()
    
    def _define_tools(self) -> List[Dict]:
        """定义 Agent 可用的工具"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "analyze_audio_for_ranges",
                    "description": "分析音频转录，找出符合需求的粗略时间范围。返回时间范围列表，token 消耗少",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "user_request": {
                                "type": "string",
                                "description": "用户的剪辑需求"
                            }
                        },
                        "required": ["user_request"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "analyze_subtitle_for_ranges",
                    "description": "分析字幕内容，找出符合需求的粗略时间范围。返回时间范围列表，token 消耗少",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "user_request": {
                                "type": "string",
                                "description": "用户的剪辑需求"
                            }
                        },
                        "required": ["user_request"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "analyze_frame_analysis_for_ranges",
                    "description": "读取已分析的帧内容 JSON（前置操作已完成），找出符合需求的帧。返回时间范围列表，token 消耗极少",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "user_request": {
                                "type": "string",
                                "description": "用户的剪辑需求"
                            }
                        },
                        "required": ["user_request"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "extract_extra_frames",
                    "description": "从原视频的指定时间范围内提取额外的帧，并使用 Qwen VL 进行视觉识别获取详细内容。当已有帧分析不够详细时使用",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "time_ranges": {
                                "type": "array",
                                "description": "需要额外分析的时间范围列表",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "start_time": {"type": "number", "description": "开始时间（秒）"},
                                        "end_time": {"type": "number", "description": "结束时间（秒）"}
                                    }
                                }
                            },
                            "frames_per_range": {
                                "type": "integer",
                                "description": "每个范围内提取的帧数，默认 3"
                            }
                        },
                        "required": ["time_ranges"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "submit_edit_analysis",
                    "description": "提交当前迭代的剪辑分析结果。每次迭代必须调用此工具提交分析，系统会在下一轮迭代中验证并改进",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "segments": {
                                "type": "array",
                                "description": "找到的片段列表",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "start_time": {"type": "number", "description": "开始时间（秒）"},
                                        "end_time": {"type": "number", "description": "结束时间（秒）"},
                                        "reason": {"type": "string", "description": "选择此片段的原因"},
                                        "relevance_score": {"type": "integer", "description": "相关性评分 0-100"},
                                        "evidence": {"type": "string", "description": "支持此片段的证据（如具体台词或画面描述）"}
                                    },
                                    "required": ["start_time", "end_time", "reason", "relevance_score"]
                                }
                            },
                            "analysis_summary": {
                                "type": "string",
                                "description": "本次分析的整体总结"
                            },
                            "confidence": {
                                "type": "integer",
                                "description": "对本次分析结果的信心程度 0-100"
                            }
                        },
                        "required": ["segments", "analysis_summary", "confidence"]
                    }
                }
            }
        ]
    
    def load_frame_analysis(self) -> bool:
        """加载帧分析结果"""
        frame_analysis_path = os.path.join(TEMP_DIR, "frame_analysis.json")
        
        if not os.path.exists(frame_analysis_path):
            print(f"帧分析文件不存在：{frame_analysis_path}")
            return False
        
        try:
            with open(frame_analysis_path, 'r', encoding='utf-8') as f:
                self.frame_analysis_data = json.load(f)
            print(f"成功加载帧分析结果：{self.frame_analysis_data.get('total_frames', 0)} 帧")
            return True
        except Exception as e:
            print(f"加载帧分析失败：{e}")
            return False
    
    def analyze_video(self, 
                     user_request: str,
                     audio_segments: Optional[List[Dict]] = None,
                     subtitles: Optional[List[Dict]] = None,
                     frames: Optional[List[Dict]] = None,
                     video_path: Optional[str] = None,
                     video_description: str = "") -> Dict:
        """
        使用 Agent 架构分析视频 - N 次迭代 * M 次工具调用模式
        
        Args:
            user_request: 用户的剪辑需求
            audio_segments: 音频转录分段
            subtitles: 字幕列表
            frames: 均匀提取的帧
            video_path: 视频文件路径
            video_description: 用户提供的视频简介（可选）
            
        Returns:
            分析结果字典
        """
        NUM_ITERATIONS = AGENT_NUM_ITERATIONS
        TOOLS_PER_ITERATION = AGENT_TOOLS_PER_ITERATION
        
        print(f"\n【Agent 分析】开始使用迭代式 Agent 架构分析视频...")
        print(f"用户需求：{user_request}")
        print(f"迭代配置：{NUM_ITERATIONS} 次迭代，每次最多 {TOOLS_PER_ITERATION} 次工具调用")
        print(f"可用资源：音频段数={len(audio_segments or [])}, 字幕数={len(subtitles or [])}, 帧数={len(frames or [])}")
        if video_description:
            print(f"📝 视频简介：{video_description}")
        
        self.current_resources = {
            "audio_segments": audio_segments,
            "subtitles": subtitles,
            "frames": frames,
            "video_path": video_path,
            "video_description": video_description,
            "original_user_request": user_request
        }
        
        video_duration = 0
        if audio_segments:
            video_duration = audio_segments[-1].get('end', 0) if audio_segments else 0
        self.current_resources["video_duration"] = video_duration
        
        total_prompt_tokens = 0
        total_completion_tokens = 0
        
        all_submitted_analyses = []
        final_analysis = None
        intermediate_results = {"audio": [], "subtitle": [], "frame": []}
        
        system_prompt = f"""你是一个智能视频分析 Agent。你的目标是找出视频中符合用户需求的片段。

【视频信息】
- 总时长: {video_duration:.1f} 秒 ({video_duration/60:.1f} 分钟)
- **重要**: 所有时间戳必须在 0 到 {video_duration:.1f} 秒之间，超出范围的时间戳无效！

【工作模式】
你将进行 {NUM_ITERATIONS} 轮迭代分析，每轮迭代：
1. 可以调用最多 {TOOLS_PER_ITERATION} 次工具进行分析（可以提前结束）
2. 必须在迭代结束前调用 submit_edit_analysis 提交你的分析结果
3. 所有 {NUM_ITERATIONS} 轮迭代都必须完成，每轮都要提交结果
4. 后续迭代应基于前一轮的结果进行改进和验证

【可用工具】
1. analyze_audio_for_ranges - 分析音频转录找时间范围（低成本）
2. analyze_subtitle_for_ranges - 分析字幕找时间范围（低成本）
3. analyze_frame_analysis_for_ranges - 读取已分析的帧 JSON 找时间范围（极低成本）
4. extract_extra_frames - 从原视频提取额外帧并分析（高成本，谨慎使用）
5. submit_edit_analysis - 提交本轮剪辑分析结果（每轮必须调用）

【关键规则】
1. **时间戳验证**：返回的时间范围内必须包含与需求相关的实际内容（台词或画面）
2. **证据要求**：每个片段必须提供 evidence 字段，说明该时间点的具体内容
3. **迭代改进**：如果上一轮有错误，下一轮必须修正
4. **诚实原则**：如果找不到相关内容，confidence 设为低分，不要编造

用户优先级配置：""" + str(self.config["analysis_priority"]) + """
"""
        
        if video_description:
            system_prompt += f"\n\n【视频简介】\n{video_description}"
        
        messages = [{"role": "system", "content": system_prompt}]
        
        for iteration in range(1, NUM_ITERATIONS + 1):
            print(f"\n{'='*60}")
            print(f"[迭代 {iteration}/{NUM_ITERATIONS}]")
            print(f"{'='*60}")
            
            if iteration == 1:
                messages.append({
                    "role": "user",
                    "content": f"用户剪辑需求：{user_request}\n\n请开始第 1 轮分析。使用工具进行分析后，务必调用 submit_edit_analysis 提交结果。"
                })
            else:
                prev_analysis = all_submitted_analyses[-1] if all_submitted_analyses else {}
                feedback = self._generate_feedback(prev_analysis, user_request)
                messages.append({
                    "role": "user",
                    "content": f"第 {iteration} 轮分析开始。\n\n【上一轮分析结果】\n{json.dumps(prev_analysis, ensure_ascii=False, indent=2)}\n\n【验证反馈】\n{feedback}\n\n请基于以上反馈改进分析，然后调用 submit_edit_analysis 提交改进后的结果。"
                })
            
            tool_calls_count = 0
            submitted = False
            
            while tool_calls_count < TOOLS_PER_ITERATION and not submitted:
                print(f"\n{'='*60}")
                print(f"[LLM 请求] 迭代 {iteration}, 工具调用: {tool_calls_count}/{TOOLS_PER_ITERATION}")
                print(f"[LLM 请求] Model: {self.model}")
                print(f"[LLM 请求] Messages 数量: {len(messages)}")
                for i, msg in enumerate(messages[-3:]):
                    role = msg.get('role', 'unknown')
                    content_preview = str(msg.get('content', '')) if msg.get('content') else ''
                    print(f"[LLM 请求] Message[{i}] role={role}: {content_preview}...")
                sys.stdout.flush()
                
                max_retries = 3
                last_error = None
                
                for retry in range(max_retries):
                    try:
                        response = self.client.chat.completions.create(
                            model=self.model,
                            messages=messages,
                            tools=self.tools,
                            tool_choice="auto",
                            temperature=AGENT_LLM_TEMPERATURE,
                            timeout=AGENT_LLM_TIMEOUT
                        )
                        print(f"[LLM 响应] 状态: 成功")
                        break
                    except Exception as e:
                        last_error = e
                        error_msg = str(e)
                        print(f"[LLM 响应] 错误 (尝试 {retry + 1}/{max_retries}): {error_msg}")
                        
                        if retry < max_retries - 1:
                            wait_time = 2 ** retry
                            print(f"⚠️ 等待 {wait_time} 秒后重试...")
                            time.sleep(wait_time)
                        else:
                            if "429" in error_msg or "thought_signature" in error_msg:
                                print(f"⚠️ API 兼容性错误，尝试简化消息后重试...")
                                messages.append({"role": "user", "content": "请继续分析并提交结果。"})
                                try:
                                    response = self.client.chat.completions.create(
                                        model=self.model,
                                        messages=messages,
                                        tools=self.tools,
                                        tool_choice="auto",
                                        temperature=AGENT_LLM_TEMPERATURE,
                                        timeout=AGENT_LLM_TIMEOUT
                                    )
                                    print(f"[LLM 响应] 重试成功")
                                except Exception as e2:
                                    print(f"[LLM 响应] 重试失败：{e2}")
                                    if final_analysis:
                                        break
                                    return {"analysis": f"API 调用失败：{e2}", "matching_segments": [], "recommendations": ""}
                            else:
                                print(f"[LLM 响应] 失败：{e}")
                                if final_analysis:
                                    break
                                return {"analysis": f"API 调用失败：{e}", "matching_segments": [], "recommendations": ""}
                
                if hasattr(response, 'usage'):
                    total_prompt_tokens += response.usage.prompt_tokens
                    total_completion_tokens += response.usage.completion_tokens
                    print(f"[LLM 响应] Token: Prompt={response.usage.prompt_tokens}, Completion={response.usage.completion_tokens}")
                
                finish_reason = response.choices[0].finish_reason
                print(f"[LLM 响应] finish_reason: {finish_reason}")
                
                if finish_reason in ["end_turn", "stop"]:
                    content = response.choices[0].message.content or ""
                    print(f"[LLM 响应] Agent 回复:\n{content}")
                    print(f"{'='*60}\n")
                    messages.append({"role": "assistant", "content": content})
                    
                    if not submitted:
                        print("⚠️ 本轮迭代未提交分析，自动结束本轮")
                    break
                
                elif finish_reason == "tool_calls":
                    tool_calls = response.choices[0].message.tool_calls
                    
                    if response.choices[0].message.content:
                        print(f"[LLM 响应] Agent 备注: {response.choices[0].message.content}")
                    
                    messages.append({
                        "role": "assistant",
                        "content": response.choices[0].message.content or "",
                        "tool_calls": [{"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}} for tc in tool_calls]
                    })
                    
                    print(f"[工具调用] 共 {len(tool_calls)} 个工具:")
                    
                    for i, tool_call in enumerate(tool_calls, 1):
                        tool_calls_count += 1
                        print(f"\n  ┌─ [{tool_calls_count}] {tool_call.function.name}")
                        
                        try:
                            args = json.loads(tool_call.function.arguments)
                            print(f"  │  完整参数：{json.dumps(args, ensure_ascii=False)}")
                            
                            if tool_call.function.name == "submit_edit_analysis":
                                result = self._handle_submit_analysis(args)
                                final_analysis = args
                                all_submitted_analyses.append(args)
                                submitted = True
                                print(f"  │  ✓ 已提交本轮分析")
                            else:
                                result = self._execute_tool(tool_call.function.name, args)
                                if tool_call.function.name == "analyze_audio_for_ranges":
                                    intermediate_results["audio"] = result.get("ranges", [])
                                elif tool_call.function.name == "analyze_subtitle_for_ranges":
                                    intermediate_results["subtitle"] = result.get("ranges", [])
                                elif tool_call.function.name == "analyze_frame_analysis_for_ranges":
                                    intermediate_results["frame"] = result.get("ranges", [])
                            
                            result_str = json.dumps(result, ensure_ascii=False, indent=2)
                            print(f"  │  完整结果：{result_str}")
                            print(f"  └─ 完成")
                            
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "name": tool_call.function.name,
                                "content": json.dumps(result, ensure_ascii=False)
                            })
                            
                        except Exception as e:
                            print(f"  │  ✗ 错误：{e}")
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "name": tool_call.function.name,
                                "content": json.dumps({"error": str(e)}, ensure_ascii=False)
                            })
                        
                        print(f"  └─ 完成")
                        
                        if submitted:
                            break
                
                elif finish_reason in ["content_filter", "length"]:
                    print(f"⚠️ API 返回 {finish_reason}")
                    messages.append({"role": "assistant", "content": response.choices[0].message.content or ""})
                    messages.append({"role": "user", "content": "请继续"})
                    continue
            
            if not submitted:
                print(f"\n⚠️ 迭代 {iteration} 未提交分析结果，自动生成默认分析")
                default_analysis = self._extract_default_analysis(intermediate_results, user_request)
                all_submitted_analyses.append(default_analysis)
                final_analysis = default_analysis
        
        print(f"\n{'='*60}")
        print(f"【Agent 分析完成】")
        print(f"总 Token: Prompt={total_prompt_tokens}, Completion={total_completion_tokens}")
        print(f"{'='*60}")
        
        if final_analysis:
            return {
                "analysis": final_analysis.get("analysis_summary", ""),
                "matching_segments": final_analysis.get("segments", []),
                "recommendations": f"信心度: {final_analysis.get('confidence', 0)}%",
                "iterations": NUM_ITERATIONS,
                "all_analyses": all_submitted_analyses
            }
        
        all_ranges = intermediate_results["audio"] + intermediate_results["subtitle"] + intermediate_results["frame"]
        if all_ranges:
            seen = set()
            unique_ranges = []
            for r in all_ranges:
                key = (r.get("start_time"), r.get("end_time"))
                if key not in seen:
                    seen.add(key)
                    start_t = r.get("start_time", 0)
                    end_t = r.get("end_time", 0)
                    if 0 <= start_t <= video_duration and 0 <= end_t <= video_duration:
                        unique_ranges.append(r)
                    else:
                        print(f"    [过滤] 无效时间戳 {start_t}s-{end_t}s (视频时长: {video_duration}s)")
            unique_ranges.sort(key=lambda x: x.get("start_time", 0))
            
            print(f"\n📋 使用中间结果: {len(unique_ranges)} 个片段")
            return {
                "analysis": "Agent 分析中断，使用已收集的中间结果",
                "matching_segments": unique_ranges,
                "recommendations": "分析未完成，结果可能不完整",
                "iterations": iteration,
                "partial": True
            }
        
        return {"analysis": "未能完成分析", "matching_segments": [], "recommendations": ""}
    
    def _handle_submit_analysis(self, args: Dict) -> Dict:
        """处理提交分析的工具调用"""
        segments = args.get("segments", [])
        summary = args.get("analysis_summary", "")
        confidence = args.get("confidence", 0)
        
        print(f"\n  📋 提交分析:")
        print(f"     总结: {summary[:100]}")
        print(f"     信心度: {confidence}%")
        print(f"     片段数: {len(segments)}")
        
        for i, seg in enumerate(segments, 1):
            start = seg.get("start_time", 0)
            end = seg.get("end_time", 0)
            reason = seg.get("reason", "")
            evidence = seg.get("evidence", "无证据")
            print(f"     [{i}] {start}s-{end}s: {reason}")
            print(f"         证据: {evidence[:50]}")
        
        return {"status": "submitted", "segments_count": len(segments)}
    
    def _extract_default_analysis(self, intermediate_results: Dict, user_request: str) -> Dict:
        """从中间结果提取默认分析（当 agent 未提交时）"""
        all_ranges = (
            intermediate_results.get("audio", []) + 
            intermediate_results.get("subtitle", []) + 
            intermediate_results.get("frame", [])
        )
        
        if not all_ranges:
            return {
                "analysis_summary": "未能找到匹配片段",
                "segments": [],
                "confidence": 0
            }
        
        seen = set()
        unique_ranges = []
        for r in all_ranges:
            key = (r.get("start_time"), r.get("end_time"))
            if key not in seen:
                seen.add(key)
                unique_ranges.append(r)
        
        unique_ranges.sort(key=lambda x: x.get("start_time", 0))
        
        segments = []
        for r in unique_ranges[:10]:
            segments.append({
                "start_time": r.get("start_time", 0),
                "end_time": r.get("end_time", 0),
                "reason": r.get("reason", ""),
                "relevance_score": r.get("relevance_score", 50),
                "evidence": r.get("evidence", "")
            })
        
        avg_score = sum(s["relevance_score"] for s in segments) / len(segments) if segments else 0
        
        return {
            "analysis_summary": f"从中间结果提取了 {len(segments)} 个片段",
            "segments": segments,
            "confidence": int(avg_score)
        }
    
    def _generate_feedback(self, prev_analysis: Dict, user_request: str) -> str:
        """生成对上一轮分析的反馈"""
        segments = prev_analysis.get("segments", [])
        if not segments:
            return "上一轮未找到任何片段，请重新分析。"
        
        feedback_parts = []
        audio_segments = self.current_resources.get("audio_segments", [])
        
        for seg in segments:
            start = seg.get("start_time", 0)
            end = seg.get("end_time", 0)
            evidence = seg.get("evidence", "")
            
            matched_texts = []
            for asg in audio_segments:
                if asg.get("start", 0) >= start - 2 and asg.get("end", 0) <= end + 2:
                    text = asg.get("text", "").strip()
                    if text:
                        matched_texts.append(text)
            
            actual_content = " ".join(matched_texts[:3]) if matched_texts else "无对应音频"
            
            keywords = self._extract_keywords(user_request)
            has_keyword = any(kw in actual_content for kw in keywords)
            
            if not has_keyword and not evidence:
                feedback_parts.append(f"⚠️ 片段 {start}s-{end}s: 时间范围内未找到相关关键词，实际内容为「{actual_content[:50]}」，请验证或删除")
            elif not evidence:
                feedback_parts.append(f"⚠️ 片段 {start}s-{end}s: 缺少 evidence 字段，请补充该时间点的具体内容")
        
        if feedback_parts:
            return "\n".join(feedback_parts)
        return "上一轮分析基本正确，可以进一步优化或确认提交。"
    
    def _extract_keywords(self, text: str) -> List[str]:
        """从用户需求中提取关键词"""
        keywords = []
        import re
        matches = re.findall(r'[\u4e00-\u9fa5]{2,}', text)
        keywords.extend(matches[:5])
        return keywords
    
    def _execute_tool(self, tool_name: str, args: Dict) -> Any:
        """执行工具"""
        original_request = self.current_resources.get("original_user_request", "")
        agent_request = args.get("user_request", "")
        
        if agent_request and agent_request != original_request:
            combined_request = f"【用户原始需求】{original_request}\n\n【分析重点】{agent_request}"
            print(f"    [DEBUG] 合并需求: {combined_request[:100]}")
        else:
            combined_request = original_request
        
        if tool_name == "analyze_audio_for_ranges":
            return self._analyze_audio_for_ranges(combined_request)
        
        elif tool_name == "analyze_subtitle_for_ranges":
            return self._analyze_subtitle_for_ranges(combined_request)
        
        elif tool_name == "analyze_frame_analysis_for_ranges":
            return self._analyze_frame_analysis_for_ranges(combined_request)
        
        elif tool_name == "extract_extra_frames":
            return self._extract_extra_frames(
                args.get("time_ranges", []),
                args.get("frames_per_range", 3)
            )
        
        else:
            return {"error": f"未知工具：{tool_name}"}
    
    def _fix_truncated_json(self, content: str) -> dict:
        """修复被截断的 JSON，尝试提取所有完整的 range 对象"""
        import re
        
        # 方法1: 找到最后一个完整的对象（以 }, 或 } 结尾），补全 ]}
        last_complete = max(content.rfind('},'), content.rfind('}\n    ]'))
        if last_complete > 0:
            # 截断到最后一个完整对象，补全 ]}
            truncated = content[:last_complete + 1]
            # 确保 ranges 数组和外层对象都闭合
            fixed = truncated.rstrip(',').rstrip() + '\n    ]\n}'
            try:
                return json.loads(fixed)
            except Exception:
                pass
        
        # 方法2: 用正则提取所有完整的 range 对象
        pattern = r'\{\s*"start_time"\s*:\s*([\d.]+)\s*,\s*"end_time"\s*:\s*([\d.]+)\s*,\s*"reason"\s*:\s*"([^"]*)"\s*\}'
        matches = re.findall(pattern, content)
        if matches:
            ranges = [
                {"start_time": float(m[0]), "end_time": float(m[1]), "reason": m[2]}
                for m in matches
            ]
            print(f"    [DEBUG] 正则提取到 {len(ranges)} 个完整 range 对象")
            return {"ranges": ranges}
        
        return {"ranges": []}

    def _analyze_audio_for_ranges(self, user_request: str) -> Dict:
        """分析音频找时间范围"""
        print("    [执行] _analyze_audio_for_ranges 开始...")
        audio_segments = self.current_resources.get("audio_segments", [])
        if not audio_segments:
            return {"ranges": [], "note": "无可用音频数据"}
        
        total = len(audio_segments)
        last_end = audio_segments[-1].get('end', 0) if audio_segments else 0
        print(f"    [DEBUG] 音频段总数：{total}，时间范围：0s - {last_end:.1f}s")
        
        audio_text = "\n".join([
            f"[{seg.get('start', 0):.1f}s] {seg.get('text', '')}"
            for seg in audio_segments
        ])
        
        prompt = f"""用户剪辑需求：{user_request}

音频转录内容（时间戳单位为秒，覆盖整个视频）：
{audio_text}

请分析音频内容，找出符合用户需求的**最重要 3-5 个**时间范围。返回 JSON 格式：
{{
    "ranges": [
        {{"start_time": 开始时间（纯数字，单位秒）, "end_time": 结束时间（纯数字，单位秒）, "reason": "简短原因（20 字以内）, "relevance_score": 0-100的整数}}
    ]
}}

【重要规则】：
1. start_time 和 end_time 必须是纯数字（秒），来自上方转录文本中实际出现的时间戳
2. **必须验证**：返回的时间范围内，音频文本中必须包含与需求相关的关键词，否则不要返回
3. relevance_score 根据音频内容与需求的匹配程度打分（100=完全符合，0=完全不符）
4. 如果找不到符合需求的内容，返回空的 ranges 列表，不要猜测或编造时间
5. reason 要简短精炼（20 字以内）
6. 只返回 JSON，不要其他内容"""
        
        print(f"\n    {'='*50}")
        print(f"    [LLM 请求] 音频分析")
        print(f"    [LLM 请求] Model: {self.model}")
        print(f"    [LLM 请求] Prompt:\n{prompt}")
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=ANALYSIS_LLM_MAX_TOKENS,
                temperature=ANALYSIS_LLM_TEMPERATURE
            )
            
            content = response.choices[0].message.content
            print(f"    [LLM 响应] 完整内容:\n{content}")
            print(f"    {'='*50}\n")
            
            content = content.replace('```json', '').replace('```', '').strip()
            
            try:
                result = json.loads(content)
            except json.JSONDecodeError as e:
                print(f"    [DEBUG] 音频分析 JSON 解析错误：{e}，尝试修复...")
                result = self._fix_truncated_json(content)
                if not result.get('ranges'):
                    print(f"    [DEBUG] 修复失败，完整内容：{content}")
                    return {"ranges": [], "note": f"分析失败：{e}"}
            
            ranges = result.get('ranges', [])
            print(f"    [执行] 音频分析找到 {len(ranges)} 个时间范围")
            
            for i, r in enumerate(ranges):
                start_t = r.get('start_time', 0)
                end_t = r.get('end_time', 0)
                matched_texts = []
                for seg in audio_segments:
                    seg_start = seg.get('start', 0)
                    seg_end = seg.get('end', 0)
                    if seg_start >= start_t - 2 and seg_end <= end_t + 2:
                        text = seg.get('text', '').strip()
                        if text:
                            matched_texts.append(f"[{seg_start:.1f}s] {text}")
                if matched_texts:
                    print(f"      [{i+1}] {start_t:.1f}s - {end_t:.1f}s: {r.get('reason', '')}")
                    for mt in matched_texts[:5]:
                        print(f"          {mt}")
                    if len(matched_texts) > 5:
                        print(f"          ... 等共 {len(matched_texts)} 句")
            
            return {"ranges": ranges, "count": len(ranges)}
            
        except Exception as e:
            print(f"    [执行] 音频分析失败：{e}")
            return {"ranges": [], "note": f"分析失败：{e}"}
    
    def _analyze_subtitle_for_ranges(self, user_request: str) -> Dict:
        """分析字幕找时间范围"""
        print("    [执行] _analyze_subtitle_for_ranges 开始...")
        subtitles = self.current_resources.get("subtitles", [])
        if not subtitles:
            return {"ranges": [], "note": "无可用字幕数据"}
        
        total = len(subtitles)
        print(f"    [DEBUG] 字幕总数：{total}")
        
        subtitle_text = "\n".join([
            f"[{sub.get('start_time', 0):.1f}s] {sub.get('text', '')}"
            for sub in subtitles
        ])
        
        prompt = f"""用户剪辑需求：{user_request}

字幕内容（时间戳单位为秒，覆盖整个视频）：
{subtitle_text}

请分析字幕内容，找出符合用户需求的**最重要 3-5 个**时间范围。返回 JSON 格式：
{{
    "ranges": [
        {{"start_time": 开始时间（纯数字，单位秒）, "end_time": 结束时间（纯数字，单位秒）, "reason": "简短原因（20 字以内）, "relevance_score": 0-100的整数}}
    ]
}}

【重要规则】：
1. start_time 和 end_time 必须是纯数字（秒），来自上方字幕中实际出现的时间戳
2. **必须验证**：返回的时间范围内，字幕文本中必须包含与需求相关的关键词，否则不要返回
3. relevance_score 根据字幕内容与需求的匹配程度打分（100=完全符合，0=完全不符）
4. 如果找不到符合需求的内容，返回空的 ranges 列表，不要猜测或编造时间
5. reason 要简短精炼（20 字以内）
6. 只返回 JSON，不要其他内容"""
        
        print(f"\n    {'='*50}")
        print(f"    [LLM 请求] 字幕分析")
        print(f"    [LLM 请求] Model: {self.model}")
        print(f"    [LLM 请求] Prompt:\n{prompt}")
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=ANALYSIS_LLM_MAX_TOKENS,
                temperature=ANALYSIS_LLM_TEMPERATURE
            )
            
            content = response.choices[0].message.content
            print(f"    [LLM 响应] 完整内容:\n{content}")
            print(f"    {'='*50}\n")
            
            content = content.replace('```json', '').replace('```', '').strip()
            
            try:
                result = json.loads(content)
            except json.JSONDecodeError as e:
                print(f"    [DEBUG] 字幕分析 JSON 解析错误：{e}，尝试修复...")
                result = self._fix_truncated_json(content)
                if not result.get('ranges'):
                    print(f"    [DEBUG] 修复失败，完整内容：{content}")
                    return {"ranges": [], "note": f"分析失败：{e}"}
            
            ranges = result.get('ranges', [])
            print(f"    [执行] 字幕分析找到 {len(ranges)} 个时间范围")
            
            for i, r in enumerate(ranges):
                start_t = r.get('start_time', 0)
                end_t = r.get('end_time', 0)
                matched_texts = []
                for sub in subtitles:
                    sub_start = sub.get('start_time', 0)
                    sub_end = sub.get('end_time', 0)
                    if sub_start >= start_t - 2 and sub_end <= end_t + 2:
                        text = sub.get('text', '').strip()
                        if text:
                            matched_texts.append(f"[{sub_start:.1f}s] {text}")
                if matched_texts:
                    print(f"      [{i+1}] {start_t:.1f}s - {end_t:.1f}s: {r.get('reason', '')}")
                    for mt in matched_texts[:5]:
                        print(f"          {mt}")
                    if len(matched_texts) > 5:
                        print(f"          ... 等共 {len(matched_texts)} 句")
            
            return {"ranges": ranges, "count": len(ranges)}
            
        except Exception as e:
            print(f"    [执行] 字幕分析失败：{e}")
            return {"ranges": [], "note": f"分析失败：{e}"}
    
    def _analyze_frame_analysis_for_ranges(self, user_request: str) -> Dict:
        """从已分析的帧 JSON 中查找符合需求的帧"""
        print("    [执行] _analyze_frame_analysis_for_ranges 开始...")
        
        if not self.frame_analysis_data:
            self.load_frame_analysis()
        
        if not self.frame_analysis_data:
            return {"ranges": [], "note": "无可用帧分析数据"}
        
        frames = self.frame_analysis_data.get('frames', [])
        if not frames:
            return {"ranges": [], "note": "帧分析数据为空"}
        
        frame_texts = []
        for frame in frames[:100]:
            ts = frame.get('timestamp', 0)
            analysis = frame.get('analysis', {})
            if analysis.get('success'):
                scene_type = analysis.get('analysis', {}).get('scene_type', '')
                objects = analysis.get('analysis', {}).get('main_objects', [])
                action = analysis.get('analysis', {}).get('action', '')
                desc = analysis.get('analysis', {}).get('description', '')
                
                text = f"[{ts}] 场景:{scene_type}, 物体:{','.join(objects[:5])}, 动作:{action}, 描述:{desc}"
                frame_texts.append(text)
        
        frame_text = "\n".join(frame_texts)
        
        prompt = f"""用户剪辑需求：{user_request}

已分析的视频帧内容：
{frame_text}

请找出符合用户需求的帧，返回**最重要 3-5 个**时间范围。返回 JSON 格式：
{{
    "ranges": [
        {{"start_time": 开始时间（秒）, "end_time": 结束时间（秒）, "reason": "简短原因（20 字以内）, "relevance_score": 0-100的整数}}
    ]
}}

要求：
1. **必须验证**：返回的时间范围内，帧画面描述必须与需求相关，否则不要返回
2. relevance_score 根据帧画面与需求的匹配程度打分（100=完全符合，0=完全不符）
3. 只返回最相关的 3-5 个时间范围
4. reason 要简短精炼（20 字以内）
5. 只返回 JSON，不要其他内容"""
        
        print(f"\n    {'='*50}")
        print(f"    [LLM 请求] 帧分析查询")
        print(f"    [LLM 请求] Model: {self.model}")
        print(f"    [LLM 请求] Prompt:\n{prompt}")
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=ANALYSIS_LLM_MAX_TOKENS,
                temperature=ANALYSIS_LLM_TEMPERATURE
            )
            
            content = response.choices[0].message.content
            print(f"    [LLM 响应] 完整内容:\n{content}")
            print(f"    {'='*50}\n")
            
            content = content.replace('```json', '').replace('```', '').strip()
            
            try:
                result = json.loads(content)
            except json.JSONDecodeError as e:
                print(f"    [DEBUG] JSON 解析错误：{e}，尝试修复...")
                result = self._fix_truncated_json(content)
                if not result.get('ranges'):
                    print(f"    [DEBUG] 修复失败，完整内容：{content}")
                    return {"ranges": [], "note": f"分析失败：{e}"}
            
            ranges = result.get('ranges', [])
            print(f"    [执行] 从帧分析中找到 {len(ranges)} 个时间范围")
            
            temp_dir = Path(self.output_dir)
            for i, r in enumerate(ranges):
                start_t = r.get('start_time', 0)
                end_t = r.get('end_time', 0)
                
                frame_files = []
                for f in temp_dir.glob("frame_*.jpg"):
                    try:
                        name = f.stem.replace('frame_', '').replace('s', '')
                        ts = float(name)
                        if start_t - 5 <= ts <= end_t + 5:
                            frame_files.append(f.name)
                    except:
                        pass
                
                frame_files.sort(key=lambda x: float(x.replace('frame_', '').replace('s.jpg', '').replace('.jpg', '')))
                
                print(f"      [{i+1}] {start_t:.1f}s - {end_t:.1f}s: {r.get('reason', '')}")
                if frame_files:
                    for fn in frame_files[:3]:
                        print(f"          {fn}")
                    if len(frame_files) > 3:
                        print(f"          ... 等共 {len(frame_files)} 张")
            
            return {"ranges": ranges, "count": len(ranges)}
            
        except Exception as e:
            print(f"    [执行] 分析帧分析数据失败：{e}")
            return {"ranges": [], "note": f"分析失败：{e}"}
    
    def _extract_extra_frames(self, time_ranges: List[Dict], frames_per_range: int = 3) -> Dict:
        """从原视频提取额外帧并使用 AI 模型分析"""
        print(f"    [执行] _extract_extra_frames 开始，范围数：{len(time_ranges)}")
        
        video_path = self.current_resources.get("video_path")
        video_duration = self.current_resources.get("video_duration", 0)
        
        if not video_path:
            return {"frames": [], "note": "无视频路径"}
        
        from openai import OpenAI
        from config import OPENAI_API_KEY, OPENAI_BASE_URL, MODEL_NAME
        
        ai_client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
        ai_model = MODEL_NAME
        
        extra_frames = []
        
        for tr in time_ranges:
            start = tr.get('start_time', 0)
            end = tr.get('end_time', 0)
            
            if end <= start:
                continue
            
            if start > video_duration or end > video_duration:
                print(f"    [警告] 范围 {start:.1f}s-{end:.1f}s 超出视频时长 {video_duration:.1f}s，跳过")
                continue
            
            start = max(0, start)
            end = min(end, video_duration)
            
            duration = end - start
            num_frames = max(2, min(frames_per_range, int(duration / 2)))
            
            print(f"    [执行] 范围 {start:.1f}s-{end:.1f}s，提取 {num_frames} 帧")
            
            for i in range(num_frames):
                ts = start + (duration * i / num_frames)
                ts_rounded = round(ts, 2)
                
                output_path = os.path.join(TEMP_DIR, f"frame_{ts_rounded}s.jpg")
                
                cmd = [
                    FFMPEG_PATH, '-y', '-ss', str(ts),
                    '-i', video_path,
                    '-vframes', '1', '-q:v', '2',
                    output_path
                ]
                
                try:
                    subprocess.run(cmd, capture_output=True, timeout=10)
                    if os.path.exists(output_path):
                        # 使用 Gemini 分析
                        with open(output_path, 'rb') as f:
                            import base64
                            image_data = f.read()
                            base64_image = base64.b64encode(image_data).decode('utf-8')
                        
                        # 调用 AI API
                        response = ai_client.chat.completions.create(
                            model=ai_model,
                            messages=[{
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
                                        "text": """请分析这个视频帧的内容，返回 JSON 格式：
{"scene_type":"场景类型","main_objects":["物体1"],"text_visible":"可见文字","action":"动作","description":"简短描述"}
要求：1. 只返回 JSON，不要其他内容；2. 所有字符串用双引号；3. 确保 JSON 完整（包含所有闭合括号）"""
                                    }
                                ]
                            }],
                            max_tokens=1000  # 增加 token 限制，避免 JSON 被截断
                        )
                        
                        content = response.choices[0].message.content
                        print(f"      [DEBUG] AI 返回：{content}")
                        
                        # 解析 JSON
                        try:
                            content = content.replace('```json', '').replace('```', '').strip()
                            
                            # 尝试解析完整内容
                            try:
                                analysis = json.loads(content)
                            except json.JSONDecodeError as e1:
                                # 如果失败，尝试修复被截断的 JSON 或多个 JSON 对象
                                print(f"      [DEBUG] JSON 解析失败，尝试修复：{e1}")
                                
                                # 方案 1: 处理多个 JSON 对象（取第一个）
                                # 找到第一个 { 和最后一个 } 之间的内容
                                import re
                                
                                # 尝试提取单个 JSON 对象
                                match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', content)
                                if match:
                                    fixed_content = match.group(0)
                                else:
                                    fixed_content = content
                                
                                # 移除不完整的行
                                lines = fixed_content.split('\n')
                                fixed_lines = []
                                for line in lines:
                                    # 跳过明显不完整的行（引号数量 < 4）
                                    if line.count('"') < 4:
                                        continue
                                    fixed_lines.append(line.rstrip(','))
                                
                                fixed_content = '\n'.join(fixed_lines)
                                
                                # 补全缺失的闭合括号
                                open_braces = fixed_content.count('{')
                                close_braces = fixed_content.count('}')
                                if open_braces > close_braces:
                                    fixed_content += '}' * (open_braces - close_braces)
                                
                                print(f"      [DEBUG] 修复后：{fixed_content}")
                                
                                analysis = json.loads(fixed_content)
                            
                            extra_frames.append({
                                "timestamp": ts_rounded,
                                "time_range": f"{start:.1f}s-{end:.1f}s",
                                "analysis": analysis,
                                "image_path": output_path,
                                "model": ai_model
                            })
                            print(f"      ✓ AI 分析帧 {ts:.1f}s: {analysis.get('description', '')[:50]}")
                        
                        except Exception as e:
                            print(f"      ✗ 解析 AI 响应失败：{e}")
                            print(f"      [DEBUG] 原始内容：{content}")
                            
                            # 即使解析失败，也保存原始文本作为 description
                            extra_frames.append({
                                "timestamp": ts_rounded,
                                "time_range": f"{start:.1f}s-{end:.1f}s",
                                "analysis": {
                                    "description": content,
                                    "scene_type": "未知",
                                    "main_objects": []
                                },
                                "image_path": output_path,
                                "model": ai_model,
                                "parse_error": str(e)
                            })
                
                except Exception as e:
                    print(f"    [执行] 提取或分析帧 {ts:.1f}s 失败：{e}")
                    continue
        
        print(f"    [执行] 额外分析了 {len(extra_frames)} 帧（使用 {ai_model}）")
        return {"frames": extra_frames, "count": len(extra_frames), "model": ai_model}
    
    def _extract_final_result(self, messages: List[Dict]) -> Dict:
        """从消息历史中提取最终结果"""
        # 收集所有 tool 返回的 ranges
        all_ranges = []
        
        for msg in reversed(messages):
            if msg.get("role") == "tool":
                try:
                    content = json.loads(msg["content"])
                    # 查找 ranges 或 segments
                    ranges = content.get("ranges", content.get("segments", []))
                    if ranges:
                        all_ranges.extend(ranges)
                except:
                    pass
        
        # 如果有 ranges，返回结果
        if all_ranges:
            # 去重（按 start_time）
            unique_ranges = []
            seen_starts = set()
            for r in all_ranges:
                start = r.get("start_time", 0)
                if start not in seen_starts:
                    unique_ranges.append(r)
                    seen_starts.add(start)
            
            # 转换为 segments 格式
            segments = []
            for r in unique_ranges:
                score = r.get("relevance_score")
                if score is None:
                    score = 60  # 如果 AI 没有返回评分，使用默认值 60
                segments.append({
                    "start_time": r.get("start_time", 0),
                    "end_time": r.get("end_time", 0),
                    "reason": r.get("reason", ""),
                    "relevance_score": score
                })
            
            return {
                "analysis": "Agent 分析完成",
                "matching_segments": segments,
                "recommendations": f"根据对视频内容的分析，为您找到了 {len(segments)} 个相关片段。"
            }
        
        return {
            "analysis": "无法提取结果",
            "matching_segments": [],
            "recommendations": ""
        }
    
    def set_config(self, config: Dict):
        """设置 Agent 配置"""
        self.config.update(config)
        print(f"Agent 配置已更新：{self.config}")
