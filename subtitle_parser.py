import re
from typing import List, Dict


def parse_srt(srt_path: str) -> List[Dict]:
    """解析SRT字幕文件"""
    with open(srt_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    pattern = r'(\d+)\s+(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})\s*\n([\s\S]*?)(?=\n\d+\s+\d{2}:\d{2}:\d{2}|\Z)'
    
    matches = re.findall(pattern, content)
    
    subtitles = []
    for match in matches:
        index = int(match[0])
        start_time = _time_to_seconds(match[1])
        end_time = _time_to_seconds(match[2])
        text = match[3].strip().replace('\n', ' ')
        
        subtitles.append({
            "index": index,
            "start_time": start_time,
            "end_time": end_time,
            "text": text
        })
    
    return subtitles


def _time_to_seconds(time_str: str) -> float:
    """将SRT时间格式转换为秒"""
    hours, minutes, seconds = time_str.split(':')
    seconds, milliseconds = seconds.split(',')
    
    total = (
        int(hours) * 3600 +
        int(minutes) * 60 +
        int(seconds) +
        int(milliseconds) / 1000
    )
    return round(total, 2)


def get_subtitle_at_time(subtitles: List[Dict], timestamp: float) -> str:
    """获取指定时间点的字幕"""
    for sub in subtitles:
        if sub["start_time"] <= timestamp <= sub["end_time"]:
            return sub["text"]
    return ""


def get_subtitles_in_range(subtitles: List[Dict], start: float, end: float) -> List[Dict]:
    """获取指定时间范围内的所有字幕"""
    result = []
    for sub in subtitles:
        if sub["start_time"] <= end and sub["end_time"] >= start:
            result.append(sub)
    return result


def format_subtitles_for_prompt(subtitles: List[Dict], max_items: int = 50) -> str:
    """格式化字幕用于AI提示"""
    if not subtitles:
        return "无字幕信息"
    
    if len(subtitles) > max_items:
        step = len(subtitles) // max_items
        subtitles = subtitles[::step][:max_items]
    
    lines = []
    for sub in subtitles:
        time_str = f"[{sub['start_time']:.1f}s]"
        lines.append(f"{time_str} {sub['text']}")
    
    return "\n".join(lines)
