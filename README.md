# AgentMontage - AI智能视频剪辑工具

## 📋 项目简介

AgentMontage 是一个基于AI的智能视频剪辑工具（甚至它本身都是ai生成的），可以自动分析视频内容并根据你的描述剪辑出符合主题的片段。

## 🚀 快速开始

### 1. 安装依赖

在项目目录下运行：

```bash
pip install -r requirements.txt
```

如果提示找不到pip，尝试：
```bash
python -m pip install -r requirements.txt
```

### 2. 配置API密钥

确保 `.env` 文件已正确配置：

```
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://api.vectorengine.ai/v1
MODEL_NAME=gemini-3-flash-preview
```

### 3. 启动应用

**Windows用户：**
- 双击 `start.bat` 文件
- 或在命令提示符中运行：`python main.py`

**Mac/Linux用户：**
```bash
python main.py
```

### 4. 访问应用

应用启动后，在浏览器中打开：
```
http://localhost:7860
```

## 📖 使用说明

### 步骤1：上传视频
点击"上传视频"按钮，选择你要剪辑的视频文件。

### 步骤2：处理视频
点击"处理视频"按钮，系统会提取视频的关键帧并显示视频信息。

### 步骤3：描述剪辑需求
在"剪辑内容描述"框中输入你想要剪辑的内容，例如：
- "找出所有有猫出现的片段"
- "提取所有人物对话的场景"
- "找出所有户外运动的镜头"
- "剪辑所有有汽车的画面"

### 步骤4：分析视频
点击"🔍 分析视频"按钮，AI会分析视频内容并找出匹配的片段。

### 步骤5：查看分析结果
查看分析结果，包括：
- 整体分析
- 匹配的片段列表（包含时间、描述、相关性评分）
- 剪辑建议

### 步骤6：调整参数（可选）
- **合并为一个视频**：勾选则将所有片段合并为一个视频，否则输出多个独立片段
- **最低相关性评分**：只剪辑相关性评分高于此值的片段（0-100）
- **淡入淡出时长**：片段之间的转场效果时长（秒）

### 步骤7：开始剪辑
点击"✂️ 开始剪辑"按钮，等待剪辑完成。

### 步骤8：下载视频
剪辑完成后，可以在线预览或下载输出视频。

## 📁 项目结构

```
AiCut/
├── main.py              # 主程序入口（Gradio Web界面）
├── video_processor.py   # 视频处理模块（帧提取、信息获取）
├── ai_analyzer.py       # AI分析模块（使用vlm视觉分析）
├── agent_analyzer.py    # 代理分析模块（音频和字幕分析）
├── video_editor.py      # 视频剪辑模块（片段提取、合并）
├── funasr_asr.py        # 音频识别模块（使用FunASR）
├── subtitle_parser.py   # 字幕解析模块
├── smart_frame_analyzer_v2.py  # 智能帧分析模块
├── config.py            # 配置文件
├── run_with_log.py      # 带日志的运行脚本
├── requirements.txt     # 依赖包列表
├── .env                 # 环境变量配置（API密钥等）
├── .env.example         # 环境变量示例
├── start.bat            # Windows启动脚本
├── temp/                # 临时文件目录（视频帧）
├── output/              # 输出视频目录
└── logs/                # 日志文件目录
```

## ⚙️ 配置说明

### API配置

在 `.env` 文件中配置：

```
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://api.vectorengine.ai/v1
MODEL_NAME=gemini-3-flash-preview
```

### 其他配置

在 `config.py` 中可以调整：

- `FRAME_EXTRACTION_INTERVAL`: 帧提取间隔（秒）
- `MAX_FRAMES_PER_ANALYSIS`: 每次分析的最大帧数
- `TEMP_DIR`: 临时文件目录
- `OUTPUT_DIR`: 输出文件目录

## 🔧 故障排除

### 问题1：ModuleNotFoundError

**错误信息**：`ModuleNotFoundError: No module named 'xxx'`

**解决方案**：
```bash
pip install -r requirements.txt
```

### 问题2：API调用失败

**错误信息**：`API调用错误` 或 `Invalid API key`

**解决方案**：
- 检查 `.env` 文件中的API密钥是否正确
- 确认API服务是否可用
- 检查网络连接

### 问题3：视频处理失败

**错误信息**：`无法提取视频帧` 或 `视频处理错误`

**解决方案**：
- 确认视频文件格式是否支持（MP4, AVI, MOV, MKV等）
- 检查视频文件是否损坏
- 尝试转换视频格式

### 问题4：剪辑输出失败

**错误信息**：`剪辑失败` 或 `导出视频错误`

**解决方案**：
- 检查是否有足够的磁盘空间
- 确认输出目录有写入权限
- 尝试降低相关性评分以获得更多片段

## 📝 注意事项

1. **API费用**：使用AI分析会产生API费用，请注意控制使用量
2. **视频时长**：长视频分析时间较长，建议先测试短视频
3. **描述准确性**：描述越具体，AI分析越准确
4. **相关性评分**：相关性评分越高，匹配越精准，但可能片段较少
5. **临时文件**：`temp` 目录会存储大量图片，定期清理以节省空间
6. **日志文件**：`logs` 目录会存储分析日志，可用于排查问题

## 🎯 使用技巧

1. **分步测试**：先用短视频测试，确认功能正常后再处理长视频
2. **精确描述**：使用具体的描述，如"红色汽车"比"汽车"更准确
3. **调整参数**：根据分析结果调整相关性评分，找到最佳平衡点
4. **多次分析**：可以多次分析同一视频，使用不同的描述
5. **保存结果**：分析结果以JSON格式显示，可以保存供后续使用

## 📞 技术支持

如遇到问题，请联系nacalorea@outlook.com


## 📄 许可证

本项目开源

---

**祝你使用愉快！** 🎬
