import os
from dotenv import load_dotenv

load_dotenv()

# OpenAI 配置
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", None)
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-3-flash-preview")

# Agent 配置
AGENT_NUM_ITERATIONS = int(os.getenv("AGENT_NUM_ITERATIONS", "3"))
AGENT_TOOLS_PER_ITERATION = int(os.getenv("AGENT_TOOLS_PER_ITERATION", "25"))

# Agent LLM 配置
AGENT_LLM_TEMPERATURE = float(os.getenv("AGENT_LLM_TEMPERATURE", "0.3"))
AGENT_LLM_TIMEOUT = int(os.getenv("AGENT_LLM_TIMEOUT", "120"))

# 分析工具 LLM 配置
ANALYSIS_LLM_TEMPERATURE = float(os.getenv("ANALYSIS_LLM_TEMPERATURE", "0.3"))
ANALYSIS_LLM_MAX_TOKENS = int(os.getenv("ANALYSIS_LLM_MAX_TOKENS", "8000"))

# 其他配置
FRAME_EXTRACTION_INTERVAL = 5
MAX_FRAMES_PER_ANALYSIS = 20
TEMP_DIR = "temp"
OUTPUT_DIR = "output"
