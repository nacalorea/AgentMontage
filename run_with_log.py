import sys
import os
from datetime import datetime

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

log_dir = os.path.join(script_dir, "logs")
os.makedirs(log_dir, exist_ok=True)

now = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(log_dir, f"aicut_{now}.log")

print(f"Log file: {log_file}")
print(f"Starting AiCut at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 50)

log_handle = open(log_file, "w", encoding="utf-8")
log_handle.write(f"Starting AiCut at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
log_handle.write("=" * 50 + "\n")
log_handle.flush()

class TeeOutput:
    def __init__(self, file, original):
        self.file = file
        self.original = original

    def write(self, text):
        self.original.write(text)
        self.file.write(text)
        self.file.flush()

    def flush(self):
        self.original.flush()
        self.file.flush()

    def isatty(self):
        return self.original.isatty()

original_stdout = sys.stdout
original_stderr = sys.stderr
sys.stdout = TeeOutput(log_handle, original_stdout)
sys.stderr = TeeOutput(log_handle, original_stderr)

try:
    # 直接运行 main.py 的启动逻辑
    import runpy
    runpy.run_path("main.py", run_name="__main__")
except KeyboardInterrupt:
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    print("\nInterrupted by user")
    log_handle.write("\nInterrupted by user\n")
except Exception as e:
    import traceback
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    error_msg = f"\nError: {e}\n{traceback.format_exc()}"
    print(error_msg)
    log_handle.write(error_msg)
finally:
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    log_handle.write("\n" + "=" * 50 + "\n")
    log_handle.write(f"Ended at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_handle.close()

print(f"\nEnded at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Log saved to: {log_file}")
