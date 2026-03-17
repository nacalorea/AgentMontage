@echo off
cd /d "%~dp0"
echo Starting AiCut Application...
echo.

:: 优先使用 conda 环境的 python，若不存在则回退到系统 PATH 中的 python
if exist "C:\ProgramData\anaconda3\python.exe" (
    set PYTHON_EXE=C:\ProgramData\anaconda3\python.exe
) else if exist "%USERPROFILE%\anaconda3\python.exe" (
    set PYTHON_EXE=%USERPROFILE%\anaconda3\python.exe
) else if exist "%USERPROFILE%\miniconda3\python.exe" (
    set PYTHON_EXE=%USERPROFILE%\miniconda3\python.exe
) else (
    set PYTHON_EXE=python
)

echo Using Python: %PYTHON_EXE%
echo.

%PYTHON_EXE% run_with_log.py

if %ERRORLEVEL% neq 0 (
    echo.
    echo Error: Application exited with error code %ERRORLEVEL%
)
pause
