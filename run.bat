@echo off
rem 使用本机 Pytorch 环境运行 demo；如路径不同请修改 PY
set "PY=C:\Users\ixiuy\miniconda3\envs\Pytorch\python.exe"
"%PY%" "%~dp0run_demo.py" %*
pause
