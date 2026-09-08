@echo off
cd /d "%~dp0"
echo Installing / updating dependencies...
python -m pip install -r requirements.txt --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org -q
if errorlevel 1 (
    echo pip install failed. Try: python -m pip install -r requirements.txt
    pause
    exit /b 1
)
echo Starting Gradio app...
python app.py
pause
