@echo off
cd /d "%~dp0"
python -m pip install --upgrade pip --trusted-host pypi.org --trusted-host files.pythonhosted.org
python -m pip install -r requirements.txt --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org
echo.
echo Done. Run: python app.py   or double-click run.bat
pause
