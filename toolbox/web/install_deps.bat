@echo off
REM Copyright (c) 2026 Tang Hubocheng. All rights reserved.
REM Original project code: no use or redistribution without written permission.
REM See the repository LICENSE for scope, exceptions and third-party rights.
cd /d "%~dp0"
python -m pip install --upgrade pip --trusted-host pypi.org --trusted-host files.pythonhosted.org
python -m pip install -r requirements.txt --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org
echo.
echo Done. Run: python app.py   or double-click run.bat
pause
