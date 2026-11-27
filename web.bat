@echo off
rem Double-click to open the web app. Live AI calls need OPENAI_API_KEY set.
cd /d "%~dp0"
".venv\Scripts\python.exe" web.py %*
if errorlevel 1 pause
