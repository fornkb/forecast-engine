@echo off
rem Double-click to see the web app working without an API key.
rem Sample answers written by a script, on a copy of the data. Nothing real is changed.
cd /d "%~dp0"
".venv\Scripts\python.exe" web.py --preview %*
if errorlevel 1 pause
