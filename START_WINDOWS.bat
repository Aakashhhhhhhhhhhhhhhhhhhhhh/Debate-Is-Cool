@echo off
title Debate is Cool — Monitoring System
color 0A
echo.
echo  ==================================================
echo  DEBATE IS COOL - Monitoring System
echo  Civic Space Nepal Pvt. Ltd.
echo  ==================================================
echo.

python --version >nul 2>&1
if %errorlevel% NEQ 0 (
    echo  Python is NOT installed on this computer.
    echo.
    echo  Steps to fix:
    echo  1. Go to:  https://www.python.org/downloads/
    echo  2. Click the yellow Download button
    echo  3. Run the installer
    echo  4. TICK the box "Add Python to PATH"
    echo  5. Double-click this file again
    echo.
    start https://www.python.org/downloads/
    pause
    exit /b 1
)

echo  Python found. Starting app...
echo.
echo  *** If your browser does not open automatically ***
echo  *** open Chrome or Edge and go to:             ***
echo  ***   http://localhost:8501                    ***
echo.

python launcher.py

echo.
echo  App has stopped. Press any key to close.
pause >nul
