@echo off
title Mol_Smell AI Server
echo ==============================================
echo       Starting Mol_Smell Web Application
echo ==============================================
echo.
echo Please wait while the AI models (SmellGNN & VAE) load into memory...
echo Keep this window open. Closing it will stop the web server!
echo.

:: Automatically open the default web browser to the correct address
timeout /t 3 /nobreak > NUL
start "" "http://localhost:8000"

:: Start the Python FastAPI backend
python app.py

:: If the app crashes or stops, keep the window open so you can see the error
pause
