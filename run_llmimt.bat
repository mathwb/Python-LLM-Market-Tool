@echo off
setlocal enabledelayedexpansion

:: Change directory to the script location
cd /d "%~dp0"

:: Check if virtual environment exists, if not create it
if not exist ".venv" (
    echo [INFO] Creating Python virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        exit /b 1
    )
    
    echo [INFO] Activating virtual environment...
    call .venv\Scripts\activate.bat
    
    echo [INFO] Upgrading pip...
    python -m pip install --upgrade pip
    
    echo [INFO] Installing dependencies from requirements.txt...
    pip install -r requirements.txt --only-binary :all:
    if errorlevel 1 (
        echo [ERROR] Dependency installation failed.
        exit /b 1
    )
) else (
    :: Activate existing virtual environment
    call .venv\Scripts\activate.bat
)

:: Run the core orchestrator script forwarding all arguments
echo [INFO] Executing llmimt_main.py...
python llmimt_main.py %*
set EXIT_CODE=%ERRORLEVEL%

echo [INFO] LLMIMT completed with exit code: %EXIT_CODE%
exit /b %EXIT_CODE%
