@echo off
echo ============================================
echo   ml-sharp Auto Install Script
echo ============================================
echo.

:: Check if conda is available
where conda >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo ERROR: conda not found. Please install Anaconda or Miniconda first.
    echo Download from: https://docs.conda.io/en/latest/miniconda.html
    pause
    exit /b 1
)

:: Clone ml-sharp if not present
set MLSHARP_DIR=%~dp0ml-sharp
if not exist "%MLSHARP_DIR%" (
    echo Cloning ml-sharp repository...
    git clone https://github.com/apple/ml-sharp "%MLSHARP_DIR%"
    if %ERRORLEVEL% neq 0 (
        echo ERROR: Failed to clone ml-sharp repository.
        pause
        exit /b 1
    )
) else (
    echo ml-sharp directory already exists, skipping clone.
)

:: Create conda environment
echo.
echo Creating conda environment 'sharp'...
conda env list | findstr /c:"sharp" >nul 2>nul
if %ERRORLEVEL% equ 0 (
    echo Conda environment 'sharp' already exists.
    echo To recreate, run: conda env remove -n sharp
) else (
    conda create -n sharp python=3.10 -y
    if %ERRORLEVEL% neq 0 (
        echo ERROR: Failed to create conda environment.
        pause
        exit /b 1
    )
)

:: Install ml-sharp in the conda environment
echo.
echo Installing ml-sharp...
conda run -n sharp --no-banner pip install -e "%MLSHARP_DIR%"
if %ERRORLEVEL% neq 0 (
    echo ERROR: Failed to install ml-sharp.
    pause
    exit /b 1
)

echo.
echo ============================================
echo   Installation complete!
echo   Conda environment: sharp
echo   Usage: conda run -n sharp sharp predict -i INPUT -o OUTPUT
echo ============================================
pause
