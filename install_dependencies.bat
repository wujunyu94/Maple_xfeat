@echo off
setlocal
cd /d "%~dp0"

echo === Maple_xfeat dependency setup ===
if "%MAPLE_XFEAT_VENV%"=="" set "MAPLE_XFEAT_VENV=.venv"
if exist "%MAPLE_XFEAT_VENV%\Scripts\python.exe" goto :venv_created
where py >nul 2>nul
if errorlevel 1 goto :create_venv_with_python
py -3.11 -m venv "%MAPLE_XFEAT_VENV%"
if not errorlevel 1 goto :venv_created
py -3 -m venv "%MAPLE_XFEAT_VENV%"
if not errorlevel 1 goto :venv_created

:create_venv_with_python
where python >nul 2>nul
if errorlevel 1 (
    echo Python 3.10 or newer was not found. Install Python and retry.
    goto :fail
)
python -m venv --without-pip "%MAPLE_XFEAT_VENV%"
if errorlevel 1 goto :fail
if not exist "%MAPLE_XFEAT_VENV%\Scripts\python.exe" goto :fail

:venv_created
set "VENV_PYTHON=%MAPLE_XFEAT_VENV%\Scripts\python.exe"
"%VENV_PYTHON%" -m pip --version >nul 2>nul
if errorlevel 1 "%VENV_PYTHON%" -m ensurepip --upgrade
if errorlevel 1 (
    echo Could not initialize pip in the virtual environment.
    goto :fail
)
"%VENV_PYTHON%" -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)"
if errorlevel 1 (
    echo Python 3.10 or newer is required.
    goto :fail
)
"%VENV_PYTHON%" -m pip install -r requirements.txt
if errorlevel 1 goto :fail

echo.
echo Choose the feature backend to prepare:
echo   1. SIFT CPU - no PyTorch
echo   2. XFeat CPU - install CPU-only PyTorch
echo   3. XFeat CUDA - install the current PyTorch wheel, then check CUDA availability
set /p BACKEND_CHOICE="Enter 1, 2 or 3 [1]: "
if "%BACKEND_CHOICE%"=="" set "BACKEND_CHOICE=1"

if "%BACKEND_CHOICE%"=="1" goto :ready
if "%BACKEND_CHOICE%"=="2" goto :fetch_xfeat_cpu
if "%BACKEND_CHOICE%"=="3" goto :fetch_xfeat_cuda
echo Invalid selection.
goto :fail

:fetch_xfeat_cpu
git submodule update --init --recursive
if errorlevel 1 (
    echo Could not fetch the XFeat submodule. Check Git and network access, then retry.
    goto :fail
)
python -m no_minimap_lab.install_xfeat
if errorlevel 1 goto :fail
goto :install_cpu

:fetch_xfeat_cuda
git submodule update --init --recursive
if errorlevel 1 (
    echo Could not fetch the XFeat submodule. Check Git and network access, then retry.
    goto :fail
)
python -m no_minimap_lab.install_xfeat
if errorlevel 1 goto :fail
goto :install_cuda

:install_cpu
"%VENV_PYTHON%" -m pip install "tqdm>=4.66"
if errorlevel 1 goto :fail
"%VENV_PYTHON%" -m pip install torch --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 goto :fail
goto :ready

:install_cuda
"%VENV_PYTHON%" -m pip install -r requirements-xfeat.txt
if errorlevel 1 goto :fail
"%VENV_PYTHON%" -c "import torch,sys; print('PyTorch:',torch.__version__,'CUDA build:',torch.version.cuda,'GPU available:',torch.cuda.is_available()); sys.exit(0 if torch.cuda.is_available() else 1)"
if errorlevel 1 (
    echo PyTorch installed, but CUDA is not available in this environment.
    echo Check the NVIDIA driver, then choose a compatible Windows CUDA build at:
    echo https://pytorch.org/get-started/locally/
    goto :fail
)

:ready
echo.
echo Setup complete. Activate with: call "%MAPLE_XFEAT_VENV%\Scripts\activate.bat"
echo Start the GUI with: "%VENV_PYTHON%" -m no_minimap_lab.run --map-id 101000000 --backend sift-cpu
pause
exit /b 0

:fail
echo.
echo Setup did not complete. Read the message above, fix the issue, and run this script again.
pause
exit /b 1
