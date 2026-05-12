@echo off
title MindAI GPU Server
color 0A

echo.
echo  ╔══════════════════════════════════════╗
echo  ║        MindAI GPU Server             ║
echo  ║  Нейронная сеть с биологическим      ║
echo  ║  обучением (STDP, 500k нейронов)     ║
echo  ╚══════════════════════════════════════╝
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo  [!] Python не найден.
    echo  [!] Скачай с https://python.org  ^(3.10 или новее^)
    echo  [!] При установке поставь галочку "Add to PATH"
    pause
    exit /b 1
)

echo  [OK] Python найден
python --version

:: Create venv if not exists
if not exist "mindai_server_env\Scripts\activate.bat" (
    echo.
    echo  [>>] Создание виртуального окружения...
    python -m venv mindai_server_env
)

:: Activate venv
call mindai_server_env\Scripts\activate.bat

:: Install/update deps silently
echo.
echo  [>>] Проверка зависимостей...

python -c "import torch; print('  [OK] torch', torch.__version__)" 2>nul || (
    echo  [>>] Установка PyTorch с CUDA поддержкой...
    pip install torch --index-url https://download.pytorch.org/whl/cu121 -q
)

python -c "import fastapi"    2>nul || pip install fastapi uvicorn websockets -q
python -c "import websocket"  2>nul || pip install websocket-client -q
python -c "import numpy"   2>nul || pip install numpy -q
python -c "import scipy"   2>nul || pip install scipy -q

echo  [OK] Все зависимости установлены

:: Show local IP so user can share it
echo.
echo  ══════════════════════════════════════
echo  Твой IP (скинь другу):
ipconfig | findstr /i "IPv4"
echo  ══════════════════════════════════════
echo.
echo  Друг запускает:
echo    python main_agent.py --remote http://ТВОЙ_IP:8000
echo.
echo  Скачать веса после обучения:
echo    python main_agent.py --download http://ТВОЙ_IP:8000
echo  ══════════════════════════════════════
echo.

:: Run server
python server.py

pause
