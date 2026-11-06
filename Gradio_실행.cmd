@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  python -m venv .venv
  if errorlevel 1 goto fail
)
if not exist ".venv\ready.flag" (
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt
  if errorlevel 1 goto fail
  ".venv\Scripts\python.exe" STEP3_Local_Deploy/titanic_dashboard/train_all.py
  if errorlevel 1 goto fail
  echo ready>".venv\ready.flag"
)
".venv\Scripts\python.exe" STEP2_Model_Serving/demo.py
goto end
:fail
echo 실행에 실패했습니다. 위 오류를 확인하세요.
pause
:end
