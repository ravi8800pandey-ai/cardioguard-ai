@echo off
echo =========================================================================
echo    Starting CardioGuard AI Streamlit Application...
echo =========================================================================
cd /d "%~dp0"
python -m streamlit run app.py
pause
