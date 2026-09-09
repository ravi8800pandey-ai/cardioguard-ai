@echo off
setlocal
echo =========================================================================
echo    CardioGuard AI - 1-Click Push to GitHub
echo =========================================================================
echo.
cd /d "%~dp0.."

set "GIT_EXE=C:\Users\ravi8\AppData\Local\Programs\Git\cmd\git.exe"
if not exist "%GIT_EXE%" (
    where git >nul 2>&1
    if errorlevel 1 (
        echo [ERROR] Git is not installed or not found.
        pause
        exit /b 1
    )
    set "GIT_EXE=git"
)

:: Check if remote origin already exists
for /f "delims=" %%i in ('"%GIT_EXE%" remote get-url origin 2^>nul') do set "EXISTING_ORIGIN=%%i"

if not "%EXISTING_ORIGIN%"=="" (
    echo Found configured remote origin:
    echo   %EXISTING_ORIGIN%
    echo.
    echo Pushing latest commits to GitHub (main)...
    "%GIT_EXE%" push origin main
) else (
    echo No remote origin detected.
    echo Make sure you have created an empty repository on GitHub first!
    echo (e.g. at https://github.com/new with name 'cardioguard-ai')
    echo.
    set /p REPO_URL="Enter your GitHub Repository URL: "

    if "%REPO_URL%"=="" (
        echo [ERROR] No URL provided. Aborted.
        pause
        exit /b 1
    )

    echo.
    echo [1/3] Setting remote origin to %REPO_URL%...
    "%GIT_EXE%" remote add origin %REPO_URL%

    echo [2/3] Setting main branch...
    "%GIT_EXE%" branch -M main

    echo [3/3] Pushing to GitHub...
    "%GIT_EXE%" push -u origin main
)

if errorlevel 1 (
    echo.
    echo =========================================================================
    echo [!] Push encountered an issue.
    echo     If prompted, sign in via browser or use a Personal Access Token (PAT).
    echo =========================================================================
) else (
    echo.
    echo =========================================================================
    echo [SUCCESS] Code pushed to GitHub successfully!
    echo.
    echo Streamlit Cloud will automatically detect the commit and deploy:
    echo https://cardioguard-ai-fddqjjqlecngokyx2uvpj9.streamlit.app/
    echo =========================================================================
)

echo.
pause

