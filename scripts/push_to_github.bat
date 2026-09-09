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
"%GIT_EXE%" remote remove origin >nul 2>&1
"%GIT_EXE%" remote add origin %REPO_URL%

echo [2/3] Setting main branch...
"%GIT_EXE%" branch -M main

echo [3/3] Pushing to GitHub...
"%GIT_EXE%" push -u origin main

if errorlevel 1 (
    echo.
    echo [!] Push encountered an issue. If GitHub prompted for authentication,
    echo     please sign in or use a GitHub Personal Access Token (PAT).
) else (
    echo.
    echo =========================================================================
    echo [SUCCESS] Code pushed to GitHub successfully!
    echo.
    echo Next step:
    echo 1. Go to https://share.streamlit.io
    echo 2. Click 'New app'
    echo 3. Select your repository and main branch
    echo 4. Set Main file path to 'app.py'
    echo 5. Click Deploy!
    echo =========================================================================
)

echo.
pause
