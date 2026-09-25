@echo off
setlocal
rem Starts Orbit64 (the library window, or a ROM given as argument / dragged
rem onto this file). Builds bin\n64.exe first if it doesn't exist yet.
cd /d "%~dp0"
if not exist "bin\n64.exe" (
    call "%~dp0build.bat"
    if errorlevel 1 (
        pause
        exit /b 1
    )
)
if not exist "bin\SDL2.dll" if exist "SDL2.dll" copy /y "SDL2.dll" "bin\" >nul
start "Orbit64" "%~dp0bin\n64.exe" %*
