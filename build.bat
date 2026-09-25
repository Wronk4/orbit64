@echo off
setlocal
rem Toolchain: tools\w64devkit by default; set W64DEVKIT to use one elsewhere.
rem SDL2: SDL2_DIR (x86_64-w64-mingw32 folder of SDL2-devel-*-mingw), see Makefile.
rem Both fall back to a sibling ..\N64_EMU\tools checkout when not found here.
if not defined W64DEVKIT set "W64DEVKIT=%~dp0tools\w64devkit"
if not exist "%W64DEVKIT%\bin\make.exe" if exist "%~dp0..\N64_EMU\tools\w64devkit\bin\make.exe" set "W64DEVKIT=%~dp0..\N64_EMU\tools\w64devkit"
if not defined SDL2_DIR if not exist "%~dp0tools\SDL2-2.30.12\x86_64-w64-mingw32" if exist "%~dp0..\N64_EMU\tools\SDL2-2.30.12\x86_64-w64-mingw32" set "SDL2_DIR=%~dp0..\N64_EMU\tools\SDL2-2.30.12\x86_64-w64-mingw32"
if not exist "%W64DEVKIT%\bin\make.exe" (
    echo [Build] w64devkit not found - set W64DEVKIT or put it in tools\w64devkit
    exit /b 1
)
if defined SDL2_DIR set "SDL2_DIR=%SDL2_DIR:\=/%"
set "PATH=%W64DEVKIT%\bin;%PATH%"
cd /d "%~dp0"
make -j%NUMBER_OF_PROCESSORS% %*
if %ERRORLEVEL% NEQ 0 (
    echo [Build] Failed with error %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)
if not exist "bin\SDL2.dll" if exist "SDL2.dll" copy /y "SDL2.dll" "bin\" >nul
echo [Build] Build successful: bin\n64.exe
