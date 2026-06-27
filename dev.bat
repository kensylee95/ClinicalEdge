@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64
echo.
echo MSVC environment ready. cl.exe and cargo build should both work in this window.
echo.