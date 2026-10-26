@echo off
echo Building EXE...

pyinstaller --onefile --windowed main.py

echo Done!
pause
