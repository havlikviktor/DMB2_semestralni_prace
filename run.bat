@echo off
REM Aktivace virtuálního prostředí
call venv\Scripts\activate.bat

REM Spuštění Python skriptu
python main.py

REM Volitelné: pozastaví okno po skončení skriptu
pause