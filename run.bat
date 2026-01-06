@echo off
REM Aktivace virtuálního prostředí
call venv\Scripts\activate.bat

REM Instalace balíčků z requirements.txt
pip install --upgrade pip
pip install -r requirements.txt

REM Spuštění Python skriptu
python main.py

REM Volitelné: pozastaví okno po skončení skriptu
pause