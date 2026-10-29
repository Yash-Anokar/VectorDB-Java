@echo off
cd /d "%~dp0"
if not exist out mkdir out
javac -d out src\vectordb\*.java || exit /b 1
java -cp out vectordb.Main
