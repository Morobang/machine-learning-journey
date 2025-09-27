@echo off
echo.
echo ========================================
echo  Titanic EDA - Generate Figures
echo ========================================
echo.

cd /d "%~dp0"
cd scripts

echo Checking Python environment...
python --version
if errorlevel 1 (
    echo Error: Python not found. Please install Python and try again.
    pause
    exit /b 1
)

echo.
echo Installing required packages...
pip install pandas numpy matplotlib seaborn scipy plotly -q

echo.
echo Generating figures...
python generate_figures.py

echo.
echo Process complete! Check the reports/figures/ directory for generated visualizations.
pause