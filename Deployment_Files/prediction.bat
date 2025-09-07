@echo off
setlocal ENABLEDELAYEDEXPANSION

echo ==========================================
echo   Running prediction and launching Streamlit dashboard...
echo ==========================================

:: Step 1 - Ask user for test CSV path
set /p "USER_FILE=Enter full path to your test CSV (or press Enter to use data/test.csv): "

if "%USER_FILE%"=="" (
    echo  Using default test file: data/test.csv
    set "TEST_ARG=data/test.csv"
    set "CUSTOM_MOUNT="
) else (
    :: Convert to absolute path
    for %%I in ("%USER_FILE%") do set "ABS_PATH=%%~fI"

    echo  Custom file found: !ABS_PATH!
    set "TEST_ARG=data/custom_test.csv"
    set "CUSTOM_MOUNT=-v !ABS_PATH!:/app/data/custom_test.csv"
)

:: Step 2 - Run prediction with optional custom mount
echo.
echo  Running prediction container...
docker run --rm ^
  -v parkinsons_data:/app/data ^
  -v parkinsons_model:/app/model ^
  -v parkinsons_output:/app/output ^
  !CUSTOM_MOUNT! ^
  pavansakleshpurlingaraju/parkinsons-app ^
  python app/predict.py --file !TEST_ARG!

:: Step 3 - Launch browser
echo.
echo  Opening dashboard at http://localhost:8501
start "" http://localhost:8501

:: Step 4 - Run Streamlit dashboard
echo.
echo  Launching Streamlit dashboard...
docker run --rm -p 8501:8501 ^
  -v parkinsons_data:/app/data ^
  -v parkinsons_model:/app/model ^
  -v parkinsons_output:/app/output ^
  pavansakleshpurlingaraju/parkinsons-app ^
  streamlit run app/interpretation_dashboard.py

pause
