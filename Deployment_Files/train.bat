@echo off
echo Creating named volume (if not already exists)...
docker volume create parkinsons_data

echo Running training container...
docker run --rm ^
  -v parkinsons_data:/app/data ^
  -v parkinsons_model:/app/model ^
  pavansakleshpurlingaraju/parkinsons-app ^
  python app/train_model.py
pause
