REM .\venv\Scripts\activate
python preprocess.py
python gen_dataset.py
python train.py
python export.py
python export_serving.py
