
# Activate virtual environment
.\venv\Scripts\activate

# Stop IIS prediction website (needs to be run as administrator)
Stop-Website -Name MarketBasketTensorflow

# Generate transactions file
$CurrentDir = Get-Location
Set-Location -Path "O:\XXX\bin"
$TargetFile = "$CurrentDir\Data\transactions.csv"
& ".\aprastenexptrn.exe" $TargetFile

# Go back to python dir.
Set-Location -Path $CurrentDir

# Delete previous model
Remove-Item -Recurse -Force model

# Generate model from data
python preprocess.py
python gen_dataset.py
python train.py --verbose=2
python export.py
python export_serving.py

# Start IIS prediction website (needs to be run as administrator)
Start-Website -Name MarketBasketTensorflow
