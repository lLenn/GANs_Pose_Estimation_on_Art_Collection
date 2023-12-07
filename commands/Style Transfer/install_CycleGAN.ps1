.\commands\prep_environment.ps1

Set-Location lib

# Clone the CycleGAN API
if(-Not (Test-Path -Path CycleGAN)) {
    git clone https://github.com/lLenn/pytorch-CycleGAN-and-pix2pix.git CycleGAN
    Set-Location CycleGAN
} else {
    Set-Location CycleGAN
    git pull
}

if(-Not (Test-Path -Path "__init__.py")) {
    New-Item -ItemType File -Name "__init__.py"
}

# Setup package
Set-Location ..
Copy-Item -Path "..\commands\Style Transfer\setup.py" -Destination .\setup.py
python .\setup.py develop
Set-Location CycleGAN

# Install dependencies
pip install -r requirements.txt

Set-Location ..\..