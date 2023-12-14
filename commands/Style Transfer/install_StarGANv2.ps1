.\commands\prep_environment.ps1

Set-Location lib

# Clone the StarGAN API
if(-Not (Test-Path -Path StarGAN)) {
    git clone https://github.com/lLenn/stylegan2-ada-pytorch.git StarGAN
    Set-Location StarGAN
} else {
    Set-Location StarGAN
    git pull
}

if(-Not (Test-Path -Path "__init__.py")) {
    New-Item -ItemType File -Name "__init__.py"
}

# Setup package
Set-Location ..
Copy-Item -Path "..\commands\Style Transfer\setup.py" -Destination .\setup.py
python .\setup.py develop
Set-Location StarGAN

# Install dependencies
pip install -r requirements.txt

Set-Location ..\..