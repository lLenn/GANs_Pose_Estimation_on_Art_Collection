.\commands\prep_environment.ps1

Set-Location lib

# Clone the AdaIN API
if(-Not (Test-Path -Path AdaIN)) {
    git clone https://github.com/lLenn/pytorch-AdaIN.git AdaIN
    Set-Location AdaIN
} else {
    Set-Location AdaIN
    git pull
}

if(-Not (Test-Path -Path "__init__.py")) {
    New-Item -ItemType File -Name "__init__.py"
}

# Setup package
Set-Location ..
Copy-Item -Path "..\commands\Style Transfer\setup.py" -Destination .\setup.py
python .\setup.py develop
Set-Location AdaIN

Set-Location ..\..