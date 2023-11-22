.\commands\prep_environment.ps1

Set-Location lib

# Clone the UGATIT API
if(-Not (Test-Path -Path UGATITLib)) {
    git clone https://github.com/lLenn/UGATIT-pytorch.git UGATITLib
    Set-Location UGATITLib
} else {
    Set-Location UGATITLib
    git pull
}

if(-Not (Test-Path -Path "__init__.py")) {
    New-Item -ItemType File -Name "__init__.py"
}

# Setup package
Set-Location ..
Copy-Item -Path "..\commands\Style Transfer\setup.py" -Destination .\setup.py
python .\setup.py develop

Set-Location ..