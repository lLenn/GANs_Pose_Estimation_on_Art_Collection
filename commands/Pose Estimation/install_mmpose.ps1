.\commands\prep_environment.ps1

Set-Location lib

# Clone the mmpose API
if(-Not (Test-Path -Path mmpose)) {
    git clone https://github.com/lLenn/mmpose.git mmpose
    Set-Location mmpose
} else {
    Set-Location mmpose
    git pull
}

# Setup package
python .\setup.py develop
Set-Location ..\..