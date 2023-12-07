.\commands\prep_environment.ps1

Set-Location lib

# Clone the DPGN API
if(-Not (Test-Path -Path DPGN)) {
    git clone https://github.com/megvii-research/DPGN.git DPGN
    Set-Location DPGN
} else {
    Set-Location DPGN
    git pull
}

if(-Not (Test-Path -Path "__init__.py")) {
    New-Item -ItemType File -Name "__init__.py"
}

# Setup package
Set-Location ..
Copy-Item -Path "..\commands\Metric Learning\setup.py" -Destination .\setup_metric_learning.py
python .\setup_metric_learning.py develop
Set-Location DPGN

Set-Location ..\..