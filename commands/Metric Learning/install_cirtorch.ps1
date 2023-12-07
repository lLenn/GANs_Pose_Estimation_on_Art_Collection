.\commands\prep_environment.ps1

Set-Location lib

# Clone the cirtorch API
if(-Not (Test-Path -Path cirtorch)) {
    git clone https://github.com/lLenn/cnnimageretrieval-pytorch.git cirtorch
    Set-Location cirtorch
} else {
    Set-Location cirtorch
    git pull
}

# Setup package
python setup.py develop

Set-Location ..\..