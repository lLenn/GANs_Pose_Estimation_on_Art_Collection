.\commands\prep_environment.ps1

Set-Location lib

# Clone the CycleGAN API
if(-Not (Test-Path -Path CycleGAN)) {
    git clone https://github.com/lLenn/pytorch-CycleGAN-and-pix2pix.git CycleGAN
    Set-Location CycleGAN
    New-Item -Name ./checkpoints/style_cezanne_pretrained -ItemType directory
    Invoke-WebRequest -Uri http://efrosgans.eecs.berkeley.edu/cyclegan/pretrained_models/style_cezanne.pth -OutFile ./checkpoints/style_cezanne_pretrained/latest_net_G.pth
} else {
    Set-Location CycleGAN
    git pull
}

# Setup package
Set-Location ..
Copy-Item -Path "..\commands\Style Transfer\setup.py" -Destination .\setup_cyclegan.py
python .\setup_cyclegan.py develop
Set-Location CycleGAN

# Install dependencies
pip install -r requirements.txt

Set-Location ..\..