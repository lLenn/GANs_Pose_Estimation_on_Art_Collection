# Update pip
python -m pip install --upgrade pip

# Install PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

& ".\commands\Pose Estimation\install_mmcv.ps1"

# Create tmp folder for install
if(-Not (Test-Path -Path lib)) {
    New-Item -Name lib -ItemType directory
}