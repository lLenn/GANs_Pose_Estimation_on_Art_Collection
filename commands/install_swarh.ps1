# Update pip
python.exe -m pip install --upgrade pip

# Create tmp folder for install
if(-Not (Test-Path -Path bin)) {
    New-Item -Name bin -ItemType directory
}
Set-Location bin

# Clone the COCO API
if(-Not (Test-Path -Path cocoapi)) {
    git clone https://github.com/cocodataset/cocoapi.git
    Set-Location cocoapi\PythonAPI
    python setup.py install
    Set-Location ..\..
}

# Clone the CrowdPose API
if(-Not (Test-Path -Path CrowdPose)) {
    git clone https://github.com/Jeff-sjtu/CrowdPose.git
    Set-Location CrowdPose\crowdpose-api\PythonAPI
    python setup.py install
    Set-Location ..\..\..
}

# Clone the fork of SWAHR-HumanPose
if(-Not (Test-Path -Path SWAHR-HumanPose)) {
    git clone https://github.com/lLenn/SWAHR-HumanPose.git
}
Set-Location SWAHR-HumanPose

# Install PyTorch
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip3 install -r requirements.txt

# Download pretrained models
pip install gdown
if(-Not (Test-Path -Path models)) {
    New-Item -Name models -ItemType directory
    gdown --folder --id 1LTC9BqodDw3qfQ2DjgH0f_n3Javds0Pe -O models\pose_coco
    gdown --folder --id 1-W9OoshMaT5UvBaW8vPhpP8pEphIcKJ7 -O models\pose_crowdpose
}

Set-Location -Path ..\..