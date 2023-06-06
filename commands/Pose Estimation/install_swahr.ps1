./commands/prep_environment.ps1

Set-Location lib

# Clone the COCO API
if(-Not (Test-Path -Path cocoapi)) {
    git clone https://github.com/lLenn/cocoapi.git
    Set-Location cocoapi\PythonAPI
} else {
    Set-Location cocoapi
    git pull
    Set-Location PythonAPI
}

python setup.py build install
Set-Location ..\..

# Clone the CrowdPose API
if(-Not (Test-Path -Path CrowdPose)) {
    git clone https://github.com/lLenn/CrowdPose.git
    Set-Location CrowdPose\crowdpose-api\PythonAPI
} else {
    Set-Location CrowdPose
    git pull
    Set-Location crowdpose-api\PythonAPI
}

python setup.py build install
Set-Location ..\..\..

# Clone the fork of SWAHR-HumanPose
if(-Not (Test-Path -Path SWAHR)) {
    git clone https://github.com/lLenn/SWAHR-HumanPose.git SWAHR
    Set-Location SWAHR
} else {
    Set-Location SWAHR
    git pull
}

# Setup package
Set-Location ..
Copy-Item -Path "..\commands\Pose Estimation\setup.py" -Destination .\setup_swahr.py
python .\setup_swahr.py develop
Set-Location SWAHR

# Install dependencies
pip install -r requirements.txt

# Download pretrained models
pip install gdown
if(-Not (Test-Path -Path models)) {
    New-Item -Name models -ItemType directory
    gdown --folder --id 1LTC9BqodDw3qfQ2DjgH0f_n3Javds0Pe -O models\pose_coco
    gdown --folder --id 1-W9OoshMaT5UvBaW8vPhpP8pEphIcKJ7 -O models\pose_crowdpose
}

Set-Location ..\..