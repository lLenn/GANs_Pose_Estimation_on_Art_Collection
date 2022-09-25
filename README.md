# Using GANs to optimize Pose Estimation on Art Collections

All commands have been tested on Ubuntu 22.04 with a Python Virtual Environment

create an environment using:  
> python3 -m venv myenv  

Access it with:  
> source myenv/bin/activate  

Make sure that your CUDA drivers are compatible with the CUDA toolkit otherwise you might get build errors while installing AlphaPose. Also adjust the install_alpha_pose.sh script on line 8 so that it installs PyTorch with the correct CUDA version. Download the right version here:
> https://developer.nvidia.com/cuda-toolkit-archive  

Use "runfile" to install it without any complications and afterwards also run it with the "--silent --drivers" options to have the right drivers installed.