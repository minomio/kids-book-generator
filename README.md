# baby-book-generator

Step 1: Create the most powerful VM on GCP you can afford :)

In our case we used:
Boot disk: c2-deeplearning-pytorch-1-11-cu113-v20220701-debian-10
Interface type: SCSI
Size: 200 GB
Type: Balanced persistent disk

Zone: us-central1-a
Preserved state size: 0 GB

Machine type: a2-highgpu-1g
CPU platform: Intel Cascade Lake
GPUs: 1x NVIDIA Tesla A100

HTTP & HTTPS traffic 

Step 2: Install the following on your machine:

pip3 lfs install
wget git clone https://huggingface.co/Cene655/ImagenT5-3B
pip3 install git+https://github.com/cene555/Imagen-pytorch.git
pip3 install git+https://github.com/openai/CLIP.git
wget git clone https://github.com/xinntao/Real-ESRGAN.git

pip3 install basicsr
pip3 install facexlib
pip3 install gfpgan

cd Real-ESRGAN
pip3 install -r requirements.txt
python3 setup.py develop
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P experiments/pretrained_models


Step 3: Run imageApp.py. It should work or prompt you to install some more dependencies. Install everything it tells you to and run imageApp.py again :)

This image generator is based on:
https://github.com/cene555/Imagen-pytorch
