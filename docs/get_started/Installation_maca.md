# Installation Guide

## Installing with pip

_TO BE IMPLEMENTED_

## Building from Source

### Prerequisites

- **Operating System**: Linux
- **Python Version**: >= 3.10
- **MACA Version**: >= 3.3.0

**OS-level prerequisites**

On Ubuntu/Debian-based systems using the following commands:

```bash
apt-get update
apt-get install -y python3 python3-dev python3-setuptools gcc zlib1g-dev build-essential cmake libedit-dev git
```

**MACA SDK**

This section describes installing MACA SDK on a native Linux environment. Or please skip to the [MACA-Pytorch image](#maca-pytorch-image) section and use maca-pytorch docker image instead.

Check out [repos.metax](https://repos.metax-tech.com/gitea/repos/index/wiki/MACA.md) and install driver/sdk/cu-bridge/pytorch

Install metax-driver and maca-sdk

``` bash
curl -fsSL https://repos.metax-tech.com/public.gpg.key | apt-key add -
echo "deb [arch=$(dpkg --print-architecture)] https://repos.metax-tech.com/r/maca-sdk-deb/ stable main" | tee /etc/apt/sources.list.d/maca-sdk-deb.list
echo "deb [arch=$(dpkg --print-architecture)] https://repos.metax-tech.com/r/metax-driver-ubuntu/ stable main" | tee /etc/apt/sources.list.d/metax-driver-ubuntu.list

apt-get update

apt-get install maca-sdk metax-driver

usermod -aG video ${USER}

reboot
```

Install cu-bridge

``` bash
export MACA_PATH=/opt/maca
git clone https://gitee.com/metax-maca/cu-bridge.git
sudo chmod 755 cu-bridge -Rf
cd cu-bridge
mkdir build && cd ./build
cmake -DCMAKE_INSTALL_PREFIX=/opt/maca/tools/cu-bridge ../
make && make install
```

Install pytorch

``` bash
pip install apex dropout_layer_norm flash_attn fused_dense_lib rotary_emb torch torchaudio torchvision triton xentropy_cuda_lib xformers -i https://repos.metax-tech.com/r/maca-pypi/simple --trusted-host repos.metax-tech.com
```

<a id="maca-pytorch-image"></a>

**MACA-Pytorch image**

Check out [Installing Docker Engine](https://docs.docker.com/engine/install/ubuntu/) and install docker prerequisites

Check out [Metax docker](https://sw-download.metax-tech.com/docker) and download pytorch image.

``` bash
docker login --username=cr_temp_user --password=eyJpbnN0YW5jZUlkIjoiY3JpLXpxYTIzejI2YTU5M3R3M2QiLCJ0aW1lIjoiMTc3MDg5NTI0MzAwMCIsInR5cGUiOiJzdWIiLCJ1c2VySWQiOiIyMDcwOTQwMTA1NjYzNDE3OTIifQ:91ecedb8bd5c4af6858745f0329d069263e1bf82 cr.metax-tech.com && docker pull cr.metax-tech.com/public-library/maca-pytorch:3.3.0.4-torch2.6-py310-ubuntu24.04-amd64

docker run -it --net=host --device=/dev/dri --device=/dev/mxcd --group-add video --name mctilelang cr.metax-tech.com/public-library/maca-pytorch:3.3.0.4-torch2.6-py310-ubuntu24.04-amd64 /bin/bash

apt-get update
apt-get install -y cmake git
```

> **Note**: Metax driver should be installed already.

**Python packages prerequisites**

``` bash
pip install z3-solver cython psutil cloudpickle tqdm torch-c-dlpack-ext
```

> **Note**: Version of z3-solver should be >= 4.13.0, include and lib directories should be found at installation path.

### Build

Clone the tilelang repository

``` bash
git clone --recursive https://gitee.com/metax-maca/mcTileLang.git
```

Build with MACA enabled, git committer must be identified.

``` bash
git config --global user.email "you@example.com"
git config --global user.name "Your Name"

cd mcTileLang
USE_MACA=ON cmake -B build
make -C build -j 32
```

Install tvm-ffi

``` bash
cd 3rdparty/tvm/3rdparty/tvm-ffi && pip install . && cd -
```

### Verify

``` bash
export MACA_PATH=/opt/maca
export LD_LIBRARY_PATH=${MACA_PATH}/lib:${MACA_PATH}/mxgpu_llvm/lib:$LD_LIBRARY_PATH
export PYTH=${MACA_PATH}/mxgpu_llvm/bin:${PATH}
export PYTHONPATH=/path/to/mcTileLang:$PYTHONPATH

python -c "import tilelang; print(tilelang.__version__)"

python path/to/mcTileLang/examples/quickstart.py
```
