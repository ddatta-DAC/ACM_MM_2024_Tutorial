# Setting up an environment for Development

Software development requires writing code, where we utilize libraries to make our work easier. In essence libraries `encapsulate` functionality, and provide `robust`, `reusable` code.
However, the process to figure out what specific packages may be needed, and how to install them requires a level of knowledge that is non-trivial. 
There are challenges such as knowledge related to environment and dependency resolution.

Most software development and ML practitioners utilize `Virtual Environment` such that the packages & dependencies for one project do not conflict with another, and do not affect global settings.



## Part 1
1. Determine 
    - Hardware one is operating on
    - Operating system --- MacOS, Linux, Windows
2. In our case, we will use a linux system, specifically Ubuntu OS running on AWS cloud EC-2 instance.
    - The set-up steps could be replicated on a local computer such
3. Key Challenge : Selecting & Installing virtual environment.
   - We choose `Anaconda` which is highly popular and easy to use across different hardware and OS configurations.

To Install Anaconda
    Visit `https://www.anaconda.com/download#downloads.     
Download the binary & create the environment.       
If you are running conda in `bash`
```
conda init bash
```
Note : `conda` must be recognized as a command, for which one may need to modify `.bashrc` or `.zshrc`. 

Thus far, while there are some complexities, it is not prohibitive. However, there are failure points and might involve a steep learning curve.

---

## Part 2

```commandline
conda create -n acm_mm python=3.9
```

```commandline
conda activate acm_mm
```
```commandline
conda install scikit-learn
conda install numpy
conda install pandas
conda install scipy
conda install matoplotlib 
```

Install additional packages
```commandline
pip3 install pyyaml 
pip3 install omegaconf 
# Install packages for code-formatting
pip3 install black 
pip3 install isort
# Jupyter Notebooks are for development
pip3 install jupyterlab  
pip3 install colorama
```

---
## Part 3 : Setting up and checking GPU configurations.

Note: One has to install NVIDIA drivers, `cuda-toolkit` for a system.       
The system we chose comes with these preinstalled, and are part of `AWS EC2` offerings.

To ensure that NVIDIA CuDA is installed
```commandline
nvcc --version
```
(Sample) Output:
```commandline
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on Mon_Apr__3_17:16:06_PDT_2023
Cuda compilation tools, release 12.1, V12.1.105
Build cuda_12.1.r12.1/compiler.32688072_0
```
It is important to note that `CuDA` can have different versions, and correct configuration can be time consuming. 

To see the GPUs & their capacities, as well as to see if any processes are running once can use:
```commandline
nvidia-smi
```
- Another utility is `nvitop`. `nvitop` is `pip` installable.

----

## Part 4

```commandline
pip3 install torch
pip3 install torchvision torchmetrics torchtext
pip3 install lightning
pip3 install transformers
pip3 install  datasets
```

Ensure CUDA is accessible inside `pytorch`

```commandline

Python 3.9.18 | packaged by conda-forge | (main, Aug 30 2023, 03:49:32)
[GCC 12.3.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import torch
>>> torch.cuda.is_available()
True
```

