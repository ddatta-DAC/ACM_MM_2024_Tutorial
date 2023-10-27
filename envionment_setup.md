### Step 1

- Install `conda`
- Site: `https://docs.conda.io/projects/conda/en/stable/`


### Create the environment:

```
conda env create -f environment.yml
```

### Activate environment:
```
conda activate acm_mm
```

-----

- Check GPU s available

```
nvcc --version
nvidia-smi
```
![Alt text](Images/nvidiasmi.png?raw=true "nvidiasmi")
### Additional utilities one requires to install.

```
pip3 install nvitop
```

```
nvitop
```
![Alt text](Images/nvitop.png?raw=true "nvitop")

-----

