## Introduction
Source code for [Root Cause Analysis of Failures in Microservices through Causal Discovery](https://proceedings.neurips.cc/paper_files/paper/2022/file/c9fcd02e6445c7dfbad6986abee53d0d-Paper-Conference.pdf).

## Setup
The following insutrctions assume that you are running Ubuntu-20.04.
#### Install python env
```bash
sudo apt update
sudo apt install -y build-essential \
                    python-dev \
                    python3-venv \
                    python3-pip \
                    libxml2 \
                    libxml2-dev \
                    zlib1g-dev \
                    python3-tk \
                    graphviz

cd ~
python3 -m venv env
source env/bin/activate
python3 -m pip install --upgrade pip
```

#### Install dependencies
```bash
git clone https://github.com/azamikram/rcd.git
cd rcd
pip install -r requirements.txt
```

#### Link modifed files
To implement RCD, we modified some code from pyAgrum and causal-learn.
Some of these changes expose some internal information for reporting results (for example number of CI tests while executing PC) or modify the existing behaviour (`local_skeleton_discovery` in `SekeletonDiscovery.py` implements the localized approach for RCD). A few of these changes also fix some minor bugs.


Assuming the rcd repository was cloned at home, execute the following;
```bash
/home/wangrunzhou/anaconda3/envs/rcd/lib/python3.10/site-packages/causallearn/__init__.py
ln -fs /home/wangrunzhou/0_warlock/rcd/pyAgrum/lib/image.py /home/wangrunzhou/anaconda3/envs/rcd/lib/python3.10/site-packages/pyAgrum/lib/
ln -fs /home/wangrunzhou/0_warlock/rcd/causallearn/search/ConstraintBased/FCI.py /home/wangrunzhou/anaconda3/envs/rcd/lib/python3.10/site-packages/causallearn/search/ConstraintBased/
ln -fs /home/wangrunzhou/0_warlock/rcd/causallearn/utils/Fas.py /home/wangrunzhou/anaconda3/envs/rcd/lib/python3.10/site-packages/causallearn/utils/
ln -fs /home/wangrunzhou/0_warlock/rcd/causallearn/utils/PCUtils/SkeletonDiscovery.py /home/wangrunzhou/anaconda3/envs/rcd/lib/python3.10/site-packages/causallearn/utils/PCUtils/ 
ln -fs /home/wangrunzhou/0_warlock/rcd/causallearn/graph/GraphClass.py /home/wangrunzhou/anaconda3/envs/rcd/lib/python3.10/site-packages/causallearn/graph/
```
ls -l /root/lab/rcd/causallearn/search/ConstraintBased/FCI.py

## Using RCD

#### Generate Synthetic Data
```sh
./gen_data.py
```

#### Executing RCD with Synthetic Data
```sh
./rcd.py --path [PATH_TO_DATA] --local --k 3
```

`--local` options enables the localized RCD while `--k` estimates the top-`k` root causes.

#### Running RCD with varying number of nodes
```sh
./compare.py

./plot_exp.py exp_results/[TIMESTAMP]
```
