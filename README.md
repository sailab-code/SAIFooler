# Messing Up 3D Virtual Environments: Transferable Adversarial 3D Objects

This repository contains the code used in the experiments for the paper _"Messing Up 3D Virtual Environments:Transferable Adversarial 3D Objects"_. 

## Article information
You can find the article on arXiv at this link <https://arxiv.org/abs/2109.08465>.

## Citation 
Bibtex citation (TBD)

## Source code
The source code is separated in three parts


### 1. Blender
This directory contains the scripts and the blender workspace for converting .fbx meshes from SAILenv into .OBJ that can be attacked. There is a sample PowerShell script that can be directly executed as such: ```./convert.ps1 <obj_name> <install_path>``` where ```<obj_name>``` is the name of the .fbx you want to convert, and ```<install_path>``` is the path where sailenv was installed.

### 2. PyTorch3D 
This directory contains the modified source code of PyTorch3D that we used for the attacks. It can be installed by following the instructions in the PyTorch3D original repository. 

### 3. SAIfooler
This directory contains the code of the Adversarial Object Generator. 

### saifooler_pgd_attack_launcher.py
This is the launcher for the experiments described in the paper. It can be executed with the command

```python saifooler_pgd_attack_launcher.py --meshes_definition <meshes_batch.json>  --port <port>```


## Requirements

Required python packages are listed into ```requirements.txt```. To execute the experiments, it is also needed to have a working installation of SAILenv. You can find installation instructions for SAILenv at their homepage <http://sailab.diism.unisi.it/sailenv>. 


Acknowledgement
---------------

This software was developed in the context of some of the activities of the PRIN 2017 project RexLearn, funded by the Italian Ministry of Education, University and Research (grant no. 2017TWNMH2).This software was developed in the context of some of the activities of the PRIN 2017 project RexLearn, funded by the Italian Ministry of Education, University and Research (grant no. 2017TWNMH2).
