# IFT6285-NLP-Project1
Repository for the Project 1 for the NLP Course IFT6285 at the University of Montreal


Environment can be set with `conda` using:

```bash
conda env create -f environment.yml
# To activate this environment, use
conda activate nlp-project-env-conda
# To deactivate an active environment, use
conda deactivate
```
```bash
conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=11.1 -c pytorch -c conda-forge
conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=10.2 -c pytorch

```

To install requirements from requirements.txt

```bash
pip install -r requirements.txt
```

Updating an environment

```bash
# update the contents of your environment.yml file accordingly and then run the following command:
conda env update --prefix ./env --file environment.yml  --prune
# The --prune option causes conda to remove any dependencies that are no longer required from the environment.
```


To remove the environment:
```bash
 conda env remove --name nlp-project-env-conda
 ```