# MolGAN-PyTorch

This is the PyTorch Implementation of [MolGAN: An implicit generative model for small molecular graphs](https://arxiv.org/abs/1805.11973)


## Environment

The environment can be install:

```bash
conda env create -f environment.yml
```

You are able to activate the environment:

```bash
conda activate molgan
```

## Download GDB-9 Dataset

Simply run a bash script in the data directory and the GDB-9 dataset will be downloaded and unzipped automatically together with the required files to compute the NP and SA scores.

```bash
cd data
bash download_dataset.sh
```

The QM9 dataset is located in the data directory as well.

Feel free to use it.

## Data Preprocessing

Simply run the python script within the data direcotry. 

You need to comment or uncomment some lines of code in the main function.

```python
python sparse_molecular_dataset.py
```

## MolGAN

Simply run the following command to train the MolGAN.

```python
python main.py
```

You are able to define the training parameters within the training block of the main function in `main.py`

## Testing Phase

Simply run the same command to test the MolGAN. 

You need to comment the training section and uncomment the testing section in the main function of `main.py`

```python
python main.py
```

## Credits
This repository refers to the following repositories:
 - [nicola-decao/MolGAN](https://github.com/nicola-decao/MolGAN)
 - [ ZhenyueQin/Implementation-MolGAN-PyTorch](https://github.com/ZhenyueQin/Implementation-MolGAN-PyTorch)
