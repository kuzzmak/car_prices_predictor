# Car prices predictor

## Installation

Before using this repository it's necessary to install required python packages. It would be best it this is done in some environment using `conda`, `virtualenv` or something else.

Once you activated your environment, install required packages by executing following command:

```bash
pip install -r requirements.txt
```

Repository structure

- `common.py` - contains useful classes and definitions which are userd throughout the repository
- `dataset.py` - everything for the construction of dataset and dataloader class used by training process
- `model.py` - contains single model definition, MLP which stands for Multi Layer Perceptron
- `train.py` - script for training the MLP predictor, has `__main__` function which launched trainig process, see options at the bottom of the file
