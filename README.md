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

There are two jupyter notebooks in this repository.

- `predict.ipynb` - contains minimal code to make predictions using existing model
- `train.ipynb` - minimal code for starting new training

## Data

After some analysis, some columns from the additional fields were taken and used as feature vectors for the model. (`mileage`, `motorSize`, `motorPower`, ...).

Some fields are missing more values than some other so the decision was to remove all data rows where some column is missing data. This can be improved in future in a way that missing values are replaced with mean values for numeric fields and most common category for categorical data.

## Model

For simplistic reason, multi layer perceptron was selected. It consists only of `Linear` layers between which is some activation, `LeakyReLU` in this case. Needs further investigation if this was good choice.

Trained model can be found in `runs` directory.

## Improvements to try

- better model (architecture, better layer configuration, different activations)
- handling of category data (reducting high dimensionality of the one hot encoded data, maybe embed with some other model)
- including more fields in feature vector (fields that make sense, that are not correlated a lot)
- study effects of different normalization or standardization techniques on the quality of model predictor
- different optimizators, loss functions, lr schedulers
- train for longer and with bigger model, try overfitting
