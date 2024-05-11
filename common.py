from enum import Enum
from typing import Tuple

import pandas as pd
import torch


# The additional fields that are currently supported in the dataset
VALID_ADDITIONAL_FIELDS = set(
    [
        'yearManufactured',
        'mileage',
        'motorSize',
        'motorPower',
        'fuelConsumption',
        'co2Emission',
        'transmissionTypeId',
    ]
)


class FieldType(Enum):
    """Represents the type of a field in a dataset."""
    NUMERICAL = 'numeric'
    CATEGORICAL = 'categorical'


class Field:
    """
    Represents a feature in the car ad dataset. Each field has a name and a type.
    """

    def __init__(self, name: str, type: FieldType):
        """
        Initializes a new instance of the Field class.

        Args:
            name (str): The name of the field.
            type (FieldType): The type of the field.

        Raises:
            ValueError: If the field name or type is not supported.
        """
        if name not in VALID_ADDITIONAL_FIELDS:
            raise ValueError('Field not supported')
        if type not in FieldType:
            raise ValueError('Field type not supported')
        
        self._name = name
        self._type = type

    @property
    def name(self) -> str:
        """
        Gets the name of the field.

        Returns:
            str: The name of the field.
        """
        return self._name
    
    @property
    def type(self) -> FieldType:
        """
        Gets the type of the field.

        Returns:
            FieldType: The type of the field.
        """
        return self._type
    
    def __str__(self) -> str:
        """
        Returns a string representation of the Field object.

        Returns:
            str: A string representation of the Field object.
        """
        return f'field - name: {self.name}, type: {self.type.value}'


class Feature:
    def __init__(self, data: pd.Series, field: Field) -> None:
        self._input_data = data
        self._field = field

        self._data = None
        self._mean = None
        self._std = None

    @property
    def type(self) -> FieldType:
        return self._field.type
    
    @property
    def data(self) -> pd.Series:
        return self._data
    
    @property
    def mean(self) -> float:
        if self._mean is None:
            raise ValueError('Mean has not been calculated yet or field type if not numerical')
        return self._mean
    
    @property
    def std(self) -> float:
        if self._std is None:
            raise ValueError('Standard deviation has not been calculated yet or field type if not numerical')
        return self._std

    def normalize(self) -> None:
        if self.type == FieldType.NUMERICAL:
            # numerical fields are z-scored
            self._data, self._mean, self._std = z_score(self._input_data)
        elif self.type == FieldType.CATEGORICAL:
            # categorical fields are one-hot encoded so it's necessary to
            # convert each category to an integer
            if self._field.name == 'transmissionTypeId':
                self._data = self._input_data.map(TRANSMISSON_MAPPING)
            else:
                raise ValueError('Future not yet supported')
        
    def to_tensor(self) -> torch.Tensor:
        if self._data is None:
            raise ValueError('Data has not been normalized yet')
        

        if self.type == FieldType.CATEGORICAL:
            tensor = torch.tensor(self._data.values, dtype=torch.int64)
            return torch.nn.functional.one_hot(tensor)
        else:
            tensor = torch.tensor(self._data.values, dtype=torch.float32)
            return tensor.reshape(-1, 1)

# Mapping of transmission types to numerical values which are necessary for
# one hot encoding
TRANSMISSON_MAPPING = {
    'Mehanički mjenjač': 0,
    'Automatski': 1,
    'Automatski sekvencijski': 2,
    'Sekvencijski mjenjač': 3
}


def z_score(series: pd.Series) -> Tuple[pd.Series, float, float]:
    """
    Calculate the z-scores of a given series.

    Parameters:
    series (pd.Series): The input series.

    Returns:
    Tuple[pd.Series, float, float]: A tuple containing the z-scores of the series,
    the mean of the series, and the standard deviation of the series.
    """
    mean = series.mean()
    std = series.std()
    res = (series - mean) / std
    return res, mean, std