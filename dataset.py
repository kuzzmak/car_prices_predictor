from functools import reduce
import json
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from common import TRANSMISSON_MAPPING, Feature, Field, FieldType, z_score


def get_additional_fields_json(df: pd.DataFrame):
    """
    Convert the 'additional_fields' column of a DataFrame into a list of JSON objects.

    Args:
        df (pd.DataFrame): The DataFrame containing the 'additional_fields' column.

    Returns:
        pd.Series: A Series containing the 'additional_fields' column as a list of JSON objects.
    """
    additional_fields = df['additional_fields']
    additional_fields_json = additional_fields.apply(json.loads)
    return additional_fields_json


def extract_field_from_json_object(json_object, field: str):
    """
    Extracts a specific field from a JSON object.

    Args:
        json_object (dict): The JSON object from which to extract the field.
        field (str): The name of the field to extract.

    Returns:
        The value of the specified field if it exists in the JSON object, None otherwise.
    """
    return json_object.get(field)


def get_series_from_additional_fields_json(additional_fields_json: pd.Series, key: str):
    """
    Extracts a specific field from a JSON object in each element of a pandas Series.

    Parameters:
    additional_fields_json (pd.Series): A pandas Series containing JSON objects.
    key (str): The key of the field to extract from the JSON objects.

    Returns:
    pd.Series: A new pandas Series containing the extracted field from each JSON object.
    """
    return additional_fields_json.apply(lambda x: extract_field_from_json_object(x, key))


class CarAdDataset(Dataset):

    def __init__(self, data_path: str, fields: List[Tuple[str, FieldType]]) -> None:
        super().__init__()

        self._data_path = data_path
        self._fields = [Field(*f) for f in fields]

        self._df = self._load_data()
        self._raw_data = self._parse_raw_data()
        self._preprocess_data()

    @property
    def fields(self) -> List[Field]:
        return self._fields

    def _load_data(self) -> pd.DataFrame:
        df = pd.read_csv(self._data_path)
        return df

    def _parse_raw_data(self) -> Dict[str, pd.Series]:
        afj = get_additional_fields_json(self._df)
        raw_data = {}
        for field in self.fields:
            prop = get_series_from_additional_fields_json(afj, field.name)
            prop.name = field.name
            raw_data[field.name] = prop

        raw_data['price'] = self._df['price']
        raw_data['condition_id'] = self._df['condition_id']

        return raw_data

    def _preprocess_data(self):
        null_entries = []
        preprocessed_data = {}
        
        train_val_df = self._raw_data[self._raw_data['condition_id'] == 20]
        print('train val size: ', len(train_val_df))

        # limit the range of values for each field because some values are
        # invalid
        for field in self.fields:
            series = self._raw_data[field.name]
            if field.name == 'mileage':
                series[series < 0] = 0
                series[series > 1e6] = 1e6
            elif field.name == 'motorSize':
                series[series > 8500] = np.NaN
            elif field.name == 'motorPower':
                series[series > 1000] = np.NaN
            elif field.name == 'fuelConsumption':
                series = series.replace('', np.NaN)
                series = series.astype(float)
                series[series > 40] = np.NaN
            elif field.name == 'co2Emission':
                series = series.replace('', np.NaN)
                series = series.astype(float)
                series[series > 500] = np.NaN
            else:
                if field.name not in ['yearManufactured', 'transmissionTypeId']:
                    raise ValueError('Field not yet supported')

            null_entries.append(series.isnull())
            preprocessed_data[field.name] = series

        # construct indices of rows that should be removed, row which should
        # be removed is the row which has invalid value in some column
        to_remove = reduce(lambda x, y: x | y, null_entries)

        # remove all rows that had in any column some null value
        preprocessed_data = {
            key: preprocessed_data[key][~to_remove] for key in preprocessed_data}

        # construct and normalize features
        features = {}
        for field in self.fields:
            f = Feature(preprocessed_data[field.name], field)
            f.normalize()
            features[field.name] = f

        # convert features to tensors
        feat_tensors = {key: features[key].to_tensor() for key in features}
        
        
        print(self._raw_data['price'])
        print(self._raw_data['condition_id'])

    def __getitem__(self, idx: int):
        pass
