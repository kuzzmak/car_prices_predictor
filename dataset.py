from functools import reduce
import json
from typing import Dict, List

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from constants import VALID_ADDITIONAL_FIELDS


def get_additional_fields_json(df: pd.DataFrame):
    additional_fields = df['additional_fields']
    additional_fields_json = additional_fields.apply(json.loads)
    return additional_fields_json


def extract_field_from_json_object(json_object, field: str):
    return json_object.get(field)


def get_series_from_additional_fields_json(additional_fields_json: pd.Series, key: str):
    return additional_fields_json.apply(lambda x: extract_field_from_json_object(x, key))


class CarAdDataset(Dataset):

    def __init__(self, data_path: str, fields: List[str]) -> None:
        super().__init__()

        self._data_path = data_path
        self._fields = fields

        self._df = self._load_data()
        self._raw_data = self._parse_raw_data()
        self._preprocess_data()

    @property
    def fields(self) -> List[str]:
        return self._fields

    def _load_data(self) -> pd.DataFrame:
        df = pd.read_csv(self._data_path)
        return df

    def _parse_raw_data(self) -> Dict[str, pd.Series]:
        afj = get_additional_fields_json(self._df)
        raw_data = {}
        for field in self.fields:
            if field not in VALID_ADDITIONAL_FIELDS:
                raise ValueError(f'Invalid field: {field}')
            prop = get_series_from_additional_fields_json(afj, field)
            prop.name = field
            raw_data[field] = prop
        return raw_data

    def _preprocess_data(self):
        null_entries = []
        preprocessed_data = {}

        for field in self.fields:
            series = self._raw_data[field]
            if field == 'mileage':
                series[series < 0] = 0
                series[series > 1e6] = 1e6
            elif field == 'motorSize':
                series[series > 8500] = np.NaN
            elif field == 'motorPower':
                series[series > 1000] = np.NaN
            elif field == 'fuelConsumption':
                series = series.replace('', np.NaN)
                series = series.astype(float)
                series[series > 40] = np.NaN
            elif field == 'co2Emission':
                series = series.replace('', np.NaN)
                series = series.astype(float)
                series[series > 500] = np.NaN
            else:
                if field not in ['yearManufactured', 'transmissionTypeId']:
                    raise ValueError('Field not yet supported')

            null_entries.append(series.isnull())
            preprocessed_data[field] = series

        # construct indices of rows that should be removed, row which should
        # be removed is the row which has invalid value in some column
        to_remove = reduce(lambda x, y: x | y, null_entries)

        # remove all rows that had in any column some null value
        self._raw_data = {
            key: self._raw_data[key][~to_remove] for key in self._raw_data}

        print(len(self._raw_data['mileage']))
        # print(to_remove)
