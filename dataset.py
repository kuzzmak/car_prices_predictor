from functools import reduce
import json
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, random_split

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
    """
    A PyTorch dataset class for loading and preprocessing car advertisement data.

    Args:
        data_path (str): The path to the CSV file containing the car advertisement data.
        fields (List[Tuple[str, FieldType]]): A list of tuples specifying the fields and their types.
        split (str): The split of the dataset ('train_val' or 'test').
        device (torch.device): The device on which to load the data.

    Attributes:
        fields (List[Field]): The list of Field objects representing the fields in the dataset.

    """

    def __init__(self, data_path: str, fields: List[Tuple[str, FieldType]], split: str, device: torch.device) -> None:
        super().__init__()

        self._data_path = data_path
        self._fields = [Field(*f) for f in fields]
        self._split = split
        self._device = device

        self._df = self._load_data()
        self._raw_data = self._prepare_raw_data(self._df)
        self._feat_tensors = self._preprocess_data()

    @property
    def fields(self) -> List[Field]:
        return self._fields

    @property
    def device(self) -> torch.device:
        return self._device

    def _load_data(self) -> pd.DataFrame:
        """
        Load the car advertisement data from the CSV file.

        Returns:
            pd.DataFrame: The loaded car advertisement data.

        """
        df = pd.read_csv(self._data_path)
        # test samples are marked with condition_id 20, while train and val with 40
        cond = 20 if self._split == 'train_val' else 40
        df = df[df['condition_id'] == cond]
        return df

    def _prepare_raw_data(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Prepare the raw data by extracting the required fields from the DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame containing the car advertisement data.

        Returns:
            Dict[str, pd.Series]: A dictionary mapping field names to their corresponding Series.

        """
        raw_data = {}
        afj = get_additional_fields_json(df)
        for field in self.fields:
            prop = get_series_from_additional_fields_json(afj, field.name)
            prop.name = field.name
            raw_data[field.name] = prop
        raw_data['price'] = df['price']
        return raw_data

    def _construct_features(self, preprocessed_data: Dict[str, pd.Series]) -> Dict[str, Feature]:
        """
        Construct the features from the preprocessed data.

        Args:
            preprocessed_data (Dict[str, pd.Series]): A dictionary mapping field names to their preprocessed Series.

        Returns:
            Dict[str, Feature]: A dictionary mapping field names to their corresponding Feature objects.

        """
        features = {}
        for field in self.fields:
            f = Feature(preprocessed_data[field.name], field)
            f.normalize()
            features[field.name] = f
        return features

    def _preprocess_data(self):
        """
        Preprocess the data by applying necessary transformations and constructing features.

        Returns:
            Dict[str, torch.Tensor]: A dictionary mapping field names to their corresponding feature tensors.

        """
        print(f'Preprocessing {self._split} data')

        null_entries = []
        preprocessed_data = {}

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
                if field.name not in ['yearManufactured', 'transmissionTypeId', 'manufacturerId']:
                    raise ValueError('Field not yet supported')

            null_entries.append(series.isnull())
            preprocessed_data[field.name] = series

        # construct indices of rows that should be removed, row which should
        # be removed is the row which has invalid value in some column
        to_remove = reduce(lambda x, y: x | y, null_entries)
        print(f'Number of rows to remove: {to_remove.sum()}')

        # remove all rows that had in any column some null value
        preprocessed_data = {
            key: preprocessed_data[key][~to_remove] for key in preprocessed_data}

        print('Constructing features...')
        features = self._construct_features(preprocessed_data)

        print('Converting features to tensors...')
        feat_tensors = {key: features[key].to_tensor() for key in features}
        feat_tensors['price'] = torch.tensor(
            self._raw_data['price'][~to_remove].values, dtype=torch.float32).reshape(-1, 1)

        print(f'Number of {self._split} samples loaded:',
              len(feat_tensors['price']))
        self._size = len(feat_tensors['price'])

        return feat_tensors

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.

        """
        return self._size

    def __getitem__(self, idx: int):
        """
        Return the features and target value for the sample at the given index.

        Args:
            idx (int): The index of the sample.

        Returns:
            List[torch.Tensor]: The features and target value for the sample.

        """
        features = [self._feat_tensors[field.name][idx]
                    for field in self.fields]
        # concatenate all features into a single tensor
        features = torch.cat(features)
        x = features.to(self.device)
        y = self._feat_tensors['price'][idx].to(self.device)
        return x, y


def get_datasets(data_path: str, fields: List[Tuple[str, FieldType]], device: torch.device, train_val_split: List[int] = [0.7, 0.3], seed: int = 42) -> Dict[str, CarAdDataset]:
    """
    Create train, validation, and test datasets for car advertisement prediction.

    Args:
        data_path (str): The path to the dataset.
        fields (List[Tuple[str, FieldType]]): A list of tuples representing the fields and their types.
        device (torch.device): The device on which to load the data.
        train_val_split (List[int], optional): A list of two integers representing the train-validation split ratio. Defaults to [0.7, 0.3].
        seed (int, optional): The random seed for reproducibility. Defaults to 42.

    Returns:
        Dict[str, CarAdDataset]: A dictionary containing the train, validation, and test datasets.
    """
    train_val_dataset = CarAdDataset(data_path, fields, 'train_val', device)
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        train_val_dataset, train_val_split, generator=generator)
    test_dataset = CarAdDataset(data_path, fields, 'test', device)
    return {'train': train_dataset, 'val': val_dataset, 'test': test_dataset}


def get_data_loaders(datasets: Dict[str, CarAdDataset], batch_size: int, num_workers: int = 0) -> Dict[str, torch.utils.data.DataLoader]:
    """
    Create data loaders for the train, validation, and test datasets.

    Args:
        datasets (Dict[str, CarAdDataset]): A dictionary containing the train, validation, and test datasets.
        batch_size (int): The batch size for the data loaders.
        num_workers (int, optional): The number of workers for the data loaders. Defaults to 0.

    Returns:
        Dict[str, torch.utils.data.DataLoader]: A dictionary containing the train, validation, and test data loaders.
    """
    return {split: DataLoader(dataset, batch_size=batch_size, num_workers=num_workers) for split, dataset in datasets.items()}
