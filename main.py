import json
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from model import MLP

TRANSMISSON_MAPPING = {
    'Mehanički mjenjač': 0,
    'Automatski': 1,
    'Automatski sekvencijski': 2,
    'Sekvencijski mjenjač': 3
}


def z_score(series: pd.Series) -> Tuple[pd.Series, float, float]:
    mean = series.mean()
    std = series.std()
    res = (series - mean) / std
    return res, mean, std


def load_data(data_path: str) -> pd.DataFrame:
    df = pd.read_csv(data_path)

    # 98081
    num_rows = len(df)

    # 'Unnamed: 0', 'ad_id', 'condition_id', 'ad_title', 'additional_fields', 'price'
    cols = df.columns.tolist()

    def get_additional_fields_json(df: pd.DataFrame):
        additional_fields = df['additional_fields']
        additional_fields_json = additional_fields.apply(json.loads)
        return additional_fields_json

    additional_fields_json = get_additional_fields_json(df)
    additional_fields_keys = list(additional_fields_json.iloc[0].keys())
    # for key in additional_fields_keys:
    #     print(key)

    def extract_field_from_json_object(json_object, field):
        return json_object.get(field)

    def get_series_from_additional_fields_json(key: str):
        return additional_fields_json.apply(
            lambda x: extract_field_from_json_object(x, key))

    year_manufactured = get_series_from_additional_fields_json(
        'yearManufactured')
    year_manufactured.name = 'yearManufactured'
    year_manufactured_nan_vals = year_manufactured.isnull()
    print('number of missing values for yearManufactured:',
          year_manufactured_nan_vals.sum())
    print('median yearManufactured:', year_manufactured.median())
    print('max yearManufactured:', year_manufactured.max())
    print('min yearManufactured:', year_manufactured.min())

    mileage = get_series_from_additional_fields_json('mileage')
    mileage.name = 'mileage'
    mileage[mileage < 0] = 0
    mileage[mileage > 1e6] = 1e6
    mileage_nan_vals = mileage.isnull()
    print('number of missing values for mileage:', mileage_nan_vals.sum())
    print('median mileage:', mileage.median())
    print('max mileage:', mileage.max())
    print('min mileage:', mileage.min())

    motor_size = get_series_from_additional_fields_json('motorSize')
    motor_size.name = 'motorSize'
    motor_size[motor_size > 8500] = np.NaN
    motor_size_nan_vals = motor_size.isnull()
    print('number of missing values for motorSize:', motor_size_nan_vals.sum())
    print('median motorSize:', motor_size.median())
    print('max motorSize:', motor_size.max())
    print('min motorSize:', motor_size.min())

    motor_power = get_series_from_additional_fields_json('motorPower')
    motor_power.name = 'motorPower'
    motor_power[motor_power > 1000] = np.NaN
    motor_power_nan_vals = motor_power.isnull()
    print('number of missing values for motorPower:', motor_power_nan_vals.sum())
    print('median motoPower:', motor_power.median())
    print('max motorPower:', motor_power.max())
    print('min motorPower:', motor_power.min())

    fuel_consumption = get_series_from_additional_fields_json(
        'fuelConsumption')
    fuel_consumption.name = 'fuelConsumption'
    fuel_consumption = fuel_consumption.replace('', np.NaN)
    fuel_consumption = fuel_consumption.astype(float)
    fuel_consumption[fuel_consumption > 40] = np.NaN
    fuel_consumption_nan_vals = fuel_consumption.isnull()
    print('number of missing values for fuelConsumption:',
          fuel_consumption_nan_vals.sum())
    print('median fuelConsumption:', fuel_consumption.median())
    print('max fuelConsumption:', fuel_consumption.max())
    print('min fuelConsumption:', fuel_consumption.min())

    co2_emission = get_series_from_additional_fields_json('co2Emission')
    co2_emission.name = 'co2Emission'
    co2_emission = co2_emission.replace('', np.NaN)
    co2_emission = co2_emission.astype(float)
    co2_emission[co2_emission > 500] = np.NaN
    co2_emission_nan_vals = co2_emission.isnull()
    print('number of missing values for co2Emission:',
          co2_emission_nan_vals.sum())
    print('median co2Emission:', co2_emission.median())
    print('max co2Emission:', co2_emission.max())
    print('min co2Emission:', co2_emission.min())
    # print('10 largest:')
    # print(co2_emission.nlargest(50))

    transmission_type = get_series_from_additional_fields_json(
        'transmissionTypeId')
    transmission_type.name = 'transmissionTypeId'
    transmission_type_nan_vals = transmission_type.isnull()
    print('number of missing values for transmissionTypeId:',
          transmission_type_nan_vals.sum())

    # | fuel_consumption_nan_vals | co2_emission_nan_vals | transmission_type_nan_vals
    # | fuel_consumption_nan_vals | co2_emission_nan_vals | transmission_type_nan_vals
    to_remove = year_manufactured_nan_vals | mileage_nan_vals | motor_size_nan_vals | motor_power_nan_vals
    df = df[~to_remove]

    df_train_val = df[df['condition_id'] == 20]
    df_test = df[df['condition_id'] == 40]

    print('number of train and val exmaples:', len(df_train_val))
    print('number of test examples:', len(df_test))

    # df = pd.concat([year_manufactured, mileage], axis=1)

    year_manufactured = year_manufactured[~to_remove]
    mileage = mileage[~to_remove]
    motor_size = motor_size[~to_remove]
    motor_power = motor_power[~to_remove]
    fuel_consumption = fuel_consumption[~to_remove]
    co2_emission = co2_emission[~to_remove]
    transmission_type = transmission_type[~to_remove]

    year_manufactured, year_manufactured_mean, year_manufactured_std = z_score(
        year_manufactured)
    mileage, mileage_mean, mileage_std = z_score(mileage)
    motor_size, motor_size_mean, motor_size_std = z_score(motor_size)
    motor_power, motor_power_mean, motor_power_std = z_score(motor_power)
    fuel_consumption, fuel_consumption_mean, fuel_consumption_std = z_score(
        fuel_consumption)
    co2_emission, co2_emission_mean, co2_emission_std = z_score(co2_emission)
    transmission_type = transmission_type.map(TRANSMISSON_MAPPING)

    year_manufactured = torch.tensor(year_manufactured.values).reshape(-1, 1)
    mileage = torch.tensor(mileage.values).reshape(-1, 1)
    motor_size = torch.tensor(motor_size.values).reshape(-1, 1)
    motor_power = torch.tensor(motor_power.values).reshape(-1, 1)
    # fuel_consumption = torch.tensor(fuel_consumption.values).reshape(-1, 1)
    # co2_emission = torch.tensor(co2_emission.values).reshape(-1, 1)
    # transmission_type = F.one_hot(torch.tensor(transmission_type.values))

    X = torch.hstack((year_manufactured, mileage, motor_size,
                     motor_power)).to(torch.float32)

    # plt.figure(figsize=(10, 8))
    # sns.countplot(y=year_manufactured)
    # sns.countplot(y=transmission_type)
    # plt.show()

    return X


if __name__ == '__main__':
    data_path = r'ML_zadatak_auti.csv'

    X = load_data(data_path)

    model = MLP()

    res = model(X)

    print(res.shape)

    # normalized_mileage, mileage_mean, mileage_std = z_score(mileage)

    # 20, 40
    # condition_id_vals = df['condition_id'].unique()
    # ad_titles = df['ad_title']
    # prices = df['price']
    # normalize car prices by z score normalization
    # normalized_prices, price_mean, price_std = z_score(prices)

    # duplicated_additional_fields = df.iloc[additional_fields[additional_fields.duplicated()].index]

    # duplicated_titles = df[df.duplicated(subset=['ad_title'], keep=False)]
    # print(duplicated_titles)
    # print('len duplicated additional fields', len(duplicated_additional_fields))
    # for idx, row in duplicated_additional_fields.iterrows():
    #     print(row['ad_title'])
    #     print(row['additional_fields'])
    #     print(idx)

    # print(df.iloc[additional_fields[additional_fields.duplicated()].index])
    # for idx, row in ad_titles.items():
    #     print(row)
    #     if idx > 20:
    #         break
