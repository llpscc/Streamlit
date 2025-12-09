import re
import pandas as pd
import numpy as np


def new_torque_extract(column):
  # используем все ту же библиотеку re для работы с регулярными выражениями
  try:
    '''
    # функция re.search заданное регулярное выражения в строке, в данном случае число с точкой в качестве разделителя.
    при этом tourqe будет первым значением
    в случае, если rpm задан одним числом, то он будет вторым значением match
    если rpm задан диапазоном, вторым значением будет начало диапазона, а третьим конец
    '''
    # убираем разделитель разрядов
    column = column.replace(',', '')
    match = re.findall(r'[\d.]+', column)
    if not match:
      return None

    torque_value = float(match[0]) # берем первое числовое значение из найденных
    if 'kgm' in column.lower():
      torque_value = float(round(torque_value * 9.80665, 2)) # конвертируем kgm в nm
    if len(match) == 3:
      rpm_value = (float(match[1]) + float(match[2])) / 2 # берем среднее от диапазона, если он есть
    # добавляю условие при длине match 2 отдельно
    elif len(match) == 2:
      rpm_value = float(match[1]) # берем значение rpm если оно представлено без диапазона
    else:
      rpm_value = None
    return torque_value, rpm_value
  except:
    return None

def name_extract(column):
    try:
        # разделяем на слова
        match = re.findall(r'\b\w+\b', column)
        if not match:
            return None

        brand = match[0]
        model = match[1]
        conf = ' '.join(match[2:])

        return brand, model, conf
    except:
        return None
      
def standardize_mileage(row):
    if row['mileage_unit'] == 'km/kg':
        if row['fuel'] == 'CNG':
            return (row['mileage_val'] / 1.25)
        elif row['fuel'] == 'LPG':
            return (row['mileage_val'] / 1.35)
        else:
            return np.nan
    elif row['mileage_unit'] == 'kmpl':
        return (row['mileage_val'])
    else:
        return np.nan
      
# функция препроцессинга для пайплайна
def full_preprocessing(df):
  df = df.copy() # копия датасета

  # engine, max_power -> num
  df['engine'] = pd.to_numeric(df['engine'].astype(str).str.extract(r'(\d+\.?\d*)')[0], errors='coerce')
  df['max_power'] = pd.to_numeric(df['max_power'].astype(str).str.extract(r'(\d+\.?\d*)')[0], errors='coerce')

  # mileage -> num
  df[['mileage_val', 'mileage_unit']] = df['mileage'].astype(str).str.extract(r'([\d\.]+)\s*(kmpl|km/kg)', expand=True)
  df['mileage_val'] = df['mileage_val'].astype(float)
  df['mileage'] = df.apply(standardize_mileage, axis=1)
  df.drop(columns=['mileage_val', 'mileage_unit'], inplace=True)

  # разбиваем torque
  df[['torque_val', 'max_torque_rpm']] = df['torque'].apply(new_torque_extract).apply(pd.Series)
  df.drop(columns=['torque'], inplace=True)
  
  # приведение seats к num
  df['seats'] = pd.to_numeric(df['seats'])

  # обработка name
  df['new_name'] = df['name'].astype(str).str.replace(r'\b(Diesel|Petrol|LPG|CNG)\b', '', regex=True)
  df[['brand', 'model', 'conf']] = df['new_name'].apply(name_extract).apply(pd.Series)
  df= df.drop(columns=['name', 'new_name'])

  return df
