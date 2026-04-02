import pandas as pd
import numpy as np
data_set = pd.read_csv('student_lifestyle_100k.csv')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

print("Размер датасета:", data_set.shape)
print("\nПервые 3 строки:")
print(data_set.head(3))

print("Базовая информация о дата сете: ")
data_set.info()

print("Проверка пропусков:")
print(data_set.isnull().sum())

print("Проверка дубликатов:")
print(data_set.duplicated().sum())

print("Проверка столбцов на логику: ")
errors = {}


invalid_age = data_set[(data_set['Age'] < 0) | (data_set['Age'] > 100)]
errors['Age'] = len(invalid_age)


invalid_cgpa = data_set[(data_set['CGPA'] < 0) | (data_set['CGPA'] > 4.0)]
errors['CGPA'] = len(invalid_cgpa)


invalid_sleep = data_set[(data_set['Sleep_Duration'] < 0) | (data_set['Sleep_Duration'] > 24)]
errors['Sleep_Duration'] = len(invalid_sleep)


invalid_study = data_set[(data_set['Study_Hours'] < 0) | (data_set['Study_Hours'] > 24)]
errors['Study_Hours'] = len(invalid_study)


invalid_social = data_set[(data_set['Social_Media_Hours'] < 0) | (data_set['Social_Media_Hours'] > 24)]
errors['Social_Media_Hours'] = len(invalid_social)


invalid_phys = data_set[(data_set['Physical_Activity'] < 0)]
errors['Physical_Activity'] = len(invalid_phys)


invalid_stress = data_set[(data_set['Stress_Level'] < 1) | (data_set['Stress_Level'] > 10)]
errors['Stress_Level'] = len(invalid_stress)

for col, count in errors.items():
    status = "Trut" if count == 0 else "False"
    print(f"  {col:20}: {status}")

if sum(errors.values()) == 0:
    print("Все значения находятся в допустимых пределах")
