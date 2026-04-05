import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

data_set = pd.read_csv('student_lifestyle_100k.csv')

print("Размер датасета:", data_set.shape)
print("\nПервые 3 строки:")
print(data_set.head(3))

print("\nБазовая информация о датасете: ")
data_set.info()

print("\nПроверка пропусков:")
print(data_set.isnull().sum())

print("\nПроверка дубликатов:")
print(data_set.duplicated().sum())

print("\nПроверка столбцов на логику: ")
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
    status = "True" if count == 0 else "False"
    print(f"  {col:20}: {status}")

if sum(errors.values()) == 0:
    print("Все значения находятся в допустимых пределах")

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)

numeric_cols = ['Age', 'CGPA', 'Sleep_Duration', 'Study_Hours', 
                'Social_Media_Hours', 'Physical_Activity', 'Stress_Level']

fig, axes = plt.subplots(2, 4, figsize=(12,12))
axes = axes.ravel()

for i, col in enumerate(numeric_cols):
    axes[i].hist(data_set[col], bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    axes[i].axvline(data_set[col].mean(), color='red', linestyle='--', 
                    label=f'Mean: {data_set[col].mean():.2f}')
    axes[i].axvline(data_set[col].median(), color='green', linestyle='--', 
                    label=f'Median: {data_set[col].median():.2f}')
    axes[i].set_title(f'Распределение {col}')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Частота')
    axes[i].legend()

fig.delaxes(axes[7])
plt.suptitle('Распределение числовых признаков', fontsize=16)
plt.tight_layout()
plt.show()

numeric_df = data_set.select_dtypes(include=[np.number])
corr_matrix = numeric_df.corr()

plt.figure(figsize=(6, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
            fmt='.2f', square=True, linewidths=0.5)
plt.title('Корреляционная матрица', fontsize=14)
plt.tight_layout()
plt.show()

print("\nКорреляции с целевыми переменными:")
print("-" * 50)
depression_numeric = data_set['Depression'].astype(int)
for col in numeric_cols:
    corr_cgpa = data_set[col].corr(data_set['CGPA'])
    corr_dep = data_set[col].corr(depression_numeric)
    print(f"{col:20} | CGPA: {corr_cgpa:6.3f} | Depression: {corr_dep:6.3f}")

fig, axes = plt.subplots(1, 2, figsize=(6, 6))

dep_counts = data_set['Depression'].value_counts()
colors = ['#2ecc71', '#e74c3c']
bars = axes[0].bar(['Нет депрессии', 'Депрессия'], dep_counts.values, 
                   color=colors, alpha=0.8, edgecolor='black')
axes[0].set_title('Количество студентов с депрессией', fontsize=14)
axes[0].set_ylabel('Количество')

for bar, val in zip(bars, dep_counts.values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100, 
                 f'{val}\n({val/len(data_set)*100:.1f}%)', 
                 ha='center', va='bottom')

axes[1].pie(dep_counts.values, labels=['Нет депрессии', 'Депрессия'], 
            autopct='%1.1f%%', colors=colors, explode=(0, 0.05))
axes[1].set_title('Процентное соотношение', fontsize=14)

plt.suptitle('Анализ целевой переменной Depression', fontsize=16)
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(1, 1, figsize=(6, 6))

stress_dep = pd.crosstab(data_set['Stress_Level'], data_set['Depression'], normalize='index') * 100
stress_dep.columns = ['Нет депрессии', 'Депрессия']
stress_dep.plot(kind='bar', ax=ax, color=['#2ecc71', '#e74c3c'], alpha=0.8, edgecolor='black')
ax.set_title('Зависимость депрессии от уровня стресса', fontsize=14)
ax.set_xlabel('Уровень стресса')
ax.set_ylabel('Процент (%)')
ax.legend(loc='upper left')
ax.tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.show()

print("\n=== Дополнительная статистика по Depression ===")
print(f"Процент студентов с депрессией: {dep_counts[True] / len(data_set) * 100:.2f}%")