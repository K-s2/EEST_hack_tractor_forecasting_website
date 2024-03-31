import joblib
import os
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE, ADASYN, SMOTENC
from sklearn.metrics import f1_score

if __name__ == '__main__':
    #drive.mount('/content/drive')

    #path_analyse = '/content/drive/MyDrive/tractor_forecasting/data/для анализа'
    path = '/mnt/c/project/ml_model_deployment_example/dataset_problems.csv' 
    data_problems = pd.read_csv(path, sep=';')
    path1 = '/mnt/c/project/ml_model_deployment_example/dataset_normal.csv' 
    data_normal =pd.read_csv(path1, sep=';')

    data_problems['broken'] = 1  # some_values - это значения, которые вы хотите добавить
    data_normal['broken'] = 0 # other_values - это другой набор значений для второй таблицы

    result = pd.concat([data_problems, data_normal], axis=0)

    result.sample(20)

    feature_to_drop = ['Нагрузка на двигатель, %', 'iButton2', 'Уровень топлива % (spn96)', 'Стояночный тормоз (spn3842)', 'Засоренность фильтра рулевого управления (spn3844)', 'Засоренность фильтра навесного оборудования (spn3851)', 'Выход блока управления двигателем (spn3852)', 'Включение тормозков (spn3859)', 'Засоренность фильтра слива (spn3858)', 'Аварийное давление масла КПП (spn3857)', 'Аварийная температура масла ДВС(spn3856)', 'Неисправность тормозной системы (spn3863)', 'Термостарт (spn3862)', 'Разрешение запуска двигателя (spn3861)', 'Низкий уровень ОЖ (spn3860)', 'Аварийная температура масла ГТР (spn3867)', 'Необходимость сервисного обслуживания (spn3866)', 'Подогрев топливного фильтра (spn3865)', 'Вода в топливе (spn3864)', 'Холодный старт (spn3871)', 'Крутящий момент (spn513), Нм', 'Положение рейки ТНВД (spn51), %', 'Расход топлива (spn183), л/ч', 'Давление наддувочного воздуха двигателя (spn106), кПа', 'Температура масла гидравлики (spn5536), С', 'Педаль слива (spn598)']
    result = result.drop(feature_to_drop, axis=1)

    result = result[result['Сост.пед.сцепл.'] != '-']
    result = result[result['Полож.пед.акселер.,%'] != '-']

    result = result.drop(['Дата и время', 'Текущая передача (spn523)', 'ДВС. Температура наддувочного воздуха, °С', 'Значение счетчика моточасов, час:мин', 'Сост.пед.сцепл.', 'Нейтраль КПП (spn3843)', 'Аварийная температура охлаждающей жидкости (spn3841)', 'Засоренность воздушного фильтра (spn3840)',
           'Засоренность фильтра КПП (spn3847)',
           'Аварийное давление масла ДВС (spn3846)', 'Засоренность фильтра ДВС (spn3845)',
           'Недопустимый уровень масла в гидробаке (spn3850)',
           'Аварийная температура масла в гидросистеме (spn3849)',
           'Аварийное давление в I контуре тормозной системы (spn3848)',
           'Аварийное давление в II контуре тормозной системы (spn3855)',
           'Зарядка АКБ (spn3854)', 'Отопитель (spn3853)',
           'ДВС. Температура наддувочного воздуха, °С',
           'Текущая передача (spn523)'], axis=1)

    result = result.replace(r'^-+$', np.nan, regex=True)

    result = result.dropna()

    for col in result.columns:
        if result[col].dtype == 'object':
            result[col] = result[col].str.replace(',', '.')
            result[col] = result[col].str.replace(r'(?!^-)-', '', regex=True)
            result[col] = result[col].str.replace(r'^[a-zA-Z]+$', '', regex=True)

    # Превращаем все значения столбца в числа, ошибки конвертации заменяем на NaN
    numeric_values = pd.to_numeric(result['ДВС. Частота вращения коленчатого вала'], errors='coerce')

    # Теперь можем посчитать среднее, игнорируя NaN
    average = numeric_values.mean()

    print(f"Среднее значение чисел в столбце: {average}")

    result['Давл.масла двиг.,кПа'] = result['Давл.масла двиг.,кПа'].fillna(411.1471917501907)
    result['Скорость'] = result['Скорость'].fillna(5.63885736828516)
    result['ДВС. Давление смазки'] = result['ДВС. Давление смазки'].fillna(411.14496089567706)
    result['ДВС. Частота вращения коленчатого вала'] = result['ДВС. Частота вращения коленчатого вала'].fillna(1557.7539549885962)

    int_features = ['Давление в пневмостистеме (spn46), кПа', 'ДВС. Температура охлаждающей жидкости', 'ДВС. Давление смазки', 'КПП. Давление масла в системе смазки', 'Давл.масла двиг.,кПа']
    float_features = ['ДВС. Частота вращения коленчатого вала', 'Электросистема. Напряжение', 'Скорость', 'КПП. Температура масла', 'Обор.двиг.,об/мин', 'Темп.масла двиг.,°С', 'Полож.пед.акселер.,%']

    result['Полож.пед.акселер.,%'] = pd.to_numeric(result['Полож.пед.акселер.,%'], errors='coerce')
    result.dropna(subset=['Полож.пед.акселер.,%'], inplace=True)

    result['Давл.масла двиг.,кПа'] = pd.to_numeric(result['Давл.масла двиг.,кПа'], errors='coerce')
    result.dropna(subset=['Давл.масла двиг.,кПа'], inplace=True)

    result['Темп.масла двиг.,°С'] = pd.to_numeric(result['Темп.масла двиг.,°С'], errors='coerce')
    result.dropna(subset=['Темп.масла двиг.,°С'], inplace=True)

    result['Обор.двиг.,об/мин'] = pd.to_numeric(result['Обор.двиг.,об/мин'], errors='coerce')
    result.dropna(subset=['Обор.двиг.,об/мин'], inplace=True)

    def to_int(x):
      return str(x).replace(',', '.')
    for col in ['КПП. Температура масла', 'КПП. Давление масла в системе смазки', 'Скорость', 'ДВС. Давление смазки', 'ДВС. Температура охлаждающей жидкости', 'Давление в пневмостистеме (spn46), кПа', 'Электросистема. Напряжение', 'ДВС. Частота вращения коленчатого вала']:
      result[col] = pd.to_numeric(result[col], errors='coerce')
      result.dropna(subset=[col], inplace=True)

    data = result.drop('broken', axis=1)
    target = result['broken']

    data.drop('ДВС. Частота вращения коленчатого вала', inplace=True, axis=1)

    data.drop(['Электросистема. Напряжение', 'Давл.масла двиг.,кПа'], inplace=True, axis=1)

    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

    from imblearn.under_sampling import RandomUnderSampler

    rus = RandomUnderSampler(random_state=42)
    X_rus, y_rus = rus.fit_resample(X_train, y_train)

    model = CatBoostClassifier(iterations=215)

    # Обучение модели
    model.fit(X_train, y_train, eval_set=(X_test, y_test))
    joblib.dump(model, "./model.joblib")
# Оценка модели
#predictions = model.predict(X_test)
#F1_score = f1_score(y_test, predictions)
#print(f'F1 score: {F1_score}')
