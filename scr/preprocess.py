# data processing
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def pre_data():
    pre_data = pd.read_excel('2_курс,1_сем_2020_21_уч_г_зима_от_27_01_2022_долги_все_институты.xlsx')
    pre_data.to_csv('data.csv', index=False)

def data_processing(df: pd.DataFrame):
    df = df.drop(['ФИО', 'Номер ЛД', 'Институт', 'Учебная группа', 'Статус'], axis=1)
    df['Долги'] = df['Долги'] * -1
    df.drop(index=df.index[0], axis=0, inplace=True)
    df.replace({'зачтено': 5, 'Отлично': 5,
                'Хорошо': 4,
                'Удовлетворительно': 3,
                'Неудовлетворительно': 2, 'Неявка': 2, 'не зачтено': 2,
                'Неявка по ув.причине': 0
                },
               inplace=True)

    df.replace({'да': 1, 'нет': 0}, inplace=True)

    df = df.astype({'Долги': int})
    df['result'] = df['result'].replace({'да': 1, 'нет': 0})
    df.fillna(0, inplace=True)
    X = df.drop('result', axis=1)
    return df


def data_split(df: pd.DataFrame):
    X = X.round().astype(int)
    y = df['result']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    return X_train, X_test, y_train, y_test

def scaling_data(X_train, X_test):
    X_train_std = StandardScaler().fit_transform(X_train)
    X_test_std = StandardScaler().transform(X_test)
    return X_train_std, X_test_std

