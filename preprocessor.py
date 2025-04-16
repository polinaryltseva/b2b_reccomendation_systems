import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from typing import List 
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor


def hierarchical_median_impute(df, columns_to_fill, group_hierarchy):
    for col in columns_to_fill:
        is_null = df[col].isnull()
        for groups in group_hierarchy:
            if len(groups) == 0:
                global_median = df[col].median()
                df.loc[is_null, col] = global_median
                break
            medians = df.groupby(groups)[col].transform('median')
            df.loc[is_null, col] = df.loc[is_null, col].fillna(medians)
            is_null = df[col].isnull()
            if not is_null.any():
                break
    return df


group_hierarchy = [['Сфера деятельности', 'Регион поставки'],['Сфера деятельности'],['Регион поставки'],[]]

def preprocess_tender_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Выполняет полную предобработку данных тендеров
    - Преобразование 'Победитель' и 'Статус допуска' в бинарный формат
    - Заполнение пропусков в 'Тип торгов', 'Стоимость(руб.) Заказчик' с использованием иерархии медиан
    - Инжиниринг признаков на основе исторических данных (win_rate, статистика по регионам/заказчикам/сферам,
      avg_price_drop, recent_activity_ratio, статистика по конкурентам и т.д.)
      ВАЖНО: Статистика рассчитывается на данных до 2024 года ('train_data') и применяется ко всему набору данных
    - Заполнение пропусков в созданных признаках с помощью KNNImputer
    - Создание дополнительных временных признаков ('is_january', 'days_between', пики публикаций/заявок)
    Args:
        data (pd.DataFrame): Исходный DataFrame с данными тендеров.
                               Ожидаются столбцы: 'Реестровый номер публикации', 'ИНН поставщика',
                               'Статус допуска', 'Победитель', 'Дата публикации', 'Стоимость(руб.) Заказчик',
                               'Сфера деятельности', 'ИНН заказчика', 'Регион поставки', 'Тип торгов',
                               'Идентификационный код закупки', 'Снижение на торгах,%', 'Заказчик',
                               'РНП сейчас', 'Дата окончания приема заявок / Дата планового окончания исполнения контракта / Плановая дата публикации лота по ППГ',
                               'Дата начала подачи заявок / Дата начала исполнения контракта / Дата публикации ППГ',
                               'Дата окончания проведения торгов'
    Returns:
        pd.DataFrame: Предобработанный DataFrame
    """

    df = data.copy() 
    if "Реестровый номер публикации" in df.columns and "ИНН поставщика" in df.columns:
        tender_participants = df.groupby("Реестровый номер публикации")["ИНН поставщика"].transform("count")
        df["participants_count"] = tender_participants

    if "Статус допуска" in df.columns and "participants_count" in df.columns:
      initial_rows = df.shape[0]
      df = df[~((df["participants_count"] == 1) & (df["Статус допуска"] == "Не допущен"))]
      df = df[~((df["Победитель"] == "Победитель") & (df["Статус допуска"] == "Не допущен"))]
      print(f"Удалено строк после фильтрации невалидных тендеров: {initial_rows - df.shape[0]}")

    if "Победитель" in df.columns and "participants_count" in df.columns and "Статус допуска" in df.columns:
        df["Победитель"] = np.where(
            (df["Победитель"] == "Победитель") | (df["participants_count"] == 1) | ((df["Победитель"] == "Победитель") & (df["Статус допуска"] == "Неизвестно")),
            1, 0
        )

    if 'Сводный риск' in df.columns:
        df.loc[df['Сводный риск'] == 'Высокий риск', 'Сводный риск'] = 3
        df.loc[df['Сводный риск'] == 'Средний риск', 'Сводный риск'] = 2
        df.loc[df['Сводный риск'] == 'Низкий риск', 'Сводный риск'] = 1
        df.loc[(df['Сводный риск'] == 'Риск не определен') | (df['Сводный риск'] == 'Недействующий'), 'Сводный риск'] = 0
        df['Сводный риск'] = df['Сводный риск'].fillna(0)

    if "Статус допуска" in df.columns:
        df["Статус допуска"] = np.where(df["Статус допуска"] == "Допущен", 1, 0)

    if 'Тип торгов' in df.columns:
        df['Тип торгов'] = df['Тип торгов'].fillna('Неизвестно')

    if 'Сфера деятельности' in df.columns:
        initial_rows = df.shape[0]
        df = df[df['Сфера деятельности'].notna()]
        print(f"  Удалено строк с отсутствующей Сферой деятельности: {initial_rows - df.shape[0]}")

    if 'Стоимость(руб.) Заказчик' in df.columns and 'Выручка' in df.columns:
        imputer = IterativeImputer(estimator=RandomForestRegressor(n_estimators=100, random_state=42, bootstrap=True), max_iter=10)
        df[['Стоимость(руб.) Заказчик', 'Выручка']] = imputer.fit_transform(df[['Стоимость(руб.) Заказчик', 'Выручка']])

    if 'Дата публикации' in df.columns:
        df['Дата публикации'] = pd.to_datetime(df['Дата публикации'])
        df['year'] = df['Дата публикации'].dt.year
        df['month'] = df['Дата публикации'].dt.month

    if 'year' in df.columns and 'Победитель' in df.columns and 'ИНН поставщика' in df.columns:
        train_data = df[df['year'].isin(range(2019, 2024))].copy()
        print(f"Размер 'train_data' для расчета статистики: {train_data.shape[0]} строк")
        df['win_rate'] = train_data.groupby(['ИНН поставщика'])['Победитель'].transform(lambda x: (x == 1).sum() / len(x) if len(x) > 0 else 0)

        if 'Регион поставки' in df.columns:
            df['region_activity'] = train_data.groupby(['ИНН поставщика', 'Регион поставки'])['Победитель'].transform('count')
            df['region_wins'] = train_data.groupby(['ИНН поставщика', 'Регион поставки'])['Победитель'].transform(lambda x: (x == 1).sum())
            df['region_win_rate'] = (df['region_wins'] / df['region_activity']).fillna(0) 

        if 'Заказчик' in df.columns: 
            df['customer_activity'] = train_data.groupby(['ИНН поставщика', 'Заказчик'])['Победитель'].transform('count')
            df['customer_wins'] = train_data.groupby(['ИНН поставщика', 'Заказчик'])['Победитель'].transform(lambda x: (x == 1).sum())
            df['customer_win_rate'] = (df['customer_wins'] / df['customer_activity']).fillna(0)

        if 'Сфера деятельности' in df.columns:
            df['sphere_activity'] = train_data.groupby(['ИНН поставщика', 'Сфера деятельности'])['Победитель'].transform('count')
            df['sphere_wins'] = train_data.groupby(['ИНН поставщика', 'Сфера деятельности'])['Победитель'].transform(lambda x: (x == 1).sum())
            df['sphere_win_rate'] = (df['sphere_wins'] / df['sphere_activity']).fillna(0)

        df['total_activity'] = train_data.groupby('ИНН поставщика')['Победитель'].transform('count')
        df['total_wins'] = train_data.groupby('ИНН поставщика')['Победитель'].transform(lambda x: (x == 1).sum())

        if 'Снижение на торгах,%' in df.columns:
            df['avg_price_drop'] = train_data.groupby('ИНН поставщика')['Снижение на торгах,%'].transform('mean')

        if 'Дата публикации' in df.columns:
             df["last_activity_date"] = train_data.groupby("ИНН поставщика")["Дата публикации"].transform("max")
 
        if 'Реестровый номер публикации' in df.columns:
            df['competitors_per_tender'] = df.groupby('Реестровый номер публикации')['ИНН поставщика'].transform('nunique')
            train_data_with_competitors = df[df['year'].isin(range(2019, 2024))].copy()
            if 'Регион поставки' in df.columns:
                df['avg_competitors_in_region'] = train_data_with_competitors.groupby('Регион поставки')['competitors_per_tender'].transform('mean')
            if 'Сфера деятельности' in df.columns:
                df['avg_competitors_in_sphere'] = train_data_with_competitors.groupby('Сфера деятельности')['competitors_per_tender'].transform('mean')
            if 'Заказчик' in df.columns:
                df['avg_competitors_in_customer'] = train_data_with_competitors.groupby('Заказчик')['competitors_per_tender'].transform('mean')

        if 'Заказчик' in df.columns:
            df['customer_loyalty'] = train_data.groupby('Заказчик')['ИНН поставщика'].transform('nunique')

    else:
        print("Отсутствуют базовые колонки ('year', 'Победитель', 'ИНН поставщика') для добавления признаков")

    cols_to_drop_intermediate = [
        'РНП сейчас', 
        'participants_count', 'Идентификационный код закупки', 
        'region_activity', 'customer_activity', 'sphere_activity', 'total_activity', 'customer_loyalty' 
    ]
    df = df.drop(columns=[col for col in cols_to_drop_intermediate if col in df.columns], errors='ignore')


    columns_to_impute = ['win_rate', 'region_win_rate',
    'customer_win_rate', 'sphere_win_rate', 'avg_price_drop']
    existing_columns_to_impute = [col for col in columns_to_impute if col in df.columns]

    if existing_columns_to_impute:
        print(f"Колонки для KNNImputer: {existing_columns_to_impute}")
        imputer = KNNImputer(n_neighbors=11)
        df[existing_columns_to_impute] = imputer.fit_transform(df[existing_columns_to_impute])
    else:
        print("Нет колонок для заполнения с помощью KNNImputer")

    median_columns = ['region_wins',
    'customer_wins', 'sphere_wins', 'total_wins', 'competitors_per_tender',
    'avg_competitors_in_region', 'avg_competitors_in_sphere',
    'avg_competitors_in_customer']

    df = hierarchical_median_impute(df, median_columns, group_hierarchy)

    if 'month' in df.columns:
        df['is_january'] = df['month'].apply(lambda x: 1 if x == 1 else 0)

    if 'year' in df.columns:
        df['is_2020'] = df['year'].apply(lambda x: 1 if x == 2020 else 0)

    date_end_col = 'Дата окончания приема заявок / Дата планового окончания исполнения контракта / Плановая дата публикации лота по ППГ'
    date_start_col = 'Дата начала подачи заявок / Дата начала исполнения контракта / Дата публикации ППГ'
    date_trade_end_col = 'Дата окончания проведения торгов'
    date_pub_col = 'Дата публикации' 

    if date_end_col in df.columns and date_start_col in df.columns:
        df[date_end_col] = pd.to_datetime(df[date_end_col])
        df[date_start_col] = pd.to_datetime(df[date_start_col])
        df['days_between'] = (df[date_end_col] - df[date_start_col]).dt.days

    if date_pub_col in df.columns:
        df['publication_peak_9_11'] = df[date_pub_col].dt.hour.between(9, 11, inclusive='both').astype(int)
    if date_end_col in df.columns:
        df['app_end_noon_12'] = (df[date_end_col].dt.hour == 12).astype(int)
    if date_start_col in df.columns:
        df['app_start_night_12'] = (df[date_start_col].dt.hour == 0).astype(int)
    if date_trade_end_col in df.columns:
        df['trade_end_night_12'] = (df[date_trade_end_col].dt.hour == 0).astype(int)

    df['log_profit'] = np.log1p(df['Выручка'])
    df['log_cost'] = np.log1p(df['Стоимость(руб.) Заказчик'])

    return df