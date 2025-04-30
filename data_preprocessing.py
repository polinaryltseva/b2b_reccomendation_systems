import pandas as pd

def preprocess_data(dff):
    """
    Предобработка данных для модели рекомендаций
    """    
    df = dff.copy()
    
    categorical_columns = [
        'Регион поставки',
        'Сфера деятельности',
        'Поставщик',
        'Статус допуска',
        'Уровень'
    ]

    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].fillna('Неизвестно')
    
    if 'Победитель' in df.columns:
        df['Target'] = (df['Победитель'] == 'Победитель').astype(int)
    else:
        df['Target'] = 0
    
    if 'Дата публикации' in df.columns:
        df['Дата публикации'] = pd.to_datetime(df['Дата публикации'])
        df['Год'] = df['Дата публикации'].dt.year
        df['Месяц'] = df['Дата публикации'].dt.month

    return df