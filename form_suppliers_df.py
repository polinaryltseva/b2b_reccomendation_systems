import pandas as pd
import numpy as np
import random
from typing import List, Dict, Any, Set

def form_suppliers_df(data: pd.DataFrame) -> pd.DataFrame:
    new_data = data[data['Дата публикации'].dt.year < 2024].copy()
    new_data['is_winner'] = (new_data['Победитель'] == 'Победитель').astype(int)

    new_data = new_data.sort_values(by=['ИНН поставщика', 'Дата публикации'], ascending=[True, True])
    aggregations = {
    'total_participations': ('ИНН поставщика', 'size'),
    'total_wins_calculated': ('is_winner', 'sum'),
    'first_participation_date': ('Дата публикации', 'min'),
    'last_participation_date': ('Дата публикации', 'max'),
    'avg_price_drop_hist': ('Снижение на торгах,%', 'mean'), 
    'avg_tender_value_hist': ('Стоимость(руб.) Заказчик', 'mean'), 
    'regions_set': ('Регион поставки', lambda x: set(x.dropna())),
    'spheres_set': ('Сфера деятельности', lambda x: set(x.dropna())),
    'win_rate': ('win_rate', 'last'),
    'region_wins': ('region_wins', 'last'),
    'region_win_rate': ('region_win_rate', 'last'),
    'customer_wins': ('customer_wins', 'last'),
    'customer_win_rate': ('customer_win_rate', 'last'),
    'sphere_wins': ('sphere_wins', 'last'),
    'sphere_win_rate': ('sphere_win_rate', 'last'),
    'total_wins': ('total_wins', 'last'),
    'recent_activity_ratio': ('recent_activity_ratio', 'last'),
    'competitors_per_tender': ('competitors_per_tender', 'last'), 
    'avg_competitors_in_region': ('avg_competitors_in_region', 'last'),
    'avg_competitors_in_sphere': ('avg_competitors_in_sphere', 'last'),
    'avg_competitors_in_customer': ('avg_competitors_in_customer', 'last'),
    'РНП ранее': ('РНП ранее', 'last') 
    }

    supplier_historical_data = new_data.groupby('ИНН поставщика').agg(**aggregations).reset_index()
    supplier_historical_data['win_rate_calculated'] = (
    supplier_historical_data['total_wins_calculated'] / supplier_historical_data['total_participations']
    ).fillna(0)

    supplier_historical_data['win_rate'] = supplier_historical_data['win_rate_calculated']
    supplier_historical_data['avg_price_drop_hist'] = supplier_historical_data['avg_price_drop_hist'].fillna(0)
    supplier_historical_data['avg_tender_value_hist'] = supplier_historical_data['avg_tender_value_hist'].fillna(0)

    supplier_historical_data = supplier_historical_data.rename(columns={'avg_price_drop_hist': 'avg_price_drop'})
    columns_to_drop_agg = ['total_wins_calculated']
    supplier_historical_data = supplier_historical_data.drop(columns=[col for col in columns_to_drop_agg if col in supplier_historical_data.columns])
    supplier_historical_data = supplier_historical_data.rename(columns={'total_wins_calculated': 'total_wins'})
    supplier_historical_data = supplier_historical_data.drop(columns=[col for col in columns_to_drop_agg if col in supplier_historical_data.columns])

    print(f"Создано {supplier_historical_data.shape[0]} профилей с историческими данными поставщиков.")
    return supplier_historical_data

def filter_suppliers(suppliers_df, tender_info_df):
    tender_region = tender_info_df['Регион поставки'].iloc[0]
    tender_sphere = tender_info_df['Сфера деятельности'].iloc[0]

    filtered_suppliers = suppliers_df[
        suppliers_df['regions_set'].apply(lambda regions: tender_region in regions) &
        suppliers_df['spheres_set'].apply(lambda spheres: tender_sphere in spheres)
    ].copy()

    print(f"Получено {filtered_suppliers.shape[0]} поставщиков по региону '{tender_region}' и сфере '{tender_sphere}'")
    return filtered_suppliers

def select_test_tenders(data: pd.DataFrame, num_tenders: int = 5) -> List[pd.DataFrame]:
    """
    Выбирает несколько валидных тендеров для тестирования (все участники которого присутствуют
    в наборе данных - обучающей выборке)
    """
    all_known_suppliers: Set[str] = set(data['ИНН поставщика'].unique())
    print(f"{len(all_known_suppliers)} уникальных поставщиков в данных")

    def get_supplier_stats(inn: str) -> Dict[str, int]:
        supplier_records = data[data['ИНН поставщика'] == inn]
        wins = 0
        if 'Победитель' in supplier_records.columns:
            wins = int(supplier_records['Победитель'].sum())
        return {
            'records_count': len(supplier_records),
            'wins_count': wins
        }

    valid_tenders: List[Dict[str, Any]] = []
    all_tender_ids = data['Реестровый номер публикации'].unique()

    processed_count = 0
    for tender_id in all_tender_ids:
        tender_data = data[data['Реестровый номер публикации'] == tender_id]
        participants = tender_data['ИНН поставщика'].unique()
        num_participants = len(participants)
        if num_participants > 1:
            if all(inn in all_known_suppliers for inn in participants):
                winner_data = tender_data[tender_data['Победитель'] == 1]
                if not winner_data.empty:
                    winner_inn = winner_data['ИНН поставщика'].iloc[0]
                    valid_tenders.append({
                        'tender_id': tender_id,
                        'participants_count': num_participants,
                        'winner_inn': winner_inn
                    })
        processed_count += 1


    print(f"\nНайдено {len(valid_tenders)} валидных тендеров.")
    selected_tenders_data: List[pd.DataFrame] = []
    if valid_tenders:
        n_select = min(num_tenders, len(valid_tenders))
        selected_sample = random.sample(valid_tenders, n_select)
        for i, selected_info in enumerate(selected_sample):
            tender_id = selected_info['tender_id']
            print(f"\nТендер {i+1}/{n_select}:")
            print(f"ID: {tender_id}")
            print(f"Количество участников: {selected_info['participants_count']}")

            tender_data_full = data[data['Реестровый номер публикации'] == tender_id].copy()

            print("Информация об участниках:")
            for _, participant_row in tender_data_full.iterrows():
                inn = participant_row['ИНН поставщика']
                stats = get_supplier_stats(inn)
                is_winner = participant_row['Победитель'] == 1
                print(f"{'* ПОБЕДИТЕЛЬ *' if is_winner else '- Участник -'}")
                print(f"ИНН: {inn}")
                print(f"Участий в данных: {stats['records_count']}")
                print(f"Побед в данных:   {stats['wins_count']}")
                print("-" * 40)

            selected_tenders_data.append(tender_data_full)
            print("=" * 70)

    else:
        print("Не найдено подходящих тендеров")

    return selected_tenders_data


def format_tender_info_dict(tender_df: pd.DataFrame) -> Dict[str, Any]:
    first_row = tender_df.iloc[0]
    tender_info_dict = {
            'Регион поставки': first_row['Регион поставки'],
            'Город поставки': first_row['Город поставки'],
            'Сфера деятельности': first_row['Сфера деятельности'],
            'Заказчик': first_row['Заказчик'],
            'ИНН заказчика': first_row['ИНН заказчика'],
            'Стоимость(руб.) Заказчик': first_row['Стоимость(руб.) Заказчик'],
            'Дата публикации': str(first_row['Дата публикации']),
            'Дата окончания приема заявок / Дата планового окончания исполнения контракта / Плановая дата публикации лота по ППГ':
                str(first_row['Дата окончания приема заявок / Дата планового окончания исполнения контракта / Плановая дата публикации лота по ППГ']),
            'Дата начала подачи заявок / Дата начала исполнения контракта / Дата публикации ППГ':
                str(first_row['Дата начала подачи заявок / Дата начала исполнения контракта / Дата публикации ППГ']),
            'Дата окончания проведения торгов': str(first_row['Дата окончания проведения торгов']),
            'Форма публикации': first_row['Форма публикации'],
            'Снижение на торгах,%': 0,
            'Тип торгов': first_row['Тип торгов'],
            'Статус допуска': 0,
            'is_january': first_row['is_january'], 
            'days_between': first_row['days_between'], 
            'publication_peak_9_11': first_row['publication_peak_9_11'],
            'app_end_noon_12': first_row['app_end_noon_12'], 
            'app_start_night_12': first_row['app_start_night_12'], 
            'trade_end_night_12': first_row['trade_end_night_12'],
            'year': first_row['year'],
            'month': first_row['month']
        }
    return tender_info_dict
