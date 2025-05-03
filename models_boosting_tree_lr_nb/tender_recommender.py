import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from joblib import load

from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score, average_precision_score
import random

def calculate_company_features(company_data, historical_data):
    """
    Расчет признаков для компании на основе исторических данных
    """
    features = {}
    company_inn = company_data['ИНН поставщика']
    
    company_history = historical_data[(historical_data['ИНН поставщика'] == company_inn) | 
                                    (historical_data['Поставщик'] == company_data['Поставщик'])]
    
    features['Кол-во конкурентов в тендере'] = None
    features['Процент побед'] = None
    features['Процент побед по региону'] = None
    
    if len(company_history) > 0:
        company_wins = (company_history['Победитель'] == 'Победитель').sum()
        total_participations = len(company_history)
        
        if total_participations > 0:
            features['Процент побед'] = company_wins / total_participations
        
        current_region = company_data.get('Регион поставки')
        if current_region and not pd.isna(current_region):
            region_history = company_history[company_history['Регион поставки'] == current_region]
            if len(region_history) > 0:
                region_wins = (region_history['Победитель'] == 'Победитель').sum()
                features['Процент побед по региону'] = region_wins / len(region_history)
    
    customer_inn = company_data.get('ИНН заказчика')
    customer_name = company_data.get('Заказчик')
    
    if (customer_inn is None or pd.isna(customer_inn)) and customer_name:
        if 'Заказчик' in historical_data.columns:
            customer_data = historical_data[historical_data['Заказчик'] == customer_name]
            if not customer_data.empty and 'ИНН заказчика' in customer_data.columns:
                customer_inn = customer_data['ИНН заказчика'].iloc[0]
    
    if customer_inn and not pd.isna(customer_inn):
        customer_tenders = historical_data[historical_data['ИНН заказчика'] == customer_inn]
        if not customer_tenders.empty:
            competitors_per_tender = customer_tenders.groupby('Реестровый номер публикации')['ИНН поставщика'].nunique()
            features['Среднее кол-во конкурентов у заказчика'] = competitors_per_tender.mean()
    
    for key in features:
        if pd.isna(features[key]) or features[key] is None:
            features[key] = 0.0
        elif isinstance(features[key], (int, float)):
            features[key] = float(features[key])
    
    return features


def fill_median(data, target_column, group_columns=['Сфера деятельности', 'Регион поставки']):
    """
    Заполняет пропущенные значения медианами по группам
    """
    result = data[target_column].copy()
    
    if not result.isna().any():
        return result
    
    if result.notna().sum() == 0:
        print(f"Предупреждение: в столбце '{target_column}' все значения отсутствуют")
        return pd.Series([0.0] * len(data), index=data.index)
    
    temp_data = pd.DataFrame(index=data.index)
    
    available_group_columns = []
    for col in group_columns:
        if col in data.columns:
            if data[col].notna().any():
                temp_data[col] = data[col]
                available_group_columns.append(col)
            else:
                print(f"Предупреждение: колонка '{col}' содержит только пустые значения")
    
    temp_data[target_column] = result
    
    if available_group_columns:
        group_medians = temp_data.groupby(available_group_columns)[target_column].transform(
            lambda x: x.median() if len(x) > 0 else np.nan
        )
        result = result.fillna(group_medians)
  
        
        if result.isna().any():
            for group in available_group_columns:
                group_medians = temp_data.groupby(group)[target_column].transform(
                        lambda x: x.median() if len(x) > 0 else np.nan
                    )
                result = result.fillna(group_medians)

    if result.isna().any():
        overall_median = result.median()
        if pd.isna(overall_median):
            print(f"Предупреждение: не удалось вычислить медиану для '{target_column}', заполняем нулями")
            result = result.fillna(0.0)
        else:
            result = result.fillna(overall_median)
    
    if result.isna().any():
        print(f"Предупреждение: остались пропущенные значения в '{target_column}', заполняем нулями")
        result = result.fillna(0.0)
    
    return result


def prepare_tender_features(tender_data, historical_data):
   """
   Подготовка признаков для тендера
   """
   features = {}

   features.update({
       'Уровень': tender_data['Уровень'],
       'Регион поставки': tender_data['Регион поставки'],
       'Город поставки': tender_data['Город поставки'],
       'Сфера деятельности': tender_data['Сфера деятельности'],
       'Тип торгов': tender_data.get('Тип торгов', None),
       'Заказчик': tender_data.get('Заказчик', None),

   })
  
   region_tenders = historical_data[historical_data['Регион поставки'] == tender_data['Регион поставки']]
   sphere_tenders = historical_data[historical_data['Сфера деятельности'] == tender_data['Сфера деятельности']]
  
   features['Кол-во уникальных поставщиков в регионе'] = region_tenders['ИНН поставщика'].nunique()
   features['Кол-во уникальных поставщиков в сфере'] = sphere_tenders['ИНН поставщика'].nunique()
  
   return features


def get_candidate_companies(tender_data, companies_db, historical_data, top_n=20):
    """
    Получает список компаний-кандидатов для тендера на основе:
    1. Совпадения региона поставки с субъектом поставщика
    2. Наличия опыта в сфере деятельности тендера
    """
    if historical_data is None or len(historical_data) == 0:
        print("Предупреждение: Отсутствуют исторические данные")
        return pd.DataFrame()
    
    region_match = companies_db['Субъект поставщика'] == tender_data['Регион поставки']
    candidates = companies_db[region_match].copy()
    
    if len(candidates) == 0:
        return pd.DataFrame()
    
    if 'Сфера деятельности' in tender_data:
        tender_sector = tender_data['Сфера деятельности']
        
        sector_experience = historical_data[
            (historical_data['Сфера деятельности'] == tender_sector)
        ].groupby('ИНН поставщика').size().reset_index(name='sector_tenders')

        candidates = candidates.merge(
            sector_experience,
            on='ИНН поставщика',
            how='left'
        )
        
        candidates['sector_tenders'] = candidates['sector_tenders'].fillna(0)
        
        candidates = candidates[candidates['sector_tenders'] > 0].copy()
        
        if len(candidates) == 0:
            return pd.DataFrame()
        
        candidates = candidates.rename(columns={'sector_tenders': 'Количество тендеров в сфере'})
    
    return candidates[['ИНН поставщика', 'Субъект поставщика', 'Количество тендеров в сфере'] + 
                     [col for col in candidates.columns if col not in 
                      ['ИНН поставщика', 'Субъект поставщика', 'Количество тендеров в сфере']]]


def get_model_components(model):
    """
    Извлекает компоненты модели (pipeline, препроцессор, классификатор)
    """
    if isinstance(model, dict) and 'model' in model:
        pipeline = model['model']
        if 'best_params' in model:
            print(f"Лучшие параметры: {model['best_params']}")
        if 'cv_best_score' in model:
            print(f"Лучший CV score: {model['cv_best_score']:.4f}")
        if 'features' in model:
            print(f"Признаки при обучении ({len(model['features'])}): {model['features'][:10]}...")
    elif hasattr(model, 'best_estimator_'):
        pipeline = model.best_estimator_
    else:
        pipeline = model
    
    print(f"Тип модели: {type(pipeline)}")
    
    preprocessor = None
    classifier = None
    
    if hasattr(pipeline, 'named_steps'):
        print(f"Шаги пайплайна: {list(pipeline.named_steps.keys())}")
        if 'preprocessor' in pipeline.named_steps:
            preprocessor = pipeline.named_steps['preprocessor']
        if 'classifier' in pipeline.named_steps:
            classifier = pipeline.named_steps['classifier']
    elif hasattr(pipeline, 'steps'):
        for name, step in pipeline.steps:
            if name in ['preprocessor', 'preprocessing', 'transform']:
                preprocessor = step
            elif name in ['classifier', 'estimator', 'model']:
                classifier = step
    elif hasattr(pipeline, '_final_estimator'):
        classifier = pipeline._final_estimator
        if hasattr(pipeline, 'transformers_'):
            preprocessor = pipeline.transformers_
    
    if classifier is None:
        classifier = pipeline
    
    return pipeline, preprocessor, classifier


def get_recommendations(tender_data=None, df=None, model=None, top_n=20, tender_id=None):
    """
    Получение рекомендаций поставщиков для тендера
    """
    if tender_id is not None:
        tender_rows = df[df['Реестровый номер публикации'] == tender_id]
        if len(tender_rows) == 0:
            print(f"Тендер с реестровым номером '{tender_id}' не найден")
            return []
        
        sample_tender_row = tender_rows.iloc[0]
        tender_data = {
            'Сфера деятельности': sample_tender_row['Сфера деятельности'],
            'Регион поставки': sample_tender_row['Регион поставки'],
            'Уровень': sample_tender_row['Уровень'],
            'Заказчик': sample_tender_row.get('Заказчик'),
            'ИНН заказчика': sample_tender_row.get('ИНН заказчика'),
            'Реестровый номер публикации': tender_id
        }

        print(f"Сфера деятельности: {tender_data['Сфера деятельности']}")
        print(f"Регион поставки: {tender_data['Регион поставки']}")
    
    pipeline, preprocessor, classifier = get_model_components(model)
    print(f"Тип модели: {type(classifier)}")
    
    supplier_id_column = 'ИНН поставщика'
    
    company_columns = [col for col in [supplier_id_column, 'Поставщик', 'Регион поставки', 
                                     'Сфера деятельности', 'Субъект поставщика', 'Адрес'] 
                      if col in df.columns]
    
    companies_db = df[company_columns].drop_duplicates()
    print(f"\nВсего уникальных компаний в базе: {len(companies_db)}")
    
    candidates = get_candidate_companies(tender_data, companies_db, df)
    print(f"Отобрано {len(candidates)} кандидатов для тендера")
    
    if len(candidates) == 0:
        print("Не найдено подходящих кандидатов для тендера")
        return []
    
    for idx, company in candidates.iterrows():
        company_features = calculate_company_features(company, df)
        for key, value in company_features.items():
            candidates.at[idx, key] = value
    
    required_features = [
        'Процент побед',
        'Процент побед по региону',
        'Среднее кол-во конкурентов у заказчика',
        'Кол-во конкурентов в тендере',
        'Уровень',
        'Допущен'
    ]
    
    for feature in required_features:
        if feature not in candidates.columns:
            if feature == 'Уровень':
                candidates[feature] = tender_data.get('Уровень', 1)
            elif feature == 'Допущен':
                candidates[feature] = None
            else:
                candidates[feature] = None
    
    iterative_columns = [
        'Процент побед',
        'Процент побед по региону'
    ]
    
    group_median_columns = [
        'Среднее кол-во конкурентов у заказчика',
        'Кол-во конкурентов в тендере',
        'Допущен'
    ]
    
    for col in candidates.columns:
        if col in iterative_columns + group_median_columns or pd.api.types.is_numeric_dtype(candidates[col]):
            candidates[col] = pd.to_numeric(candidates[col], errors='coerce')
    
    available_iterative_columns = [col for col in iterative_columns if col in candidates.columns]
    available_iterative_columns_with_nans = [col for col in available_iterative_columns if candidates[col].isna().any()]
    
    if available_iterative_columns_with_nans:
        print(f"\nПрименяем IterativeImputer для заполнения: {available_iterative_columns_with_nans}")
        
        iterative_imputer = IterativeImputer(
            estimator=RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
            initial_strategy='median',
            max_iter=10,
            random_state=42
        )
        
        imp_df = candidates[available_iterative_columns].copy()
        
        imputed_values = iterative_imputer.fit_transform(imp_df)
            
        for i, col in enumerate(available_iterative_columns):
            candidates[col] = imputed_values[:, i]
    
    available_group_median_columns = [col for col in group_median_columns if col in candidates.columns]
    
    for col in available_group_median_columns:
        if candidates[col].isna().any():
            missing_count = candidates[col].isna().sum()
            candidates[col] = fill_median(candidates, col)

    is_catboost = 'catboost' in str(type(classifier)).lower()
    
    if is_catboost or hasattr(classifier, 'feature_names_in_'):
        X_pred = candidates[required_features].copy()
        
        for col in X_pred.columns:
            X_pred[col] = pd.to_numeric(X_pred[col], errors='coerce')
        
        if hasattr(classifier, 'predict_proba'):
            probabilities = classifier.predict_proba(X_pred)[:, 1]
        else:
            probabilities = classifier.predict(X_pred)
    else:
        X_pred = candidates.copy()
        
        numerical_cols = []
        
        if hasattr(preprocessor, 'transformers_'):
            for name, transformer, cols in preprocessor.transformers_:
                numerical_cols.extend(cols)
        
        X_transformed = preprocessor.transform(X_pred)
        
        if hasattr(classifier, 'predict_proba'):
            probabilities = classifier.predict_proba(X_transformed)[:, 1]
        else:
            probabilities = classifier.predict(X_transformed)

    
    results = candidates.copy()
    results['probability'] = probabilities
    
    results = results.sort_values('probability', ascending=False).head(top_n)
    
    recommendations = []
    print(f"\nТоп-{top_n} результаты:")  
    
    for i, (_, row) in enumerate(results.iterrows(), 1):
        supplier_name = row.get('Поставщик', f"ИНН: {row['ИНН поставщика']}")
        supplier_inn = row['ИНН поставщика']
        print(f"{i}. {supplier_name} (ИНН: {supplier_inn}) - {row['probability']:.4f}; адрес: {row['Адрес']}")
        
        recommendation = {
            'inn': str(row['ИНН поставщика']),
            'probability': round(float(row['probability']) * 100, 4),
        }
        
        if 'Поставщик' in row:
            recommendation['name'] = str(row['Поставщик'])
        if 'Регион поставки' in row:
            recommendation['region'] = str(row['Регион поставки'])
        if 'Сфера деятельности' in row:
            recommendation['sector'] = str(row['Сфера деятельности'])
        if 'Процент побед' in row:
            recommendation['win_rate'] = round(float(row['Процент побед']) * 100, 4)
        if 'Адрес' in row and not pd.isna(row['Адрес']):
            recommendation['address'] = str(row['Адрес'])
        
        recommendations.append(recommendation)
    
    
    return recommendations


def evaluate_recommendations(df, model, num_tenders=100, k_values=[1, 3, 5, 10, 20], random_seed=42, min_suppliers=1):
    """
    Оценивает качество рекомендаций на исторических данных.
    """
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    def precision_at_k(y_true, y_pred, k):
        """Precision@k - доля релевантных элементов среди топ-k рекомендаций"""
        if len(y_pred) < k:
            return 0.0
        return np.mean(y_true[np.argsort(-y_pred)[:k]])
    
    def recall_at_k(y_true, y_pred, k):
        """Recall@k - доля релевантных элементов из топ-k от всех релевантных"""
        if np.sum(y_true) == 0:
            return 0.0 
        if len(y_pred) < k:
            return np.sum(y_true[np.argsort(-y_pred)]) / np.sum(y_true)
        return np.sum(y_true[np.argsort(-y_pred)[:k]]) / np.sum(y_true)
    
    def ndcg_at_k(y_true, y_pred, k):
        """
        Normalized Discounted Cumulative Gain @ k
        Учитывает не только релевантность, но и позицию в ранжировании
        """
        if len(y_pred) < k:
            k = len(y_pred)
        if np.sum(y_true) == 0:
            return 0.0 
            
        idx = np.argsort(-y_pred)[:k]
        dcg = np.sum(y_true[idx] / np.log2(np.arange(2, k + 2)))
        
        idx_ideal = np.argsort(-y_true)[:k]
        idcg = np.sum(y_true[idx_ideal] / np.log2(np.arange(2, k + 2)))
        
        return dcg / idcg if idcg > 0 else 0.0

    print("Проверяем доступность колонок в данных...")
    
    tender_id_column = 'Реестровый номер публикации'
    target_column = 'Target'
    supplier_id_column = 'ИНН поставщика'
    
    numerical_features = [
        'Кол-во конкурентов в тендере',
        'Среднее кол-во конкурентов у заказчика',
        'Процент побед',
        'Процент побед по региону',
        'Допущен'
    ]
    ordinal_features = ['Уровень']
    
    companies_db = df[[supplier_id_column, 'Поставщик', 'Регион поставки', 
                      'Сфера деятельности', 'Субъект поставщика']].drop_duplicates()
    print(f"Всего уникальных компаний в базе: {len(companies_db)}")
    
    winner_counts = df.groupby(tender_id_column)[target_column].sum()
    valid_tenders = winner_counts[winner_counts > 0].index.tolist()
    print(f"\nНайдено {len(valid_tenders)} тендеров с победителями")
    
    filtered_df = df[df[tender_id_column].isin(valid_tenders)].copy()
    unique_tenders = filtered_df.drop_duplicates(subset=[tender_id_column])
    
    if len(unique_tenders) > num_tenders:
        test_tenders = unique_tenders.sample(n=num_tenders, random_state=random_seed)
    else:
        test_tenders = unique_tenders
    
    print(f"Будет оценено {len(test_tenders)} тендеров")
    
    metrics = {k: {'precision': [], 'recall': [], 'ndcg': []} for k in k_values}
    metrics['auc'] = []
    metrics['ap'] = []
    
    processed_tenders = 0
    skipped_tenders = 0
    successful_tenders = 0
    
    for i, (_, tender) in enumerate(test_tenders.iterrows()):
        processed_tenders += 1
        tender_id = tender[tender_id_column]

        tender_data = {
            'Сфера деятельности': tender['Сфера деятельности'],
            'Регион поставки': tender['Регион поставки'],
            'Дата публикации': tender['Дата публикации'],
            'Уровень': tender.get('Уровень', 1),
            'Тип торгов': tender.get('Тип торгов', 'Не указан'),
            'ИНН заказчика': tender.get('ИНН заказчика'),
            'Заказчик': tender.get('Заказчик')
        }
        
        actual_data = filtered_df[filtered_df[tender_id_column] == tender_id].copy()
        actual_winners = actual_data[actual_data[target_column] == 1][supplier_id_column].unique()
        
        if len(actual_winners) == 0:
            skipped_tenders += 1
            continue
        
        try:
            candidates = get_candidate_companies(tender_data, companies_db, df, top_n=1000)
            
            if len(candidates) < min_suppliers:
                skipped_tenders += 1
                continue
            
            for idx, company in candidates.iterrows():
                company_features = calculate_company_features(company, df)
                for key, value in company_features.items():
                    candidates.at[idx, key] = value
            
            predictions_df = pd.DataFrame()
            tender_df = pd.DataFrame([tender_data])
            
            for _, candidate in candidates.iterrows():
                tender_copy = tender_df.copy()
                for col in candidate.index:
                    if col != supplier_id_column:
                        tender_copy[col] = candidate[col]
                predictions_df = pd.concat([predictions_df, tender_copy])
            
            for feature in numerical_features + ordinal_features:
                if feature not in predictions_df.columns:
                    if feature in ['Допущен']:
                        predictions_df[feature] = 1 
                    else:
                        predictions_df[feature] = 0 
            
            if isinstance(model, dict) and 'model' in model:
                pipeline = model['model']
            elif hasattr(model, 'best_estimator_'):
                pipeline = model.best_estimator_
            else:
                pipeline = model
            
            preprocessor = None
            classifier = None
            
            if hasattr(pipeline, 'named_steps'):
                if 'preprocessor' in pipeline.named_steps:
                    preprocessor = pipeline.named_steps['preprocessor']
                if 'classifier' in pipeline.named_steps:
                    classifier = pipeline.named_steps['classifier']
            elif hasattr(pipeline, 'steps'):
                for name, step in pipeline.steps:
                    if name in ['preprocessor', 'preprocessing', 'transform']:
                        preprocessor = step
                    elif name in ['classifier', 'estimator', 'model']:
                        classifier = step
            elif hasattr(pipeline, '_final_estimator'):
                classifier = pipeline._final_estimator
                if hasattr(pipeline, 'transformers_'):
                    preprocessor = pipeline.transformers_
            
            if classifier is None:
                classifier = pipeline
            
            if preprocessor is not None:
                categorical_cols = []
                numerical_cols = []
                
                if hasattr(preprocessor, 'transformers_'):
                    for name, transformer, cols in preprocessor.transformers_:
                        if name == 'categorical':
                            categorical_cols = cols
                        elif name in ['numeric', 'numerical']:
                            numerical_cols.extend(cols)
                
                X_pred = predictions_df.copy()
                
                for col in numerical_features:
                    if col in X_pred.columns:
                        X_pred[col] = pd.to_numeric(X_pred[col], errors='coerce').fillna(0)
                
                for col in categorical_cols:
                    if col in X_pred.columns:
                        X_pred[col] = X_pred[col].fillna('Other')
                
                X_transformed = preprocessor.transform(X_pred)
                
                if hasattr(classifier, 'predict_proba'):
                    y_pred = classifier.predict_proba(X_transformed)[:, 1]
                else:
                    y_pred = classifier.predict(X_transformed)
            else:
                if hasattr(classifier, 'predict_proba'):
                    y_pred = classifier.predict_proba(predictions_df)[:, 1]
                else:
                    y_pred = classifier.predict(predictions_df)
            
            y_true = np.zeros(len(candidates))
            for i, candidate in enumerate(candidates[supplier_id_column]):
                if candidate in actual_winners:
                    y_true[i] = 1
            
            for k in k_values:
                if k <= len(y_pred):
                    precision = precision_at_k(y_true, y_pred, k)
                    recall = recall_at_k(y_true, y_pred, k)
                    ndcg = ndcg_at_k(y_true, y_pred, k)
                    
                    metrics[k]['precision'].append(precision)
                    metrics[k]['recall'].append(recall)
                    metrics[k]['ndcg'].append(ndcg)
                    
            if len(np.unique(y_true)) > 1:
                try:
                    auc = roc_auc_score(y_true, y_pred)
                    ap = average_precision_score(y_true, y_pred)
                    metrics['auc'].append(auc)
                    metrics['ap'].append(ap)
                except Exception as e:
                    print(f"Ошибка при расчете AUC/AP: {str(e)}")
            
            successful_tenders += 1
            
        except Exception as e:
            print(f"Ошибка при обработке тендера: {str(e)}")
            skipped_tenders += 1
            continue
    
    results = {
        'processed_tenders': processed_tenders,
        'skipped_tenders': skipped_tenders,
        'successful_tenders': successful_tenders,
        'total_tenders': len(test_tenders)
    }
    
    if successful_tenders > 0:
        for k in k_values:
            if metrics[k]['precision']:
                results[f'precision@{k}'] = float(np.mean(metrics[k]['precision']))
                results[f'recall@{k}'] = float(np.mean(metrics[k]['recall']))
                results[f'ndcg@{k}'] = float(np.mean(metrics[k]['ndcg']))
                
                results[f'precision@{k}_std'] = float(np.std(metrics[k]['precision']))
                results[f'recall@{k}_std'] = float(np.std(metrics[k]['recall']))
                results[f'ndcg@{k}_std'] = float(np.std(metrics[k]['ndcg']))
                
                results[f'precision@{k}_min'] = float(np.min(metrics[k]['precision']))
                results[f'precision@{k}_max'] = float(np.max(metrics[k]['precision']))
                results[f'recall@{k}_min'] = float(np.min(metrics[k]['recall']))
                results[f'recall@{k}_max'] = float(np.max(metrics[k]['recall']))
            else:
                results[f'precision@{k}'] = 0.0
                results[f'recall@{k}'] = 0.0
                results[f'ndcg@{k}'] = 0.0
                results[f'precision@{k}_std'] = 0.0
                results[f'recall@{k}_std'] = 0.0
                results[f'ndcg@{k}_std'] = 0.0
                results[f'precision@{k}_min'] = 0.0
                results[f'precision@{k}_max'] = 0.0
                results[f'recall@{k}_min'] = 0.0
                results[f'recall@{k}_max'] = 0.0
        
        all_recalls = []
        for k in k_values:
            if metrics[k]['recall']:
                all_recalls.extend(metrics[k]['recall'])
        results['mean_recall'] = float(np.mean(all_recalls)) if all_recalls else 0.0
        results['mean_recall_std'] = float(np.std(all_recalls)) if all_recalls else 0.0
        
        results['roc_auc'] = float(np.mean(metrics['auc'])) if metrics['auc'] else 0.0
        results['roc_auc_std'] = float(np.std(metrics['auc'])) if metrics['auc'] else 0.0
        results['average_precision'] = float(np.mean(metrics['ap'])) if metrics['ap'] else 0.0
        results['average_precision_std'] = float(np.std(metrics['ap'])) if metrics['ap'] else 0.0
    else:
        for k in k_values:
            results[f'precision@{k}'] = 0.0
            results[f'recall@{k}'] = 0.0
            results[f'ndcg@{k}'] = 0.0
            results[f'precision@{k}_std'] = 0.0
            results[f'recall@{k}_std'] = 0.0
            results[f'ndcg@{k}_std'] = 0.0
            results[f'precision@{k}_min'] = 0.0
            results[f'precision@{k}_max'] = 0.0
            results[f'recall@{k}_min'] = 0.0
            results[f'recall@{k}_max'] = 0.0
        
        results['mean_recall'] = 0.0
        results['mean_recall_std'] = 0.0
        results['roc_auc'] = 0.0
        results['roc_auc_std'] = 0.0
        results['average_precision'] = 0.0
        results['average_precision_std'] = 0.0
    
    return results




def visualize_evaluation_results(results, title="Оценка качества рекомендаций"):
    """
    Визуализирует результаты оценки рекомендаций с помощью графиков
    """

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(title, fontsize=16)
    
    has_metrics = results.get('successful_tenders', 0) > 0
    has_error = 'error' in results
    
    k_values = [1, 3, 5, 10, 20]
    
    ax1 = axes[0]
    if has_metrics:
        precision_values = [results.get(f'precision@{k}', 0) for k in k_values]
        recall_values = [results.get(f'recall@{k}', 0) for k in k_values]
        
        x = np.arange(len(k_values))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, precision_values, width, label='Precision@K', color='blue', alpha=0.7)
        bars2 = ax1.bar(x + width/2, recall_values, width, label='Recall@K', color='green', alpha=0.7)
        
        ax1.set_xlabel('K')
        ax1.set_ylabel('Значение метрики')
        ax1.set_title('Precision@K и Recall@K')
        ax1.set_xticks(x)
        ax1.set_xticklabels(k_values)
        ax1.legend()
        
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax1.annotate(f'{height:.2f}',
                                xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, 3),
                                textcoords="offset points",
                                ha='center', va='bottom')
    else:
        ax1.text(0.5, 0.5, 'Нет данных для отображения', ha='center', va='center', fontsize=12)
        ax1.axis('off')
    
    ax2 = axes[1]
    if has_metrics:
        ndcg_values = [results.get(f'ndcg@{k}', 0) for k in k_values]
        ax2.plot(k_values, ndcg_values, 'o-', color='purple', linewidth=2, markersize=8)
        ax2.set_xlabel('K')
        ax2.set_ylabel('NDCG')
        ax2.set_title('NDCG@K')
        ax2.set_xticks(k_values)
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        for i, txt in enumerate(ndcg_values):
            if txt > 0:
                ax2.annotate(f'{txt:.2f}', (k_values[i], ndcg_values[i]), 
                            textcoords="offset points", xytext=(0, 10), ha='center')
    else:
        ax2.text(0.5, 0.5, 'Нет данных для отображения', ha='center', va='center', fontsize=12)
        ax2.axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig



def print_evaluation_results(results):
    """
    Выводит результаты оценки рекомендаций в отформатированном виде

    """

    if results['successful_tenders'] > 0:
        print("\n=== Метрики качества ===")
        
        k_values = [1, 3, 5, 10, 20]
        for k in k_values:
            if f'precision@{k}' in results:
                print(f"\nМетрики для top-{k}:")
                print(f"Precision@{k}: {results[f'precision@{k}']:.4f} ± {results[f'precision@{k}_std']:.4f}")
                print(f"Recall@{k}: {results[f'recall@{k}']:.4f} ± {results[f'recall@{k}_std']:.4f}")
                print(f"NDCG@{k}: {results[f'ndcg@{k}']:.4f} ± {results[f'ndcg@{k}_std']:.4f}")
        
        print("\n=== Глобальные метрики ===")
        if 'roc_auc' in results:
            print(f"ROC AUC: {results['roc_auc']:.4f} ± {results['roc_auc_std']:.4f}")
        if 'average_precision' in results:
            print(f"Average Precision: {results['average_precision']:.4f} ± {results['average_precision_std']:.4f}")
        if 'mean_recall' in results:
            print(f"Mean Recall: {results['mean_recall']:.4f} ± {results['mean_recall_std']:.4f}")
    
  
