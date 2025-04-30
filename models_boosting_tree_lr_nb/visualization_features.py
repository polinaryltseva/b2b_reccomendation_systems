import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.inspection import permutation_importance


def get_feature_names_nb(preprocessor, numerical_features, percent_features, time_features, categorical_features):
    """
    Получает имена признаков после препроцессинга для LightGBM модели
    
    Args:
        preprocessor: ColumnTransformer для LightGBM
        numerical_features: список числовых признаков
        ordinal_features: список порядковых признаков
    
    Returns:
        list: список имен признаков
    """
    feature_names = []
    
    feature_names.extend(numerical_features)
    feature_names.extend(percent_features)
    feature_names.extend(time_features)
    feature_names.extend(categorical_features)
    
    return feature_names


def plot_feature_importance(importance_df, title="Feature Importance", top_n=20):
    importance_df = importance_df.sort_values(by='importance', ascending=True).tail(top_n)
    
    plt.figure(figsize=(12, 8))
    colors = plt.cm.viridis_r(np.linspace(0.4, 1, len(importance_df)))
    bars = plt.barh(importance_df['feature'], importance_df['importance'], color=colors, edgecolor='black')
    
    for bar in bars:
        width = bar.get_width()
        plt.text(width * 1.005, bar.get_y() + bar.get_height()/2,
                 f'{width:.3f}',
                 va='center', ha='left', fontsize=9)
    
    plt.title(title, fontsize=14, pad=20)
    plt.xlabel('Importance Score', fontsize=12)
    plt.ylabel('')
    plt.xlim(right=importance_df['importance'].max() * 1.15)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    
    plt.show()

    print("\nТоп важных признаков:")
    print(importance_df.sort_values(by='importance', ascending=False))



def evaluate_model(name, y_true, y_pred, y_proba):
    print(f"{name} Metrics:")
    print("Accuracy:", round(accuracy_score(y_true, y_pred), 4))
    print("Precision:", round(precision_score(y_true, y_pred), 4))
    print("Recall:", round(recall_score(y_true, y_pred), 4))
    print("F1-Score:", round(f1_score(y_true, y_pred), 4))
    print("ROC-AUC:", round(roc_auc_score(y_true, y_proba), 4))
    
    cm = confusion_matrix(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    tpr_manual = tp / (tp + fn)
    print(f"TPR: {round(tpr_manual, 4)}")
    print(confusion_matrix(y_true, y_pred))

    print("\nConfusion Matrix:")
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()
