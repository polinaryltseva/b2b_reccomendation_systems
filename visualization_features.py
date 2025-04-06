import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

def get_feature_importance(model, numerical_features, time_features, categorical_features, categorical_encoder=None):
    feature_importance = {}
    importance_values = model.coef_[0] if hasattr(model, 'coef_') else model.feature_importances_
    
    current_position = 0
    for feature in numerical_features + time_features:
        feature_importance[feature] = abs(importance_values[current_position])
        current_position += 1
    
    if categorical_encoder is not None:
        for feature in categorical_features:
            n_categories = len(categorical_encoder.categories_[categorical_features.index(feature)]) - 1
            feature_importance[feature] = sum(abs(importance_values[current_position:current_position + n_categories]))
            current_position += n_categories
    
    importance_df = pd.DataFrame({
        'feature': list(feature_importance.keys()),
        'importance': list(feature_importance.values())
    }).sort_values('importance', ascending=False)
    
    return importance_df

def plot_feature_importance(importance_df, title="Feature Importance"):
    plt.figure(figsize=(12, 6))
    plt.bar(importance_df['feature'], importance_df['importance'])
    plt.xticks(rotation=45, ha='right')
    plt.title(title)
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.tight_layout()
    plt.show()
    
    print("\nВажность признаков:")
    print(importance_df)


def evaluate_model(name, y_true, y_pred, y_proba):
    print(f"{name} Metrics:")
    print("Accuracy:", round(accuracy_score(y_true, y_pred), 4))
    print("Precision:", round(precision_score(y_true, y_pred), 4))
    print("Recall:", round(recall_score(y_true, y_pred), 4))
    print("F1-Score:", round(f1_score(y_true, y_pred), 4))
    print("ROC-AUC:", round(roc_auc_score(y_true, y_proba), 4))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()