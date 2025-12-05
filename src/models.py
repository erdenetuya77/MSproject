"""
Naive Bayes загварын модуль

Зээлийн эрсдэлийн үнэлгээнд Naive Bayes загварыг
сургаж үнэлэх функцууд
"""

import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


def train_naive_bayes(X_train, y_train):
    """
    Gaussian Naive Bayes загвар сургах

    Parameters:
    -----------
    X_train : array-like
        Train features
    y_train : array-like
        Train target

    Returns:
    --------
    GaussianNB
        Сургагдсан загвар
    """
    model = GaussianNB()
    model.fit(X_train, y_train)
    print("✓ Naive Bayes загвар сургагдлаа")
    return model


def evaluate_model(model, X_test, y_test, model_name="Naive Bayes"):
    """
    Загварыг үнэлэх

    Parameters:
    -----------
    model : sklearn model
        Үнэлэх загвар
    X_test : array-like
        Test features
    y_test : array-like
        Test target
    model_name : str
        Загварын нэр

    Returns:
    --------
    dict
        Үнэлгээний үр дүн
    """
    # Таамаглал хийх
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Үзүүлэлтүүд тооцоолох
    metrics = {
        'Загвар': model_name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'AUC': roc_auc_score(y_test, y_proba)
    }

    return {
        'metrics': metrics,
        'y_pred': y_pred,
        'y_proba': y_proba,
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }


def print_evaluation(evaluation_results):
    """
    Үнэлгээний үр дүнг хэвлэх

    Parameters:
    -----------
    evaluation_results : dict
        evaluate_model()-ээс буцаасан үр дүн
    """
    metrics = evaluation_results['metrics']
    cm = evaluation_results['confusion_matrix']

    print(f"\n{'='*50}")
    print(f"Загвар: {metrics['Загвар']}")
    print(f"{'='*50}")
    print(f"Accuracy : {metrics['Accuracy']:.4f}")
    print(f"Precision: {metrics['Precision']:.4f}")
    print(f"Recall   : {metrics['Recall']:.4f}")
    print(f"F1-Score : {metrics['F1-Score']:.4f}")
    print(f"AUC      : {metrics['AUC']:.4f}")
    print(f"\nConfusion Matrix:")
    print(cm)
    print(f"{'='*50}\n")


if __name__ == "__main__":
    # Жишээ ашиглалт
    print("Энэ модуль нь Naive Bayes загварыг сургах болон үнэлэх функцуудыг агуулна")
    print("preprocessing.py модультой хамт ашиглана")
