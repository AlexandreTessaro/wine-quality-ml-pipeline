"""
Treinamento de Modelos de Machine Learning
Implementa 3 algoritmos diferentes: Random Forest, SVM e Gradient Boosting
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Modelos
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

# Utilitários
from sklearn.model_selection import cross_val_score
import joblib

# Configuração de diretórios
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data' / 'processed'
MODELS_DIR = BASE_DIR / 'models'

# Criar diretório de modelos
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    """Carrega os dados preparados para treinamento"""
    print("Carregando dados preparados...")
    
    X_train = pd.read_csv(DATA_DIR / 'X_train.csv')
    X_test = pd.read_csv(DATA_DIR / 'X_test.csv')
    y_train = pd.read_csv(DATA_DIR / 'y_train.csv').squeeze()
    y_test = pd.read_csv(DATA_DIR / 'y_test.csv').squeeze()
    
    print(f"✓ Dados carregados:")
    print(f"  - Treino: {len(X_train)} amostras")
    print(f"  - Teste: {len(X_test)} amostras")
    
    return X_train, X_test, y_train, y_test


def train_random_forest(X_train, y_train):
    """
    Treina modelo Random Forest Classifier
    Random Forest é um ensemble de árvores de decisão que reduz overfitting
    """
    print("\n" + "=" * 60)
    print("TREINANDO: Random Forest Classifier")
    print("=" * 60)
    
    # Hiperparâmetros otimizados
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    
    print("\nTreinando modelo...")
    model.fit(X_train, y_train)
    
    # Validação cruzada
    print("\nRealizando validação cruzada (5 folds)...")
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"  Acurácia média (CV): {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Salvar modelo
    model_path = MODELS_DIR / 'random_forest_model.pkl'
    joblib.dump(model, model_path)
    print(f"\n✓ Modelo salvo em: {model_path}")
    
    return model


def train_svm(X_train, y_train):
    """
    Treina modelo Support Vector Machine (SVM)
    SVM encontra o hiperplano ótimo para separar as classes
    """
    print("\n" + "=" * 60)
    print("TREINANDO: Support Vector Machine (SVM)")
    print("=" * 60)
    
    # SVM com kernel RBF (Radial Basis Function)
    # Usando C menor e gamma menor para reduzir tempo de treinamento
    model = SVC(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        random_state=42,
        verbose=True
    )
    
    print("\nTreinando modelo (pode levar alguns minutos)...")
    model.fit(X_train, y_train)
    
    # Validação cruzada
    print("\nRealizando validação cruzada (5 folds)...")
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"  Acurácia média (CV): {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Salvar modelo
    model_path = MODELS_DIR / 'svm_model.pkl'
    joblib.dump(model, model_path)
    print(f"\n✓ Modelo salvo em: {model_path}")
    
    return model


def train_gradient_boosting(X_train, y_train):
    """
    Treina modelo Gradient Boosting Classifier
    Gradient Boosting combina múltiplos modelos fracos sequencialmente
    """
    print("\n" + "=" * 60)
    print("TREINANDO: Gradient Boosting Classifier")
    print("=" * 60)
    
    # Hiperparâmetros otimizados
    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        verbose=1
    )
    
    print("\nTreinando modelo...")
    model.fit(X_train, y_train)
    
    # Validação cruzada
    print("\nRealizando validação cruzada (5 folds)...")
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"  Acurácia média (CV): {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Salvar modelo
    model_path = MODELS_DIR / 'gradient_boosting_model.pkl'
    joblib.dump(model, model_path)
    print(f"\n✓ Modelo salvo em: {model_path}")
    
    return model


def main():
    """
    Executa o treinamento de todos os modelos
    """
    print("\n" + "=" * 60)
    print("TREINAMENTO DE MODELOS DE MACHINE LEARNING")
    print("=" * 60)
    
    # Carregar dados
    X_train, X_test, y_train, y_test = load_data()
    
    # Treinar modelos
    models = {}
    
    models['Random Forest'] = train_random_forest(X_train, y_train)
    models['SVM'] = train_svm(X_train, y_train)
    models['Gradient Boosting'] = train_gradient_boosting(X_train, y_train)
    
    print("\n" + "=" * 60)
    print("TREINAMENTO CONCLUÍDO!")
    print("=" * 60)
    print(f"\nModelos treinados e salvos em: {MODELS_DIR}")
    print("\nModelos disponíveis:")
    for name in models.keys():
        print(f"  - {name}")
    
    return models


if __name__ == "__main__":
    models = main()

