"""
Demonstração completa do deploy
"""
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

def demo_deploy():
    print("=" * 70)
    print("DEMONSTRAÇÃO DO DEPLOY - Modelos de Qualidade de Vinhos")
    print("=" * 70)
    
    # 1. Carregar modelo
    print("\n[1] Carregando modelo deployado...")
    package = joblib.load('deploy/svm_joblib.pkl')  # Usando SVM (menor arquivo)
    
    model = package['model']
    scaler = package['scaler']
    
    print(f"   ✓ Modelo: {package['model_name']}")
    print(f"   ✓ Versão: {package['version']}")
    
    # 2. Preparar dados de exemplo - ORDEM CORRETA DAS FEATURES
    print("\n[2] Preparando dados de exemplo...")
    # IMPORTANTE: A ordem deve ser EXATAMENTE a mesma do treinamento
    exemplo_vinho = pd.DataFrame({
        'fixed acidity': [7.4],
        'volatile acidity': [0.7],
        'citric acid': [0.0],
        'residual sugar': [1.9],
        'chlorides': [0.076],
        'free sulfur dioxide': [11.0],
        'total sulfur dioxide': [34.0],
        'density': [0.9978],
        'pH': [3.51],
        'sulphates': [0.56],
        'alcohol': [9.4],
        'total_acidity': [8.1],  # Calculada: fixed + volatile
        'alcohol_acidity_ratio': [1.16],  # Calculada: alcohol / total_acidity
        'wine_type_encoded': [0]  # 0 = tinto, 1 = branco
    })
    
    # Garantir a ordem correta das colunas (mesma ordem do treinamento)
    colunas_ordenadas = [
        'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
        'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
        'pH', 'sulphates', 'alcohol', 'total_acidity', 
        'alcohol_acidity_ratio', 'wine_type_encoded'
    ]
    exemplo_vinho = exemplo_vinho[colunas_ordenadas]
    
    print("   Features do vinho:")
    for col in colunas_ordenadas:
        print(f"     {col}: {exemplo_vinho[col].values[0]}")
    
    # 3. Normalizar
    print("\n[3] Normalizando features...")
    exemplo_normalizado = scaler.transform(exemplo_vinho)
    print(f"   ✓ Dados normalizados (shape: {exemplo_normalizado.shape})")
    
    # 4. Fazer previsão
    print("\n[4] Fazendo previsão...")
    qualidade = model.predict(exemplo_normalizado)
    probabilidades = None
    
    if hasattr(model, 'predict_proba'):
        probabilidades = model.predict_proba(exemplo_normalizado)[0]
    
    print(f"   ✓ Qualidade prevista: {qualidade[0]}/10")
    
    if probabilidades is not None:
        print("\n   Probabilidades por classe:")
        for i, prob in enumerate(probabilidades):
            print(f"     Classe {i}: {prob*100:.2f}%")
    
    # 5. Comparar modelos
    print("\n[5] Comparando modelos deployados:")
    modelos = {
        'Random Forest': 'random_forest_joblib.pkl',
        'SVM': 'svm_joblib.pkl',
        'Gradient Boosting': 'gradient_boosting_joblib.pkl'
    }
    
    for nome, arquivo in modelos.items():
        try:
            pkg = joblib.load(f'deploy/{arquivo}')
            pred = pkg['model'].predict(exemplo_normalizado)
            size = Path(f'deploy/{arquivo}').stat().st_size / (1024 * 1024)
            print(f"   {nome:20} → Qualidade: {pred[0]} | Tamanho: {size:.2f} MB")
        except Exception as e:
            print(f"   {nome:20} → Erro: {str(e)[:50]}")
    
    print("\n" + "=" * 70)
    print("DEMONSTRAÇÃO CONCLUÍDA!")
    print("=" * 70)

if __name__ == "__main__":
    demo_deploy()