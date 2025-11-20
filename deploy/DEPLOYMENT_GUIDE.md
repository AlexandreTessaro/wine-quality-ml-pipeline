
# Guia de Deploy - Modelos de Qualidade de Vinhos

## Arquivos Deployados

Os modelos foram deployados usando duas técnicas:

### 1. Pickle (pickle.pkl)
- Biblioteca padrão do Python
- Boa para objetos Python simples
- Compatível com todas as versões do Python

### 2. Joblib (joblib.pkl)
- Otimizado para arrays NumPy grandes
- Melhor compressão
- Mais rápido para modelos scikit-learn

## Como Usar

### Carregar modelo com Pickle:
```python
import pickle

with open('deploy/random_forest_pickle.pkl', 'rb') as f:
    package = pickle.load(f)

model = package['model']
scaler = package['scaler']
```

### Carregar modelo com Joblib:
```python
import joblib

package = joblib.load('deploy/random_forest_joblib.pkl')

model = package['model']
scaler = package['scaler']
```

### Fazer Previsão:
```python
import pandas as pd
import numpy as np

# Preparar features (mesma ordem do treinamento)
features = pd.DataFrame({
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
    'wine_type_encoded': [0],
    'total_acidity': [8.1],
    'alcohol_acidity_ratio': [1.16]
})

# Normalizar
features_scaled = scaler.transform(features)

# Prever
prediction = model.predict(features_scaled)
print(f"Qualidade prevista: {prediction[0]}")
```

## Comparação Pickle vs Joblib

| Característica | Pickle | Joblib |
|---------------|--------|--------|
| Velocidade | Média | Rápida |
| Compressão | Básica | Avançada |
| Tamanho arquivo | Maior | Menor |
| Compatibilidade | Universal | scikit-learn |
| Uso recomendado | Objetos Python gerais | Modelos ML com arrays NumPy |

## Modelos Disponíveis

1. Random Forest Classifier
2. Support Vector Machine (SVM)
3. Gradient Boosting Classifier

Todos os modelos incluem:
- Modelo treinado
- Scaler para normalização
- Metadados (nome, versão)
