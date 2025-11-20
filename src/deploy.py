"""
Deploy de Modelos usando Pickle e Joblib
Serializa modelos treinados para uso em produção
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler

# Configuração de diretórios
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data' / 'processed'
MODELS_DIR = BASE_DIR / 'models'
DEPLOY_DIR = BASE_DIR / 'deploy'

# Criar diretório de deploy
DEPLOY_DIR.mkdir(parents=True, exist_ok=True)


def load_scaler():
    """Carrega o scaler usado no pré-processamento"""
    # O scaler foi salvo durante o pipeline de dados
    # Vamos recriá-lo a partir dos dados de treino
    X_train = pd.read_csv(DATA_DIR / 'X_train.csv')
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def deploy_with_pickle(model, model_name, scaler):
    """
    Deploy usando Pickle (biblioteca padrão do Python)
    Pickle é nativo do Python e funciona bem para objetos Python simples
    """
    print(f"\n{'=' * 60}")
    print(f"DEPLOY COM PICKLE: {model_name}")
    print(f"{'=' * 60}")
    
    # Criar pacote de deploy
    deploy_package = {
        'model': model,
        'scaler': scaler,
        'model_name': model_name,
        'version': '1.0.0'
    }
    
    # Salvar com Pickle
    pickle_path = DEPLOY_DIR / f'{model_name.lower().replace(" ", "_")}_pickle.pkl'
    
    with open(pickle_path, 'wb') as f:
        pickle.dump(deploy_package, f)
    
    file_size = pickle_path.stat().st_size / (1024 * 1024)  # MB
    print(f"\n✓ Modelo salvo com Pickle:")
    print(f"  Arquivo: {pickle_path}")
    print(f"  Tamanho: {file_size:.2f} MB")
    
    # Testar carregamento
    print("\nTestando carregamento...")
    with open(pickle_path, 'rb') as f:
        loaded_package = pickle.load(f)
    
    print(f"✓ Modelo carregado com sucesso!")
    print(f"  Modelo: {loaded_package['model_name']}")
    print(f"  Versão: {loaded_package['version']}")
    
    return pickle_path


def deploy_with_joblib(model, model_name, scaler):
    """
    Deploy usando Joblib (otimizado para arrays NumPy)
    Joblib é mais eficiente para modelos scikit-learn com arrays NumPy grandes
    """
    print(f"\n{'=' * 60}")
    print(f"DEPLOY COM JOBLIB: {model_name}")
    print(f"{'=' * 60}")
    
    # Criar pacote de deploy
    deploy_package = {
        'model': model,
        'scaler': scaler,
        'model_name': model_name,
        'version': '1.0.0'
    }
    
    # Salvar com Joblib (usa compressão por padrão)
    joblib_path = DEPLOY_DIR / f'{model_name.lower().replace(" ", "_")}_joblib.pkl'
    
    joblib.dump(deploy_package, joblib_path, compress=3)  # compress=3 para melhor compressão
    
    file_size = joblib_path.stat().st_size / (1024 * 1024)  # MB
    print(f"\n✓ Modelo salvo com Joblib:")
    print(f"  Arquivo: {joblib_path}")
    print(f"  Tamanho: {file_size:.2f} MB")
    print(f"  Compressão: nível 3 (otimizado)")
    
    # Testar carregamento
    print("\nTestando carregamento...")
    loaded_package = joblib.load(joblib_path)
    
    print(f"✓ Modelo carregado com sucesso!")
    print(f"  Modelo: {loaded_package['model_name']}")
    print(f"  Versão: {loaded_package['version']}")
    
    return joblib_path


def create_prediction_function(model_path, method='pickle'):
    """
    Cria função de exemplo para fazer previsões com o modelo deployado
    """
    print(f"\n{'=' * 60}")
    print(f"CRIANDO FUNÇÃO DE PREDIÇÃO ({method.upper()})")
    print(f"{'=' * 60}")
    
    if method == 'pickle':
        with open(model_path, 'rb') as f:
            package = pickle.load(f)
    else:
        package = joblib.load(model_path)
    
    model = package['model']
    scaler = package['scaler']
    
    def predict_wine_quality(features):
        """
        Função para prever qualidade do vinho
        
        Args:
            features: array ou DataFrame com features do vinho
        
        Returns:
            qualidade prevista (0-10)
        """
        # Normalizar features
        features_scaled = scaler.transform(features)
        
        # Fazer previsão
        prediction = model.predict(features_scaled)
        
        return prediction[0] if len(prediction) == 1 else prediction
    
    return predict_wine_quality


def test_deployment():
    """
    Testa o modelo deployado com dados de exemplo
    """
    print(f"\n{'=' * 60}")
    print("TESTANDO DEPLOY")
    print(f"{'=' * 60}")
    
    # Carregar dados de teste
    X_test = pd.read_csv(DATA_DIR / 'X_test.csv')
    y_test = pd.read_csv(DATA_DIR / 'y_test.csv').squeeze()
    
    # Testar com Pickle
    pickle_path = DEPLOY_DIR / 'random_forest_pickle.pkl'
    if pickle_path.exists():
        print("\n1. Testando modelo deployado com Pickle...")
        predict_fn = create_prediction_function(pickle_path, 'pickle')
        
        # Testar com primeira amostra
        sample = X_test.iloc[0:1]
        prediction = predict_fn(sample)
        actual = y_test.iloc[0]
        
        print(f"   Amostra de teste:")
        print(f"   Previsão: {prediction}")
        print(f"   Real: {actual}")
        print(f"   ✓ Previsão funcionando corretamente!")
    
    # Testar com Joblib
    joblib_path = DEPLOY_DIR / 'random_forest_joblib.pkl'
    if joblib_path.exists():
        print("\n2. Testando modelo deployado com Joblib...")
        predict_fn = create_prediction_function(joblib_path, 'joblib')
        
        # Testar com primeira amostra
        sample = X_test.iloc[0:1]
        prediction = predict_fn(sample)
        actual = y_test.iloc[0]
        
        print(f"   Amostra de teste:")
        print(f"   Previsão: {prediction}")
        print(f"   Real: {actual}")
        print(f"   ✓ Previsão funcionando corretamente!")


def create_deployment_guide():
    """
    Cria guia de uso dos modelos deployados
    """
    guide = """
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
"""
    
    guide_path = DEPLOY_DIR / 'DEPLOYMENT_GUIDE.md'
    with open(guide_path, 'w', encoding='utf-8') as f:
        f.write(guide)
    
    print(f"\n✓ Guia de deploy salvo em: {guide_path}")


def main():
    """
    Executa o deploy de todos os modelos usando Pickle e Joblib
    """
    print("\n" + "=" * 60)
    print("DEPLOY DE MODELOS - PICKLE E JOBLIB")
    print("=" * 60)
    
    # Carregar scaler
    print("\nCarregando scaler...")
    scaler = load_scaler()
    print("✓ Scaler carregado")
    
    # Carregar modelos
    model_files = {
        'Random Forest': 'random_forest_model.pkl',
        'SVM': 'svm_model.pkl',
        'Gradient Boosting': 'gradient_boosting_model.pkl'
    }
    
    deployed_models = []
    
    for model_name, model_file in model_files.items():
        model_path = MODELS_DIR / model_file
        
        if not model_path.exists():
            print(f"\n⚠ {model_name} não encontrado. Pulando...")
            continue
        
        print(f"\n{'=' * 60}")
        print(f"PROCESSANDO: {model_name}")
        print(f"{'=' * 60}")
        
        # Carregar modelo
        model = joblib.load(model_path)
        
        # Deploy com Pickle
        pickle_path = deploy_with_pickle(model, model_name, scaler)
        
        # Deploy com Joblib
        joblib_path = deploy_with_joblib(model, model_name, scaler)
        
        deployed_models.append({
            'name': model_name,
            'pickle': pickle_path,
            'joblib': joblib_path
        })
    
    # Criar guia de deploy
    create_deployment_guide()
    
    # Testar deployment
    test_deployment()
    
    print("\n" + "=" * 60)
    print("DEPLOY CONCLUÍDO!")
    print("=" * 60)
    print(f"\nModelos deployados em: {DEPLOY_DIR}")
    print("\nArquivos criados:")
    for model_info in deployed_models:
        print(f"\n  {model_info['name']}:")
        print(f"    - Pickle: {model_info['pickle'].name}")
        print(f"    - Joblib: {model_info['joblib'].name}")
    
    print(f"\n✓ Guia de uso: {DEPLOY_DIR / 'DEPLOYMENT_GUIDE.md'}")


if __name__ == "__main__":
    main()

