# Exemplo de Uso Rápido

## Instalação

```bash
# Criar ambiente virtual (recomendado)
python -m venv venv

# Ativar ambiente virtual
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Instalar dependências
pip install -r requirements.txt
```

## Execução Completa

Execute o pipeline completo com um único comando:

```bash
python main.py
```

Este comando executa todas as etapas:
1. Pipeline de dados (ETL)
2. Treinamento dos 3 modelos
3. Avaliação com 3 métricas
4. Deploy usando Pickle e Joblib

## Execução por Etapas

### 1. Pipeline de Dados

```bash
python src/data_pipeline.py
```

Gera:
- Dados brutos em `data/raw/`
- Dados processados em `data/processed/`

### 2. Treinamento de Modelos

```bash
python src/train_models.py
```

Treina e salva:
- Random Forest
- SVM
- Gradient Boosting

Modelos salvos em `models/`

### 3. Avaliação

```bash
python src/evaluate.py
```

Gera:
- Relatório de métricas
- Gráficos de comparação
- Matrizes de confusão
- Tabela de resultados em CSV

Resultados em `results/`

### 4. Deploy

```bash
python src/deploy.py
```

Cria arquivos deployados em `deploy/`:
- Versões Pickle (.pkl)
- Versões Joblib (.pkl)
- Guia de uso

## Usar Modelo Deployado

```python
import joblib
import pandas as pd
import numpy as np

# Carregar modelo deployado
package = joblib.load('deploy/random_forest_joblib.pkl')

model = package['model']
scaler = package['scaler']

# Preparar dados de exemplo
# (usar as mesmas features do treinamento)
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
    'wine_type_encoded': [0],  # 0=red, 1=white
    'total_acidity': [8.1],
    'alcohol_acidity_ratio': [1.16]
})

# Normalizar
features_scaled = scaler.transform(features)

# Prever
prediction = model.predict(features_scaled)
print(f"Qualidade prevista: {prediction[0]}")
```

## Estrutura de Diretórios

```
ml-pipeline/
├── data/
│   ├── raw/              # Dados brutos (Bronze)
│   ├── processed/         # Dados processados (Silver/Gold)
│   └── external/          # Dados externos
├── src/
│   ├── data_pipeline.py   # ETL
│   ├── train_models.py    # Treinamento
│   ├── evaluate.py        # Avaliação
│   └── deploy.py          # Deploy
├── models/                # Modelos treinados
├── results/               # Resultados e gráficos
├── deploy/                # Modelos deployados
├── main.py                # Script principal
├── requirements.txt       # Dependências
└── README.md              # Documentação
```

## Troubleshooting

### Erro ao baixar dados
Se houver problema ao baixar dados da UCI, os dados serão salvos localmente após primeira execução bem-sucedida.

### Erro de memória
Para datasets grandes, considere reduzir o número de estimadores nos modelos ou usar amostragem.

### Modelos não encontrados
Certifique-se de executar `train_models.py` antes de `evaluate.py` e `deploy.py`.

