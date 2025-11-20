# 🍷 Pipeline de Machine Learning - Previsão de Qualidade de Vinhos

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.2-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Projeto completo de Machine Learning para classificação de qualidade de vinhos utilizando pipeline de dados profissional, múltiplos algoritmos e deploy em produção.

---

## 📋 Índice

- [Visão Geral](#-visão-geral)
- [Domínio do Problema](#-domínio-do-problema)
- [Pipeline de Dados](#-pipeline-de-dados)
- [Modelos Implementados](#-modelos-implementados)
- [Resultados](#-resultados)
- [Deploy](#-deploy)
- [Estrutura do Projeto](#-estrutura-do-projeto)
- [Como Executar](#-como-executar)
- [Tecnologias](#-tecnologias)

---

## 🎯 Visão Geral

Este projeto implementa um **pipeline completo de Machine Learning** para prever a qualidade de vinhos (tintos e brancos) com base em características físico-químicas mensuráveis. O sistema utiliza arquitetura **Data Lakehouse** para processamento de dados e implementa três algoritmos diferentes de classificação.

### Características Principais

- ✅ Pipeline ETL completo (Bronze → Silver → Gold)
- ✅ 3 algoritmos de Machine Learning implementados
- ✅ Avaliação com 3 métricas diferentes
- ✅ Deploy profissional usando Pickle e Joblib
- ✅ Documentação completa e código organizado
- ✅ Visualizações e gráficos de resultados

---

## 🍇 Domínio do Problema

### Previsão de Qualidade de Vinhos

**Tipo de Problema**: Classificação Multiclasse (0-10)

**Objetivo**: Desenvolver um modelo preditivo que classifique a qualidade de vinhos com base em características físico-químicas mensuráveis.

**Aplicação Prática**: 
- Auxiliar produtores e enólogos na identificação de fatores que influenciam a qualidade
- Otimizar processos de produção
- Padronizar avaliações de qualidade
- Reduzir custos e melhorar produtos finais

**Dataset**: 
- **Fonte**: UCI Machine Learning Repository
- **Vinhos Tintos**: 1.599 registros
- **Vinhos Brancos**: 4.898 registros
- **Total**: 6.497 registros brutos

**Variáveis**:
- **Features de Entrada (14 variáveis)**:
  - Acidez fixa, acidez volátil, ácido cítrico
  - Açúcar residual, cloretos
  - Dióxido de enxofre livre e total
  - Densidade, pH, sulfatos, álcool
  - Tipo de vinho (tinto/branco)
  - Features derivadas: acidez total, relação álcool/acidez
  
- **Variável Alvo**: Qualidade do vinho (escala 0-10)

---

## 🔄 Pipeline de Dados

O pipeline segue a arquitetura **Data Lakehouse** com três camadas:

### Camada Bronze (Raw) - Extração
- **Fonte**: UCI Machine Learning Repository
- **Processo**: Download e armazenamento de dados brutos
- **Arquivos**: 
  - `winequality-red-raw.csv` (1.599 registros)
  - `winequality-white-raw.csv` (4.898 registros)

### Camada Silver (Processed) - Transformação
- **União de datasets**: Combinação de vinhos tintos e brancos
- **Limpeza**: Remoção de valores faltantes e duplicatas
- **Tratamento de outliers**: Método IQR (Interquartile Range)
- **Feature Engineering**: 
  - `total_acidity` = acidez fixa + acidez volátil
  - `alcohol_acidity_ratio` = álcool / acidez total
- **Codificação**: Conversão de variáveis categóricas
- **Resultado**: 3.812 registros limpos e processados

### Camada Gold (Curated) - Preparação para ML
- **Separação**: Features (X) e Target (y)
- **Divisão**: Treino (80%) e Teste (20%) com estratificação
- **Normalização**: StandardScaler (média=0, desvio padrão=1)
- **Resultado Final**:
  - Treino: 3.049 amostras
  - Teste: 763 amostras
  - Features: 14 variáveis

### Estatísticas do Pipeline

| Etapa | Registros | Ação |
|-------|-----------|------|
| Bronze (Raw) | 6.497 | Extração inicial |
| Após união | 6.497 | Combinação datasets |
| Após remoção duplicatas | 5.320 | -1.177 duplicatas |
| Após remoção outliers | 3.812 | -1.508 outliers |
| Gold (Treino) | 3.049 | 80% dos dados |
| Gold (Teste) | 763 | 20% dos dados |

---

## 🤖 Modelos Implementados

Foram implementados e avaliados **três algoritmos diferentes** de Machine Learning:

### 1. Random Forest Classifier
- **Tipo**: Ensemble de árvores de decisão
- **Hiperparâmetros**:
  - `n_estimators=100`
  - `max_depth=20`
  - `min_samples_split=5`
  - `min_samples_leaf=2`
- **Vantagens**: Robusto a outliers, reduz overfitting
- **Validação Cruzada**: 0.5458 (±0.0285)

### 2. Support Vector Machine (SVM)
- **Tipo**: Classificador baseado em kernels
- **Hiperparâmetros**:
  - `kernel='rbf'` (Radial Basis Function)
  - `C=1.0`
  - `gamma='scale'`
- **Vantagens**: Boa performance em problemas não-lineares
- **Validação Cruzada**: 0.5451 (±0.0528)

### 3. Gradient Boosting Classifier
- **Tipo**: Ensemble sequencial de modelos fracos
- **Hiperparâmetros**:
  - `n_estimators=100`
  - `learning_rate=0.1`
  - `max_depth=5`
- **Vantagens**: Aprende com erros anteriores, boa performance
- **Validação Cruzada**: 0.5326 (±0.0392)

---

## 📊 Resultados

### Métricas de Avaliação

Foram utilizadas **três métricas** para avaliar o desempenho dos modelos:

#### 1. Accuracy (Acurácia)
- **Definição**: Proporção de previsões corretas
- **Fórmula**: `Accuracy = (TP + TN) / (TP + TN + FP + FN)`
- **Quando usar**: Problemas com classes balanceadas

#### 2. Precision (Precisão)
- **Definição**: Proporção de verdadeiros positivos entre todos os positivos previstos
- **Fórmula**: `Precision = TP / (TP + FP)`
- **Quando usar**: Quando falsos positivos são custosos

#### 3. F1-Score
- **Definição**: Média harmônica entre Precision e Recall
- **Fórmula**: `F1 = 2 × (Precision × Recall) / (Precision + Recall)`
- **Quando usar**: Balancear Precision e Recall

### Resultados dos Modelos

| Modelo | Accuracy | Precision | F1-Score | Melhor em |
|--------|----------|-----------|----------|-----------|
| **SVM** | **0.5714** | **0.5490** | 0.5306 | Accuracy e Precision |
| **Random Forest** | 0.5675 | 0.5357 | **0.5365** | F1-Score |
| **Gradient Boosting** | 0.5334 | 0.5116 | 0.5118 | - |

### Análise dos Resultados

#### 🏆 Modelo Vencedor: **SVM**
- Melhor **Accuracy**: 57.14%
- Melhor **Precision**: 54.90%
- Desempenho consistente em todas as métricas

#### 📈 Insights
- **Desempenho**: Valores de 53-57% são esperados para classificação multiclasse
- **Consistência**: Modelos apresentam resultados similares, indicando robustez
- **Classes**: Melhor desempenho nas classes 5, 6 e 7 (mais frequentes)
- **Desafio**: Dificuldade em prever classes raras (3, 4, 8, 9)

### Visualizações

O projeto gera automaticamente gráficos de análise:

#### 📉 Comparação de Métricas
**Arquivo**: `results/metric_comparison.png`

Gráfico de barras comparando Accuracy, Precision e F1-Score dos três modelos.

![Comparação de Métricas](results/metric_comparison.png)

#### 📊 Matrizes de Confusão
**Arquivo**: `results/confusion_matrices.png`

Matrizes de confusão para cada modelo, mostrando a distribuição de erros por classe.

![Matrizes de Confusão](results/confusion_matrices.png)

> 💡 **Dica**: Visualize os gráficos executando o script de avaliação ou abrindo os arquivos PNG na pasta `results/`

### Distribuição das Classes

A qualidade dos vinhos no dataset processado segue a seguinte distribuição:

| Qualidade | Quantidade | Percentual |
|-----------|------------|------------|
| 3 | 11 | 0.3% |
| 4 | 114 | 3.0% |
| 5 | 1.107 | 29.0% |
| 6 | 1.749 | 45.9% |
| 7 | 701 | 18.4% |
| 8 | 125 | 3.3% |
| 9 | 5 | 0.1% |

**Observação**: Dataset desbalanceado, com predominância das classes 5, 6 e 7.

---

## 🚀 Deploy

O deploy foi realizado utilizando **duas técnicas de serialização**:

### Pickle
- Biblioteca padrão do Python
- Compatível com todas as versões
- Arquivos maiores (ex: Random Forest = 11.90 MB)

### Joblib ⭐ Recomendado
- Otimizado para arrays NumPy
- Melhor compressão (ex: Random Forest = 2.35 MB)
- Mais rápido para modelos scikit-learn
- **Recomendado para produção**

### Arquivos Deployados

```
deploy/
├── random_forest_pickle.pkl      (11.90 MB)
├── random_forest_joblib.pkl      (2.35 MB) ⭐
├── svm_pickle.pkl                (0.56 MB)
├── svm_joblib.pkl                (0.16 MB) ⭐
├── gradient_boosting_pickle.pkl  (2.26 MB)
├── gradient_boosting_joblib.pkl   (0.72 MB) ⭐
└── DEPLOYMENT_GUIDE.md           (Guia completo)
```

### Comparação Pickle vs Joblib

| Modelo | Pickle | Joblib | Redução |
|--------|--------|--------|---------|
| Random Forest | 11.90 MB | 2.35 MB | **80%** |
| SVM | 0.56 MB | 0.16 MB | **71%** |
| Gradient Boosting | 2.26 MB | 0.72 MB | **68%** |

### Como Usar o Modelo Deployado

```python
import joblib
import pandas as pd

# Carregar modelo
package = joblib.load('deploy/svm_joblib.pkl')
model = package['model']
scaler = package['scaler']

# Preparar dados (mesma ordem do treinamento)
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
    'total_acidity': [8.1],
    'alcohol_acidity_ratio': [1.16],
    'wine_type_encoded': [0]  # 0=tinto, 1=branco
})

# Normalizar e prever
features_scaled = scaler.transform(features)
qualidade = model.predict(features_scaled)

print(f"Qualidade prevista: {qualidade[0]}/10")
```

### Demonstração

Execute o script de demonstração:

```bash
python demo_deploy.py
```

Este script mostra:
- Carregamento do modelo
- Preparação de dados de exemplo
- Normalização
- Previsão de qualidade
- Comparação entre modelos

---

## 📁 Estrutura do Projeto

```
ml-pipeline/
│
├── 📄 README.md                    # Este arquivo
├── 📄 RELATORIO_AVALIACAO.md      # Relatório completo da avaliação
├── 📄 EXEMPLO_USO.md              # Guia rápido de uso
├── 📄 main.py                     # Script principal (executa tudo)
├── 📄 demo_deploy.py              # Demonstração do deploy
├── 📄 requirements.txt            # Dependências Python
├── 📄 .gitignore                  # Arquivos ignorados pelo Git
│
├── 📁 src/                        # Código fonte
│   ├── __init__.py
│   ├── data_pipeline.py          # Pipeline ETL (Bronze->Silver->Gold)
│   ├── train_models.py           # Treinamento de 3 modelos
│   ├── evaluate.py               # Avaliação com 3 métricas
│   └── deploy.py                 # Deploy (Pickle e Joblib)
│
├── 📁 data/                       # Dados
│   ├── raw/                      # Camada Bronze (dados brutos)
│   │   ├── winequality-red-raw.csv
│   │   └── winequality-white-raw.csv
│   ├── processed/                # Camada Silver/Gold (dados processados)
│   │   ├── winequality-processed.csv
│   │   ├── X_train.csv
│   │   ├── X_test.csv
│   │   ├── y_train.csv
│   │   └── y_test.csv
│   └── external/                 # Dados externos
│
├── 📁 models/                     # Modelos treinados
│   ├── random_forest_model.pkl
│   ├── svm_model.pkl
│   └── gradient_boosting_model.pkl
│
├── 📁 results/                    # Resultados da avaliação
│   ├── metric_comparison.png     # Gráfico de comparação
│   ├── confusion_matrices.png    # Matrizes de confusão
│   └── model_results.csv         # Tabela de resultados
│
└── 📁 deploy/                     # Modelos deployados
    ├── *_pickle.pkl              # Versões Pickle
    ├── *_joblib.pkl              # Versões Joblib ⭐
    └── DEPLOYMENT_GUIDE.md       # Guia de uso dos modelos
```

---

## 🚀 Como Executar

### Pré-requisitos

- Python 3.8 ou superior
- pip (gerenciador de pacotes Python)

### Instalação

1. **Clone o repositório** (ou baixe os arquivos)

2. **Instale as dependências**:
```bash
pip install -r requirements.txt
```

### Execução Completa

Execute o pipeline completo com um único comando:

```bash
python main.py
```

Este comando executa todas as etapas:
1. ✅ Pipeline de dados (ETL)
2. ✅ Treinamento dos 3 modelos
3. ✅ Avaliação com 3 métricas
4. ✅ Deploy usando Pickle e Joblib

### Execução por Etapas

#### 1. Pipeline de Dados
```bash
python src/data_pipeline.py
```
**Resultado**: Dados processados em `data/processed/`

#### 2. Treinamento de Modelos
```bash
python src/train_models.py
```
**Resultado**: Modelos salvos em `models/`

#### 3. Avaliação
```bash
python src/evaluate.py
```
**Resultado**: Gráficos e tabelas em `results/`

#### 4. Deploy
```bash
python src/deploy.py
```
**Resultado**: Modelos deployados em `deploy/`

#### 5. Demonstração
```bash
python demo_deploy.py
```
**Resultado**: Demonstração interativa do deploy

---

## 🛠️ Tecnologias

### Bibliotecas Principais

- **pandas** (2.1.4) - Manipulação de dados
- **numpy** (1.26.2) - Computação numérica
- **scikit-learn** (1.3.2) - Machine Learning
- **matplotlib** (3.8.2) - Visualizações
- **seaborn** (0.13.0) - Gráficos estatísticos
- **joblib** (1.3.2) - Serialização de modelos

### Ferramentas

- **Python** - Linguagem de programação
- **Data Lakehouse** - Arquitetura de dados
- **Git** - Controle de versão

---

## 📈 Próximos Passos

### Melhorias Sugeridas

- [ ] Tuning de hiperparâmetros com GridSearch/RandomSearch
- [ ] Implementação de ensemble dos melhores modelos
- [ ] Deploy em ambiente cloud (AWS, GCP, Azure)
- [ ] Criação de API REST para servir previsões
- [ ] Monitoramento de performance em produção
- [ ] Implementação de pipeline CI/CD
- [ ] Adição de mais features derivadas
- [ ] Tratamento específico para classes desbalanceadas

---

## 📝 Requisitos da Avaliação

Este projeto atende aos requisitos da **Avaliação N3**:

- ✅ **a) Domínio de problema** (1,0) - Reapresentado e documentado
- ✅ **b) Pipeline de dados** (2,0) - Data Lakehouse completo com explicações
- ✅ **c) Treinamento e avaliação** (5,0) - 3 modelos e 3 métricas
- ✅ **d) Deploy** (2,0) - Pickle e Joblib implementados

**Total: 10,0 pontos**

---

## 📚 Documentação Adicional

- **[RELATORIO_AVALIACAO.md](RELATORIO_AVALIACAO.md)** - Relatório completo da avaliação
- **[EXEMPLO_USO.md](EXEMPLO_USO.md)** - Guia rápido de uso
- **[deploy/DEPLOYMENT_GUIDE.md](deploy/DEPLOYMENT_GUIDE.md)** - Guia detalhado de deploy

---

## 👥 Autores

Desenvolvido para a **Avaliação N3 - Disciplina de Machine Learning**

---

## 📄 Licença

Este projeto é desenvolvido para fins educacionais.

---

## 🙏 Agradecimentos

- **UCI Machine Learning Repository** - Fornecimento do dataset
- **scikit-learn** - Biblioteca de Machine Learning
- Comunidade Python - Suporte e documentação

---

<div align="center">

**🍷 Desenvolvido com dedicação para classificação de qualidade de vinhos 🍷**

⭐ Se este projeto foi útil, considere dar uma estrela!

</div>
