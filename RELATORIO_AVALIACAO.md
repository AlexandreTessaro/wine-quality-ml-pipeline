# Relatório de Avaliação N3 - Pipeline de Machine Learning

## Domínio de Problema

### a) Reapresentação do Domínio de Problema (1,0)

**Problema**: Previsão de Qualidade de Vinhos

Este projeto desenvolve um modelo preditivo para classificar a qualidade de vinhos (tinto ou branco) com base em características físico-químicas. O problema é de **classificação multiclasse**, onde buscamos prever a qualidade do vinho em uma escala de 0 a 10.

**Contexto e Objetivos**:
- **Objetivo Principal**: Prever a qualidade de vinhos utilizando atributos físico-químicos mensuráveis
- **Tipo de Problema**: Classificação multiclasse (qualidade de 0 a 10)
- **Aplicação Prática**: Auxiliar produtores e enólogos a identificar fatores que influenciam a qualidade do vinho, permitindo otimização do processo de produção
- **Impacto**: Reduzir custos de produção, melhorar a qualidade dos produtos finais e padronizar processos de avaliação

**Variáveis do Dataset**:
- **Features de Entrada**: 
  - Acidez fixa, acidez volátil, ácido cítrico
  - Açúcar residual, cloretos
  - Dióxido de enxofre livre e total
  - Densidade, pH, sulfatos, álcool
  - Tipo de vinho (tinto/branco)
  - Features derivadas (acidez total, relação álcool/acidez)
- **Variável Alvo**: Qualidade do vinho (0-10)

**Repositório de Dados**: Data Lakehouse (UCI Machine Learning Repository)

---

## Pipeline de Dados

### b) Apresentação do Pipeline de Dados (2,0)

O pipeline de dados segue a arquitetura **Data Lakehouse**, organizada em três camadas:

#### **Camada Bronze (Raw) - Extração**
**Elemento**: Extração de dados brutos do repositório

**Descrição**: 
- Carregamento dos datasets de vinhos tintos e brancos do repositório UCI Machine Learning Repository
- Dados são salvos em formato CSV sem nenhum processamento
- Preserva a integridade dos dados originais para rastreabilidade

**Arquivos gerados**:
- `winequality-red-raw.csv` (1.599 registros)
- `winequality-white-raw.csv` (4.898 registros)

#### **Camada Silver (Processed) - Transformação**
**Elemento**: Limpeza, validação e transformação dos dados

**Descrição**:
- **União de datasets**: Combinação dos dados de vinhos tintos e brancos em um único dataset
- **Tratamento de valores faltantes**: Verificação e remoção de registros com valores nulos
- **Remoção de duplicatas**: Identificação e remoção de registros duplicados
- **Tratamento de outliers**: Uso do método IQR (Interquartile Range) para identificar e remover outliers extremos
- **Feature Engineering**: Criação de features derivadas:
  - `total_acidity`: Soma da acidez fixa e volátil
  - `alcohol_acidity_ratio`: Relação entre álcool e acidez total
- **Codificação de variáveis categóricas**: Conversão do tipo de vinho (red/white) para valores numéricos (0/1)

**Arquivos gerados**:
- `winequality-processed.csv` (dataset limpo e transformado)

#### **Camada Gold (Curated) - Preparação para ML**
**Elemento**: Preparação final dos dados para algoritmos de Machine Learning

**Descrição**:
- **Separação de features e target**: Isolamento da variável alvo (quality) das features preditoras
- **Divisão treino/teste**: Separação dos dados em conjunto de treino (80%) e teste (20%) com estratificação para manter a distribuição das classes
- **Normalização**: Aplicação do StandardScaler para normalizar todas as features (média=0, desvio padrão=1), essencial para algoritmos sensíveis à escala como SVM
- **Persistência**: Salvamento dos dados processados para uso nos modelos

**Arquivos gerados**:
- `X_train.csv`, `X_test.csv` (features normalizadas)
- `y_train.csv`, `y_test.csv` (variáveis alvo)

**Fluxo do Pipeline**:
```
Repositório UCI → Bronze (Raw) → Silver (Processed) → Gold (Curated) → Modelos ML
```

---

## Treinamento e Avaliação de Modelos

### c) Resultados do Treinamento e Teste (5,0)

Foram implementados e avaliados **três algoritmos diferentes** de Machine Learning:

#### **1. Random Forest Classifier**

**Características do Algoritmo**:
- Ensemble de múltiplas árvores de decisão
- Reduz overfitting através de bagging e feature randomness
- Robusto a outliers e não requer normalização (mas beneficia dela)

**Hiperparâmetros utilizados**:
- `n_estimators=100`: Número de árvores na floresta
- `max_depth=20`: Profundidade máxima das árvores
- `min_samples_split=5`: Mínimo de amostras para dividir um nó
- `min_samples_leaf=2`: Mínimo de amostras em uma folha

#### **2. Support Vector Machine (SVM)**

**Características do Algoritmo**:
- Encontra o hiperplano ótimo para separar classes
- Kernel RBF (Radial Basis Function) para problemas não-lineares
- Sensível à escala dos dados (requer normalização)

**Hiperparâmetros utilizados**:
- `kernel='rbf'`: Função de kernel radial
- `C=1.0`: Parâmetro de regularização
- `gamma='scale'`: Coeficiente do kernel RBF

#### **3. Gradient Boosting Classifier**

**Características do Algoritmo**:
- Ensemble sequencial que combina modelos fracos
- Aprende com erros dos modelos anteriores
- Boa performance em problemas de classificação

**Hiperparâmetros utilizados**:
- `n_estimators=100`: Número de modelos base
- `learning_rate=0.1`: Taxa de aprendizado
- `max_depth=5`: Profundidade máxima das árvores base

### **Métricas de Desempenho**

Foram utilizadas **três métricas** para avaliar o desempenho dos modelos:

#### **1. Accuracy (Acurácia)**

**Definição**: Proporção de previsões corretas em relação ao total de previsões realizadas.

**Fórmula**: 
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

**Características**:
- **Vantagens**: 
  - Fácil de interpretar e comunicar
  - Boa métrica geral quando as classes estão balanceadas
  - Não requer ajuste de threshold
- **Limitações**: 
  - Pode ser enganosa em datasets desbalanceados
  - Não considera a distribuição de erros entre classes
- **Quando usar**: Melhor para problemas com classes balanceadas

**Interpretação**: Valores próximos a 1.0 indicam alta taxa de acerto geral do modelo.

#### **2. Precision (Precisão)**

**Definição**: Proporção de verdadeiros positivos entre todos os casos classificados como positivos.

**Fórmula**: 
```
Precision = TP / (TP + FP)
```

**Características**:
- **Vantagens**: 
  - Importante quando falsos positivos são custosos
  - Mostra confiabilidade das previsões positivas
  - Útil para problemas onde queremos minimizar falsos alarmes
- **Limitações**: 
  - Não considera falsos negativos
  - Pode ser alta mesmo com muitos falsos negativos
- **Quando usar**: Ideal quando o custo de falsos positivos é alto (ex: diagnóstico médico, detecção de fraudes)

**Interpretação**: Valores altos indicam que quando o modelo prevê uma classe, ele está frequentemente correto.

#### **3. F1-Score**

**Definição**: Média harmônica entre Precision e Recall, balanceando ambas as métricas.

**Fórmula**: 
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**Características**:
- **Vantagens**: 
  - Balanceia Precision e Recall em uma única métrica
  - Boa métrica única quando precisamos considerar ambos os aspectos
  - Útil para datasets desbalanceados
  - Não é afetada por classes majoritárias
- **Limitações**: 
  - Pode não ser ideal se Precision ou Recall forem mais importantes
  - Média harmônica penaliza mais valores extremos
- **Quando usar**: Melhor quando precisamos balancear Precision e Recall

**Interpretação**: Valores altos indicam bom equilíbrio entre precisão e capacidade de encontrar casos positivos.

**Nota**: Para problemas multiclasse, utilizamos a média ponderada (`weighted`) das métricas, que considera o suporte (número de amostras) de cada classe.

### **Resultados Obtidos**

Os resultados detalhados são salvos em `results/model_results.csv` e visualizados em `results/metric_comparison.png`.

**Resumo dos Resultados** (valores aproximados após execução):

| Modelo | Accuracy | Precision | F1-Score |
|--------|----------|-----------|----------|
| Random Forest | ~0.65 | ~0.64 | ~0.64 |
| SVM | ~0.58 | ~0.57 | ~0.57 |
| Gradient Boosting | ~0.63 | ~0.62 | ~0.62 |

**Análise dos Resultados**:
- **Random Forest** apresentou o melhor desempenho geral nas três métricas
- **SVM** teve desempenho inferior, possivelmente devido à complexidade do problema multiclasse
- **Gradient Boosting** apresentou resultados intermediários, próximo ao Random Forest

**Observações**:
- A qualidade dos vinhos é um problema desafiador devido à subjetividade da avaliação humana
- Os modelos capturam padrões nas características físico-químicas, mas há limitações inerentes ao problema
- A distribuição das classes de qualidade pode afetar o desempenho dos modelos

---

## Deploy do Modelo

### d) Apresentação do Deploy (2,0)

O deploy dos modelos foi realizado utilizando **duas técnicas de serialização**:

#### **1. Pickle**

**Características**:
- Biblioteca padrão do Python (não requer instalação adicional)
- Serialização de objetos Python de forma nativa
- Compatível com todas as versões do Python
- Boa para objetos Python simples

**Implementação**:
```python
import pickle

# Salvar
with open('model_pickle.pkl', 'wb') as f:
    pickle.dump(deploy_package, f)

# Carregar
with open('model_pickle.pkl', 'rb') as f:
    package = pickle.load(f)
```

**Arquivos gerados**:
- `random_forest_pickle.pkl`
- `svm_pickle.pkl`
- `gradient_boosting_pickle.pkl`

#### **2. Joblib**

**Características**:
- Otimizado para arrays NumPy grandes (comum em modelos scikit-learn)
- Melhor compressão de arquivos
- Mais rápido para carregar/salvar modelos ML
- Suporta compressão configurável

**Implementação**:
```python
import joblib

# Salvar com compressão
joblib.dump(deploy_package, 'model_joblib.pkl', compress=3)

# Carregar
package = joblib.load('model_joblib.pkl')
```

**Arquivos gerados**:
- `random_forest_joblib.pkl`
- `svm_joblib.pkl`
- `gradient_boosting_joblib.pkl`

**Pacote de Deploy**:
Cada arquivo deployado contém:
- **Modelo treinado**: O modelo completo pronto para uso
- **Scaler**: Objeto StandardScaler para normalização de novas amostras
- **Metadados**: Nome do modelo, versão, informações adicionais

**Comparação Pickle vs Joblib**:

| Característica | Pickle | Joblib |
|---------------|--------|--------|
| Velocidade | Média | Rápida |
| Compressão | Básica | Avançada (configurável) |
| Tamanho arquivo | Maior | Menor (com compressão) |
| Compatibilidade | Universal Python | Otimizado para scikit-learn |
| Uso recomendado | Objetos Python gerais | Modelos ML com arrays NumPy |

**Uso em Produção**:
```python
# Carregar modelo deployado
import joblib
package = joblib.load('deploy/random_forest_joblib.pkl')

model = package['model']
scaler = package['scaler']

# Preparar novos dados
new_features = prepare_features(...)
features_scaled = scaler.transform(new_features)

# Fazer previsão
prediction = model.predict(features_scaled)
```

**Localização dos Arquivos**:
- Todos os modelos deployados estão em: `deploy/`
- Guia de uso completo: `deploy/DEPLOYMENT_GUIDE.md`

---

## Conclusão

Este projeto demonstrou a implementação completa de um pipeline de Machine Learning, desde a extração de dados até o deploy de modelos em produção. O pipeline segue as melhores práticas de engenharia de dados (Data Lakehouse) e ciência de dados, utilizando múltiplos algoritmos e métricas para garantir robustez e confiabilidade.

**Principais Conquistas**:
- ✅ Pipeline de dados completo e reprodutível
- ✅ Três modelos diferentes implementados e avaliados
- ✅ Avaliação robusta com múltiplas métricas
- ✅ Deploy profissional usando Pickle e Joblib
- ✅ Documentação completa e código organizado

**Próximos Passos Sugeridos**:
- Tuning de hiperparâmetros com GridSearch ou RandomSearch
- Implementação de ensemble dos melhores modelos
- Deploy em ambiente cloud (AWS, GCP, Azure)
- Criação de API REST para servir previsões
- Monitoramento de performance em produção

---

**Desenvolvido para**: Avaliação N3 - Disciplina de Machine Learning  
**Data**: 2024  
**Tecnologias**: Python, scikit-learn, pandas, numpy, matplotlib, seaborn

