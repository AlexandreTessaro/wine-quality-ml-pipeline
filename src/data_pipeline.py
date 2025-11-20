"""
Pipeline de Dados - Extração, Transformação e Carregamento (ETL)
Data Lakehouse: Bronze -> Silver -> Gold
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Configuração de diretórios
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
RAW_DIR = DATA_DIR / 'raw'
PROCESSED_DIR = DATA_DIR / 'processed'
EXTERNAL_DIR = DATA_DIR / 'external'

# Criar diretórios se não existirem
for directory in [RAW_DIR, PROCESSED_DIR, EXTERNAL_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


def download_wine_data():
    """
    Camada Bronze (Raw): Extração de dados brutos
    Baixa os dados de qualidade de vinhos do repositório UCI
    """
    print("=" * 60)
    print("CAMADA BRONZE (RAW) - Extração de Dados")
    print("=" * 60)
    
    # URLs dos datasets
    wine_red_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    wine_white_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
    
    try:
        # Carregar dados brutos
        print("\n1. Carregando dados de vinhos tintos...")
        df_red = pd.read_csv(wine_red_url, sep=';')
        df_red['wine_type'] = 'red'
        
        print("2. Carregando dados de vinhos brancos...")
        df_white = pd.read_csv(wine_white_url, sep=';')
        df_white['wine_type'] = 'white'
        
        # Salvar dados brutos (Bronze Layer)
        df_red.to_csv(RAW_DIR / 'winequality-red-raw.csv', index=False)
        df_white.to_csv(RAW_DIR / 'winequality-white-raw.csv', index=False)
        
        print(f"\n✓ Dados brutos salvos em: {RAW_DIR}")
        print(f"  - Vinhos tintos: {len(df_red)} registros")
        print(f"  - Vinhos brancos: {len(df_white)} registros")
        
        return df_red, df_white
    
    except Exception as e:
        print(f"Erro ao baixar dados: {e}")
        print("Usando dados locais se disponíveis...")
        # Tentar carregar dados locais
        if (RAW_DIR / 'winequality-red-raw.csv').exists():
            df_red = pd.read_csv(RAW_DIR / 'winequality-red-raw.csv')
            df_white = pd.read_csv(RAW_DIR / 'winequality-white-raw.csv')
            return df_red, df_white
        else:
            raise


def transform_data(df_red, df_white):
    """
    Camada Silver (Processed): Transformação e limpeza dos dados
    """
    print("\n" + "=" * 60)
    print("CAMADA SILVER (PROCESSED) - Transformação de Dados")
    print("=" * 60)
    
    # 1. Unir datasets
    print("\n1. Unindo datasets de vinhos tintos e brancos...")
    df_combined = pd.concat([df_red, df_white], ignore_index=True)
    print(f"   Total de registros: {len(df_combined)}")
    
    # 2. Verificar valores faltantes
    print("\n2. Verificando valores faltantes...")
    missing = df_combined.isnull().sum()
    if missing.sum() > 0:
        print("   Valores faltantes encontrados:")
        print(missing[missing > 0])
        df_combined = df_combined.dropna()
    else:
        print("   ✓ Nenhum valor faltante encontrado")
    
    # 3. Verificar duplicatas
    print("\n3. Verificando duplicatas...")
    duplicates = df_combined.duplicated().sum()
    if duplicates > 0:
        print(f"   Removendo {duplicates} registros duplicados...")
        df_combined = df_combined.drop_duplicates()
    else:
        print("   ✓ Nenhuma duplicata encontrada")
    
    # 4. Verificar outliers usando IQR
    print("\n4. Tratando outliers...")
    numeric_cols = df_combined.select_dtypes(include=[np.number]).columns
    numeric_cols = numeric_cols.drop('quality')  # Não tratar outliers na variável alvo
    
    outliers_removed = 0
    for col in numeric_cols:
        Q1 = df_combined[col].quantile(0.25)
        Q3 = df_combined[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        before = len(df_combined)
        df_combined = df_combined[(df_combined[col] >= lower_bound) & 
                                  (df_combined[col] <= upper_bound)]
        outliers_removed += (before - len(df_combined))
    
    print(f"   ✓ {outliers_removed} outliers removidos")
    
    # 5. Feature Engineering
    print("\n5. Criando features derivadas...")
    # Criar feature de acidez total
    df_combined['total_acidity'] = df_combined['fixed acidity'] + df_combined['volatile acidity']
    # Criar feature de relação álcool/acidez
    df_combined['alcohol_acidity_ratio'] = df_combined['alcohol'] / (df_combined['total_acidity'] + 1e-6)
    
    # 6. Codificar variável categórica (wine_type)
    print("\n6. Codificando variáveis categóricas...")
    df_combined['wine_type_encoded'] = df_combined['wine_type'].map({'red': 0, 'white': 1})
    
    # Salvar dados processados (Silver Layer)
    df_combined.to_csv(PROCESSED_DIR / 'winequality-processed.csv', index=False)
    print(f"\n✓ Dados processados salvos em: {PROCESSED_DIR}")
    print(f"  Registros finais: {len(df_combined)}")
    
    return df_combined


def prepare_ml_data(df_processed):
    """
    Camada Gold (Curated): Preparação final para Machine Learning
    """
    print("\n" + "=" * 60)
    print("CAMADA GOLD (CURATED) - Preparação para ML")
    print("=" * 60)
    
    # Separar features e target
    print("\n1. Separando features e variável alvo...")
    
    # Features: todas as colunas numéricas exceto quality
    feature_cols = [col for col in df_processed.columns 
                   if col not in ['quality', 'wine_type']]
    
    X = df_processed[feature_cols]
    y = df_processed['quality']
    
    print(f"   Features: {len(feature_cols)} variáveis")
    print(f"   Target: qualidade do vinho (0-10)")
    print(f"   Distribuição da qualidade:")
    print(y.value_counts().sort_index())
    
    # Dividir em treino e teste
    print("\n2. Dividindo dados em treino (80%) e teste (20%)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"   Treino: {len(X_train)} amostras")
    print(f"   Teste: {len(X_test)} amostras")
    
    # Normalizar features
    print("\n3. Normalizando features (StandardScaler)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Converter de volta para DataFrame
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    
    # Salvar dados prontos para ML (Gold Layer)
    X_train_scaled.to_csv(PROCESSED_DIR / 'X_train.csv', index=False)
    X_test_scaled.to_csv(PROCESSED_DIR / 'X_test.csv', index=False)
    y_train.to_csv(PROCESSED_DIR / 'y_train.csv', index=False)
    y_test.to_csv(PROCESSED_DIR / 'y_test.csv', index=False)
    
    print(f"\n✓ Dados prontos para ML salvos em: {PROCESSED_DIR}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_cols


def main():
    """
    Executa o pipeline completo de dados
    """
    print("\n" + "=" * 60)
    print("PIPELINE DE DADOS - DATA LAKEHOUSE")
    print("Bronze -> Silver -> Gold")
    print("=" * 60)
    
    # Bronze: Extração
    df_red, df_white = download_wine_data()
    
    # Silver: Transformação
    df_processed = transform_data(df_red, df_white)
    
    # Gold: Preparação para ML
    X_train, X_test, y_train, y_test, scaler, feature_cols = prepare_ml_data(df_processed)
    
    print("\n" + "=" * 60)
    print("PIPELINE CONCLUÍDO COM SUCESSO!")
    print("=" * 60)
    print(f"\nResumo:")
    print(f"  - Dados brutos: {RAW_DIR}")
    print(f"  - Dados processados: {PROCESSED_DIR}")
    print(f"  - Features: {len(feature_cols)}")
    print(f"  - Amostras de treino: {len(X_train)}")
    print(f"  - Amostras de teste: {len(X_test)}")
    
    return X_train, X_test, y_train, y_test, scaler, feature_cols


if __name__ == "__main__":
    main()

