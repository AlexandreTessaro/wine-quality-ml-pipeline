"""
Script Principal - Executa o pipeline completo
Executa todas as etapas: ETL -> Treinamento -> Avaliação -> Deploy
"""

import sys
from pathlib import Path

# Adicionar src ao path
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR / 'src'))

from data_pipeline import main as run_pipeline
from train_models import main as train_models
from evaluate import main as evaluate_models
from deploy import main as deploy_models


def main():
    """
    Executa o pipeline completo de Machine Learning
    """
    print("\n" + "=" * 70)
    print(" " * 15 + "PIPELINE COMPLETO DE MACHINE LEARNING")
    print(" " * 10 + "Previsão de Qualidade de Vinhos")
    print("=" * 70)
    
    try:
        # 1. Pipeline de Dados
        print("\n" + "=" * 70)
        print("ETAPA 1: PIPELINE DE DADOS (ETL)")
        print("=" * 70)
        X_train, X_test, y_train, y_test, scaler, feature_cols = run_pipeline()
        
        # 2. Treinamento de Modelos
        print("\n" + "=" * 70)
        print("ETAPA 2: TREINAMENTO DE MODELOS")
        print("=" * 70)
        models = train_models()
        
        # 3. Avaliação de Modelos
        print("\n" + "=" * 70)
        print("ETAPA 3: AVALIAÇÃO DE MODELOS")
        print("=" * 70)
        results = evaluate_models()
        
        # 4. Deploy de Modelos
        print("\n" + "=" * 70)
        print("ETAPA 4: DEPLOY DE MODELOS")
        print("=" * 70)
        deploy_models()
        
        print("\n" + "=" * 70)
        print(" " * 20 + "PIPELINE CONCLUÍDO COM SUCESSO!")
        print("=" * 70)
        print("\nResumo:")
        print("  ✓ Pipeline de dados executado")
        print("  ✓ 3 modelos treinados")
        print("  ✓ Modelos avaliados com 3 métricas")
        print("  ✓ Modelos deployados (Pickle e Joblib)")
        print("\nArquivos gerados:")
        print("  - Dados processados: data/processed/")
        print("  - Modelos treinados: models/")
        print("  - Resultados: results/")
        print("  - Deploy: deploy/")
        
    except Exception as e:
        print(f"\n❌ Erro durante execução: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

