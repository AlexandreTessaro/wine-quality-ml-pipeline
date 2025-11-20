"""
Avaliação de Modelos de Machine Learning
Avalia 3 modelos usando 3 métricas: Accuracy, Precision e F1-Score
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Métricas
from sklearn.metrics import accuracy_score, precision_score, f1_score, classification_report, confusion_matrix

# Visualização
import matplotlib.pyplot as plt
import seaborn as sns

# Modelos
import joblib

# Configuração de diretórios
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data' / 'processed'
MODELS_DIR = BASE_DIR / 'models'
RESULTS_DIR = BASE_DIR / 'results'

# Criar diretório de resultados
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Configurar estilo dos gráficos
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def load_data_and_models():
    """Carrega dados de teste e modelos treinados"""
    print("Carregando dados e modelos...")
    
    X_test = pd.read_csv(DATA_DIR / 'X_test.csv')
    y_test = pd.read_csv(DATA_DIR / 'y_test.csv').squeeze()
    
    models = {}
    model_names = ['random_forest_model.pkl', 'svm_model.pkl', 'gradient_boosting_model.pkl']
    display_names = ['Random Forest', 'SVM', 'Gradient Boosting']
    
    for model_file, display_name in zip(model_names, display_names):
        model_path = MODELS_DIR / model_file
        if model_path.exists():
            models[display_name] = joblib.load(model_path)
            print(f"✓ {display_name} carregado")
        else:
            print(f"⚠ {display_name} não encontrado em {model_path}")
    
    print(f"\n✓ Dados de teste: {len(X_test)} amostras")
    
    return X_test, y_test, models


def explain_metrics():
    """
    Explica as características das métricas escolhidas
    """
    print("\n" + "=" * 60)
    print("CARACTERÍSTICAS DAS MÉTRICAS DE DESEMPENHO")
    print("=" * 60)
    
    metrics_info = {
        "Accuracy (Acurácia)": {
            "definicao": "Proporção de previsões corretas em relação ao total de previsões",
            "formula": "Accuracy = (TP + TN) / (TP + TN + FP + FN)",
            "vantagens": [
                "Fácil de interpretar",
                "Boa métrica geral quando as classes estão balanceadas",
                "Não requer ajuste de threshold"
            ],
            "limitacoes": [
                "Pode ser enganosa em datasets desbalanceados",
                "Não considera a distribuição de erros entre classes"
            ],
            "uso": "Melhor para problemas com classes balanceadas"
        },
        "Precision (Precisão)": {
            "definicao": "Proporção de verdadeiros positivos entre todos os casos classificados como positivos",
            "formula": "Precision = TP / (TP + FP)",
            "vantagens": [
                "Importante quando falsos positivos são custosos",
                "Mostra confiabilidade das previsões positivas",
                "Útil para problemas onde queremos minimizar falsos alarmes"
            ],
            "limitacoes": [
                "Não considera falsos negativos",
                "Pode ser alta mesmo com muitos falsos negativos"
            ],
            "uso": "Ideal quando o custo de falsos positivos é alto (ex: diagnóstico médico)"
        },
        "F1-Score": {
            "definicao": "Média harmônica entre Precision e Recall, balanceando ambas as métricas",
            "formula": "F1 = 2 * (Precision * Recall) / (Precision + Recall)",
            "vantagens": [
                "Balanceia Precision e Recall",
                "Boa métrica única quando precisamos considerar ambos os aspectos",
                "Útil para datasets desbalanceados",
                "Não é afetada por classes majoritárias"
            ],
            "limitacoes": [
                "Pode não ser ideal se Precision ou Recall forem mais importantes",
                "Média harmônica penaliza mais valores extremos"
            ],
            "uso": "Melhor quando precisamos balancear Precision e Recall"
        }
    }
    
    for metric_name, info in metrics_info.items():
        print(f"\n{metric_name}:")
        print(f"  Definição: {info['definicao']}")
        print(f"  Fórmula: {info['formula']}")
        print(f"  Vantagens:")
        for v in info['vantagens']:
            print(f"    • {v}")
        print(f"  Limitações:")
        for l in info['limitacoes']:
            print(f"    • {l}")
        print(f"  Quando usar: {info['uso']}")


def evaluate_model(model, X_test, y_test, model_name):
    """
    Avalia um modelo usando as 3 métricas escolhidas
    """
    print(f"\n{'=' * 60}")
    print(f"AVALIANDO: {model_name}")
    print(f"{'=' * 60}")
    
    # Fazer previsões
    print("\nFazendo previsões no conjunto de teste...")
    y_pred = model.predict(X_test)
    
    # Calcular métricas
    # Para Precision e F1-Score, usar 'weighted' para média ponderada (multiclasse)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print(f"\n{'Métrica':<20} {'Valor':<15}")
    print(f"{'-' * 35}")
    print(f"{'Accuracy':<20} {accuracy:.4f}")
    print(f"{'Precision (weighted)':<20} {precision:.4f}")
    print(f"{'F1-Score (weighted)':<20} {f1:.4f}")
    
    # Relatório detalhado
    print(f"\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    # Matriz de confusão
    cm = confusion_matrix(y_test, y_pred)
    
    return {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'f1_score': f1,
        'predictions': y_pred,
        'confusion_matrix': cm
    }


def plot_results(results):
    """
    Cria visualizações dos resultados
    """
    print("\nGerando visualizações...")
    
    # Preparar dados para gráficos
    model_names = [r['model_name'] for r in results]
    accuracies = [r['accuracy'] for r in results]
    precisions = [r['precision'] for r in results]
    f1_scores = [r['f1_score'] for r in results]
    
    # Gráfico de comparação de métricas
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(model_names))
    width = 0.25
    
    ax.bar(x - width, accuracies, width, label='Accuracy', alpha=0.8)
    ax.bar(x, precisions, width, label='Precision', alpha=0.8)
    ax.bar(x + width, f1_scores, width, label='F1-Score', alpha=0.8)
    
    ax.set_xlabel('Modelos', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Comparação de Métricas entre Modelos', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'metric_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Gráfico salvo em: {RESULTS_DIR / 'metric_comparison.png'}")
    
    # Matrizes de confusão
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, result in enumerate(results):
        cm = result['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                   cbar_kws={'label': 'Quantidade'})
        axes[idx].set_title(f'Matriz de Confusão - {result["model_name"]}', 
                           fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Previsão', fontsize=10)
        axes[idx].set_ylabel('Real', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
    print(f"✓ Matrizes de confusão salvas em: {RESULTS_DIR / 'confusion_matrices.png'}")
    
    plt.close('all')


def save_results_table(results):
    """
    Salva tabela de resultados em CSV
    """
    results_df = pd.DataFrame([
        {
            'Modelo': r['model_name'],
            'Accuracy': f"{r['accuracy']:.4f}",
            'Precision': f"{r['precision']:.4f}",
            'F1-Score': f"{r['f1_score']:.4f}"
        }
        for r in results
    ])
    
    results_df.to_csv(RESULTS_DIR / 'model_results.csv', index=False)
    print(f"\n✓ Tabela de resultados salva em: {RESULTS_DIR / 'model_results.csv'}")
    
    print("\n" + "=" * 60)
    print("RESUMO DOS RESULTADOS")
    print("=" * 60)
    print(results_df.to_string(index=False))


def main():
    """
    Executa a avaliação completa de todos os modelos
    """
    print("\n" + "=" * 60)
    print("AVALIAÇÃO DE MODELOS DE MACHINE LEARNING")
    print("=" * 60)
    
    # Explicar métricas
    explain_metrics()
    
    # Carregar dados e modelos
    X_test, y_test, models = load_data_and_models()
    
    if not models:
        print("\n⚠ Nenhum modelo encontrado! Execute train_models.py primeiro.")
        return
    
    # Avaliar cada modelo
    results = []
    for model_name, model in models.items():
        result = evaluate_model(model, X_test, y_test, model_name)
        results.append(result)
    
    # Gerar visualizações
    plot_results(results)
    
    # Salvar resultados
    save_results_table(results)
    
    print("\n" + "=" * 60)
    print("AVALIAÇÃO CONCLUÍDA!")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    results = main()

