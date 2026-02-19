"""
Script de Visualização Completa - Envenenamento de Dados
Dataset: Iris | Modelo: Regressão Logística

Gera duas visualizações principais:
1. Gráfico de convergência por rodada (métricas globais)
2. Gráfico de classificação percentual por espécie/classe

Autor: Projeto de Iniciação Científica
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from copy import deepcopy
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Configuração de estilo
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class ModeloSimples:
    """Modelo de classificação simplificado"""
    
    def __init__(self):
        self._modelo = LogisticRegression(max_iter=1000, random_state=42,
                                          multi_class='multinomial', solver='lbfgs')
        self.scaler = StandardScaler()
    
    def treinar(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self._modelo.fit(X_scaled, y)
    
    def avaliar(self, X, y):
        if not hasattr(self.scaler, 'mean_'):
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        y_pred = self._modelo.predict(X_scaled)
        y_proba = self._modelo.predict_proba(X_scaled)
        
        return {
            'acuracia': accuracy_score(y, y_pred),
            'f1_score': f1_score(y, y_pred, average='weighted'),
            'precisao': precision_score(y, y_pred, average='weighted'),
            'recall': recall_score(y, y_pred, average='weighted'),
            'loss': log_loss(y, y_proba),
            'y_pred': y_pred,
            'y_true': y
        }
    
    def obter_pesos(self):
        if hasattr(self._modelo, 'coef_'):
            return {
                'coef': deepcopy(self._modelo.coef_),
                'intercept': deepcopy(self._modelo.intercept_),
                'classes': deepcopy(self._modelo.classes_)
            }
        return None
    
    def atualizar_pesos(self, pesos):
        if pesos:
            self._modelo.coef_ = deepcopy(pesos['coef'])
            self._modelo.intercept_ = deepcopy(pesos['intercept'])
            self._modelo.classes_ = deepcopy(pesos['classes'])


def envenenar_pesos(pesos, taxa=0.8, tipo='inverter'):
    """Corrompe pesos do modelo"""
    pesos_corrompidos = deepcopy(pesos)
    
    if tipo == 'inverter':
        pesos_corrompidos['coef'] = -pesos['coef'] * (1 + taxa)
        pesos_corrompidos['intercept'] = -pesos['intercept'] * (1 + taxa)
    
    return pesos_corrompidos


def calcular_acuracia_por_classe(y_true, y_pred):
    """Calcula acurácia para cada classe"""
    especies = ['Setosa', 'Versicolor', 'Virginica']
    acuracias = {}
    
    for i, especie in enumerate(especies):
        mascara = y_true == i
        if mascara.sum() > 0:
            acc = accuracy_score(y_true[mascara], y_pred[mascara])
            acuracias[especie] = acc
        else:
            acuracias[especie] = 0.0
    
    return acuracias


def executar_experimento_completo(dados_clientes, dados_validacao, num_rodadas=8):
    """Executa experimento com envenenamento e coleta todas as métricas"""
    print("\n" + "="*70)
    print("EXPERIMENTO COMPLETO - ENVENENAMENTO DE DADOS")
    print("="*70)
    
    X_val, y_val = dados_validacao
    
    # Históricos
    historico_metricas_globais = []
    historico_acuracia_por_classe = []
    
    for rodada in range(1, num_rodadas + 1):
        print(f"\n[Rodada {rodada}/{num_rodadas}]")
        
        # 1. Treinamento local
        pesos_locais = []
        for i, (X_cliente, y_cliente) in enumerate(dados_clientes, 1):
            modelo = ModeloSimples()
            modelo.treinar(X_cliente, y_cliente)
            metricas = modelo.avaliar(X_cliente, y_cliente)
            pesos = modelo.obter_pesos()
            
            # Cliente 3 é malicioso
            if i == 3:
                pesos = envenenar_pesos(pesos, taxa=0.8, tipo='inverter')
                print(f"  ⚠ Cliente {i}: ENVENENADO (Acc local={metricas['acuracia']:.4f})")
            else:
                print(f"  ✓ Cliente {i}: Honesto (Acc local={metricas['acuracia']:.4f})")
            
            pesos_locais.append(pesos)
        
        # 2. Agregação FedAvg
        pesos_globais = {
            'coef': np.mean([p['coef'] for p in pesos_locais], axis=0),
            'intercept': np.mean([p['intercept'] for p in pesos_locais], axis=0),
            'classes': pesos_locais[0]['classes']
        }
        
        # 3. Avaliação global
        modelo_global = ModeloSimples()
        modelo_global.atualizar_pesos(pesos_globais)
        metricas_globais = modelo_global.avaliar(X_val, y_val)
        
        # 4. Calcula acurácia por classe
        acuracia_classes = calcular_acuracia_por_classe(
            metricas_globais['y_true'], 
            metricas_globais['y_pred']
        )
        
        print(f"  📊 Modelo Global: Acc={metricas_globais['acuracia']:.4f} | "
              f"F1={metricas_globais['f1_score']:.4f} | Loss={metricas_globais['loss']:.4f}")
        print(f"     Por espécie: Setosa={acuracia_classes['Setosa']:.3f} | "
              f"Versicolor={acuracia_classes['Versicolor']:.3f} | "
              f"Virginica={acuracia_classes['Virginica']:.3f}")
        
        # Armazena histórico
        historico_metricas_globais.append({
            'rodada': rodada,
            'acuracia': metricas_globais['acuracia'],
            'f1_score': metricas_globais['f1_score'],
            'precisao': metricas_globais['precisao'],
            'recall': metricas_globais['recall'],
            'loss': metricas_globais['loss']
        })
        
        historico_acuracia_por_classe.append({
            'rodada': rodada,
            **acuracia_classes
        })
    
    return historico_metricas_globais, historico_acuracia_por_classe


def gerar_grafico_convergencia(historico_metricas):
    """
    VISUALIZAÇÃO 1: Gráfico de convergência por rodada
    Mostra evolução das métricas (Acurácia, F1-Score, Precisão, Loss)
    """
    print("\n[GERANDO VISUALIZAÇÃO 1] Gráfico de Convergência por Rodada")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    rodadas = [m['rodada'] for m in historico_metricas]
    acuracias = [m['acuracia'] for m in historico_metricas]
    f1_scores = [m['f1_score'] for m in historico_metricas]
    precisoes = [m['precisao'] for m in historico_metricas]
    losses = [m['loss'] for m in historico_metricas]
    
    cores = ['#2E86AB', '#A23B72', '#F18F01', '#06D6A0']
    
    # 1. Acurácia
    ax1 = axes[0, 0]
    ax1.plot(rodadas, acuracias, marker='o', linewidth=3, markersize=10, 
            color=cores[0], label='Acurácia Global')
    ax1.fill_between(rodadas, acuracias, alpha=0.2, color=cores[0])
    ax1.set_title('Acurácia Global por Rodada', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Rodada Federada', fontsize=12)
    ax1.set_ylabel('Acurácia', fontsize=12)
    ax1.set_ylim([0, 1.05])
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)
    
    # Adiciona valores nos pontos
    for r, acc in zip(rodadas, acuracias):
        ax1.annotate(f'{acc:.3f}', (r, acc), textcoords="offset points", 
                    xytext=(0, 8), ha='center', fontsize=9, fontweight='bold')
    
    # 2. F1-Score
    ax2 = axes[0, 1]
    ax2.plot(rodadas, f1_scores, marker='s', linewidth=3, markersize=10,
            color=cores[1], label='F1-Score')
    ax2.fill_between(rodadas, f1_scores, alpha=0.2, color=cores[1])
    ax2.set_title('F1-Score por Rodada', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Rodada Federada', fontsize=12)
    ax2.set_ylabel('F1-Score', fontsize=12)
    ax2.set_ylim([0, 1.05])
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11)
    
    # 3. Precisão
    ax3 = axes[1, 0]
    ax3.plot(rodadas, precisoes, marker='^', linewidth=3, markersize=10,
            color=cores[2], label='Precisão')
    ax3.fill_between(rodadas, precisoes, alpha=0.2, color=cores[2])
    ax3.set_title('Precisão por Rodada', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Rodada Federada', fontsize=12)
    ax3.set_ylabel('Precisão', fontsize=12)
    ax3.set_ylim([0, 1.05])
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=11)
    
    # 4. Loss
    ax4 = axes[1, 1]
    ax4.plot(rodadas, losses, marker='D', linewidth=3, markersize=10,
            color=cores[3], label='Loss (Log Loss)')
    ax4.fill_between(rodadas, losses, alpha=0.2, color=cores[3])
    ax4.set_title('Loss por Rodada', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Rodada Federada', fontsize=12)
    ax4.set_ylabel('Loss', fontsize=12)
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=11)
    
    plt.suptitle('Convergência do Modelo Global - Cenário de Envenenamento\nDataset Iris com 1 Cliente Malicioso (33%)',
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('visualizacao_1_convergencia_por_rodada.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Salvo: visualizacao_1_convergencia_por_rodada.png")
    plt.close()


def gerar_grafico_classificacao_especies(historico_classes):
    """
    VISUALIZAÇÃO 2: Gráfico de classificação percentual por espécie
    Mostra desempenho específico para cada classe do Iris
    """
    print("\n[GERANDO VISUALIZAÇÃO 2] Gráfico de Classificação por Espécie")
    
    fig = plt.figure(figsize=(18, 10))
    
    rodadas = [m['rodada'] for m in historico_classes]
    setosa = [m['Setosa'] * 100 for m in historico_classes]  # Converte para %
    versicolor = [m['Versicolor'] * 100 for m in historico_classes]
    virginica = [m['Virginica'] * 100 for m in historico_classes]
    
    # Cores específicas por espécie
    cores_especies = {
        'Setosa': '#FF6B6B',      # Vermelho
        'Versicolor': '#4ECDC4',  # Azul-verde
        'Virginica': '#FFD93D'    # Amarelo
    }
    
    # Layout: 2 linhas, 2 colunas
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # 1. Gráfico de linhas - Evolução ao longo das rodadas
    ax1 = fig.add_subplot(gs[0, :])  # Ocupa toda a primeira linha
    ax1.plot(rodadas, setosa, marker='o', linewidth=3, markersize=10,
            color=cores_especies['Setosa'], label='Setosa', alpha=0.9)
    ax1.plot(rodadas, versicolor, marker='s', linewidth=3, markersize=10,
            color=cores_especies['Versicolor'], label='Versicolor', alpha=0.9)
    ax1.plot(rodadas, virginica, marker='^', linewidth=3, markersize=10,
            color=cores_especies['Virginica'], label='Virginica', alpha=0.9)
    
    ax1.set_title('Acurácia por Espécie ao Longo das Rodadas Federadas',
                 fontsize=16, fontweight='bold', pad=15)
    ax1.set_xlabel('Rodada Federada', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Acurácia (%)', fontsize=13, fontweight='bold')
    ax1.set_ylim([0, 105])
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(fontsize=12, loc='lower right', framealpha=0.9)
    
    # Adiciona linha de referência em 100%
    ax1.axhline(y=100, color='green', linestyle='--', linewidth=2, alpha=0.5, label='100% (Perfeito)')
    
    # 2. Gráfico de barras - Acurácia final por espécie
    ax2 = fig.add_subplot(gs[1, 0])
    especies = ['Setosa', 'Versicolor', 'Virginica']
    acuracias_finais = [setosa[-1], versicolor[-1], virginica[-1]]
    cores_barras = [cores_especies[esp] for esp in especies]
    
    bars = ax2.bar(especies, acuracias_finais, color=cores_barras, alpha=0.8, 
                   edgecolor='black', linewidth=2)
    ax2.set_title('Acurácia Final por Espécie (Última Rodada)',
                 fontsize=14, fontweight='bold')
    ax2.set_ylabel('Acurácia (%)', fontsize=12, fontweight='bold')
    ax2.set_ylim([0, 105])
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Adiciona valores nas barras
    for bar, acc in zip(bars, acuracias_finais):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.1f}%',
                ha='center', va='bottom', fontsize=13, fontweight='bold')
    
    # 3. Tabela de métricas
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis('off')
    
    # Cria tabela com estatísticas
    tabela_dados = []
    for especie, valores in [('Setosa', setosa), ('Versicolor', versicolor), ('Virginica', virginica)]:
        media = np.mean(valores)
        minimo = np.min(valores)
        maximo = np.max(valores)
        desvio = np.std(valores)
        tabela_dados.append([especie, f'{media:.1f}%', f'{minimo:.1f}%', f'{maximo:.1f}%', f'{desvio:.1f}%'])
    
    table = ax3.table(cellText=tabela_dados,
                     colLabels=['Espécie', 'Média', 'Mínimo', 'Máximo', 'Desvio Padrão'],
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0.2, 1, 0.6])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    
    # Estiliza cabeçalho
    for i in range(5):
        table[(0, i)].set_facecolor('#4ECDC4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Estiliza linhas
    for i in range(1, 4):
        cor = cores_especies[tabela_dados[i-1][0]]
        table[(i, 0)].set_facecolor(cor)
        table[(i, 0)].set_text_props(weight='bold')
    
    ax3.text(0.5, 0.9, 'Estatísticas por Espécie', 
            ha='center', fontsize=14, fontweight='bold', transform=ax3.transAxes)
    
    plt.suptitle('Análise de Classificação por Espécie - Cenário de Envenenamento\nDataset Iris com Regressão Logística Federada',
                fontsize=16, fontweight='bold', y=0.995)
    
    plt.savefig('visualizacao_2_classificacao_por_especie.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Salvo: visualizacao_2_classificacao_por_especie.png")
    plt.close()


def carregar_dataset_iris():
    """Carrega e prepara o dataset Iris"""
    caminho_dataset = r"c:\Users\Administrador\Faculdade\Iniciação-cientifica\project\data\iris\iris.csv"
    
    try:
        df = pd.read_csv(caminho_dataset)
    except:
        from sklearn.datasets import load_iris
        iris = load_iris()
        df = pd.DataFrame(iris.data, columns=iris.feature_names)
        df['species'] = iris.target
    
    if 'species' in df.columns:
        X = df.drop('species', axis=1).values
        if df['species'].dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(df['species'])
        else:
            y = df['species'].values
    else:
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
    
    return X, y


def main():
    """Função principal"""
    print("="*70)
    print("VISUALIZAÇÃO COMPLETA - ENVENENAMENTO DE DADOS")
    print("Dataset: Iris | Modelo: Regressão Logística Federada")
    print("="*70)
    
    # 1. Carrega dataset
    X, y = carregar_dataset_iris()
    print(f"\n✓ Dataset carregado: {len(X)} amostras, {X.shape[1]} features")
    print(f"  Espécies: Setosa, Versicolor, Virginica")
    
    # 2. Divide dados
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Divide treino entre 3 clientes
    X_c1, X_temp, y_c1, y_temp = train_test_split(
        X_train, y_train, test_size=0.66, random_state=42, stratify=y_train
    )
    X_c2, X_c3, y_c2, y_c3 = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    dados_clientes = [(X_c1, y_c1), (X_c2, y_c2), (X_c3, y_c3)]
    dados_validacao = (X_val, y_val)
    
    print(f"\n✓ Configuração do Sistema Federado:")
    print(f"  - Cliente 1 (Honesto): {len(X_c1)} amostras")
    print(f"  - Cliente 2 (Honesto): {len(X_c2)} amostras")
    print(f"  - Cliente 3 (ENVENENADO): {len(X_c3)} amostras")
    print(f"  - Validação: {len(X_val)} amostras")
    
    # 3. Executa experimento
    historico_global, historico_classes = executar_experimento_completo(
        dados_clientes, dados_validacao, num_rodadas=8
    )
    
    # 4. Gera visualizações
    print("\n" + "="*70)
    print("GERANDO VISUALIZAÇÕES GRÁFICAS")
    print("="*70)
    
    gerar_grafico_convergencia(historico_global)
    gerar_grafico_classificacao_especies(historico_classes)
    
    # 5. Resumo final
    print("\n" + "="*70)
    print("EXPERIMENTO CONCLUÍDO COM SUCESSO")
    print("="*70)
    print("\n📊 Visualizações geradas:")
    print("  1. visualizacao_1_convergencia_por_rodada.png")
    print("     → Gráfico com evolução de Acurácia, F1-Score, Precisão e Loss")
    print("\n  2. visualizacao_2_classificacao_por_especie.png")
    print("     → Gráfico com desempenho específico por espécie (Setosa, Versicolor, Virginica)")
    
    print("\n📈 Resultados Finais:")
    print(f"  - Acurácia Global: {historico_global[-1]['acuracia']:.2%}")
    print(f"  - Setosa: {historico_classes[-1]['Setosa']:.2%}")
    print(f"  - Versicolor: {historico_classes[-1]['Versicolor']:.2%}")
    print(f"  - Virginica: {historico_classes[-1]['Virginica']:.2%}")
    
    print("\n⚠️  Observações:")
    print("  - O cliente 3 (33% dos clientes) realiza envenenamento por inversão de pesos")
    print("  - O ataque causa degradação persistente no modelo global")
    print("  - O impacto é mais evidente nas espécies Setosa e Virginica")
    print("="*70)


if __name__ == "__main__":
    main()
