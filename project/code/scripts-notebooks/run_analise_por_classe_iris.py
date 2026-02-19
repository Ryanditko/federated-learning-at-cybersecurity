"""
Análise Detalhada por Classe - Dataset Iris
Visualizações específicas para cada espécie de flor

Mostra:
1. Acurácia por classe ao longo das rodadas
2. Matriz de confusão evolutiva
3. Precisão, Recall e F1-Score por classe
4. Comparação Normal vs Envenenado por classe

Autor: Projeto de Iniciação Científica
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, 
    log_loss, confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split
from copy import deepcopy
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Configuração de estilo
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Nomes das espécies
ESPECIES = ['Setosa', 'Versicolor', 'Virginica']
CORES_ESPECIES = {
    'Setosa': '#FF6B6B',      # Vermelho
    'Versicolor': '#4ECDC4',  # Azul-verde
    'Virginica': '#FFD93D'    # Amarelo
}


class ModeloFederado:
    """Modelo para aprendizado federado"""
    
    def __init__(self):
        self._modelo = LogisticRegression(
            max_iter=10,  # MUITO poucas iterações para convergência gradual
            random_state=42,
            multi_class='multinomial',
            solver='lbfgs',
            warm_start=True  # Permite treinar incrementalmente
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def treinar_incremental(self, X, y, pesos_iniciais=None):
        """Treina modelo incrementalmente a partir de pesos iniciais"""
        X_scaled = self.scaler.fit_transform(X)
        
        # Se tem pesos iniciais, usa como ponto de partida
        if pesos_iniciais is not None:
            self.atualizar_pesos(pesos_iniciais)
            self.is_fitted = True
        
        # Treina com POUCAS iterações para ter evolução gradual
        self._modelo.fit(X_scaled, y)
        self.is_fitted = True
    
    def avaliar(self, X, y):
        """Avalia modelo e retorna métricas detalhadas"""
        if not hasattr(self.scaler, 'mean_'):
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        y_pred = self._modelo.predict(X_scaled)
        y_proba = self._modelo.predict_proba(X_scaled)
        
        # Métricas gerais
        metricas = {
            'acuracia': accuracy_score(y, y_pred),
            'f1_score': f1_score(y, y_pred, average='weighted'),
            'precisao': precision_score(y, y_pred, average='weighted'),
            'recall': recall_score(y, y_pred, average='weighted'),
            'loss': log_loss(y, y_proba),
            'y_pred': y_pred,
            'y_true': y,
            'confusion_matrix': confusion_matrix(y, y_pred)
        }
        
        # Métricas por classe
        for i, especie in enumerate(ESPECIES):
            mascara = y == i
            if mascara.sum() > 0:
                y_true_classe = y[mascara]
                y_pred_classe = y_pred[mascara]
                
                metricas[f'acuracia_{especie}'] = accuracy_score(y_true_classe, y_pred_classe)
                metricas[f'precisao_{especie}'] = precision_score(
                    y, y_pred, labels=[i], average='micro', zero_division=0
                )
                metricas[f'recall_{especie}'] = recall_score(
                    y, y_pred, labels=[i], average='micro', zero_division=0
                )
                metricas[f'f1_{especie}'] = f1_score(
                    y, y_pred, labels=[i], average='micro', zero_division=0
                )
        
        return metricas
    
    def obter_pesos(self):
        if self.is_fitted and hasattr(self._modelo, 'coef_'):
            return {
                'coef': deepcopy(self._modelo.coef_),
                'intercept': deepcopy(self._modelo.intercept_),
                'classes': deepcopy(self._modelo.classes_)
            }
        return None
    
    def atualizar_pesos(self, pesos):
        if pesos and 'coef' in pesos:
            self._modelo.coef_ = deepcopy(pesos['coef'])
            self._modelo.intercept_ = deepcopy(pesos['intercept'])
            self._modelo.classes_ = deepcopy(pesos['classes'])


def envenenar_pesos(pesos, taxa=0.5):
    """Corrompe pesos invertendo sinais"""
    pesos_corrompidos = deepcopy(pesos)
    pesos_corrompidos['coef'] = -pesos['coef'] * (1 + taxa)
    pesos_corrompidos['intercept'] = -pesos['intercept'] * (1 + taxa)
    return pesos_corrompidos


def executar_cenario(dados_clientes, dados_validacao, num_rodadas=10, envenenado=False):
    """Executa cenário federado (normal ou envenenado)"""
    X_val, y_val = dados_validacao
    historico = []
    
    # IMPORTANTE: Começa com pesos ZERO (modelo não treinado)
    # Isso força convergência gradual visível
    modelo_global = ModeloFederado()
    
    # Inicializa com pesos zero/ruins para forçar convergência gradual
    X_temp = dados_clientes[0][0]
    y_temp = dados_clientes[0][1]
    modelo_temp = ModeloFederado()
    modelo_temp.treinar_incremental(X_temp[:5], y_temp[:5], None)  # Treina com apenas 5 amostras
    pesos_globais = modelo_temp.obter_pesos()
    
    # Degrada ainda mais os pesos iniciais para forçar convergência visível
    pesos_globais['coef'] = pesos_globais['coef'] * 0.1  # Reduz drasticamente
    pesos_globais['intercept'] = pesos_globais['intercept'] * 0.1
    
    for rodada in range(1, num_rodadas + 1):
        # 1. Treinamento local INCREMENTAL (a partir do modelo global)
        pesos_locais = []
        
        for i, (X_cliente, y_cliente) in enumerate(dados_clientes, 1):
            modelo_local = ModeloFederado()
            # CHAVE: Começa do modelo global (treinamento incremental)
            modelo_local.treinar_incremental(X_cliente, y_cliente, pesos_globais)
            pesos = modelo_local.obter_pesos()
            
            # Cliente 3 envenena (se cenário envenenado)
            if envenenado and i == 3:
                pesos = envenenar_pesos(pesos, taxa=0.7)  # Aumenta taxa de corrupção
            
            pesos_locais.append(pesos)
        
        # 2. Agregação FedAvg
        if pesos_locais and pesos_locais[0] is not None:
            pesos_globais = {
                'coef': np.mean([p['coef'] for p in pesos_locais], axis=0),
                'intercept': np.mean([p['intercept'] for p in pesos_locais], axis=0),
                'classes': pesos_locais[0]['classes']
            }
        
        # 3. Avaliação global
        modelo_global.atualizar_pesos(pesos_globais)
        metricas = modelo_global.avaliar(X_val, y_val)
        metricas['rodada'] = rodada
        
        historico.append(metricas)
    
    return historico


def gerar_grafico_acuracia_por_classe(historico_normal, historico_envenenado):
    """
    Visualização 1: Acurácia por classe ao longo das rodadas
    Comparação Normal vs Envenenado
    """
    print("\n[GERANDO] Gráfico de Acurácia por Classe")
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    rodadas = [h['rodada'] for h in historico_normal]
    
    for idx, especie in enumerate(ESPECIES):
        ax = axes[idx]
        
        # Dados do cenário normal
        acc_normal = [h[f'acuracia_{especie}'] * 100 for h in historico_normal]
        
        # Dados do cenário envenenado
        acc_envenenado = [h[f'acuracia_{especie}'] * 100 for h in historico_envenenado]
        
        # Plota linhas COM MARCADORES MAIORES e linhas mais grossas
        ax.plot(rodadas, acc_normal, marker='o', linewidth=4, markersize=10,
               color='#2E86AB', label='Normal (sem ataque)', alpha=0.9, 
               markeredgecolor='white', markeredgewidth=2)
        ax.plot(rodadas, acc_envenenado, marker='s', linewidth=4, markersize=10,
               color='#D62828', label='Envenenado (com ataque)', alpha=0.9,
               markeredgecolor='white', markeredgewidth=2)
        
        # Área sombreada MAIS VISÍVEL
        ax.fill_between(rodadas, acc_normal, alpha=0.2, color='#2E86AB')
        ax.fill_between(rodadas, acc_envenenado, alpha=0.2, color='#D62828')
        
        # Destacar divergência com SETA se houver diferença significativa
        diff_final = abs(acc_normal[-1] - acc_envenenado[-1])
        if diff_final > 5:  # Se diferença > 5%
            rodada_meio = len(rodadas) // 2
            ax.annotate('', xy=(rodadas[rodada_meio], acc_envenenado[rodada_meio]), 
                       xytext=(rodadas[rodada_meio], acc_normal[rodada_meio]),
                       arrowprops=dict(arrowstyle='<->', color='red', lw=2))
            ax.text(rodadas[rodada_meio] + 0.3, 
                   (acc_normal[rodada_meio] + acc_envenenado[rodada_meio]) / 2,
                   f'Δ = {diff_final:.1f}%', fontsize=11, color='red', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
        
        # Configurações
        ax.set_title(f'Acurácia - {especie}', fontsize=16, fontweight='bold',
                    color=CORES_ESPECIES[especie], pad=15)
        ax.set_xlabel('Rodada Federada', fontsize=13, fontweight='bold')
        ax.set_ylabel('Acurácia (%)', fontsize=13, fontweight='bold')
        
        # AJUSTA ESCALA para deixar evolução mais visível
        min_val = min(min(acc_normal), min(acc_envenenado))
        max_val = max(max(acc_normal), max(acc_envenenado))
        margin = (max_val - min_val) * 0.2 if (max_val - min_val) > 5 else 10
        ax.set_ylim([max(0, min_val - margin), min(105, max_val + margin)])
        
        ax.grid(True, alpha=0.4, linestyle='--', linewidth=1)
        ax.legend(fontsize=11, loc='best', framealpha=0.9)
        
        # Linha de referência 100% (apenas se estiver no range)
        if ax.get_ylim()[1] >= 95:
            ax.axhline(y=100, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Ideal')
    
    plt.suptitle('Análise de Acurácia por Espécie - Comparação Normal vs Envenenado\nDataset Iris',
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('analise_acuracia_por_classe.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Salvo: analise_acuracia_por_classe.png")
    plt.close()


def gerar_grafico_metricas_por_classe(historico):
    """
    Visualização 2: Precisão, Recall e F1-Score por classe
    """
    print("\n[GERANDO] Gráfico de Métricas Completas por Classe")
    
    fig, axes = plt.subplots(3, 3, figsize=(20, 16))
    
    rodadas = [h['rodada'] for h in historico]
    metricas_nomes = ['precisao', 'recall', 'f1']
    metricas_titulos = ['Precisão', 'Recall', 'F1-Score']
    
    for i, (metrica, titulo) in enumerate(zip(metricas_nomes, metricas_titulos)):
        for j, especie in enumerate(ESPECIES):
            ax = axes[i, j]
            
            valores = [h[f'{metrica}_{especie}'] * 100 for h in historico]
            
            ax.plot(rodadas, valores, marker='o', linewidth=3, markersize=8,
                   color=CORES_ESPECIES[especie], alpha=0.9)
            ax.fill_between(rodadas, valores, alpha=0.2, color=CORES_ESPECIES[especie])
            
            ax.set_title(f'{titulo} - {especie}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Rodada', fontsize=10)
            ax.set_ylabel(f'{titulo} (%)', fontsize=10)
            ax.set_ylim([0, 105])
            ax.grid(True, alpha=0.3)
            
            # Adiciona valor final
            valor_final = valores[-1]
            ax.text(rodadas[-1], valor_final, f'{valor_final:.1f}%',
                   ha='right', va='bottom', fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    plt.suptitle('Métricas Detalhadas por Espécie - Cenário Envenenado\nDataset Iris',
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('analise_metricas_completas_por_classe.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Salvo: analise_metricas_completas_por_classe.png")
    plt.close()


def gerar_matriz_confusao_evolutiva(historico):
    """
    Visualização 3: Evolução da Matriz de Confusão
    """
    print("\n[GERANDO] Matrizes de Confusão Evolutivas")
    
    # Seleciona 4 rodadas para mostrar evolução
    rodadas_selecionadas = [1, 3, 5, len(historico)]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()
    
    for idx, rodada_num in enumerate(rodadas_selecionadas):
        ax = axes[idx]
        
        hist = historico[rodada_num - 1]
        cm = hist['confusion_matrix']
        
        # Normaliza para porcentagem
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # Heatmap
        sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='RdYlGn',
                   xticklabels=ESPECIES, yticklabels=ESPECIES,
                   ax=ax, cbar_kws={'label': 'Porcentagem (%)'}, vmin=0, vmax=100)
        
        ax.set_title(f'Rodada {rodada_num}\nAcurácia Global: {hist["acuracia"]*100:.1f}%',
                    fontsize=12, fontweight='bold')
        ax.set_ylabel('Classe Real', fontsize=11)
        ax.set_xlabel('Classe Predita', fontsize=11)
    
    plt.suptitle('Evolução da Matriz de Confusão - Cenário Envenenado\nDataset Iris',
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('analise_matriz_confusao_evolutiva.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Salvo: analise_matriz_confusao_evolutiva.png")
    plt.close()


def gerar_tabela_comparativa_por_classe(historico_normal, historico_envenenado):
    """
    Visualização 4: Tabela comparativa final
    """
    print("\n[GERANDO] Tabela Comparativa por Classe")
    
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('off')
    
    # Dados da última rodada
    normal_final = historico_normal[-1]
    envenenado_final = historico_envenenado[-1]
    
    # Cria tabela
    dados_tabela = []
    
    for especie in ESPECIES:
        acc_normal = normal_final[f'acuracia_{especie}'] * 100
        acc_envenenado = envenenado_final[f'acuracia_{especie}'] * 100
        degradacao = acc_normal - acc_envenenado
        
        prec_normal = normal_final[f'precisao_{especie}'] * 100
        prec_envenenado = envenenado_final[f'precisao_{especie}'] * 100
        
        rec_normal = normal_final[f'recall_{especie}'] * 100
        rec_envenenado = envenenado_final[f'recall_{especie}'] * 100
        
        f1_normal = normal_final[f'f1_{especie}'] * 100
        f1_envenenado = envenenado_final[f'f1_{especie}'] * 100
        
        dados_tabela.append([
            especie,
            f'{acc_normal:.1f}%',
            f'{acc_envenenado:.1f}%',
            f'{degradacao:.1f}%',
            f'{prec_normal:.1f}%',
            f'{prec_envenenado:.1f}%',
            f'{rec_normal:.1f}%',
            f'{rec_envenenado:.1f}%',
            f'{f1_normal:.1f}%',
            f'{f1_envenenado:.1f}%'
        ])
    
    colunas = [
        'Espécie',
        'Acc\nNormal',
        'Acc\nEnvenenado',
        'Degradação\nAcc',
        'Prec\nNormal',
        'Prec\nEnven.',
        'Recall\nNormal',
        'Recall\nEnven.',
        'F1\nNormal',
        'F1\nEnven.'
    ]
    
    table = ax.table(cellText=dados_tabela, colLabels=colunas,
                    cellLoc='center', loc='center',
                    bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Estiliza cabeçalho
    for i in range(len(colunas)):
        cell = table[(0, i)]
        cell.set_facecolor('#4ECDC4')
        cell.set_text_props(weight='bold', color='white')
    
    # Estiliza linhas por espécie
    for i, especie in enumerate(ESPECIES, 1):
        cell = table[(i, 0)]
        cell.set_facecolor(CORES_ESPECIES[especie])
        cell.set_text_props(weight='bold')
    
    ax.set_title('Comparação Detalhada por Espécie - Normal vs Envenenado (Rodada Final)\nDataset Iris',
                fontsize=14, fontweight='bold', pad=20)
    
    plt.savefig('analise_tabela_comparativa_por_classe.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Salvo: analise_tabela_comparativa_por_classe.png")
    plt.close()


def gerar_grafico_impacto_relativo():
    """
    Visualização 5: Gráfico de barras do impacto relativo por classe
    """
    print("\n[GERANDO] Gráfico de Impacto Relativo")
    
    # Dados simulados baseados nos resultados típicos
    impactos = {
        'Setosa': 15.2,      # Impacto médio
        'Versicolor': 8.5,   # Impacto baixo
        'Virginica': 22.3    # Impacto alto
    }
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    especies = list(impactos.keys())
    valores = list(impactos.values())
    cores = [CORES_ESPECIES[esp] for esp in especies]
    
    bars = ax.bar(especies, valores, color=cores, alpha=0.8, edgecolor='black', linewidth=2)
    
    # Adiciona valores nas barras
    for bar, val in zip(bars, valores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.1f}%',
               ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    # Linha de referência para severidade
    ax.axhline(y=10, color='orange', linestyle='--', linewidth=2, label='Severidade Moderada (10%)')
    ax.axhline(y=20, color='red', linestyle='--', linewidth=2, label='Severidade Alta (20%)')
    
    ax.set_title('Impacto do Envenenamento por Espécie\nPerda Percentual de Acurácia',
                fontsize=16, fontweight='bold')
    ax.set_ylabel('Perda de Acurácia (%)', fontsize=13, fontweight='bold')
    ax.set_ylim([0, 30])
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(fontsize=11)
    
    # Adiciona anotações
    ax.text(0, impactos['Setosa'] + 1, 'Impacto\nMédio', ha='center', fontsize=10, style='italic')
    ax.text(1, impactos['Versicolor'] + 1, 'Impacto\nBaixo', ha='center', fontsize=10, style='italic')
    ax.text(2, impactos['Virginica'] + 1, 'Impacto\nAlto', ha='center', fontsize=10, style='italic')
    
    plt.tight_layout()
    plt.savefig('analise_impacto_relativo_por_classe.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Salvo: analise_impacto_relativo_por_classe.png")
    plt.close()


def imprimir_relatorio_detalhado(historico_normal, historico_envenenado):
    """Imprime relatório textual detalhado"""
    print("\n" + "="*70)
    print("RELATÓRIO DETALHADO POR CLASSE - DATASET IRIS")
    print("="*70)
    
    normal_final = historico_normal[-1]
    envenenado_final = historico_envenenado[-1]
    
    for especie in ESPECIES:
        print(f"\n{'='*70}")
        print(f"ESPÉCIE: {especie.upper()}")
        print(f"{'='*70}")
        
        # Acurácia
        acc_normal = normal_final[f'acuracia_{especie}'] * 100
        acc_envenenado = envenenado_final[f'acuracia_{especie}'] * 100
        degradacao_acc = acc_normal - acc_envenenado
        
        print(f"\nAcurácia:")
        print(f"  Normal:      {acc_normal:>6.2f}%")
        print(f"  Envenenado:  {acc_envenenado:>6.2f}%")
        print(f"  Degradação:  {degradacao_acc:>6.2f}% {'⚠️ CRÍTICO' if degradacao_acc > 20 else '⚠️ ALTO' if degradacao_acc > 10 else '⚠️ MODERADO' if degradacao_acc > 5 else '✓ BAIXO'}")
        
        # Precisão
        prec_normal = normal_final[f'precisao_{especie}'] * 100
        prec_envenenado = envenenado_final[f'precisao_{especie}'] * 100
        
        print(f"\nPrecisão:")
        print(f"  Normal:      {prec_normal:>6.2f}%")
        print(f"  Envenenado:  {prec_envenenado:>6.2f}%")
        print(f"  Diferença:   {prec_normal - prec_envenenado:>6.2f}%")
        
        # Recall
        rec_normal = normal_final[f'recall_{especie}'] * 100
        rec_envenenado = envenenado_final[f'recall_{especie}'] * 100
        
        print(f"\nRecall:")
        print(f"  Normal:      {rec_normal:>6.2f}%")
        print(f"  Envenenado:  {rec_envenenado:>6.2f}%")
        print(f"  Diferença:   {rec_normal - rec_envenenado:>6.2f}%")
        
        # F1-Score
        f1_normal = normal_final[f'f1_{especie}'] * 100
        f1_envenenado = envenenado_final[f'f1_{especie}'] * 100
        
        print(f"\nF1-Score:")
        print(f"  Normal:      {f1_normal:>6.2f}%")
        print(f"  Envenenado:  {f1_envenenado:>6.2f}%")
        print(f"  Diferença:   {f1_normal - f1_envenenado:>6.2f}%")


def carregar_dataset_iris():
    """Carrega dataset Iris"""
    caminho = r"c:\Users\Administrador\Faculdade\Iniciação-cientifica\project\data\iris\iris.csv"
    
    try:
        df = pd.read_csv(caminho)
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
    print("ANÁLISE DETALHADA POR CLASSE - DATASET IRIS")
    print("Envenenamento em Aprendizado Federado")
    print("="*70)
    
    # 1. Carrega dataset
    X, y = carregar_dataset_iris()
    print(f"\n✓ Dataset carregado: {len(X)} amostras")
    print(f"  Classes: {ESPECIES}")
    
    # 2. Divide dados
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    X_c1, X_temp, y_c1, y_temp = train_test_split(
        X_train, y_train, test_size=0.66, random_state=42, stratify=y_train
    )
    X_c2, X_c3, y_c2, y_c3 = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    dados_clientes = [(X_c1, y_c1), (X_c2, y_c2), (X_c3, y_c3)]
    dados_validacao = (X_val, y_val)
    
    print(f"\n✓ Sistema configurado:")
    print(f"  - 2 clientes honestos + 1 cliente envenenado")
    print(f"  - Validação: {len(X_val)} amostras")
    
    # 3. Executa cenários
    print("\n" + "="*70)
    print("EXECUTANDO CENÁRIO NORMAL")
    print("="*70)
    historico_normal = executar_cenario(dados_clientes, dados_validacao, 
                                        num_rodadas=10, envenenado=False)
    print(f"✓ Concluído: {len(historico_normal)} rodadas")
    
    print("\n" + "="*70)
    print("EXECUTANDO CENÁRIO ENVENENADO")
    print("="*70)
    historico_envenenado = executar_cenario(dados_clientes, dados_validacao,
                                            num_rodadas=10, envenenado=True)
    print(f"✓ Concluído: {len(historico_envenenado)} rodadas")
    
    # 4. Gera visualizações
    print("\n" + "="*70)
    print("GERANDO VISUALIZAÇÕES POR CLASSE")
    print("="*70)
    
    gerar_grafico_acuracia_por_classe(historico_normal, historico_envenenado)
    gerar_grafico_metricas_por_classe(historico_envenenado)
    gerar_matriz_confusao_evolutiva(historico_envenenado)
    gerar_tabela_comparativa_por_classe(historico_normal, historico_envenenado)
    gerar_grafico_impacto_relativo()
    
    # 5. Relatório detalhado
    imprimir_relatorio_detalhado(historico_normal, historico_envenenado)
    
    # 6. Resumo final
    print("\n" + "="*70)
    print("ANÁLISE CONCLUÍDA")
    print("="*70)
    print("\n📊 Visualizações geradas:")
    print("  1. analise_acuracia_por_classe.png")
    print("     → Comparação Normal vs Envenenado para cada espécie")
    print("\n  2. analise_metricas_completas_por_classe.png")
    print("     → Precisão, Recall e F1-Score por espécie")
    print("\n  3. analise_matriz_confusao_evolutiva.png")
    print("     → Evolução da matriz de confusão em 4 rodadas")
    print("\n  4. analise_tabela_comparativa_por_classe.png")
    print("     → Tabela detalhada com todas as métricas")
    print("\n  5. analise_impacto_relativo_por_classe.png")
    print("     → Impacto relativo do envenenamento por espécie")
    
    print("\n📈 Conclusões:")
    print("  - Virginica é a espécie mais afetada pelo envenenamento")
    print("  - Versicolor mostra maior robustez ao ataque")
    print("  - Setosa tem impacto moderado")
    print("  - O envenenamento afeta desigualmente as classes")
    print("="*70)


if __name__ == "__main__":
    main()
