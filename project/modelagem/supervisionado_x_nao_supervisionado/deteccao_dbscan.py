"""
Análise Completa da Detecção DBSCAN
====================================
Visualizações detalhadas para o experimento com 15 clientes.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, log_loss, roc_auc_score
)
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from copy import deepcopy
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

RESULTADOS_DIR = "resultados/"
NUM_RODADAS = 12
NUM_CLIENTES = 15
NUM_MALICIOSOS = 3
CLIENTES_MALICIOSOS = [13, 14, 15]


def carregar_dataset():
    """Carrega e preprocessa o dataset Bank Marketing completo"""
    caminho = r"c:\Users\Administrador\Faculdade\Iniciação-cientifica\project\data\bank-marketing\bank.csv"
    df = pd.read_csv(caminho, sep=';')
    df = df.drop_duplicates()

    df['job'] = df.apply(
        lambda r: ('management' if r['education'] == 'tertiary' else 'blue-collar')
        if r['job'] == 'unknown' else r['job'], axis=1)
    df['education'] = df.apply(
        lambda r: ('tertiary' if r['job'] in ['entrepreneur', 'self-employed', 'management']
                   else 'primary' if r['job'] == 'housemaid' else 'secondary')
        if r['education'] == 'unknown' else r['education'], axis=1)

    if 'poutcome' in df.columns:
        df['poutcome'] = df['poutcome'].replace('unknown', 'nonexistent')
    if 'contact' in df.columns:
        df['contact'] = df['contact'].replace('unknown', 'non-call')
    if 'balance' in df.columns:
        df = df.drop('balance', axis=1)

    if 'duration' in df.columns:
        df['duration_log'] = np.log1p(df['duration'])
        df['duration_high'] = (df['duration'] > 300).astype(int)
    if 'pdays' in df.columns:
        df['previously_contacted'] = (df['pdays'] != 999).astype(int)
    if 'campaign' in df.columns:
        df['campaign_low'] = (df['campaign'] <= 2).astype(int)

    cat_cols = [c for c in df.select_dtypes(include='object').columns if c != 'y']
    for col in cat_cols:
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
        df = pd.concat([df, dummies], axis=1).drop(col, axis=1)

    df['y'] = df['y'].map({'yes': 1, 'no': 0})

    X = df.drop('y', axis=1).values
    y = df['y'].values
    return X, y


def distribuir_dados_15clientes(X, y):
    """Distribui dados entre 15 clientes"""
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    clientes = []
    X_restante, y_restante = X_tr, y_tr

    for i in range(NUM_CLIENTES):
        if i < NUM_CLIENTES - 1:
            test_size = (NUM_CLIENTES - i - 1) / (NUM_CLIENTES - i)
            X_cli, X_restante, y_cli, y_restante = train_test_split(
                X_restante, y_restante, test_size=test_size, random_state=42+i, stratify=y_restante
            )
        else:
            X_cli, y_cli = X_restante, y_restante

        clientes.append((X_cli, y_cli))

    return clientes, (X_val, y_val)


def treinar_cliente(X, y, pesos_globais=None, rodada=1, seed=42):
    """Treina modelo local"""
    rng = np.random.RandomState(seed + rodada * 7)
    n = max(150, int(len(X) * 0.40))
    idx = rng.choice(len(X), size=n, replace=False)
    X_sub, y_sub = X[idx], y[idx]

    scaler = MinMaxScaler()
    X_sc = scaler.fit_transform(X_sub)

    iters_por_rodada = min(10 + rodada * 5, 80)

    modelo = LogisticRegression(
        max_iter=iters_por_rodada, solver='saga', class_weight='balanced',
        C=0.5, penalty='l2', random_state=seed, warm_start=True
    )

    if pesos_globais is not None:
        try:
            modelo.fit(X_sc, y_sub)
            modelo.coef_ = deepcopy(pesos_globais['coef'])
            modelo.intercept_ = deepcopy(pesos_globais['intercept'])
            modelo.classes_ = deepcopy(pesos_globais['classes'])
            modelo.fit(X_sc, y_sub)
        except Exception:
            modelo.fit(X_sc, y_sub)
    else:
        modelo.fit(X_sc, y_sub)

    return {
        'coef': deepcopy(modelo.coef_),
        'intercept': deepcopy(modelo.intercept_),
        'classes': deepcopy(modelo.classes_),
        'vetor': np.concatenate([modelo.coef_.flatten(), modelo.intercept_])
    }


def envenenar_pesos(pesos, taxa=0.9):
    """Sign Flipping Attack"""
    p = deepcopy(pesos)
    p['coef'] = -pesos['coef'] * (1 + taxa)
    p['intercept'] = -pesos['intercept'] * (1 + taxa)
    p['vetor'] = np.concatenate([p['coef'].flatten(), p['intercept']])
    return p


def agregar_fedavg(lista_pesos):
    """Agrega pesos"""
    return {
        'coef': np.mean([p['coef'] for p in lista_pesos], axis=0),
        'intercept': np.mean([p['intercept'] for p in lista_pesos], axis=0),
        'classes': lista_pesos[0]['classes']
    }


def avaliar_modelo_global(pesos_globais, X_val, y_val):
    """Avalia modelo global"""
    scaler = MinMaxScaler()
    X_sc = scaler.fit_transform(X_val)

    modelo = LogisticRegression(max_iter=1, warm_start=True, solver='saga')
    modelo.fit(X_sc, y_val)
    modelo.coef_ = pesos_globais['coef']
    modelo.intercept_ = pesos_globais['intercept']
    modelo.classes_ = pesos_globais['classes']

    y_pred = modelo.predict(X_sc)
    y_proba = modelo.predict_proba(X_sc)

    return {
        'acuracia': accuracy_score(y_val, y_pred),
        'f1': f1_score(y_val, y_pred, average='binary', zero_division=0),
        'precisao': precision_score(y_val, y_pred, average='binary', zero_division=0),
        'recall': recall_score(y_val, y_pred, average='binary', zero_division=0),
        'auc': roc_auc_score(y_val, y_proba[:, 1]),
        'loss': log_loss(y_val, y_proba),
        'cm': confusion_matrix(y_val, y_pred),
        'y_pred': y_pred
    }


def detectar_maliciosos_dbscan(pesos_lista, pca=None, eps=0.8, min_samples=2):
    """Detecta outliers com DBSCAN"""
    vetores = np.array([p['vetor'] for p in pesos_lista])

    if pca is None:
        pca = PCA(n_components=min(15, vetores.shape[1], vetores.shape[0]-1), random_state=42)
        vetores_pca = pca.fit_transform(vetores)
    else:
        vetores_pca = pca.transform(vetores)

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(vetores_pca)

    suspeitos = [i for i, label in enumerate(labels) if label == -1]

    return suspeitos, labels, vetores_pca, pca


def executar_federado_com_defesa_dbscan(clientes, dados_val, envenenado=True):
    """Executa aprendizado com defesa DBSCAN"""
    X_val, y_val = dados_val
    historico = []

    pesos_globais = treinar_cliente(clientes[0][0][:300], clientes[0][1][:300], rodada=0)
    pca = None

    for rodada in range(1, NUM_RODADAS + 1):
        pesos_locais = []
        detectados = []

        todos_pesos = []
        for idx, (X_cli, y_cli) in enumerate(clientes, 1):
            pesos = treinar_cliente(X_cli, y_cli, pesos_globais, rodada=rodada, seed=42 + idx)

            if envenenado and idx in CLIENTES_MALICIOSOS:
                pesos = envenenar_pesos(pesos)

            todos_pesos.append(pesos)

        suspeitos_indices, labels, vetores_pca, pca = detectar_maliciosos_dbscan(todos_pesos, pca, eps=1.0, min_samples=3)

        for idx in range(NUM_CLIENTES):
            e_malicioso = idx in suspeitos_indices
            detectados.append({
                'cliente': idx + 1,
                'detectado': e_malicioso,
                'label_cluster': labels[idx]
            })

            if not e_malicioso:
                pesos_locais.append(todos_pesos[idx])

        if len(pesos_locais) == 0:
            pesos_locais = todos_pesos

        pesos_globais = agregar_fedavg(pesos_locais)
        metricas = avaliar_modelo_global(pesos_globais, X_val, y_val)
        metricas['rodada'] = rodada
        metricas['n_clientes_usados'] = len(pesos_locais)
        metricas['detectados'] = detectados
        historico.append(metricas)

    return historico


def executar_federado_sem_defesa(clientes, dados_val, envenenado=True):
    """Executa aprendizado sem defesa"""
    X_val, y_val = dados_val
    historico = []

    pesos_globais = treinar_cliente(clientes[0][0][:300], clientes[0][1][:300], rodada=0)

    for rodada in range(1, NUM_RODADAS + 1):
        pesos_locais = []

        for idx, (X_cli, y_cli) in enumerate(clientes, 1):
            pesos = treinar_cliente(X_cli, y_cli, pesos_globais, rodada=rodada, seed=42 + idx)

            if envenenado and idx in CLIENTES_MALICIOSOS:
                pesos = envenenar_pesos(pesos)

            pesos_locais.append(pesos)

        pesos_globais = agregar_fedavg(pesos_locais)
        metricas = avaliar_modelo_global(pesos_globais, X_val, y_val)
        metricas['rodada'] = rodada
        historico.append(metricas)

    return historico


# ==============================================================================
# NOVAS VISUALIZAÇÕES
# ==============================================================================

def plot_matriz_confusao_por_rodada(hist_sem_defesa, hist_com_defesa):
    """Matrizes de confusão selecionadas (rodadas 1, 4, 8, 12)"""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    rodadas_analisar = [1, 4, 8, 12]

    for idx, rodada in enumerate(rodadas_analisar):
        # Sem defesa
        ax = axes[0, idx]
        cm_sem = hist_sem_defesa[rodada - 1]['cm']
        sns.heatmap(cm_sem, annot=True, fmt='d', cmap='Reds', ax=ax,
                    cbar=False, xticklabels=['Não', 'Sim'], yticklabels=['Não', 'Sim'],
                    annot_kws={'size': 12, 'weight': 'bold'})
        ax.set_title(f'Sem Defesa - Rodada {rodada}', fontweight='bold', fontsize=12)
        if idx == 0:
            ax.set_ylabel('Real', fontweight='bold')

        # Com defesa DBSCAN
        ax = axes[1, idx]
        cm_com = hist_com_defesa[rodada - 1]['cm']
        sns.heatmap(cm_com, annot=True, fmt='d', cmap='Greens', ax=ax,
                    cbar=False, xticklabels=['Não', 'Sim'], yticklabels=['Não', 'Sim'],
                    annot_kws={'size': 12, 'weight': 'bold'})
        ax.set_title(f'Com DBSCAN - Rodada {rodada}', fontweight='bold', fontsize=12)
        if idx == 0:
            ax.set_ylabel('Real', fontweight='bold')
        ax.set_xlabel('Predito', fontweight='bold')

    plt.suptitle('Evolução das Matrizes de Confusão - DBSCAN 15 Clientes\nRodadas 1, 4, 8 e 12',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{RESULTADOS_DIR}dbscan_matriz_confusao_evolutiva.png', dpi=300, bbox_inches='tight')
    print(f"  Salvo: dbscan_matriz_confusao_evolutiva.png")
    plt.close()


def plot_metricas_individuais_por_rodada(hist_sem_defesa, hist_com_defesa, hist_normal):
    """Plot detalhado de cada métrica individualmente"""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()

    metricas = ['acuracia', 'f1', 'precisao', 'recall', 'auc', 'loss']
    titulos = ['Acurácia', 'F1-Score', 'Precisão', 'Recall', 'AUC-ROC', 'Loss']

    for idx, (metrica, titulo) in enumerate(zip(metricas, titulos)):
        ax = axes[idx]

        rodadas = [h['rodada'] for h in hist_normal]
        fator = 100 if metrica != 'auc' and metrica != 'loss' else 1

        v_normal = [h[metrica] * fator for h in hist_normal]
        v_sem = [h[metrica] * fator for h in hist_sem_defesa]
        v_com = [h[metrica] * fator for h in hist_com_defesa]

        ax.plot(rodadas, v_normal, marker='o', linewidth=3, color='#2E86AB',
                label='Normal', markersize=8, markeredgewidth=2, markeredgecolor='white')
        ax.plot(rodadas, v_sem, marker='s', linewidth=3, color='#D62828',
                label='Atacado (sem defesa)', markersize=8, linestyle='--', markeredgewidth=2, markeredgecolor='white')
        ax.plot(rodadas, v_com, marker='^', linewidth=3, color='#06D6A0',
                label='Atacado (com DBSCAN)', markersize=8, markeredgewidth=2, markeredgecolor='white')

        ax.fill_between(rodadas, v_normal, alpha=0.1, color='#2E86AB')
        ax.fill_between(rodadas, v_com, alpha=0.1, color='#06D6A0')

        ax.set_title(titulo, fontsize=14, fontweight='bold')
        ax.set_xlabel('Rodada Federada', fontweight='bold', fontsize=11)
        sufixo = '%' if metrica in ['acuracia', 'f1', 'precisao', 'recall'] else ''
        ax.set_ylabel(f'{titulo} {sufixo}', fontweight='bold', fontsize=11)
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3, linestyle='--')

    plt.suptitle('Análise Detalhada de Métricas - DBSCAN 15 Clientes\nTodas as 12 Rodadas Federadas',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{RESULTADOS_DIR}dbscan_metricas_detalhadas.png', dpi=300, bbox_inches='tight')
    print(f"  Salvo: dbscan_metricas_detalhadas.png")
    plt.close()


def plot_tabela_comparativa_final(hist_normal, hist_sem_defesa, hist_com_defesa):
    """Tabela comparativa de todas as métricas finais"""
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.axis('off')

    final_normal = hist_normal[-1]
    final_sem = hist_sem_defesa[-1]
    final_com = hist_com_defesa[-1]

    table_data = [
        ['Métrica', 'Normal', 'Atacado (sem defesa)', 'Atacado (com DBSCAN)', 'Recuperação'],
        ['Acurácia (%)', f"{final_normal['acuracia']*100:.2f}", f"{final_sem['acuracia']*100:.2f}",
         f"{final_com['acuracia']*100:.2f}",
         f"{((final_com['acuracia']-final_sem['acuracia'])/(final_normal['acuracia']-final_sem['acuracia'])*100 if final_normal['acuracia']!=final_sem['acuracia'] else 0):.1f}%"],
        ['F1-Score (%)', f"{final_normal['f1']*100:.2f}", f"{final_sem['f1']*100:.2f}",
         f"{final_com['f1']*100:.2f}",
         f"{((final_com['f1']-final_sem['f1'])/(final_normal['f1']-final_sem['f1'])*100 if final_normal['f1']!=final_sem['f1'] else 0):.1f}%"],
        ['Precisão (%)', f"{final_normal['precisao']*100:.2f}", f"{final_sem['precisao']*100:.2f}",
         f"{final_com['precisao']*100:.2f}",
         f"{((final_com['precisao']-final_sem['precisao'])/(final_normal['precisao']-final_sem['precisao'])*100 if final_normal['precisao']!=final_sem['precisao'] else 0):.1f}%"],
        ['Recall (%)', f"{final_normal['recall']*100:.2f}", f"{final_sem['recall']*100:.2f}",
         f"{final_com['recall']*100:.2f}",
         f"{((final_com['recall']-final_sem['recall'])/(final_normal['recall']-final_sem['recall'])*100 if final_normal['recall']!=final_sem['recall'] else 0):.1f}%"],
        ['AUC-ROC', f"{final_normal['auc']:.4f}", f"{final_sem['auc']:.4f}",
         f"{final_com['auc']:.4f}",
         f"{((final_com['auc']-final_sem['auc'])/(final_normal['auc']-final_sem['auc'])*100 if final_normal['auc']!=final_sem['auc'] else 0):.1f}%"],
        ['Loss', f"{final_normal['loss']:.4f}", f"{final_sem['loss']:.4f}",
         f"{final_com['loss']:.4f}",
         f"{((final_sem['loss']-final_com['loss'])/(final_sem['loss']-final_normal['loss'])*100 if final_sem['loss']!=final_normal['loss'] else 0):.1f}%"],
    ]

    table = ax.table(cellText=table_data[1:], colLabels=table_data[0], cellLoc='center',
                     loc='center', colWidths=[0.2, 0.18, 0.22, 0.22, 0.18])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 3)

    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#2E86AB')
        table[(0, i)].set_text_props(color='white', fontweight='bold', size=12)

    for i in range(1, len(table_data)):
        table[(i, 0)].set_facecolor('#E8F4F8')
        table[(i, 0)].set_text_props(weight='bold')
        table[(i, 1)].set_facecolor('#D4EDDA')
        table[(i, 2)].set_facecolor('#F8D7DA')
        table[(i, 3)].set_facecolor('#D1E7F7')
        table[(i, 4)].set_facecolor('#FFF3CD')

    ax.set_title('Tabela Comparativa de Métricas Finais (Rodada 12)\nDBSCAN 15 Clientes',
                 fontsize=15, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(f'{RESULTADOS_DIR}dbscan_tabela_comparativa_final.png', dpi=300, bbox_inches='tight')
    print(f"  Salvo: dbscan_tabela_comparativa_final.png")
    plt.close()


def plot_degradacao_vs_recuperacao(hist_normal, hist_sem_defesa, hist_com_defesa):
    """Gráfico mostrando degradação e recuperação"""
    fig, ax = plt.subplots(figsize=(14, 8))

    metricas = ['Acurácia', 'F1-Score', 'Precisão', 'Recall', 'AUC']
    metrica_keys = ['acuracia', 'f1', 'precisao', 'recall', 'auc']

    degradacoes = []
    recuperacoes = []

    for key in metrica_keys:
        v_normal = hist_normal[-1][key]
        v_sem = hist_sem_defesa[-1][key]
        v_com = hist_com_defesa[-1][key]

        degradacao = (v_normal - v_sem) * 100
        recuperacao = ((v_com - v_sem) / (v_normal - v_sem)) * 100 if v_normal != v_sem else 0

        degradacoes.append(degradacao)
        recuperacoes.append(recuperacao)

    x = np.arange(len(metricas))
    width = 0.35

    bars1 = ax.bar(x - width/2, degradacoes, width, label='Degradação pelo Ataque', color='#D62828', alpha=0.8)
    bars2 = ax.bar(x + width/2, recuperacoes, width, label='Recuperação com DBSCAN', color='#06D6A0', alpha=0.8)

    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)

    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)

    ax.set_ylabel('Variação de Métrica (%)', fontweight='bold', fontsize=12)
    ax.set_title('Degradação do Ataque vs Recuperação com DBSCAN\nDBSCAN 15 Clientes',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metricas, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='black', linewidth=1)

    plt.tight_layout()
    plt.savefig(f'{RESULTADOS_DIR}dbscan_degradacao_vs_recuperacao.png', dpi=300, bbox_inches='tight')
    print(f"  Salvo: dbscan_degradacao_vs_recuperacao.png")
    plt.close()


def plot_acuracia_por_cliente_malicioso(hist_com_defesa):
    """Mostra acurácia geral e detecções de maliciosos"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Gráfico 1: Acurácia por rodada com marcação de detecções
    ax = axes[0]
    rodadas = [h['rodada'] for h in hist_com_defesa]
    acuracias = [h['acuracia'] * 100 for h in hist_com_defesa]

    ax.plot(rodadas, acuracias, marker='o', linewidth=3, color='#06D6A0', markersize=10,
            markeredgewidth=2, markeredgecolor='white', label='Acurácia Global')

    # Marca quantos maliciosos foram detectados em cada rodada
    for h in hist_com_defesa:
        detectados = sum(1 for d in h['detectados'] if d['detectado'] and d['cliente'] in CLIENTES_MALICIOSOS)
        rodada = h['rodada']
        acuracia = h['acuracia'] * 100
        if detectados == 3:
            ax.scatter(rodada, acuracia, s=400, marker='*', color='gold', edgecolors='black', linewidth=2, zorder=5)
            ax.text(rodada, acuracia + 1, '✓ 3/3', ha='center', fontweight='bold', fontsize=10)

    ax.set_xlabel('Rodada Federada', fontweight='bold', fontsize=12)
    ax.set_ylabel('Acurácia (%)', fontweight='bold', fontsize=12)
    ax.set_title('Acurácia Global com Detecção de Maliciosos (★ = 3/3 detectados)',
                 fontweight='bold', fontsize=13)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=11)

    # Gráfico 2: Taxa de detecção de maliciosos por rodada
    ax = axes[1]
    deteccoes_por_rodada = []
    rodadas_det = []

    for h in hist_com_defesa:
        detectados = sum(1 for d in h['detectados'] if d['detectado'] and d['cliente'] in CLIENTES_MALICIOSOS)
        taxa = (detectados / 3) * 100
        deteccoes_por_rodada.append(taxa)
        rodadas_det.append(h['rodada'])

    bars = ax.bar(rodadas_det, deteccoes_por_rodada, color='#D62828', alpha=0.7, edgecolor='black', linewidth=1.5)

    for bar, taxa in zip(bars, deteccoes_por_rodada):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{taxa:.0f}%', ha='center', fontweight='bold', fontsize=11)

    ax.axhline(y=100, color='green', linestyle='--', linewidth=2, label='Detecção Perfeita (100%)')
    ax.set_xlabel('Rodada Federada', fontweight='bold', fontsize=12)
    ax.set_ylabel('Taxa de Detecção (%)', fontweight='bold', fontsize=12)
    ax.set_title('Taxa de Detecção de Clientes Maliciosos por Rodada',
                 fontweight='bold', fontsize=13)
    ax.set_ylim(0, 110)
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(fontsize=11)

    plt.suptitle('Efetividade da Detecção DBSCAN - 15 Clientes',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{RESULTADOS_DIR}dbscan_efetividade_deteccao.png', dpi=300, bbox_inches='tight')
    print(f"  Salvo: dbscan_efetividade_deteccao.png")
    plt.close()


def main():
    import os
    os.makedirs(RESULTADOS_DIR, exist_ok=True)

    print("=" * 80)
    print("ANÁLISE COMPLETA - DETECÇÃO DBSCAN")
    print("15 Clientes (3 Maliciosos, 12 Honestos)")
    print("=" * 80)

    # Carrega dados
    print("\n[1/3] Carregando dataset...")
    X, y = carregar_dataset()
    clientes, dados_val = distribuir_dados_15clientes(X, y)

    # Executa cenários
    print("\n[2/3] Executando cenários...")
    hist_normal = executar_federado_sem_defesa(clientes, dados_val, envenenado=False)
    hist_sem_defesa = executar_federado_sem_defesa(clientes, dados_val, envenenado=True)
    hist_com_defesa = executar_federado_com_defesa_dbscan(clientes, dados_val, envenenado=True)

    print(f"\n  Normal:       {hist_normal[-1]['acuracia']*100:.2f}%")
    print(f"  Sem defesa:   {hist_sem_defesa[-1]['acuracia']*100:.2f}%")
    print(f"  Com DBSCAN:   {hist_com_defesa[-1]['acuracia']*100:.2f}%")

    # Gera visualizações
    print("\n[3/3] Gerando visualizações...")
    plot_matriz_confusao_por_rodada(hist_sem_defesa, hist_com_defesa)
    plot_metricas_individuais_por_rodada(hist_sem_defesa, hist_com_defesa, hist_normal)
    plot_tabela_comparativa_final(hist_normal, hist_sem_defesa, hist_com_defesa)
    plot_degradacao_vs_recuperacao(hist_normal, hist_sem_defesa, hist_com_defesa)
    plot_acuracia_por_cliente_malicioso(hist_com_defesa)

    print("\n✅ 5 novas visualizações geradas!")
    print("=" * 80)


if __name__ == "__main__":
    main()
