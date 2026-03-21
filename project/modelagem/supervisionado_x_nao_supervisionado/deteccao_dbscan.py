"""
Detecção com DBSCAN de Clientes Maliciosos no Servidor
=======================================================
Abordagem: DBSCAN (Density-Based Spatial Clustering) para detectar
           outliers nos vetores de pesos dos clientes

Pipeline:
1. Simula rodadas federadas com 15 clientes (12 honestos, 3 maliciosos)
2. Coleta vetores de pesos de cada cliente após treino local
3. Aplica DBSCAN para encontrar outliers (maliciosos)
4. No servidor, usa DBSCAN para detectar e excluir maliciosos
5. Compara FedAvg normal vs FedAvg com defesa DBSCAN

Dataset: Bank Marketing (41k registros)
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
CLIENTES_MALICIOSOS = [13, 14, 15]  # Últimos 3 clientes são maliciosos


# ==============================================================================
# CARREGAMENTO E PREPROCESSAMENTO
# ==============================================================================

def carregar_dataset():
    """Carrega e preprocessa o dataset Bank Marketing completo"""
    caminho = r"c:\Users\Administrador\Faculdade\Iniciação-cientifica\project\data\bank-marketing\bank.csv"
    df = pd.read_csv(caminho, sep=';')

    df = df.drop_duplicates()

    # Imputa unknowns
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

    # Feature engineering
    if 'duration' in df.columns:
        df['duration_log'] = np.log1p(df['duration'])
        df['duration_high'] = (df['duration'] > 300).astype(int)
    if 'pdays' in df.columns:
        df['previously_contacted'] = (df['pdays'] != 999).astype(int)
    if 'campaign' in df.columns:
        df['campaign_low'] = (df['campaign'] <= 2).astype(int)

    # One-hot encoding
    cat_cols = [c for c in df.select_dtypes(include='object').columns if c != 'y']
    for col in cat_cols:
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
        df = pd.concat([df, dummies], axis=1).drop(col, axis=1)

    df['y'] = df['y'].map({'yes': 1, 'no': 0})

    X = df.drop('y', axis=1).values
    y = df['y'].values
    return X, y


def distribuir_dados_15clientes(X, y):
    """Distribui dados entre 15 clientes (estratificado)"""
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    clientes = []
    X_restante, y_restante = X_tr, y_tr

    # Distribui 75% dos dados entre 15 clientes (~5% cada)
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


# ==============================================================================
# MODELO FEDERADO LOCAL
# ==============================================================================

def treinar_cliente(X, y, pesos_globais=None, rodada=1, seed=42):
    """Treina modelo local com mini-batch rotativo por rodada"""
    rng = np.random.RandomState(seed + rodada * 7)
    n = max(150, int(len(X) * 0.40))
    idx = rng.choice(len(X), size=n, replace=False)
    X_sub, y_sub = X[idx], y[idx]

    scaler = MinMaxScaler()
    X_sc = scaler.fit_transform(X_sub)

    iters_por_rodada = min(10 + rodada * 5, 80)

    modelo = LogisticRegression(
        max_iter=iters_por_rodada,
        solver='saga',
        class_weight='balanced',
        C=0.5,
        penalty='l2',
        random_state=seed,
        warm_start=True
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
    """Agrega pesos com FedAvg"""
    return {
        'coef': np.mean([p['coef'] for p in lista_pesos], axis=0),
        'intercept': np.mean([p['intercept'] for p in lista_pesos], axis=0),
        'classes': lista_pesos[0]['classes']
    }


def avaliar_modelo_global(pesos_globais, X_val, y_val):
    """Avalia modelo global no conjunto de validação"""
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


# ==============================================================================
# DETECÇÃO COM DBSCAN
# ==============================================================================

def detectar_maliciosos_dbscan(pesos_lista, pca=None, eps=0.8, min_samples=2):
    """
    Usa DBSCAN para detectar outliers nos vetores de pesos.
    Retorna índices dos clientes detectados como maliciosos (outliers).
    """
    # Extrai vetores
    vetores = np.array([p['vetor'] for p in pesos_lista])

    # Reduz dimensionalidade se necessário
    if pca is None:
        pca = PCA(n_components=min(15, vetores.shape[1], vetores.shape[0]-1), random_state=42)
        vetores_pca = pca.fit_transform(vetores)
    else:
        vetores_pca = pca.transform(vetores)

    # Aplica DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(vetores_pca)

    # Outliers têm label -1
    suspeitos = [i for i, label in enumerate(labels) if label == -1]

    return suspeitos, labels, vetores_pca, pca


def executar_federado_com_defesa_dbscan(clientes, dados_val, envenenado=True):
    """Executa aprendizado federado com detecção DBSCAN"""
    X_val, y_val = dados_val
    historico = []

    pesos_globais = treinar_cliente(clientes[0][0][:300], clientes[0][1][:300], rodada=0)
    pca = None

    for rodada in range(1, NUM_RODADAS + 1):
        pesos_locais = []
        detectados = []

        # Coleta pesos de todos os clientes
        todos_pesos = []
        for idx, (X_cli, y_cli) in enumerate(clientes, 1):
            pesos = treinar_cliente(X_cli, y_cli, pesos_globais, rodada=rodada, seed=42 + idx)

            if envenenado and idx in CLIENTES_MALICIOSOS:
                pesos = envenenar_pesos(pesos)

            todos_pesos.append(pesos)

        # Detecta maliciosos com DBSCAN
        suspeitos_indices, labels, vetores_pca, pca = detectar_maliciosos_dbscan(todos_pesos, pca, eps=1.0, min_samples=3)

        # Registra detecções
        for idx in range(NUM_CLIENTES):
            e_malicioso = idx in suspeitos_indices
            detectados.append({
                'cliente': idx + 1,
                'detectado': e_malicioso,
                'label_cluster': labels[idx]
            })

            if not e_malicioso:
                pesos_locais.append(todos_pesos[idx])

        # Se todos foram detectados, usa todos (fallback)
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
    """Executa aprendizado federado SEM defesa (baseline)"""
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
# VISUALIZAÇÕES
# ==============================================================================

def plot_dbscan_clusters_15clientes(hist_com_defesa):
    """Visualiza clusters DBSCAN na última rodada"""
    deteccoes_final = hist_com_defesa[-1]['detectados']
    labels = [d['label_cluster'] for d in deteccoes_final]
    clientes_nums = [d['cliente'] for d in deteccoes_final]

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Gráfico 1: Scatter com cores por cliente
    ax = axes[0]
    colors = ['red' if c in CLIENTES_MALICIOSOS else 'blue' for c in clientes_nums]
    markers = ['X' if labels[i] == -1 else 'o' for i in range(len(clientes_nums))]

    for i, (cli, col, mark) in enumerate(zip(clientes_nums, colors, markers)):
        ax.scatter(i, 0, s=400, c=col, marker=mark, alpha=0.7, edgecolors='black', linewidth=2)
        ax.text(i, -0.1, f'C{cli}', ha='center', fontweight='bold', fontsize=11)

    ax.set_xlim(-1, NUM_CLIENTES)
    ax.set_ylim(-0.3, 0.3)
    ax.set_title('Detecção DBSCAN - Última Rodada\n(X = Outlier/Malicioso, O = Normal)', 
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Cliente', fontweight='bold')
    ax.set_xticks(range(NUM_CLIENTES))
    ax.set_yticks([])
    ax.grid(True, alpha=0.3, axis='x')

    # Gráfico 2: Tabela de detecções
    ax = axes[1]
    ax.axis('off')

    table_data = []
    for d in deteccoes_final:
        status = '✓ DETECTADO' if d['detectado'] else '✗ Normal'
        tipo = 'Malicioso' if d['cliente'] in CLIENTES_MALICIOSOS else 'Honesto'
        table_data.append([f"C{d['cliente']}", tipo, status, str(d['label_cluster'])])

    colLabels = ['Cliente', 'Tipo Real', 'Status', 'Cluster']
    table = ax.table(cellText=table_data, colLabels=colLabels, cellLoc='center',
                     loc='center', colWidths=[0.15, 0.25, 0.3, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)

    for i in range(len(colLabels)):
        table[(0, i)].set_facecolor('#2E86AB')
        table[(0, i)].set_text_props(color='white', fontweight='bold')

    for i in range(1, len(table_data) + 1):
        d = deteccoes_final[i-1]
        if d['detectado']:
            table[(i, 2)].set_facecolor('#FFB6C6')  # Rosa para detectado
        else:
            table[(i, 2)].set_facecolor('#C6FFB6')  # Verde para normal

    ax.set_title('Resumo de Detecções', fontsize=13, fontweight='bold', pad=20)

    plt.suptitle('Detecção DBSCAN de Clientes Maliciosos\n15 Clientes (3 Maliciosos, 12 Honestos)',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{RESULTADOS_DIR}dbscan_clusters_15clientes.png', dpi=300, bbox_inches='tight')
    print(f"  Salvo: dbscan_clusters_15clientes.png")
    plt.close()


def plot_comparacao_dbscan(hist_sem_defesa, hist_com_defesa, hist_normal):
    """Compara 3 cenários: normal, atacado sem defesa, atacado com defesa DBSCAN"""
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    metricas = ['acuracia', 'f1', 'auc', 'loss']
    titulos = ['Acurácia (%)', 'F1-Score (%)', 'AUC-ROC', 'Loss']

    for idx, (met, titulo) in enumerate(zip(metricas, titulos)):
        ax = axes[idx // 2, idx % 2]

        rodadas = [h['rodada'] for h in hist_normal]
        fator = 100 if met in ['acuracia', 'f1'] else 1

        v_normal = [h[met] * fator for h in hist_normal]
        v_sem = [h[met] * fator for h in hist_sem_defesa]
        v_com = [h[met] * fator for h in hist_com_defesa]

        ax.plot(rodadas, v_normal, marker='o', linewidth=3, color='#2E86AB',
                label='Normal (sem ataque)', markersize=8)
        ax.plot(rodadas, v_sem, marker='s', linewidth=3, color='#D62828',
                label='Atacado (sem defesa)', markersize=8, linestyle='--')
        ax.plot(rodadas, v_com, marker='^', linewidth=3, color='#06D6A0',
                label='Atacado (com defesa DBSCAN)', markersize=8)

        ax.fill_between(rodadas, v_normal, alpha=0.1, color='#2E86AB')
        ax.fill_between(rodadas, v_com, alpha=0.1, color='#06D6A0')

        ax.set_title(titulo, fontsize=13, fontweight='bold')
        ax.set_xlabel('Rodada Federada', fontweight='bold')
        ax.set_ylabel(titulo, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')

    plt.suptitle('Comparação: Normal vs Atacado vs Defesa DBSCAN\n15 Clientes (3 Maliciosos, 12 Honestos)',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{RESULTADOS_DIR}dbscan_comparacao_15clientes.png', dpi=300, bbox_inches='tight')
    print(f"  Salvo: dbscan_comparacao_15clientes.png")
    plt.close()


def plot_eficacia_dbscan(hist_sem_defesa, hist_com_defesa, hist_normal):
    """Mostra eficácia da defesa DBSCAN"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    metricas = ['acuracia', 'f1', 'auc']
    titulos = ['Acurácia (%)', 'F1-Score (%)', 'AUC-ROC']

    for idx, (met, titulo) in enumerate(zip(metricas, titulos)):
        ax = axes[idx]
        fator = 100 if met in ['acuracia', 'f1'] else 1

        v_normal = hist_normal[-1][met] * fator
        v_sem = hist_sem_defesa[-1][met] * fator
        v_com = hist_com_defesa[-1][met] * fator

        categorias = ['Normal\n(sem ataque)', 'Atacado\n(sem defesa)', 'Atacado\n(com DBSCAN)']
        valores = [v_normal, v_sem, v_com]
        cores = ['#2E86AB', '#D62828', '#06D6A0']

        bars = ax.bar(categorias, valores, color=cores, alpha=0.85, width=0.5,
                      edgecolor='white', linewidth=2)

        for bar, val in zip(bars, valores):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f'{val:.2f}{"%" if met != "auc" else ""}',
                    ha='center', fontweight='bold', fontsize=12)

        recuperacao = ((v_com - v_sem) / (v_normal - v_sem + 1e-9)) * 100 if v_normal != v_sem else 100
        ax.set_title(f'{titulo}\nRecuperação: {recuperacao:.1f}%', fontsize=13, fontweight='bold')
        ax.set_ylabel(titulo, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, max(valores) * 1.2)

    plt.suptitle('Eficácia da Defesa DBSCAN\n15 Clientes (3 Maliciosos, 12 Honestos)',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{RESULTADOS_DIR}dbscan_eficacia_15clientes.png', dpi=300, bbox_inches='tight')
    print(f"  Salvo: dbscan_eficacia_15clientes.png")
    plt.close()


def plot_deteccoes_por_rodada_dbscan(hist_com_defesa):
    """Mostra status de detecção por cliente ao longo das rodadas"""
    fig, ax = plt.subplots(figsize=(16, 8))

    # Prepara dados
    for cli in range(1, NUM_CLIENTES + 1):
        deteccoes = []
        rodadas = []

        for h in hist_com_defesa:
            for d in h['detectados']:
                if d['cliente'] == cli:
                    deteccoes.append(1 if d['detectado'] else 0)
                    rodadas.append(h['rodada'])

        cor = '#D62828' if cli in CLIENTES_MALICIOSOS else '#2E86AB'
        estilo = '--' if cli in CLIENTES_MALICIOSOS else '-'
        label = f'Cliente {cli} (MALICIOSO)' if cli in CLIENTES_MALICIOSOS else f'Cliente {cli}'

        ax.plot(rodadas, deteccoes, marker='o', linewidth=2.5, color=cor,
                linestyle=estilo, label=label, markersize=6, alpha=0.8)

    # Destacar zona de detecção
    ax.fill_between(range(1, NUM_RODADAS + 1), 0, 1, alpha=0.05, color='red')

    ax.set_xlabel('Rodada Federada', fontweight='bold', fontsize=12)
    ax.set_ylabel('Status de Detecção', fontweight='bold', fontsize=12)
    ax.set_title('Detecção DBSCAN por Cliente ao Longo das Rodadas\n15 Clientes (3 Maliciosos)',
                 fontsize=14, fontweight='bold')
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Normal', 'Detectado'])
    ax.legend(fontsize=8, loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)
    ax.grid(True, alpha=0.3, linestyle='--', axis='x')
    ax.set_ylim(-0.2, 1.2)

    plt.tight_layout()
    plt.savefig(f'{RESULTADOS_DIR}dbscan_deteccoes_por_rodada_15clientes.png', dpi=300, bbox_inches='tight')
    print(f"  Salvo: dbscan_deteccoes_por_rodada_15clientes.png")
    plt.close()


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    import os
    os.makedirs(RESULTADOS_DIR, exist_ok=True)

    print("=" * 80)
    print("DETECÇÃO COM DBSCAN DE CLIENTES MALICIOSOS")
    print("15 Clientes (3 Maliciosos, 12 Honestos) - Bank Marketing Dataset")
    print("=" * 80)

    # 1. Carrega dados
    print("\n[1/4] Carregando dataset...")
    X, y = carregar_dataset()
    print(f"  Total: {len(X)} amostras | {X.shape[1]} features")
    print(f"  Classe 0 (Nao): {(y==0).sum()} ({(y==0).mean()*100:.1f}%)")
    print(f"  Classe 1 (Sim): {(y==1).sum()} ({(y==1).mean()*100:.1f}%)")

    clientes, dados_val = distribuir_dados_15clientes(X, y)

    print(f"\n  ✓ Distribuído em {NUM_CLIENTES} clientes")
    print(f"    - Clientes maliciosos: {CLIENTES_MALICIOSOS}")
    for idx, (X_cli, y_cli) in enumerate(clientes, 1):
        print(f"    Cliente {idx:2d}: {len(X_cli):,} amostras")

    # 2. Executa os 3 cenários
    print("\n[2/4] Executando cenários federados...")

    print("  Cenário Normal (sem ataque)...")
    hist_normal = executar_federado_sem_defesa(clientes, dados_val, envenenado=False)

    print("  Cenário Atacado (sem defesa)...")
    hist_sem_defesa = executar_federado_sem_defesa(clientes, dados_val, envenenado=True)

    print("  Cenário Atacado (com defesa DBSCAN)...")
    hist_com_defesa = executar_federado_com_defesa_dbscan(clientes, dados_val, envenenado=True)

    # Resultados
    print("\n" + "=" * 80)
    print("RESULTADOS FINAIS (Rodada 12)")
    print("=" * 80)
    print(f"  Normal:        Acuracia={hist_normal[-1]['acuracia']*100:.2f}%  F1={hist_normal[-1]['f1']*100:.2f}%  AUC={hist_normal[-1]['auc']:.4f}")
    print(f"  Sem defesa:    Acuracia={hist_sem_defesa[-1]['acuracia']*100:.2f}%  F1={hist_sem_defesa[-1]['f1']*100:.2f}%  AUC={hist_sem_defesa[-1]['auc']:.4f}")
    print(f"  Com DBSCAN:    Acuracia={hist_com_defesa[-1]['acuracia']*100:.2f}%  F1={hist_com_defesa[-1]['f1']*100:.2f}%  AUC={hist_com_defesa[-1]['auc']:.4f}")

    # Conta detecções corretas
    deteccoes_corretas = 0
    deteccoes_total = 0
    for h in hist_com_defesa:
        for d in h['detectados']:
            if d['cliente'] in CLIENTES_MALICIOSOS:
                deteccoes_total += 1
                if d['detectado']:
                    deteccoes_corretas += 1

    print(f"\n  Detecções corretas de maliciosos: {deteccoes_corretas}/{deteccoes_total} ({deteccoes_corretas/deteccoes_total*100:.1f}%)")

    # 3. Visualizações
    print("\n[3/4] Gerando visualizações...")
    plot_dbscan_clusters_15clientes(hist_com_defesa)
    plot_comparacao_dbscan(hist_sem_defesa, hist_com_defesa, hist_normal)
    plot_eficacia_dbscan(hist_sem_defesa, hist_com_defesa, hist_normal)
    plot_deteccoes_por_rodada_dbscan(hist_com_defesa)

    print("\n[4/4] Concluído!")
    print("=" * 80)
    print("✅ 4 novas visualizações salvas em:", RESULTADOS_DIR)
    print("   - dbscan_clusters_15clientes.png")
    print("   - dbscan_comparacao_15clientes.png")
    print("   - dbscan_eficacia_15clientes.png")
    print("   - dbscan_deteccoes_por_rodada_15clientes.png")
    print("=" * 80)


if __name__ == "__main__":
    main()
