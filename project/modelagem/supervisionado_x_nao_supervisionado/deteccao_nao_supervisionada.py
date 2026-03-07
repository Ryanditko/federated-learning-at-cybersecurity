"""
Detecção Não Supervisionada de Clientes Maliciosos no Servidor
==============================================================
Abordagem: K-Means clusteriza vetores de pesos de clientes.
           O cluster "anômalo" (menor, mais distante do centro) é marcado como suspeito.

Pipeline:
1. Coleta vetores de pesos de cada cliente após treino local (sem rótulos)
2. Aplica K-Means (k=2) nos vetores de pesos
3. Identifica cluster suspeito (menor tamanho / maior distância ao centroide global)
4. Exclui clientes do cluster suspeito da agregação FedAvg
5. Compara FedAvg normal vs FedAvg com defesa não supervisionada

Dataset: Bank Marketing (41k registros)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_auc_score, log_loss, silhouette_score
)
from sklearn.model_selection import train_test_split
from copy import deepcopy
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

RESULTADOS_DIR = "resultados/"
NUM_RODADAS = 12
NUM_CLIENTES = 3
CLIENTE_MALICIOSO = 3


# ==============================================================================
# CARREGAMENTO E PREPROCESSAMENTO
# ==============================================================================

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


def distribuir_dados(X, y):
    """Separa validação global e distribui entre clientes"""
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    X_c1, X_tmp, y_c1, y_tmp = train_test_split(X_tr, y_tr, test_size=0.6, random_state=42, stratify=y_tr)
    X_c2, X_c3, y_c2, y_c3 = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp)
    return [(X_c1, y_c1), (X_c2, y_c2), (X_c3, y_c3)], (X_val, y_val)


# ==============================================================================
# MODELO FEDERADO LOCAL
# ==============================================================================

def treinar_cliente(X, y, pesos_globais=None):
    """Treina modelo local e retorna pesos"""
    scaler = MinMaxScaler()
    X_sc = scaler.fit_transform(X)

    modelo = LogisticRegression(
        max_iter=500, solver='saga', class_weight='balanced',
        C=0.5, penalty='l2', random_state=42, warm_start=True
    )

    if pesos_globais is not None:
        try:
            modelo.fit(X_sc, y)
            modelo.coef_ = deepcopy(pesos_globais['coef'])
            modelo.intercept_ = deepcopy(pesos_globais['intercept'])
            modelo.classes_ = deepcopy(pesos_globais['classes'])
            modelo.fit(X_sc, y)
        except Exception:
            modelo.fit(X_sc, y)
    else:
        modelo.fit(X_sc, y)

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
    }


# ==============================================================================
# DEFESA NÃO SUPERVISIONADA COM K-MEANS
# ==============================================================================

def detectar_malicioso_kmeans(pesos_lista, n_clusters=2):
    """
    Aplica K-Means nos vetores de pesos.
    O cluster suspeito é o que tem menor tamanho OU maior distância ao centroide médio.
    Retorna: índices suspeitos, labels, centroides, pca para visualização
    """
    vetores = np.array([p['vetor'] for p in pesos_lista])

    # Reduz dimensionalidade para K-Means ser mais estável
    n_comp = min(10, vetores.shape[1], vetores.shape[0] - 1)
    pca = PCA(n_components=n_comp, random_state=42)
    vetores_pca = pca.fit_transform(vetores)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(vetores_pca)
    centroides = kmeans.cluster_centers_

    # Identifica cluster suspeito: menor número de membros
    tamanhos = [np.sum(labels == i) for i in range(n_clusters)]
    cluster_suspeito = int(np.argmin(tamanhos))

    # Clientes no cluster suspeito são marcados como maliciosos
    suspeitos = [i for i, label in enumerate(labels) if label == cluster_suspeito]

    # Distância de cada vetor ao centroide do cluster mais populoso
    cluster_honesto = 1 - cluster_suspeito
    distancias = np.linalg.norm(vetores_pca - centroides[cluster_honesto], axis=1)

    return suspeitos, labels, centroides, pca, vetores_pca, distancias


# ==============================================================================
# EXECUÇÃO DOS CENÁRIOS
# ==============================================================================

def executar_federado_sem_defesa(clientes, dados_val, envenenado=True):
    """FedAvg sem nenhuma defesa"""
    X_val, y_val = dados_val
    historico = []
    pesos_globais = treinar_cliente(clientes[0][0][:500], clientes[0][1][:500])

    for rodada in range(1, NUM_RODADAS + 1):
        pesos_locais = []
        for idx, (X_cli, y_cli) in enumerate(clientes, 1):
            pesos = treinar_cliente(X_cli, y_cli, pesos_globais)
            if envenenado and idx == CLIENTE_MALICIOSO:
                pesos = envenenar_pesos(pesos)
            pesos_locais.append(pesos)

        pesos_globais = agregar_fedavg(pesos_locais)
        metricas = avaliar_modelo_global(pesos_globais, X_val, y_val)
        metricas['rodada'] = rodada
        historico.append(metricas)

    return historico


def executar_federado_com_kmeans(clientes, dados_val, envenenado=True):
    """FedAvg com defesa K-Means: exclui cluster suspeito"""
    X_val, y_val = dados_val
    historico = []
    historico_kmeans = []  # Para visualizações
    pesos_globais = treinar_cliente(clientes[0][0][:500], clientes[0][1][:500])

    for rodada in range(1, NUM_RODADAS + 1):
        pesos_locais_todos = []

        for idx, (X_cli, y_cli) in enumerate(clientes, 1):
            pesos = treinar_cliente(X_cli, y_cli, pesos_globais)
            if envenenado and idx == CLIENTE_MALICIOSO:
                pesos = envenenar_pesos(pesos)
            pesos_locais_todos.append({'pesos': pesos, 'cliente': idx})

        # DEFESA K-MEANS: detecta cluster suspeito
        suspeitos, labels, centroides, pca_kmeans, vetores_pca, distancias = detectar_malicioso_kmeans(
            [p['pesos'] for p in pesos_locais_todos]
        )

        # Usa apenas clientes não suspeitos
        pesos_aprovados = [
            p['pesos'] for i, p in enumerate(pesos_locais_todos) if i not in suspeitos
        ]

        if len(pesos_aprovados) == 0:
            pesos_aprovados = [pesos_locais_todos[0]['pesos']]

        historico_kmeans.append({
            'rodada': rodada,
            'suspeitos': suspeitos,
            'labels': labels,
            'distancias': distancias,
            'vetores_pca': vetores_pca,
            'centroides': centroides,
            'n_aprovados': len(pesos_aprovados)
        })

        pesos_globais = agregar_fedavg(pesos_aprovados)
        metricas = avaliar_modelo_global(pesos_globais, X_val, y_val)
        metricas['rodada'] = rodada
        metricas['suspeitos'] = [p['cliente'] for i, p in enumerate(pesos_locais_todos) if i in suspeitos]
        historico.append(metricas)

    return historico, historico_kmeans


# ==============================================================================
# VISUALIZAÇÕES
# ==============================================================================

def plot_clusters_pca(historico_kmeans):
    """Visualiza os clusters K-Means na última rodada via PCA 2D"""
    import os

    # Usa rodada final e rodada 1 para comparação
    rodadas_plot = [0, len(historico_kmeans) // 2, -1]
    titulos = ['Rodada 1', f'Rodada {len(historico_kmeans)//2+1}', f'Rodada {len(historico_kmeans)} (Final)']

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for i, (r_idx, titulo) in enumerate(zip(rodadas_plot, titulos)):
        ax = axes[i]
        info = historico_kmeans[r_idx]
        vetores = info['vetores_pca']
        labels = info['labels']
        suspeitos = info['suspeitos']

        # Reduz para 2D para visualização
        if vetores.shape[1] >= 2:
            pca2d = PCA(n_components=2, random_state=42)
            v2d = pca2d.fit_transform(vetores)
        else:
            v2d = np.column_stack([vetores[:, 0], np.zeros(len(vetores))])

        cores = ['#D62828' if j in suspeitos else '#2E86AB' for j in range(len(labels))]
        marcadores = ['s' if j in suspeitos else 'o' for j in range(len(labels))]

        for j in range(len(v2d)):
            label_txt = f'Cliente {j+1} (SUSPEITO)' if j in suspeitos else f'Cliente {j+1}'
            ax.scatter(v2d[j, 0], v2d[j, 1], c=cores[j], marker=marcadores[j],
                       s=250, edgecolors='white', linewidths=2, zorder=5)
            ax.annotate(f'  C{j+1}', (v2d[j, 0], v2d[j, 1]), fontsize=11, fontweight='bold')

        ax.set_title(f'{titulo}', fontsize=13, fontweight='bold')
        ax.set_xlabel('PCA 1', fontweight='bold')
        ax.set_ylabel('PCA 2', fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Legenda
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#2E86AB', label='Honesto'),
            Patch(facecolor='#D62828', label='Suspeito (excluído)')
        ]
        ax.legend(handles=legend_elements, fontsize=9)

    plt.suptitle('K-Means: Clustering de Vetores de Pesos dos Clientes\nDetecção Não Supervisionada de Clientes Maliciosos',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{RESULTADOS_DIR}nao_supervisionado_clusters_pca.png', dpi=300, bbox_inches='tight')
    print(f"  Salvo: nao_supervisionado_clusters_pca.png")
    plt.close()


def plot_comparacao_defesa(hist_sem_defesa, hist_com_defesa, hist_normal):
    """Compara 3 cenários: normal, atacado sem defesa, atacado com K-Means"""
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
                label='Normal (sem ataque)', markersize=7)
        ax.plot(rodadas, v_sem, marker='s', linewidth=3, color='#D62828',
                label='Atacado (sem defesa)', markersize=7, linestyle='--')
        ax.plot(rodadas, v_com, marker='^', linewidth=3, color='#118AB2',
                label='Atacado (com defesa K-Means)', markersize=7)

        ax.fill_between(rodadas, v_normal, alpha=0.1, color='#2E86AB')
        ax.fill_between(rodadas, v_com, alpha=0.1, color='#118AB2')

        ax.set_title(titulo, fontsize=13, fontweight='bold')
        ax.set_xlabel('Rodada Federada', fontweight='bold')
        ax.set_ylabel(titulo, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, linestyle='--')

    plt.suptitle('Comparação: Normal vs Atacado vs Defesa K-Means\nBank Marketing - Aprendizado Federado',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{RESULTADOS_DIR}nao_supervisionado_comparacao_cenarios.png', dpi=300, bbox_inches='tight')
    print(f"  Salvo: nao_supervisionado_comparacao_cenarios.png")
    plt.close()


def plot_distancias_anomalia(historico_kmeans):
    """Mostra distância ao centroide honesto por cliente ao longo das rodadas"""
    fig, ax = plt.subplots(figsize=(14, 6))

    for cli in range(NUM_CLIENTES):
        distancias = [h['distancias'][cli] for h in historico_kmeans]
        rodadas = [h['rodada'] for h in historico_kmeans]

        cor = '#D62828' if cli + 1 == CLIENTE_MALICIOSO else '#2E86AB' if cli == 0 else '#06D6A0'
        estilo = '--' if cli + 1 == CLIENTE_MALICIOSO else '-'
        label = f'Cliente {cli+1} (MALICIOSO)' if cli + 1 == CLIENTE_MALICIOSO else f'Cliente {cli+1} (honesto)'

        ax.plot(rodadas, distancias, marker='o', linewidth=3, color=cor,
                linestyle=estilo, label=label, markersize=8)

    ax.set_xlabel('Rodada Federada', fontweight='bold', fontsize=12)
    ax.set_ylabel('Distância ao Centroide do Cluster Honesto', fontweight='bold', fontsize=12)
    ax.set_title('Distância ao Centroide por Cliente\nDetecção Não Supervisionada (K-Means)',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(f'{RESULTADOS_DIR}nao_supervisionado_distancias.png', dpi=300, bbox_inches='tight')
    print(f"  Salvo: nao_supervisionado_distancias.png")
    plt.close()


def plot_eficacia_recuperacao(hist_sem_defesa, hist_com_defesa, hist_normal):
    """Mostra o quanto a defesa K-Means recuperou o desempenho"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    metricas = ['acuracia', 'f1', 'auc']
    titulos = ['Acurácia (%)', 'F1-Score (%)', 'AUC-ROC']

    for idx, (met, titulo) in enumerate(zip(metricas, titulos)):
        ax = axes[idx]
        fator = 100 if met in ['acuracia', 'f1'] else 1

        v_normal = hist_normal[-1][met] * fator
        v_sem = hist_sem_defesa[-1][met] * fator
        v_com = hist_com_defesa[-1][met] * fator

        categorias = ['Normal\n(sem ataque)', 'Atacado\n(sem defesa)', 'Atacado\n(K-Means)']
        valores = [v_normal, v_sem, v_com]
        cores = ['#2E86AB', '#D62828', '#118AB2']

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

    plt.suptitle('Eficácia da Defesa K-Means (Não Supervisionada)\nBank Marketing - Aprendizado Federado',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{RESULTADOS_DIR}nao_supervisionado_eficacia_recuperacao.png', dpi=300, bbox_inches='tight')
    print(f"  Salvo: nao_supervisionado_eficacia_recuperacao.png")
    plt.close()


def plot_deteccoes_por_rodada(historico_kmeans):
    """Mostra quais clientes foram detectados como suspeitos em cada rodada"""
    fig, ax = plt.subplots(figsize=(14, 5))

    for rodada_info in historico_kmeans:
        rodada = rodada_info['rodada']
        suspeitos = rodada_info['suspeitos']

        for cli in range(1, NUM_CLIENTES + 1):
            detectado = (cli - 1) in suspeitos
            cor = '#D62828' if detectado else '#06D6A0'
            marcador = 'X' if detectado else 'o'
            ax.scatter(rodada, cli, c=cor, marker=marcador, s=200,
                       zorder=5, edgecolors='white', linewidths=1.5)

    ax.set_xlabel('Rodada Federada', fontweight='bold', fontsize=12)
    ax.set_ylabel('Cliente', fontweight='bold', fontsize=12)
    ax.set_yticks([1, 2, 3])
    ax.set_yticklabels(['Cliente 1\n(honesto)', 'Cliente 2\n(honesto)', f'Cliente 3\n(MALICIOSO)'])
    ax.set_title('Detecção de Suspeitos por Rodada - K-Means\nX = Detectado como Suspeito | O = Aprovado',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#D62828', label='Detectado (excluído)'),
        Patch(facecolor='#06D6A0', label='Aprovado (usado na agregação)')
    ]
    ax.legend(handles=legend_elements, fontsize=11, loc='upper right')

    plt.tight_layout()
    plt.savefig(f'{RESULTADOS_DIR}nao_supervisionado_deteccoes_por_rodada.png', dpi=300, bbox_inches='tight')
    print(f"  Salvo: nao_supervisionado_deteccoes_por_rodada.png")
    plt.close()


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    import os
    os.makedirs(RESULTADOS_DIR, exist_ok=True)

    print("=" * 70)
    print("DETECÇÃO NÃO SUPERVISIONADA DE CLIENTES MALICIOSOS (K-MEANS)")
    print("Bank Marketing Dataset - Aprendizado Federado")
    print("=" * 70)

    # 1. Carrega dados
    print("\n[1/4] Carregando dataset...")
    X, y = carregar_dataset()
    print(f"  Total: {len(X)} amostras | {X.shape[1]} features")
    clientes, dados_val = distribuir_dados(X, y)

    # 2. Executa os 3 cenários
    print("\n[2/4] Executando cenários federados...")

    print("  Cenário Normal (sem ataque)...")
    hist_normal = executar_federado_sem_defesa(clientes, dados_val, envenenado=False)

    print("  Cenário Atacado sem defesa...")
    hist_sem_defesa = executar_federado_sem_defesa(clientes, dados_val, envenenado=True)

    print("  Cenário Atacado com defesa K-Means...")
    hist_com_defesa, hist_kmeans = executar_federado_com_kmeans(clientes, dados_val, envenenado=True)

    # Resultados
    print("\n" + "=" * 70)
    print("RESULTADOS FINAIS (Rodada 12)")
    print("=" * 70)
    print(f"  Normal:       Acuracia={hist_normal[-1]['acuracia']*100:.2f}%  F1={hist_normal[-1]['f1']*100:.2f}%  AUC={hist_normal[-1]['auc']:.4f}")
    print(f"  Sem defesa:   Acuracia={hist_sem_defesa[-1]['acuracia']*100:.2f}%  F1={hist_sem_defesa[-1]['f1']*100:.2f}%  AUC={hist_sem_defesa[-1]['auc']:.4f}")
    print(f"  Com K-Means:  Acuracia={hist_com_defesa[-1]['acuracia']*100:.2f}%  F1={hist_com_defesa[-1]['f1']*100:.2f}%  AUC={hist_com_defesa[-1]['auc']:.4f}")

    # Detecções
    deteccoes_corretas = sum(
        1 for h in hist_kmeans if (CLIENTE_MALICIOSO - 1) in h['suspeitos']
    )
    print(f"\n  Deteccoes corretas do cliente malicioso: {deteccoes_corretas}/{NUM_RODADAS} rodadas")

    # 3. Visualizações
    print("\n[3/4] Gerando visualizações...")
    plot_clusters_pca(hist_kmeans)
    plot_comparacao_defesa(hist_sem_defesa, hist_com_defesa, hist_normal)
    plot_distancias_anomalia(hist_kmeans)
    plot_eficacia_recuperacao(hist_sem_defesa, hist_com_defesa, hist_normal)
    plot_deteccoes_por_rodada(hist_kmeans)

    print("\n[4/4] Concluido!")
    print(f"\n✅ 5 visualizações salvas em: {RESULTADOS_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
