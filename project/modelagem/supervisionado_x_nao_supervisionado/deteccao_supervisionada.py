"""
Detecção Supervisionada de Clientes Maliciosos no Servidor
===========================================================
Abordagem: Regressão Logística treinada para classificar
           vetores de pesos de clientes como bom (0) ou malicioso (1)

Pipeline:
1. Simula rodadas federadas com clientes honestos e maliciosos
2. Coleta vetores de pesos de cada cliente após treino local
3. Treina classificador supervisionado (X = pesos, y = 0/1)
4. No servidor, usa o classificador para detectar e excluir maliciosos
5. Compara FedAvg normal vs FedAvg com defesa supervisionada

Dataset: Bank Marketing (41k registros)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, roc_auc_score, log_loss
)
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
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


def distribuir_dados(X, y):
    """Separa validação global e distribui entre clientes"""
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    X_c1, X_tmp, y_c1, y_tmp = train_test_split(X_tr, y_tr, test_size=0.6, random_state=42, stratify=y_tr)
    X_c2, X_c3, y_c2, y_c3 = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp)

    clientes = [(X_c1, y_c1), (X_c2, y_c2), (X_c3, y_c3)]
    return clientes, (X_val, y_val)


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
        'y_pred': y_pred
    }


# ==============================================================================
# GERAÇÃO DO DATASET DE TREINAMENTO DO DETECTOR SUPERVISIONADO
# ==============================================================================

def gerar_dataset_detector(clientes, n_simulacoes=200):
    """
    Simula múltiplas rodadas para coletar vetores de pesos rotulados.
    Retorna X_det (vetores de pesos), y_det (0=honesto, 1=malicioso)
    """
    print("\n[SUPERVISIONADO] Gerando dataset para treinar o detector...")

    X_det, y_det = [], []

    # Inicializa pesos globais com cliente 1
    pesos_globais = treinar_cliente(clientes[0][0][:500], clientes[0][1][:500])

    for sim in range(n_simulacoes):
        random_state = sim

        for idx, (X_cli, y_cli) in enumerate(clientes, 1):
            # Amostra aleatória do cliente para simular variação
            n = min(len(X_cli), 300)
            idxs = np.random.choice(len(X_cli), n, replace=False)
            pesos = treinar_cliente(X_cli[idxs], y_cli[idxs], pesos_globais)

            # Define se é malicioso: cliente 3 em ~50% das simulações
            e_malicioso = (idx == CLIENTE_MALICIOSO) and (sim % 2 == 0)

            if e_malicioso:
                pesos = envenenar_pesos(pesos)

            X_det.append(pesos['vetor'])
            y_det.append(1 if e_malicioso else 0)

    X_det = np.array(X_det)
    y_det = np.array(y_det)

    print(f"  Vetores coletados: {len(X_det)}")
    print(f"  Honestos (0): {(y_det == 0).sum()} | Maliciosos (1): {(y_det == 1).sum()}")

    return X_det, y_det


# ==============================================================================
# TREINAMENTO DO DETECTOR SUPERVISIONADO
# ==============================================================================

def treinar_detector_supervisionado(X_det, y_det):
    """Treina classificador supervisionado nos vetores de pesos"""
    print("\n[SUPERVISIONADO] Treinando detector de clientes maliciosos...")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_det, y_det, test_size=0.3, random_state=42, stratify=y_det
    )

    # Reduz dimensionalidade para generalizar melhor
    pca = PCA(n_components=min(20, X_tr.shape[1]), random_state=42)
    X_tr_pca = pca.fit_transform(X_tr)
    X_te_pca = pca.transform(X_te)

    detector = LogisticRegression(
        max_iter=1000, solver='saga', class_weight='balanced',
        C=1.0, random_state=42
    )
    detector.fit(X_tr_pca, y_tr)

    y_pred = detector.predict(X_te_pca)
    acc = accuracy_score(y_te, y_pred)
    f1 = f1_score(y_te, y_pred, zero_division=0)

    print(f"  Acuracia do detector: {acc*100:.2f}%")
    print(f"  F1-Score do detector: {f1*100:.2f}%")
    print(f"\n{classification_report(y_te, y_pred, target_names=['Honesto', 'Malicioso'])}")

    return detector, pca, {'acc': acc, 'f1': f1, 'y_te': y_te, 'y_pred': y_pred, 'cm': confusion_matrix(y_te, y_pred)}


# ==============================================================================
# FEDERADO COM DEFESA SUPERVISIONADA
# ==============================================================================

def executar_federado_com_defesa(clientes, dados_val, detector, pca, envenenado=True):
    """Executa aprendizado federado usando detector supervisionado para excluir maliciosos"""
    X_val, y_val = dados_val
    historico = []

    pesos_globais = treinar_cliente(clientes[0][0][:500], clientes[0][1][:500])

    for rodada in range(1, NUM_RODADAS + 1):
        pesos_locais = []
        detectados = []

        for idx, (X_cli, y_cli) in enumerate(clientes, 1):
            pesos = treinar_cliente(X_cli, y_cli, pesos_globais)

            # Aplica ataque no cliente malicioso
            if envenenado and idx == CLIENTE_MALICIOSO:
                pesos = envenenar_pesos(pesos)

            # DEFESA: classifica o vetor de pesos
            vetor_pca = pca.transform([pesos['vetor']])
            prob_malicioso = detector.predict_proba(vetor_pca)[0][1]
            e_malicioso = prob_malicioso > 0.5

            detectados.append({'cliente': idx, 'prob_malicioso': prob_malicioso, 'detectado': e_malicioso})

            # Só adiciona se não for detectado como malicioso
            if not e_malicioso:
                pesos_locais.append(pesos)

        # Agrega apenas pesos honestos
        if len(pesos_locais) == 0:
            pesos_locais = [treinar_cliente(clientes[0][0][:100], clientes[0][1][:100])]

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


# ==============================================================================
# VISUALIZAÇÕES
# ==============================================================================

def plot_detector_cm(metricas_detector):
    """Matriz de confusão do detector supervisionado"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Matriz de confusão
    ax = axes[0]
    sns.heatmap(metricas_detector['cm'], annot=True, fmt='d', cmap='Blues',
                xticklabels=['Honesto', 'Malicioso'],
                yticklabels=['Honesto', 'Malicioso'], ax=ax, cbar=False,
                annot_kws={'size': 16, 'weight': 'bold'})
    ax.set_title('Matriz de Confusão\nDetector Supervisionado', fontsize=14, fontweight='bold')
    ax.set_xlabel('Predito', fontweight='bold')
    ax.set_ylabel('Real', fontweight='bold')

    # Métricas do detector
    ax = axes[1]
    ax.axis('off')
    metricas_txt = [
        ['Métrica', 'Valor'],
        ['Acurácia', f"{metricas_detector['acc']*100:.2f}%"],
        ['F1-Score', f"{metricas_detector['f1']*100:.2f}%"],
        ['Verdadeiros Positivos', str(metricas_detector['cm'][1, 1])],
        ['Falsos Negativos', str(metricas_detector['cm'][1, 0])],
        ['Falsos Positivos', str(metricas_detector['cm'][0, 1])],
    ]
    table = ax.table(cellText=metricas_txt[1:], colLabels=metricas_txt[0],
                     cellLoc='center', loc='center', colWidths=[0.5, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2.5)
    for i in range(2):
        table[(0, i)].set_facecolor('#2E86AB')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    for i in range(1, len(metricas_txt)):
        table[(i, 0)].set_facecolor('#E8F4F8')
        table[(i, 1)].set_facecolor('#D4EDDA')

    ax.set_title('Desempenho do Detector', fontsize=14, fontweight='bold')

    plt.suptitle('Detecção Supervisionada de Clientes Maliciosos\nBank Marketing - Aprendizado Federado',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{RESULTADOS_DIR}supervisionado_detector_cm.png', dpi=300, bbox_inches='tight')
    print(f"  Salvo: supervisionado_detector_cm.png")
    plt.close()


def plot_comparacao_defesa(hist_sem_defesa, hist_com_defesa, hist_normal):
    """Compara 3 cenários: normal, atacado sem defesa, atacado com defesa"""
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
        ax.plot(rodadas, v_com, marker='^', linewidth=3, color='#06D6A0',
                label='Atacado (com defesa supervisionada)', markersize=7)

        ax.fill_between(rodadas, v_normal, alpha=0.1, color='#2E86AB')
        ax.fill_between(rodadas, v_com, alpha=0.1, color='#06D6A0')

        ax.set_title(titulo, fontsize=13, fontweight='bold')
        ax.set_xlabel('Rodada Federada', fontweight='bold')
        ax.set_ylabel(titulo, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, linestyle='--')

    plt.suptitle('Comparação: Normal vs Atacado vs Defesa Supervisionada\nBank Marketing - Aprendizado Federado',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{RESULTADOS_DIR}supervisionado_comparacao_cenarios.png', dpi=300, bbox_inches='tight')
    print(f"  Salvo: supervisionado_comparacao_cenarios.png")
    plt.close()


def plot_eficacia_recuperacao(hist_sem_defesa, hist_com_defesa, hist_normal):
    """Mostra o quanto a defesa recuperou o desempenho"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    metricas = ['acuracia', 'f1', 'auc']
    titulos = ['Acurácia (%)', 'F1-Score (%)', 'AUC-ROC']

    for idx, (met, titulo) in enumerate(zip(metricas, titulos)):
        ax = axes[idx]
        fator = 100 if met in ['acuracia', 'f1'] else 1

        v_normal = hist_normal[-1][met] * fator
        v_sem = hist_sem_defesa[-1][met] * fator
        v_com = hist_com_defesa[-1][met] * fator

        categorias = ['Normal\n(sem ataque)', 'Atacado\n(sem defesa)', 'Atacado\n(com defesa)']
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

    plt.suptitle('Eficácia da Defesa Supervisionada\nBank Marketing - Aprendizado Federado',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{RESULTADOS_DIR}supervisionado_eficacia_recuperacao.png', dpi=300, bbox_inches='tight')
    print(f"  Salvo: supervisionado_eficacia_recuperacao.png")
    plt.close()


def plot_probabilidades_deteccao(hist_com_defesa):
    """Mostra probabilidade de ser malicioso para cada cliente ao longo das rodadas"""
    fig, ax = plt.subplots(figsize=(14, 6))

    for cli in range(1, NUM_CLIENTES + 1):
        probs = []
        rodadas = []
        for h in hist_com_defesa:
            for d in h['detectados']:
                if d['cliente'] == cli:
                    probs.append(d['prob_malicioso'])
                    rodadas.append(h['rodada'])

        cor = '#D62828' if cli == CLIENTE_MALICIOSO else '#2E86AB' if cli == 1 else '#06D6A0'
        estilo = '--' if cli == CLIENTE_MALICIOSO else '-'
        label = f'Cliente {cli} (MALICIOSO)' if cli == CLIENTE_MALICIOSO else f'Cliente {cli} (honesto)'
        ax.plot(rodadas, probs, marker='o', linewidth=3, color=cor,
                linestyle=estilo, label=label, markersize=8)

    ax.axhline(y=0.5, color='orange', linestyle=':', linewidth=2, label='Limiar de detecção (0.5)')
    ax.fill_between(range(1, NUM_RODADAS + 1), 0.5, 1.0, alpha=0.05, color='red', label='Zona malicioso')

    ax.set_xlabel('Rodada Federada', fontweight='bold', fontsize=12)
    ax.set_ylabel('Probabilidade de ser Malicioso', fontweight='bold', fontsize=12)
    ax.set_title('Probabilidade de Malícia por Cliente ao Longo das Rodadas\nDetector Supervisionado no Servidor',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='center right')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    plt.savefig(f'{RESULTADOS_DIR}supervisionado_probabilidades_deteccao.png', dpi=300, bbox_inches='tight')
    print(f"  Salvo: supervisionado_probabilidades_deteccao.png")
    plt.close()


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    import os
    os.makedirs(RESULTADOS_DIR, exist_ok=True)

    print("=" * 70)
    print("DETECÇÃO SUPERVISIONADA DE CLIENTES MALICIOSOS")
    print("Bank Marketing Dataset - Aprendizado Federado")
    print("=" * 70)

    # 1. Carrega dados
    print("\n[1/5] Carregando dataset...")
    X, y = carregar_dataset()
    print(f"  Total: {len(X)} amostras | {X.shape[1]} features")
    print(f"  Classe 0 (Nao): {(y==0).sum()} ({(y==0).mean()*100:.1f}%)")
    print(f"  Classe 1 (Sim): {(y==1).sum()} ({(y==1).mean()*100:.1f}%)")

    clientes, dados_val = distribuir_dados(X, y)

    # 2. Gera dataset para treinar o detector
    print("\n[2/5] Gerando dataset de detecção...")
    X_det, y_det = gerar_dataset_detector(clientes, n_simulacoes=200)

    # 3. Treina detector supervisionado
    print("\n[3/5] Treinando detector supervisionado...")
    detector, pca, metricas_detector = treinar_detector_supervisionado(X_det, y_det)

    # 4. Executa os 3 cenários
    print("\n[4/5] Executando cenários federados...")

    print("  Cenário Normal (sem ataque)...")
    hist_normal = executar_federado_sem_defesa(clientes, dados_val, envenenado=False)

    print("  Cenário Atacado sem defesa...")
    hist_sem_defesa = executar_federado_sem_defesa(clientes, dados_val, envenenado=True)

    print("  Cenário Atacado com defesa supervisionada...")
    hist_com_defesa = executar_federado_com_defesa(clientes, dados_val, detector, pca, envenenado=True)

    # Resultados
    print("\n" + "=" * 70)
    print("RESULTADOS FINAIS (Rodada 12)")
    print("=" * 70)
    print(f"  Normal:        Acuracia={hist_normal[-1]['acuracia']*100:.2f}%  F1={hist_normal[-1]['f1']*100:.2f}%  AUC={hist_normal[-1]['auc']:.4f}")
    print(f"  Sem defesa:    Acuracia={hist_sem_defesa[-1]['acuracia']*100:.2f}%  F1={hist_sem_defesa[-1]['f1']*100:.2f}%  AUC={hist_sem_defesa[-1]['auc']:.4f}")
    print(f"  Com defesa:    Acuracia={hist_com_defesa[-1]['acuracia']*100:.2f}%  F1={hist_com_defesa[-1]['f1']*100:.2f}%  AUC={hist_com_defesa[-1]['auc']:.4f}")

    # 5. Visualizações
    print("\n[5/5] Gerando visualizações...")
    plot_detector_cm(metricas_detector)
    plot_comparacao_defesa(hist_sem_defesa, hist_com_defesa, hist_normal)
    plot_eficacia_recuperacao(hist_sem_defesa, hist_com_defesa, hist_normal)
    plot_probabilidades_deteccao(hist_com_defesa)

    print("\n✅ Concluído! 4 visualizações salvas em:", RESULTADOS_DIR)
    print("=" * 70)


if __name__ == "__main__":
    main()
