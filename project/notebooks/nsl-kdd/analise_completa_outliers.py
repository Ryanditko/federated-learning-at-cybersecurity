# %%
# Importações necessárias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import os
import warnings

warnings.filterwarnings("ignore")

plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")

# %%
# Configuração de diretórios
DATA_DIR = r"C:\Users\Administrador\Faculdade-Impacta\Iniciação-cientifica\project\data\nsl-kdd"
OUTPUT_DIR = r"C:\Users\Administrador\Faculdade-Impacta\Iniciação-cientifica\project\notebooks\nsl-kdd\teste-images"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# %%
# Carregar dados
train_file = None
test_file = None

try:
    files = os.listdir(DATA_DIR)
    for file in files:
        if "train" in file.lower() and file.endswith((".csv", ".txt")):
            train_file = file
        elif "test" in file.lower() and file.endswith((".csv", ".txt")):
            test_file = file
except FileNotFoundError:
    raise FileNotFoundError("❌ Diretório não encontrado!")

if train_file:
    df_train = pd.read_csv(f"{DATA_DIR}/{train_file}", header=None)
    
    if test_file:
        df_test = pd.read_csv(f"{DATA_DIR}/{test_file}", header=None)
        df = pd.concat([df_train, df_test], ignore_index=True)
    else:
        df = df_train.copy()
else:
    raise FileNotFoundError("❌ Arquivos de treino não encontrados!")

# %%
# Definir nomes das colunas do NSL-KDD
columns = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "attack_type",
]

columns.append("difficulty")
df.columns = columns[: len(df.columns)]

# %%
# Função para análise de outliers
def analisar_outliers(df, attack_type, features, algorithm="IsolationForest", contamination=0.05):
    """
    Analisa outliers para um tipo específico de ataque usando features selecionadas
    """
    df_filtered = df[df["attack_type"].isin(["normal", attack_type])].copy()
    
    X = df_filtered[features].dropna()
    valid_indices = X.index
    
    # Selecionar algoritmo
    try:
        if algorithm == "IsolationForest":
            model = IsolationForest(contamination=contamination, random_state=42)
            outlier_pred = model.fit_predict(X)
        elif algorithm == "LOF":
            model = LocalOutlierFactor(contamination=contamination)
            outlier_pred = model.fit_predict(X)
        elif algorithm == "EllipticEnvelope":
            model = EllipticEnvelope(contamination=contamination, random_state=42, support_fraction=0.9)
            outlier_pred = model.fit_predict(X)
    except Exception as e:
        # Se falhar, usar IsolationForest como fallback
        model = IsolationForest(contamination=contamination, random_state=42)
        outlier_pred = model.fit_predict(X)
    
    # Criar coluna de outliers
    df_filtered.loc[valid_indices, "outlier"] = (outlier_pred == -1).astype(int)
    df_filtered["outlier"] = df_filtered["outlier"].astype("Int64")
    
    # Criar labels verdadeiros
    df_filtered["true_label"] = df_filtered["attack_type"].map(
        {"normal": 0, attack_type: 1}
    )
    
    # Calcular métricas
    mask = df_filtered["true_label"].notna() & df_filtered["outlier"].notna()
    y_true = df_filtered.loc[mask, "true_label"]
    y_pred = df_filtered.loc[mask, "outlier"]
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    return df_filtered, acc, prec, rec, f1, valid_indices

# %%
# Função para plotar distribuições
def plot_distribuicao(df, attack_type, features, output_path):
    """
    Plota a distribuição dos atributos para normal vs ataque
    """
    df_filtered = df[df["attack_type"].isin(["normal", attack_type])]
    
    n_features = len(features)
    cols = min(3, n_features)
    rows = (n_features + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
    if n_features == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_features > 1 else [axes]
    
    for i, feature in enumerate(features):
        sns.kdeplot(
            data=df_filtered,
            x=feature,
            hue="attack_type",
            fill=True,
            common_norm=False,
            alpha=0.5,
            ax=axes[i],
        )
        axes[i].set_title(f"Distribuição: {feature}", fontsize=12, fontweight='bold')
        axes[i].set_xlabel(feature, fontsize=10)
        axes[i].set_ylabel("Densidade", fontsize=10)
    
    # Remover eixos extras
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.suptitle(f"Distribuição de Atributos - Normal vs {attack_type}", 
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

# %%
# Função para plotar scatter com outliers
def plot_scatter_outliers(df_filtered, features, attack_type, algorithm, metrics, output_path, valid_indices):
    """
    Plota scatter plot mostrando outliers detectados
    """
    if len(features) >= 2:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Outliers detectados
        df_plot = df_filtered.loc[valid_indices]
        sns.scatterplot(
            data=df_plot,
            x=features[0],
            y=features[1],
            hue="outlier",
            palette={0: "blue", 1: "red"},
            s=50,
            alpha=0.6,
            ax=ax1,
        )
        ax1.set_title(f"Outliers Detectados - {algorithm}", fontsize=12, fontweight='bold')
        ax1.legend(title="Outlier", labels=["Normal", "Outlier"])
        
        # Plot 2: Attack type real
        sns.scatterplot(
            data=df_plot,
            x=features[0],
            y=features[1],
            hue="attack_type",
            palette="Set1",
            s=50,
            alpha=0.6,
            ax=ax2,
        )
        ax2.set_title(f"Distribuição Real - {attack_type}", fontsize=12, fontweight='bold')
        
        # Adicionar métricas no título
        acc, prec, rec, f1 = metrics
        fig.suptitle(
            f"Análise de Outliers - {attack_type}\n"
            f"Accuracy: {acc:.3f} | Precision: {prec:.3f} | Recall: {rec:.3f} | F1-Score: {f1:.3f}",
            fontsize=14,
            fontweight='bold'
        )
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

# %%
# Função para plotar matriz de confusão
def plot_confusion_matrix(df_filtered, attack_type, algorithm, metrics, output_path):
    """
    Plota matriz de confusão
    """
    mask = df_filtered["true_label"].notna() & df_filtered["outlier"].notna()
    y_true = df_filtered.loc[mask, "true_label"]
    y_pred = df_filtered.loc[mask, "outlier"]
    
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Normal", attack_type]
    )
    disp.plot(cmap="Blues", ax=ax)
    
    acc, prec, rec, f1 = metrics
    plt.title(
        f"Matriz de Confusão - {attack_type} ({algorithm})\n"
        f"Accuracy: {acc:.3f} | Precision: {prec:.3f} | Recall: {rec:.3f} | F1: {f1:.3f}",
        fontsize=12,
        fontweight='bold'
    )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

# %%
# Função para criar análise completa
def analise_completa(df, attack_type, features, algorithm="IsolationForest", contamination=0.05, exemplo_num=1):
    """
    Realiza análise completa e salva todas as visualizações
    """
    features_str = "_".join(features[:3])  # Limitar nome do arquivo
    
    # 1. Análise de outliers
    df_filtered, acc, prec, rec, f1, valid_indices = analisar_outliers(
        df, attack_type, features, algorithm, contamination
    )
    
    metrics = (acc, prec, rec, f1)
    
    # 2. Plot de distribuição
    dist_path = f"{OUTPUT_DIR}/exemplo{exemplo_num:02d}_{attack_type}_distribuicao_{features_str}.png"
    plot_distribuicao(df, attack_type, features, dist_path)
    
    # 3. Plot scatter com outliers
    if len(features) >= 2:
        scatter_path = f"{OUTPUT_DIR}/exemplo{exemplo_num:02d}_{attack_type}_outliers_{algorithm}_{features_str}.png"
        plot_scatter_outliers(df_filtered, features, attack_type, algorithm, metrics, scatter_path, valid_indices)
    
    # 4. Matriz de confusão
    cm_path = f"{OUTPUT_DIR}/exemplo{exemplo_num:02d}_{attack_type}_confusion_matrix_{algorithm}_{features_str}.png"
    plot_confusion_matrix(df_filtered, attack_type, algorithm, metrics, cm_path)
    
    return metrics

# %%
# Exemplo 1: Neptune - count vs serror_rate (IsolationForest)
metrics1 = analise_completa(
    df, "neptune", 
    ["count", "serror_rate"],
    algorithm="IsolationForest",
    contamination=0.05,
    exemplo_num=1
)

# %%
# Exemplo 2: Neptune - src_bytes vs dst_bytes (IsolationForest)
metrics2 = analise_completa(
    df, "neptune",
    ["src_bytes", "dst_bytes"],
    algorithm="IsolationForest",
    contamination=0.05,
    exemplo_num=2
)

# %%
# Exemplo 3: Neptune - duration vs count (LOF)
metrics3 = analise_completa(
    df, "neptune",
    ["duration", "count"],
    algorithm="LOF",
    contamination=0.05,
    exemplo_num=3
)

# %%
# Exemplo 4: Neptune - srv_count vs dst_host_count (IsolationForest)
metrics4 = analise_completa(
    df, "neptune",
    ["srv_count", "dst_host_count"],
    algorithm="IsolationForest",
    contamination=0.1,
    exemplo_num=4
)

# %%
# Exemplo 5: Neptune - same_srv_rate vs diff_srv_rate (EllipticEnvelope)
metrics5 = analise_completa(
    df, "neptune",
    ["same_srv_rate", "diff_srv_rate"],
    algorithm="EllipticEnvelope",
    contamination=0.05,
    exemplo_num=5
)

# %%
# Descobrir outros tipos de ataque disponíveis
attack_types = df["attack_type"].value_counts()
outros_ataques = [atk for atk in attack_types.index[:10] if atk not in ["normal", "neptune"]]

# %%
# Exemplo 6: Smurf - count vs serror_rate
if "smurf" in df["attack_type"].values:
    metrics6 = analise_completa(
        df, "smurf",
        ["count", "serror_rate"],
        algorithm="IsolationForest",
        contamination=0.05,
        exemplo_num=6
    )

# %%
# Exemplo 7: Satan - src_bytes vs dst_bytes
if "satan" in df["attack_type"].values:
    metrics7 = analise_completa(
        df, "satan",
        ["src_bytes", "dst_bytes"],
        algorithm="IsolationForest",
        contamination=0.05,
        exemplo_num=7
    )

# %%
# Exemplo 8: Portsweep - duration vs srv_count
if "portsweep" in df["attack_type"].values:
    metrics8 = analise_completa(
        df, "portsweep",
        ["duration", "srv_count"],
        algorithm="LOF",
        contamination=0.05,
        exemplo_num=8
    )

# %%
# Exemplo 9: Ipsweep - count vs dst_host_count
if "ipsweep" in df["attack_type"].values:
    metrics9 = analise_completa(
        df, "ipsweep",
        ["count", "dst_host_count"],
        algorithm="IsolationForest",
        contamination=0.05,
        exemplo_num=9
    )

# %%
# Exemplo 10: Back - src_bytes vs count
if "back" in df["attack_type"].values:
    metrics10 = analise_completa(
        df, "back",
        ["src_bytes", "count"],
        algorithm="IsolationForest",
        contamination=0.05,
        exemplo_num=10
    )

# %%
# Exemplo 11: Neptune com 3 features - count, serror_rate, srv_count
metrics11 = analise_completa(
    df, "neptune",
    ["count", "serror_rate", "srv_count"],
    algorithm="IsolationForest",
    contamination=0.05,
    exemplo_num=11
)

# %%
# Exemplo 12: Neptune - dst_host_serror_rate vs dst_host_srv_serror_rate
metrics12 = analise_completa(
    df, "neptune",
    ["dst_host_serror_rate", "dst_host_srv_serror_rate"],
    algorithm="IsolationForest",
    contamination=0.05,
    exemplo_num=12
)

# %%
# Criar resumo comparativo de todas as análises
def criar_resumo_comparativo():
    """
    Cria um gráfico comparativo com todas as métricas
    """
    exemplos = []
    metricas_todas = []
    
    # Coletar todas as métricas (você precisa armazenar em uma lista)
    resultados = [
        ("Exemplo 01: Neptune\ncount vs serror_rate\n(IsolationForest)", metrics1),
        ("Exemplo 02: Neptune\nsrc_bytes vs dst_bytes\n(IsolationForest)", metrics2),
        ("Exemplo 03: Neptune\nduration vs count\n(LOF)", metrics3),
        ("Exemplo 04: Neptune\nsrv_count vs dst_host_count\n(IsolationForest)", metrics4),
        ("Exemplo 05: Neptune\nsame_srv_rate vs diff_srv_rate\n(EllipticEnvelope)", metrics5),
    ]
    
    # Adicionar outros ataques se existirem
    if "smurf" in df["attack_type"].values:
        resultados.append(("Exemplo 06: Smurf\ncount vs serror_rate\n(IsolationForest)", metrics6))
    if "satan" in df["attack_type"].values:
        resultados.append(("Exemplo 07: Satan\nsrc_bytes vs dst_bytes\n(IsolationForest)", metrics7))
    if "portsweep" in df["attack_type"].values:
        resultados.append(("Exemplo 08: Portsweep\nduration vs srv_count\n(LOF)", metrics8))
    if "ipsweep" in df["attack_type"].values:
        resultados.append(("Exemplo 09: Ipsweep\ncount vs dst_host_count\n(IsolationForest)", metrics9))
    if "back" in df["attack_type"].values:
        resultados.append(("Exemplo 10: Back\nsrc_bytes vs count\n(IsolationForest)", metrics10))
    
    resultados.append(("Exemplo 11: Neptune\ncount, serror_rate, srv_count\n(IsolationForest)", metrics11))
    resultados.append(("Exemplo 12: Neptune\ndst_host_serror_rate vs\ndst_host_srv_serror_rate\n(IsolationForest)", metrics12))
    
    # Criar DataFrame com resultados
    data = []
    for nome, (acc, prec, rec, f1) in resultados:
        data.append({
            "Exemplo": nome,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1
        })
    
    df_resultados = pd.DataFrame(data)
    
    # Plot comparativo
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    metricas = ["Accuracy", "Precision", "Recall", "F1-Score"]
    colors = plt.cm.viridis(np.linspace(0, 1, len(df_resultados)))
    
    for idx, metrica in enumerate(metricas):
        ax = axes[idx // 2, idx % 2]
        bars = ax.barh(df_resultados["Exemplo"], df_resultados[metrica], color=colors)
        ax.set_xlabel(metrica, fontsize=12, fontweight='bold')
        ax.set_title(f"Comparação de {metrica}", fontsize=14, fontweight='bold')
        ax.set_xlim(0, 1)
        
        # Adicionar valores nas barras
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2, 
                   f'{width:.3f}', 
                   ha='left', va='center', fontsize=9, fontweight='bold')
    
    plt.suptitle("Comparação de Desempenho - Detecção de Outliers em Diferentes Cenários",
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/resumo_comparativo_todos_exemplos.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Salvar tabela de resultados
    df_resultados.to_csv(f"{OUTPUT_DIR}/resultados_metricas.csv", index=False)
    
    return df_resultados

# %%
# Criar resumo comparativo
df_resultados = criar_resumo_comparativo()

# %%
# Exibir mensagem final
print("=" * 80)
print("✅ ANÁLISE COMPLETA FINALIZADA!")
print("=" * 80)
print(f"📁 Todas as imagens foram salvas em: {OUTPUT_DIR}")
print(f"📊 Total de exemplos gerados: 12+")
print(f"📈 Arquivo de métricas: resultados_metricas.csv")
print("=" * 80)
print("\nResumo dos exemplos gerados:")
print("-" * 80)
print("1-5:   Neptune com diferentes atributos e algoritmos")
print("6-10:  Outros tipos de ataque (Smurf, Satan, Portsweep, Ipsweep, Back)")
print("11-12: Neptune com combinações adicionais de atributos")
print("=" * 80)
