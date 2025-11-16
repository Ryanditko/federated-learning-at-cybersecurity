# %%
# imports
# Importações necessárias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import os

# Algoritmos de clustering e detecção de outliers
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import KMeans

# Técnicas de balanceamento
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN

# Métricas de avaliação
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import warnings

warnings.filterwarnings("ignore")

# Configurações de visualização
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")
plt.rcParams["figure.figsize"] = (12, 8)
# %%
# Configuração de diretórios
TARGET_ATTACK = "neptune"
DATA_DIR = r"C:\Users\Administrador\Faculdade-Impacta\Iniciação-cientifica\project\data\nsl-kdd"
OUTPUT_DIR = r"C:\Users\Administrador\Faculdade-Impacta\Iniciação-cientifica\project\notebooks\nsl-kdd\output-images"
RESULTS_DIR = r"C:\Users\Administrador\Faculdade-Impacta\Iniciação-cientifica\project\notebooks\nsl-kdd\results"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"📁 Diretório de dados: {DATA_DIR}")
print(f"📊 Diretório de gráficos: {OUTPUT_DIR}")
print(f"📋 Diretório de resultados: {RESULTS_DIR}")
# %%
# Verificar arquivos disponíveis
train_file = None
test_file = None

try:
    files = os.listdir(DATA_DIR)
    print("📂 Arquivos encontrados:")
    for file in files:
        print(f"  • {file}")

    # Identificar arquivos de treino e teste
    for file in files:
        if "train" in file.lower() and file.endswith((".csv", ".txt")):
            train_file = file
        elif "test" in file.lower() and file.endswith((".csv", ".txt")):
            test_file = file

    print(f"\n🎯 Arquivo de treino: {train_file}")
    print(f"🎯 Arquivo de teste: {test_file}")

except FileNotFoundError:
    print("❌ Diretório não encontrado! Execute primeiro o download do dataset.")
# %%
# Carregar dados
if train_file:
    df_train = pd.read_csv(f"{DATA_DIR}/{train_file}", header=None)
    print(f"✅ Dados de treino carregados: {len(df_train):,} registros")
    
    if test_file:
        df_test = pd.read_csv(f"{DATA_DIR}/{test_file}", header=None)
        print(f"✅ Dados de teste carregados: {len(df_test):,} registros")
        
        # Combinar datasets
        df = pd.concat([df_train, df_test], ignore_index=True)
    else:
        df = df_train.copy()
    
    print(f"📊 Dataset final: {len(df):,} registros, {len(df.columns)} colunas")
else:
    print("❌ Nenhum arquivo de dados encontrado! Verifique o diretório e execute o download do dataset.")
    raise FileNotFoundError("Arquivos de treino não encontrados no diretório especificado.")
# %%
# Definir nomes das colunas do NSL-KDD
columns = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
    "attack_type",
]

# Verificar se há coluna de dificuldade
# if len(df.columns) == 42:
columns.append("difficulty")

# Aplicar nomes das colunas
df.columns = columns[: len(df.columns)]

print(f"✅ Estrutura definida: {len(df.columns)} colunas")
print(f"📋 Primeiras colunas: {list(df.columns[:10])}")
print(f"🎯 Coluna target: attack_type")
# %%
# distribuição
df_filtered = df[df["attack_type"].isin(["normal", TARGET_ATTACK])]
num_cols = df_filtered.select_dtypes(include=["number"]).columns

num_cols = df_filtered.select_dtypes(include=["number"]).columns
cat_cols = df_filtered.select_dtypes(exclude=["number"]).columns.drop(
    "attack_type", errors="ignore"
)
if len(num_cols) > 0:
    n = len(num_cols)
    cols = 3
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.flatten()

    for i, col in enumerate(num_cols):
        sns.kdeplot(
            data=df_filtered,
            x=col,
            hue="attack_type",
            fill=True,
            common_norm=False,
            alpha=0.4,
            ax=axes[i],
        )
        axes[i].set_title(col)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle("Distribuição dos atributos numéricos por tipo de ataque", fontsize=14)
    plt.tight_layout()
    plt.show()

if len(cat_cols) > 0:
    n = len(cat_cols)
    cols = 3
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.flatten()

    for i, col in enumerate(cat_cols):
        sns.countplot(data=df_filtered, x=col, hue="attack_type", ax=axes[i])
        axes[i].set_title(col)
        axes[i].tick_params(axis="x", rotation=45)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(
        "Distribuição dos atributos categóricos por tipo de ataque", fontsize=14
    )
    plt.tight_layout()
    plt.show()
# %%
# detecção de outliers
X = df_filtered[["count", "serror_rate"]].dropna()
iso = IsolationForest(contamination=0.05, random_state=42)
outlier_pred = iso.fit_predict(X)
X_clustered = X.copy()
X_clustered["outlier"] = (outlier_pred == -1).astype(int)

df_filtered.loc[X.index, "outlier"] = X_clustered["outlier"]
df_filtered["outlier"] = df_filtered["outlier"].astype("Int64")

plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=df_filtered,
    x="count",
    y="serror_rate",
    hue="outlier",
    palette={0: "blue", 1: "red"},
    s=50,
)
plt.title("Detecção de Outliers com Isolation Forest")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=df_filtered,
    x="count",
    y="serror_rate",
    hue="attack_type",
    palette="tab10",
    s=50,
)

plt.title("Distribuição de attack_type (count vs serror_rate)")
plt.tight_layout()
plt.show()
# %%
# avaliação
df_filtered["true_label"] = df_filtered["attack_type"].map(
    {"normal": 0, TARGET_ATTACK: 1}
)
mask = df_filtered["true_label"].notna() & df_filtered["outlier"].notna()
y_true = df_filtered.loc[mask, "true_label"]
y_pred = df_filtered.loc[mask, "outlier"]

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm, display_labels=["Normal", f"Outlier ({TARGET_ATTACK})"]
)
disp.plot(cmap="Blues")
plt.title("Matriz de Confusão — Detecção de Outliers vs Attack Type")
plt.tight_layout()
plt.show()

print("Matriz de confusão:\n", cm)

mask = df_filtered["true_label"].notna() & df_filtered["outlier"].notna()
y_true = df_filtered.loc[mask, "true_label"]
y_pred = df_filtered.loc[mask, "outlier"]

acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(f"Acurácia:  {acc:.4f}")
print(f"Precisão:  {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1-score:  {f1:.4f}")