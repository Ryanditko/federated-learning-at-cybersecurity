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
# Tipos de ataque para testar (TOP 10, excluindo Neptune)
ATAQUES_TESTAR = [
    "satan",
    "ipsweep", 
    "portsweep",
    "smurf",
    "nmap",
    "back",
    "teardrop",
    "warezclient",
    "pod",
    "guess_passwd"
]

# Combinações de atributos para testar
COMBINACOES_ATRIBUTOS = [
    ["count", "serror_rate"],
    ["src_bytes", "dst_bytes"],
    ["duration", "count"],
    ["srv_count", "dst_host_count"],
    ["same_srv_rate", "diff_srv_rate"],
    ["dst_host_serror_rate", "dst_host_srv_serror_rate"],
    ["rerror_rate", "srv_rerror_rate"],
    ["dst_host_count", "dst_host_srv_count"],
    ["src_bytes", "count"],
    ["duration", "srv_count"],
]

# %%
# Função para análise de outliers
def analisar_outliers(df, attack_type, features, algorithm="IsolationForest", contamination=0.05):
    """
    Analisa outliers para um tipo específico de ataque usando features selecionadas
    """
    df_filtered = df[df["attack_type"].isin(["normal", attack_type])].copy()
    
    # Verificar se há dados suficientes
    attack_count = (df_filtered["attack_type"] == attack_type).sum()
    if attack_count < 10:
        return None, 0, 0, 0, 0, None
    
    X = df_filtered[features].dropna()
    valid_indices = X.index
    
    if len(X) < 10:
        return None, 0, 0, 0, 0, None
    
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
            f"Análise de Outliers - {attack_type.upper()}\n"
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
        display_labels=["Normal", attack_type.upper()]
    )
    disp.plot(cmap="Blues", ax=ax)
    
    acc, prec, rec, f1 = metrics
    plt.title(
        f"Matriz de Confusão - {attack_type.upper()} ({algorithm})\n"
        f"Accuracy: {acc:.3f} | Precision: {prec:.3f} | Recall: {rec:.3f} | F1: {f1:.3f}",
        fontsize=12,
        fontweight='bold'
    )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

# %%
# Testar todas as combinações
resultados = []
exemplo_num = 1

print("=" * 80)
print("🚀 INICIANDO ANÁLISE COMPLETA DE MÚLTIPLOS ATAQUES")
print("=" * 80)

for attack_type in ATAQUES_TESTAR:
    print(f"\n📊 Testando ataque: {attack_type.upper()}")
    print("-" * 80)
    
    melhores_metricas = {"attack": attack_type, "f1": 0, "acc": 0, "features": None, "algorithm": None}
    
    for features in COMBINACOES_ATRIBUTOS:
        features_str = "_".join(features[:2])
        
        # Testar com IsolationForest
        df_filtered, acc, prec, rec, f1, valid_indices = analisar_outliers(
            df, attack_type, features, "IsolationForest", 0.05
        )
        
        if df_filtered is not None:
            metrics = (acc, prec, rec, f1)
            
            # Salvar resultados
            resultados.append({
                "Exemplo": f"Exemplo {exemplo_num:02d}",
                "Ataque": attack_type.upper(),
                "Atributos": " vs ".join(features),
                "Algoritmo": "IsolationForest",
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1-Score": f1
            })
            
            # Verificar se é o melhor resultado para este ataque
            if f1 > melhores_metricas["f1"]:
                melhores_metricas.update({
                    "f1": f1,
                    "acc": acc,
                    "prec": prec,
                    "rec": rec,
                    "features": features,
                    "algorithm": "IsolationForest",
                    "exemplo_num": exemplo_num
                })
            
            # Salvar apenas se for bom resultado (F1 > 0.05 ou Accuracy > 0.7)
            if f1 > 0.05 or acc > 0.7:
                # Plot scatter
                scatter_path = f"{OUTPUT_DIR}/ataque_{exemplo_num:02d}_{attack_type}_outliers_IsolationForest_{features_str}.png"
                plot_scatter_outliers(df_filtered, features, attack_type, "IsolationForest", metrics, scatter_path, valid_indices)
                
                # Plot confusion matrix
                cm_path = f"{OUTPUT_DIR}/ataque_{exemplo_num:02d}_{attack_type}_confusion_matrix_IsolationForest_{features_str}.png"
                plot_confusion_matrix(df_filtered, attack_type, "IsolationForest", metrics, cm_path)
                
                print(f"  ✓ {features[0]:25s} vs {features[1]:30s} → Acc: {acc:.3f} | F1: {f1:.3f}")
            
            exemplo_num += 1

    # Mostrar melhor resultado para este ataque
    if melhores_metricas["features"]:
        print(f"\n  🏆 MELHOR para {attack_type.upper()}:")
        print(f"     Features: {' vs '.join(melhores_metricas['features'])}")
        print(f"     Accuracy: {melhores_metricas['acc']:.3f} | F1-Score: {melhores_metricas['f1']:.3f}")

# %%
# Criar DataFrame com resultados
df_resultados = pd.DataFrame(resultados)

# Ordenar por F1-Score
df_resultados_sorted = df_resultados.sort_values("F1-Score", ascending=False)

# Salvar resultados
df_resultados_sorted.to_csv(f"{OUTPUT_DIR}/resultados_todos_ataques.csv", index=False)

# %%
# Criar visualização dos TOP 15 melhores resultados
top_15 = df_resultados_sorted.head(15)

fig, axes = plt.subplots(2, 2, figsize=(20, 14))

metricas = ["Accuracy", "Precision", "Recall", "F1-Score"]
colors = plt.cm.viridis(np.linspace(0, 1, len(top_15)))

for idx, metrica in enumerate(metricas):
    ax = axes[idx // 2, idx % 2]
    
    # Criar labels para as barras
    labels = [f"{row['Ataque']}\n{row['Atributos'][:30]}" for _, row in top_15.iterrows()]
    
    bars = ax.barh(range(len(top_15)), top_15[metrica], color=colors)
    ax.set_yticks(range(len(top_15)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel(metrica, fontsize=12, fontweight='bold')
    ax.set_title(f"TOP 15 - {metrica}", fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    
    # Adicionar valores nas barras
    for i, (bar, val) in enumerate(zip(bars, top_15[metrica])):
        ax.text(val + 0.01, i, f'{val:.3f}', 
               va='center', fontsize=8, fontweight='bold')

plt.suptitle("TOP 15 MELHORES RESULTADOS - Detecção de Outliers por Tipo de Ataque",
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/top_15_melhores_resultados.png", dpi=300, bbox_inches='tight')
plt.close()

# %%
# Criar visualização por tipo de ataque
fig, axes = plt.subplots(2, 1, figsize=(18, 12))

# Agrupar por ataque e pegar o melhor resultado
melhores_por_ataque = df_resultados.loc[df_resultados.groupby("Ataque")["F1-Score"].idxmax()]
melhores_por_ataque = melhores_por_ataque.sort_values("F1-Score", ascending=False)

# Plot 1: F1-Score por ataque
ax1 = axes[0]
colors_attack = plt.cm.tab20(np.linspace(0, 1, len(melhores_por_ataque)))
bars1 = ax1.bar(melhores_por_ataque["Ataque"], melhores_por_ataque["F1-Score"], color=colors_attack)
ax1.set_ylabel("F1-Score", fontsize=12, fontweight='bold')
ax1.set_title("Melhor F1-Score por Tipo de Ataque", fontsize=14, fontweight='bold')
ax1.set_ylim(0, max(melhores_por_ataque["F1-Score"]) * 1.2)
ax1.tick_params(axis='x', rotation=45)

for bar, val in zip(bars1, melhores_por_ataque["F1-Score"]):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.3f}',
            ha='center', va='bottom', fontweight='bold')

# Plot 2: Accuracy por ataque
ax2 = axes[1]
bars2 = ax2.bar(melhores_por_ataque["Ataque"], melhores_por_ataque["Accuracy"], color=colors_attack)
ax2.set_ylabel("Accuracy", fontsize=12, fontweight='bold')
ax2.set_xlabel("Tipo de Ataque", fontsize=12, fontweight='bold')
ax2.set_title("Melhor Accuracy por Tipo de Ataque", fontsize=14, fontweight='bold')
ax2.set_ylim(0, 1.1)
ax2.tick_params(axis='x', rotation=45)

for bar, val in zip(bars2, melhores_por_ataque["Accuracy"]):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.3f}',
            ha='center', va='bottom', fontweight='bold')

plt.suptitle("Comparação de Desempenho - Melhor Resultado por Tipo de Ataque",
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/comparacao_por_tipo_ataque.png", dpi=300, bbox_inches='tight')
plt.close()

# %%
# Salvar também melhor resultado por ataque
melhores_por_ataque.to_csv(f"{OUTPUT_DIR}/melhores_por_tipo_ataque.csv", index=False)

# %%
# Exibir resumo final
print("\n" + "=" * 80)
print("✅ ANÁLISE COMPLETA FINALIZADA!")
print("=" * 80)
print(f"📁 Resultados salvos em: {OUTPUT_DIR}")
print(f"📊 Total de combinações testadas: {len(resultados)}")
print(f"🎯 Ataques analisados: {len(ATAQUES_TESTAR)}")
print("=" * 80)

print("\n🏆 TOP 5 MELHORES RESULTADOS GERAIS:")
print("-" * 80)
for idx, row in top_15.head(5).iterrows():
    print(f"{row['Exemplo']}: {row['Ataque']:15s} | {row['Atributos']:40s}")
    print(f"           Acc: {row['Accuracy']:.3f} | Prec: {row['Precision']:.3f} | Rec: {row['Recall']:.3f} | F1: {row['F1-Score']:.3f}")
    print("-" * 80)

print("\n🎯 MELHOR RESULTADO POR TIPO DE ATAQUE:")
print("-" * 80)
for _, row in melhores_por_ataque.iterrows():
    print(f"{row['Ataque']:15s} | {row['Atributos']:40s} | F1: {row['F1-Score']:.3f} | Acc: {row['Accuracy']:.3f}")

print("\n" + "=" * 80)
print("📈 Arquivos gerados:")
print("   • resultados_todos_ataques.csv - Todos os resultados")
print("   • melhores_por_tipo_ataque.csv - Melhor resultado por ataque")
print("   • top_15_melhores_resultados.png - Visualização TOP 15")
print("   • comparacao_por_tipo_ataque.png - Comparação por ataque")
print("   • ataque_XX_*.png - Imagens individuais dos melhores casos")
print("=" * 80)
