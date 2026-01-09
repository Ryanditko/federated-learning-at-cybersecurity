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
        # arquivo de treino para construir o modelo
        if "train" in file.lower() and file.endswith((".csv", ".txt")):
            train_file = file
            # arquivos de teste para validar o modelo 
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
# Filtrar dados
df_filtered = df[df["attack_type"].isin(["normal", TARGET_ATTACK])].copy()
print(f"✅ Dados filtrados: {len(df_filtered):,} registros")
print(f"   - Normal: {len(df_filtered[df_filtered['attack_type'] == 'normal']):,}")
print(f"   - {TARGET_ATTACK}: {len(df_filtered[df_filtered['attack_type'] == TARGET_ATTACK]):,}")

# %%
# TESTE DE DIFERENTES COMBINAÇÕES DE ATRIBUTOS
# Definir combinações de atributos para testar
feature_combinations = {
    "Original (count + serror_rate)": ["count", "serror_rate"],
    "Bytes transferidos": ["src_bytes", "dst_bytes"],
    "Taxas de erro": ["serror_rate", "rerror_rate"],
    "Contadores de conexão": ["count", "srv_count"],
    "Análise de destino": ["dst_host_count", "dst_host_srv_count"],
    "Taxas de mesmo serviço": ["same_srv_rate", "diff_srv_rate"],
    "Multi-dimensional (4D)": ["duration", "src_bytes", "dst_bytes", "count"],
    "Comportamento suspeito": ["hot", "num_failed_logins", "num_compromised"],
    "Rede + Erro": ["count", "srv_count", "serror_rate", "rerror_rate"],
    "Host completo": ["dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_serror_rate"],
    "DoS específico": ["count", "srv_count", "dst_host_count", "dst_host_srv_count"],
    "Taxas completas": ["same_srv_rate", "diff_srv_rate", "dst_host_same_srv_rate", "dst_host_diff_srv_rate"]
}

# DataFrame para armazenar resultados
results_list = []

print("\n" + "=" * 80)
print("🔍 TESTANDO DIFERENTES COMBINAÇÕES DE ATRIBUTOS COM ISOLATION FOREST")
print("=" * 80)

for combo_name, features in feature_combinations.items():
    print(f"\n{'─' * 80}")
    print(f"📊 Testando: {combo_name}")
    print(f"   Atributos: {', '.join(features)}")
    print(f"{'─' * 80}")
    
    try:
        # Selecionar features e remover NaN
        X = df_filtered[features].dropna()
        
        if len(X) < 100:
            print(f"⚠️  Poucos dados disponíveis ({len(X)} registros). Pulando...")
            continue
        
        # Aplicar Isolation Forest
        iso = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
        outlier_pred = iso.fit_predict(X)
        
        # Adicionar predições ao DataFrame
        X_clustered = X.copy()
        X_clustered["outlier"] = (outlier_pred == -1).astype(int)
        
        # Atualizar df_filtered
        df_filtered_copy = df_filtered.copy()
        df_filtered_copy.loc[X.index, "outlier"] = X_clustered["outlier"]
        df_filtered_copy["outlier"] = df_filtered_copy["outlier"].astype("Int64")
        
        # Criar labels verdadeiros
        df_filtered_copy["true_label"] = df_filtered_copy["attack_type"].map(
            {"normal": 0, TARGET_ATTACK: 1}
        )
        
        # Calcular métricas
        mask = df_filtered_copy["true_label"].notna() & df_filtered_copy["outlier"].notna()
        y_true = df_filtered_copy.loc[mask, "true_label"]
        y_pred = df_filtered_copy.loc[mask, "outlier"]
        
        if len(y_true) > 0:
            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            # Calcular confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            tn, fp, fn, tp = cm.ravel()
            
            # Armazenar resultados
            results_list.append({
                "Combinação": combo_name,
                "Atributos": ", ".join(features),
                "N_Features": len(features),
                "Acurácia": acc,
                "Precisão": prec,
                "Recall": rec,
                "F1-Score": f1,
                "True Positive": tp,
                "False Positive": fp,
                "True Negative": tn,
                "False Negative": fn,
                "Amostras": len(X)
            })
            
            # Imprimir métricas
            print(f"✅ Acurácia:  {acc:.4f}")
            print(f"✅ Precisão:  {prec:.4f}")
            print(f"✅ Recall:    {rec:.4f}")
            print(f"✅ F1-score:  {f1:.4f}")
            print(f"📊 TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")
            print(f"📈 Amostras:  {len(X):,}")
            
    except Exception as e:
        print(f"❌ Erro ao processar {combo_name}: {str(e)}")
        continue

# %%
# VISUALIZAR RESULTADOS COMPARATIVOS
if results_list:
    results_df = pd.DataFrame(results_list)
    
    # Salvar resultados
    results_file = os.path.join(RESULTS_DIR, f"comparison_{TARGET_ATTACK}_isolation_forest.csv")
    results_df.to_csv(results_file, index=False)
    print(f"\n💾 Resultados salvos em: {results_file}")
    
    # Ordenar por F1-Score
    results_df_sorted = results_df.sort_values("F1-Score", ascending=False)
    
    print("\n" + "=" * 80)
    print("🏆 RANKING DAS MELHORES COMBINAÇÕES (por F1-Score)")
    print("=" * 80)
    print(results_df_sorted[["Combinação", "Acurácia", "Precisão", "Recall", "F1-Score"]].to_string(index=False))
    
    # Visualizar métricas
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    metrics = ["Acurácia", "Precisão", "Recall", "F1-Score"]
    colors = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12"]
    
    for idx, (metric, color) in enumerate(zip(metrics, colors)):
        ax = axes[idx // 2, idx % 2]
        
        sorted_data = results_df.sort_values(metric, ascending=True)
        
        bars = ax.barh(sorted_data["Combinação"], sorted_data[metric], color=color, alpha=0.7)
        ax.set_xlabel(metric, fontsize=12, fontweight="bold")
        ax.set_title(f"{metric} por Combinação de Atributos", fontsize=14, fontweight="bold")
        ax.set_xlim(0, 1)
        ax.grid(axis="x", alpha=0.3)
        
        # Adicionar valores nas barras
        for i, (combo, value) in enumerate(zip(sorted_data["Combinação"], sorted_data[metric])):
            ax.text(value + 0.01, i, f"{value:.3f}", va="center", fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"comparison_metrics_{TARGET_ATTACK}.png"), dpi=300, bbox_inches="tight")
    plt.show()
    
    # Gráfico de comparação geral
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(results_df_sorted))
    width = 0.2
    
    ax.bar(x - 1.5*width, results_df_sorted["Acurácia"], width, label="Acurácia", alpha=0.8)
    ax.bar(x - 0.5*width, results_df_sorted["Precisão"], width, label="Precisão", alpha=0.8)
    ax.bar(x + 0.5*width, results_df_sorted["Recall"], width, label="Recall", alpha=0.8)
    ax.bar(x + 1.5*width, results_df_sorted["F1-Score"], width, label="F1-Score", alpha=0.8)
    
    ax.set_xlabel("Combinação de Atributos", fontsize=12, fontweight="bold")
    ax.set_ylabel("Score", fontsize=12, fontweight="bold")
    ax.set_title("Comparação de Todas as Métricas por Combinação", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(results_df_sorted["Combinação"], rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"comparison_all_metrics_{TARGET_ATTACK}.png"), dpi=300, bbox_inches="tight")
    plt.show()
    
    # Visualizar top 5 combinações
    print("\n" + "=" * 80)
    print("🎯 VISUALIZANDO TOP 5 COMBINAÇÕES")
    print("=" * 80)
    
    top_5 = results_df_sorted.head(5)
    
    for idx, row in top_5.iterrows():
        combo_name = row["Combinação"]
        features = row["Atributos"].split(", ")
        
        print(f"\n📊 {combo_name} (F1-Score: {row['F1-Score']:.4f})")
        
        # Plotar apenas se houver 2 features para visualização 2D
        if len(features) == 2:
            X = df_filtered[features].dropna()
            iso = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
            outlier_pred = iso.fit_predict(X)
            
            X_plot = X.copy()
            X_plot["outlier"] = (outlier_pred == -1).astype(int)
            X_plot["attack_type"] = df_filtered.loc[X.index, "attack_type"]
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Plot 1: Outliers detectados
            sns.scatterplot(
                data=X_plot,
                x=features[0],
                y=features[1],
                hue="outlier",
                palette={0: "blue", 1: "red"},
                s=50,
                alpha=0.6,
                ax=axes[0]
            )
            axes[0].set_title(f"Outliers Detectados - {combo_name}")
            axes[0].legend(title="Outlier", labels=["Normal", "Outlier"])
            
            # Plot 2: Ground truth
            sns.scatterplot(
                data=X_plot,
                x=features[0],
                y=features[1],
                hue="attack_type",
                palette={"normal": "green", TARGET_ATTACK: "red"},
                s=50,
                alpha=0.6,
                ax=axes[1]
            )
            axes[1].set_title(f"Ground Truth - {combo_name}")
            
            plt.tight_layout()
            safe_name = combo_name.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "")
            plt.savefig(os.path.join(OUTPUT_DIR, f"visualization_{safe_name}.png"), dpi=300, bbox_inches="tight")
            plt.show()
        
        elif len(features) > 2:
            # Usar PCA para visualização
            X = df_filtered[features].dropna()
            iso = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
            outlier_pred = iso.fit_predict(X)
            
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X)
            
            df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2"], index=X.index)
            df_pca["outlier"] = (outlier_pred == -1).astype(int)
            df_pca["attack_type"] = df_filtered.loc[X.index, "attack_type"]
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Plot 1: Outliers detectados
            sns.scatterplot(
                data=df_pca,
                x="PC1",
                y="PC2",
                hue="outlier",
                palette={0: "blue", 1: "red"},
                s=50,
                alpha=0.6,
                ax=axes[0]
            )
            axes[0].set_title(f"Outliers Detectados (PCA) - {combo_name}")
            axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} var)")
            axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} var)")
            axes[0].legend(title="Outlier", labels=["Normal", "Outlier"])
            
            # Plot 2: Ground truth
            sns.scatterplot(
                data=df_pca,
                x="PC1",
                y="PC2",
                hue="attack_type",
                palette={"normal": "green", TARGET_ATTACK: "red"},
                s=50,
                alpha=0.6,
                ax=axes[1]
            )
            axes[1].set_title(f"Ground Truth (PCA) - {combo_name}")
            axes[1].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} var)")
            axes[1].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} var)")
            
            plt.tight_layout()
            safe_name = combo_name.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "")
            plt.savefig(os.path.join(OUTPUT_DIR, f"visualization_pca_{safe_name}.png"), dpi=300, bbox_inches="tight")
            plt.show()

    print("\n" + "=" * 80)
    print("✅ ANÁLISE COMPLETA CONCLUÍDA!")
    print("=" * 80)
    print(f"📊 Total de combinações testadas: {len(results_df)}")
    print(f"🏆 Melhor combinação: {results_df_sorted.iloc[0]['Combinação']}")
    print(f"   F1-Score: {results_df_sorted.iloc[0]['F1-Score']:.4f}")
    print(f"   Recall: {results_df_sorted.iloc[0]['Recall']:.4f}")
    print(f"   Acurácia: {results_df_sorted.iloc[0]['Acurácia']:.4f}")

# %%
