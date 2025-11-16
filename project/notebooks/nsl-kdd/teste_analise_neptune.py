# TESTE DE ANÁLISE NEPTUNE - Script Python Simples
# Versão em script Python para contornar problemas com notebooks

import os
import sys
import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Configuração de diretórios
PROJECT_DIR = r"C:\Users\Administrador\Faculdade-Impacta\Iniciação-cientifica\project"
DATA_DIR = os.path.join(PROJECT_DIR, "data", "nsl-kdd")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "notebooks", "nsl-kdd", "output-images")
RESULTS_DIR = os.path.join(PROJECT_DIR, "notebooks", "nsl-kdd", "results")

# Criar diretórios se não existirem
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"🌊 ANÁLISE ESPECÍFICA DO ATAQUE NEPTUNE")
print(f"=" * 50)
print(f"📁 Diretório de dados: {DATA_DIR}")
print(f"📊 Diretório de saída: {OUTPUT_DIR}")
print(f"📋 Diretório de resultados: {RESULTS_DIR}")

# Verificar se o diretório de dados existe
if not os.path.exists(DATA_DIR):
    print(f"❌ Diretório de dados não encontrado: {DATA_DIR}")
    print("Por favor, execute o download do dataset NSL-KDD primeiro.")
    sys.exit(1)

# Listar arquivos disponíveis
print(f"\n📂 Verificando arquivos disponíveis...")
try:
    files = os.listdir(DATA_DIR)
    print(f"Arquivos encontrados: {files}")
    
    # Procurar arquivos de dados
    data_files = [f for f in files if f.endswith('.csv') or f.endswith('.txt')]
    if not data_files:
        print("❌ Nenhum arquivo de dados encontrado (.csv ou .txt)")
        sys.exit(1)
    
    # Usar o primeiro arquivo encontrado ou procurar por padrões específicos
    train_file = None
    test_file = None
    
    for file in data_files:
        if 'train' in file.lower():
            train_file = file
        elif 'test' in file.lower():
            test_file = file
    
    # Se não encontrou arquivos específicos, usar os primeiros disponíveis
    if not train_file and data_files:
        train_file = data_files[0]
    if not test_file and len(data_files) > 1:
        test_file = data_files[1]
    elif not test_file:
        test_file = train_file  # Usar o mesmo arquivo se só tiver um
    
    print(f"📄 Arquivo de treino: {train_file}")
    print(f"📄 Arquivo de teste: {test_file}")
    
except Exception as e:
    print(f"❌ Erro ao listar arquivos: {e}")
    sys.exit(1)

# Definir colunas do NSL-KDD
nsl_columns = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins',
    'logged_in', 'num_compromised', 'root_shell', 'su_attempted',
    'num_root', 'num_file_creations', 'num_shells', 'num_access_files',
    'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count',
    'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate',
    'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label'
]

def load_nsl_data(file_path):
    """Carrega dados NSL-KDD"""
    print(f"\n⏳ Carregando {file_path}...")
    try:
        # Tentar carregar com diferentes configurações
        for sep in [',', ';', '\t']:
            try:
                df = pd.read_csv(os.path.join(DATA_DIR, file_path), 
                               names=nsl_columns, sep=sep)
                if len(df.columns) == len(nsl_columns) and len(df) > 0:
                    print(f"✅ Carregado com separador '{sep}': {df.shape}")
                    return df
            except:
                continue
        
        # Última tentativa sem especificar separador
        df = pd.read_csv(os.path.join(DATA_DIR, file_path), names=nsl_columns)
        print(f"✅ Carregado com configuração padrão: {df.shape}")
        return df
        
    except Exception as e:
        print(f"❌ Erro ao carregar {file_path}: {e}")
        return None

# Carregar dados
print(f"\n📊 CARREGAMENTO DOS DADOS")
print("=" * 30)

df_train = load_nsl_data(train_file) if train_file else None
df_test = load_nsl_data(test_file) if test_file and test_file != train_file else None

if df_train is None:
    print("❌ Não foi possível carregar os dados")
    sys.exit(1)

# Combinar datasets se tiver ambos
if df_test is not None and not df_test.equals(df_train):
    df_combined = pd.concat([df_train, df_test], ignore_index=True)
    print(f"📊 Datasets combinados: {df_combined.shape}")
else:
    df_combined = df_train
    print(f"📊 Usando apenas dataset de treino: {df_combined.shape}")

# Mostrar informações básicas
print(f"\n🔍 INFORMAÇÕES BÁSICAS DO DATASET")
print("-" * 40)
print(f"Total de amostras: {len(df_combined):,}")
print(f"Total de features: {len(df_combined.columns) - 1}")  # -1 para excluir label
print(f"\nLabels únicos: {df_combined['label'].unique()[:10]}")  # Primeiros 10
print(f"\nDistribuição das labels:")
print(df_combined['label'].value_counts().head())

# Verificar se Neptune está disponível
neptune_count = len(df_combined[df_combined['label'] == 'neptune'])
normal_count = len(df_combined[df_combined['label'] == 'normal'])

print(f"\n🌊 ANÁLISE ESPECÍFICA NEPTUNE")
print("-" * 30)
print(f"Amostras Neptune: {neptune_count:,}")
print(f"Amostras Normal: {normal_count:,}")

if neptune_count == 0:
    print("❌ Nenhuma amostra de ataque Neptune encontrada!")
    # Mostrar ataques disponíveis
    print("Ataques disponíveis:")
    attack_counts = df_combined[df_combined['label'] != 'normal']['label'].value_counts()
    for attack, count in attack_counts.head(10).items():
        print(f"  • {attack}: {count:,}")
    
    # Usar o ataque mais comum se Neptune não estiver disponível
    if len(attack_counts) > 0:
        most_common_attack = attack_counts.index[0]
        print(f"\n🔄 Usando ataque mais comum: {most_common_attack}")
        attack_label = most_common_attack
        attack_count = attack_counts.iloc[0]
    else:
        print("❌ Nenhum ataque encontrado no dataset")
        sys.exit(1)
else:
    attack_label = 'neptune'
    attack_count = neptune_count

# Filtrar dados para análise (Normal + Ataque específico)
df_analysis = df_combined[df_combined['label'].isin(['normal', attack_label])].copy()

print(f"\n🎯 PREPARANDO ANÁLISE")
print(f"Ataque analisado: {attack_label}")
print(f"Total de amostras: {len(df_analysis):,}")
print(f"  • Normal: {len(df_analysis[df_analysis['label'] == 'normal']):,}")
print(f"  • {attack_label}: {len(df_analysis[df_analysis['label'] == attack_label]):,}")

# Calcular taxa de contaminação
contamination_rate = len(df_analysis[df_analysis['label'] == attack_label]) / len(df_analysis)
print(f"Taxa de contaminação: {contamination_rate:.4f} ({contamination_rate*100:.2f}%)")

# Preparar features (encoding de categóricas)
print(f"\n⚙️ PREPARAÇÃO DE FEATURES")
print("-" * 25)

# Identificar features categóricas e numéricas
categorical_features = df_analysis.select_dtypes(include=['object']).columns.tolist()
if 'label' in categorical_features:
    categorical_features.remove('label')

numeric_features = df_analysis.select_dtypes(include=[np.number]).columns.tolist()
if 'label' in numeric_features:
    numeric_features.remove('label')

print(f"Features categóricas: {len(categorical_features)}")
print(f"Features numéricas: {len(numeric_features)}")

# Encoding de variáveis categóricas
df_encoded = df_analysis.copy()
label_encoders = {}

for feature in categorical_features:
    if feature != 'label':
        le = LabelEncoder()
        df_encoded[feature] = le.fit_transform(df_encoded[feature].astype(str))
        label_encoders[feature] = le

print(f"✅ Encoding aplicado a {len(categorical_features)} features categóricas")

# Preparar dados para modelos
features = [col for col in df_encoded.columns if col != 'label']
X = df_encoded[features].values
y_true = (df_encoded['label'] == attack_label).astype(int)

# Normalizar dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"📊 Dados preparados: {X_scaled.shape}")
print(f"Labels de verdade: {np.sum(y_true)} positivos de {len(y_true)} total")

# Definir conjuntos de features para teste
feature_sets = {
    'all_features': {
        'name': 'Todas as Features',
        'features': list(range(len(features)))
    },
    'basic_network': {
        'name': 'Features Básicas de Rede',
        'features': [i for i, f in enumerate(features) if f in ['duration', 'src_bytes', 'dst_bytes', 'count', 'srv_count']]
    },
    'connection_stats': {
        'name': 'Estatísticas de Conexão',
        'features': [i for i, f in enumerate(features) if 'rate' in f or f in ['count', 'srv_count']]
    }
}

# Filtrar conjuntos que realmente têm features
valid_feature_sets = {}
for name, fset in feature_sets.items():
    if len(fset['features']) > 0:
        valid_feature_sets[name] = fset
        print(f"  ✅ {fset['name']}: {len(fset['features'])} features")
    else:
        print(f"  ❌ {fset['name']}: Sem features válidas")

# Teste de algoritmos
print(f"\n🚀 TESTE DE ALGORITMOS DE DETECÇÃO")
print("=" * 40)

algorithms = {
    'IsolationForest': IsolationForest(contamination=contamination_rate, random_state=42),
    'OneClassSVM': OneClassSVM(nu=contamination_rate, kernel='rbf'),
    'EllipticEnvelope': EllipticEnvelope(contamination=contamination_rate, random_state=42)
}

# Adicionar LOF se a contaminação não for muito baixa
if contamination_rate > 0.001:  # Só usar LOF se tiver contaminação suficiente
    algorithms['LocalOutlierFactor'] = LocalOutlierFactor(contamination=contamination_rate, novelty=True)

results = []

for fset_name, fset_info in valid_feature_sets.items():
    print(f"\n📋 Testando conjunto: {fset_info['name']}")
    
    # Selecionar features
    feature_indices = fset_info['features']
    X_subset = X_scaled[:, feature_indices]
    
    print(f"  📊 Dimensões: {X_subset.shape}")
    
    for algo_name, algorithm in algorithms.items():
        print(f"  🔄 {algo_name}...", end=" ")
        
        try:
            # Treinar
            start_time = time.time()
            algorithm.fit(X_subset)
            train_time = time.time() - start_time
            
            # Predizer
            start_time = time.time()
            y_pred = algorithm.predict(X_subset)
            pred_time = time.time() - start_time
            
            # Converter predições
            y_pred_binary = (y_pred == -1).astype(int)
            
            # Calcular métricas
            accuracy = accuracy_score(y_true, y_pred_binary)
            precision = precision_score(y_true, y_pred_binary, zero_division=0)
            recall = recall_score(y_true, y_pred_binary, zero_division=0)
            f1 = f1_score(y_true, y_pred_binary, zero_division=0)
            
            # Matriz de confusão
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
            
            result = {
                'attack_type': attack_label,
                'feature_set': fset_info['name'],
                'feature_count': len(feature_indices),
                'algorithm': algo_name,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'true_positives': int(tp),
                'false_positives': int(fp),
                'true_negatives': int(tn),
                'false_negatives': int(fn),
                'train_time': train_time,
                'prediction_time': pred_time,
                'contamination': contamination_rate
            }
            
            results.append(result)
            print(f"✅ R:{recall:.3f} P:{precision:.3f} F1:{f1:.3f}")
            
        except Exception as e:
            print(f"❌ Erro: {str(e)[:50]}...")
            continue

# Análise de resultados
print(f"\n📊 ANÁLISE DE RESULTADOS")
print("=" * 30)

if results:
    results_df = pd.DataFrame(results)
    
    # Melhores resultados por recall
    print(f"\n🏆 TOP 5 MELHORES RESULTADOS (Recall):")
    print("-" * 60)
    top_recall = results_df.nlargest(5, 'recall')
    
    for idx, row in top_recall.iterrows():
        print(f"{row['algorithm']:<20} | {row['feature_set']:<20}")
        print(f"  Recall: {row['recall']:.3f} | Precision: {row['precision']:.3f} | F1: {row['f1_score']:.3f}")
        print(f"  Features: {row['feature_count']} | TP: {row['true_positives']} | FN: {row['false_negatives']}")
        print()
    
    # Análise por algoritmo
    print(f"\n📈 PERFORMANCE MÉDIA POR ALGORITMO:")
    print("-" * 40)
    algo_stats = results_df.groupby('algorithm').agg({
        'recall': ['mean', 'max', 'std'],
        'precision': ['mean', 'max', 'std'],
        'f1_score': ['mean', 'max', 'std']
    }).round(3)
    
    for algo in results_df['algorithm'].unique():
        algo_data = results_df[results_df['algorithm'] == algo]
        print(f"\n{algo}:")
        print(f"  Recall:    μ={algo_data['recall'].mean():.3f}, max={algo_data['recall'].max():.3f}")
        print(f"  Precision: μ={algo_data['precision'].mean():.3f}, max={algo_data['precision'].max():.3f}")
        print(f"  F1-Score:  μ={algo_data['f1_score'].mean():.3f}, max={algo_data['f1_score'].max():.3f}")
    
    # Salvar resultados
    results_file = os.path.join(RESULTS_DIR, f'{attack_label}_analysis_results.csv')
    results_df.to_csv(results_file, index=False)
    print(f"\n💾 Resultados salvos em: {results_file}")
    
    print(f"\n🎉 ANÁLISE CONCLUÍDA!")
    print(f"✅ {len(results)} testes executados com sucesso")
    print(f"🌊 Ataque analisado: {attack_label}")
    print(f"📊 Taxa de contaminação: {contamination_rate:.4f}")
    
    if attack_label == 'neptune':
        print(f"🏆 Análise do ataque Neptune completada!")
    else:
        print(f"🔄 Análise adaptada para ataque {attack_label} (Neptune não disponível)")

else:
    print("❌ Nenhum resultado foi gerado")

print(f"\n📁 Arquivos de saída disponíveis em:")
print(f"  • Resultados: {RESULTS_DIR}")
print(f"  • Imagens: {OUTPUT_DIR}")
