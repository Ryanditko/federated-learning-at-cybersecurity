"""
Aprendizado Federado com Dados Distribuídos - Bank Marketing Dataset
Poisoning Attack com Distribuição Realista de Dados

Características:
- Conjunto de validação GLOBAL (servidor central)
- Conjuntos de dados DIFERENTES para cada cliente
- Distribuição estratificada de classes por cliente
- Dataset: Bank Marketing (Kaggle)

Pipeline: Dados Distribuídos → Treina Local → Corrompe Pesos → Avalia Local e Global → Agrega

Autor: Projeto de Iniciação Científica
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    log_loss, confusion_matrix, roc_auc_score, classification_report
)
from sklearn.model_selection import train_test_split
from copy import deepcopy
import warnings
warnings.filterwarnings('ignore')

# Configuração
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

NUM_RODADAS = 12
NUM_CLIENTES = 3


class ModeloFederado:
    """Modelo de Regressão Logística para Aprendizado Federado"""
    
    def __init__(self):
        self._modelo = LogisticRegression(
            max_iter=500,  # Aumentado para garantir convergência
            random_state=42,
            solver='saga',  # Saga é melhor para dados desbalanceados
            warm_start=True,
            class_weight='balanced',  # Balanceia classes desbalanceadas
            penalty='l2',  # Regularização L2
            C=0.5,  # Regularização mais forte para evitar overfitting na classe majoritária
            tol=1e-4  # Critério de parada mais rigoroso
        )
        self.scaler = MinMaxScaler()
        self.is_fitted = False
    
    def treinar_incremental(self, X, y, pesos_iniciais=None):
        """Treina modelo incrementalmente a partir dos pesos globais"""
        X_scaled = self.scaler.fit_transform(X)
        
        if pesos_iniciais is not None:
            try:
                if not hasattr(self._modelo, 'classes_') or not np.array_equal(
                    self._modelo.classes_, pesos_iniciais['classes']
                ):
                    self._modelo.fit(X_scaled, y)
                
                self._carregar_pesos(pesos_iniciais)
                self.is_fitted = True
                self._modelo.fit(X_scaled, y)
            except (ValueError, AttributeError):
                self._modelo.fit(X_scaled, y)
        else:
            self._modelo.fit(X_scaled, y)
        
        self.is_fitted = True
    
    def avaliar(self, X, y):
        """Avalia modelo e retorna métricas"""
        if not hasattr(self.scaler, 'mean_'):
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        y_pred = self._modelo.predict(X_scaled)
        y_proba = self._modelo.predict_proba(X_scaled)
        
        return {
            'acuracia': accuracy_score(y, y_pred),
            'f1_score': f1_score(y, y_pred, average='binary'),
            'precisao': precision_score(y, y_pred, average='binary', zero_division=0),
            'recall': recall_score(y, y_pred, average='binary', zero_division=0),
            'loss': log_loss(y, y_proba),
            'auc': roc_auc_score(y, y_proba[:, 1]),
            'y_pred': y_pred,
            'y_true': y,
            'y_proba': y_proba,
            'confusion_matrix': confusion_matrix(y, y_pred)
        }
    
    def obter_pesos(self):
        """Extrai pesos do modelo"""
        if self.is_fitted and hasattr(self._modelo, 'coef_'):
            return {
                'coef': deepcopy(self._modelo.coef_),
                'intercept': deepcopy(self._modelo.intercept_),
                'classes': deepcopy(self._modelo.classes_)
            }
        return None
    
    def _carregar_pesos(self, pesos):
        """Carrega pesos no modelo"""
        if pesos and 'coef' in pesos:
            self._modelo.coef_ = deepcopy(pesos['coef'])
            self._modelo.intercept_ = deepcopy(pesos['intercept'])
            self._modelo.classes_ = deepcopy(pesos['classes'])
            self.is_fitted = True


def envenenar_pesos(pesos, taxa=0.9):
    """
    Corrompe pesos do modelo (Sign Flipping Attack)
    Taxa mais alta para impacto mais visível
    """
    pesos_corrompidos = deepcopy(pesos)
    pesos_corrompidos['coef'] = -pesos['coef'] * (1 + taxa)
    pesos_corrompidos['intercept'] = -pesos['intercept'] * (1 + taxa)
    return pesos_corrompidos


def carregar_e_preprocessar_dataset():
    """
    Carrega e preprocessa o dataset Bank Marketing
    Retorna: DataFrame preprocessado
    """
    print("\n[CARREGAMENTO] Preparando dataset Bank Marketing...")
    
    caminho = r"c:\Users\Administrador\Faculdade\Iniciação-cientifica\project\data\bank-marketing\bank.csv"
    
    try:
        # Tenta primeiro com vírgula como separador
        df = pd.read_csv(caminho)
        
        # Se todas as colunas vieram em uma só, tenta com ponto-e-vírgula
        if df.shape[1] == 1:
            df = pd.read_csv(caminho, sep=';')
        
        # Se ainda tem problema, tenta forçar leitura
        if df.shape[1] == 1 or len(df.columns) < 10:
            print("  ⚠️ Formato inválido. Gerando dataset sintético...")
            df = gerar_dataset_sintetico()
        else:
            print(f"  ✓ Dataset carregado do arquivo")
    except Exception as e:
        print(f"  ⚠️ Erro ao carregar: {e}")
        print("  ⚠️ Gerando dataset sintético...")
        df = gerar_dataset_sintetico()
    
    print(f"  ✓ Shape original: {df.shape}")
    
    # Preprocessamento seguindo o notebook Kaggle
    df = preprocessar_features(df)
    
    print(f"  ✓ Shape após preprocessamento: {df.shape}")
    
    return df


def gerar_dataset_sintetico():
    """Gera dataset sintético similar ao Bank Marketing"""
    np.random.seed(42)
    n_samples = 4521
    
    df = pd.DataFrame({
        'age': np.random.randint(18, 95, n_samples),
        'job': np.random.choice(['admin.', 'technician', 'services', 'management', 
                                'retired', 'blue-collar', 'unemployed', 'entrepreneur',
                                'housemaid', 'unknown', 'self-employed', 'student'], n_samples),
        'marital': np.random.choice(['married', 'single', 'divorced'], n_samples),
        'education': np.random.choice(['primary', 'secondary', 'tertiary', 'unknown'], n_samples),
        'default': np.random.choice(['yes', 'no'], n_samples, p=[0.02, 0.98]),
        'balance': np.random.randint(-8000, 100000, n_samples),
        'housing': np.random.choice(['yes', 'no'], n_samples),
        'loan': np.random.choice(['yes', 'no'], n_samples),
        'contact': np.random.choice(['cellular', 'telephone', 'unknown'], n_samples),
        'day': np.random.randint(1, 32, n_samples),
        'month': np.random.choice(['jan', 'feb', 'mar', 'apr', 'may', 'jun', 
                                  'jul', 'aug', 'sep', 'oct', 'nov', 'dec'], n_samples),
        'duration': np.random.randint(0, 5000, n_samples),
        'campaign': np.random.randint(1, 64, n_samples),
        'pdays': np.random.choice([-1] + list(range(0, 900)), n_samples),
        'previous': np.random.randint(0, 276, n_samples),
        'poutcome': np.random.choice(['unknown', 'failure', 'success', 'other'], n_samples),
        'y': np.random.choice(['yes', 'no'], n_samples, p=[0.117, 0.883])
    })
    
    # Salva para uso futuro
    import os
    caminho = r"c:\Users\Administrador\Faculdade\Iniciação-cientifica\project\data\bank-marketing\bank.csv"
    os.makedirs(os.path.dirname(caminho), exist_ok=True)
    df.to_csv(caminho, index=False)
    
    return df


def preprocessar_features(df):
    """Preprocessa features seguindo o notebook Kaggle"""
    
    # Remove duplicatas
    df = df.drop_duplicates()
    
    # Trata valores unknown
    if 'job' in df.columns and 'education' in df.columns:
        def impute_job(row):
            if row['job'] == 'unknown':
                return 'management' if row.get('education') == 'tertiary' else 'blue-collar'
            return row['job']
        df['job'] = df.apply(impute_job, axis=1)
        
        def impute_education(row):
            if row['education'] == 'unknown':
                if row['job'] == 'housemaid':
                    return 'primary'
                elif row['job'] in ['entrepreneur', 'self-employed', 'management']:
                    return 'tertiary'
                return 'secondary'
            return row['education']
        df['education'] = df.apply(impute_education, axis=1)
    
    # Trata poutcome
    if 'poutcome' in df.columns:
        df['poutcome'] = df['poutcome'].replace('unknown', 'nonexistent')
    
    # Trata contact
    if 'contact' in df.columns:
        df['contact'] = df['contact'].replace('unknown', 'non-call')
    
    # Remove balance (muito variável)
    if 'balance' in df.columns:
        df = df.drop('balance', axis=1)
    
    # IMPORTANTE: Feature Engineering para melhorar predição da classe positiva
    if 'duration' in df.columns:
        # Duration é altamente correlacionado com sucesso
        df['duration_log'] = np.log1p(df['duration'])
        df['duration_high'] = (df['duration'] > 300).astype(int)
    
    if 'pdays' in df.columns:
        # Cliente já foi contatado antes?
        df['previously_contacted'] = (df['pdays'] != 999).astype(int)
        df['pdays_log'] = np.log1p(df['pdays'].replace(999, 0))
    
    if 'campaign' in df.columns:
        # Número de contatos na campanha atual
        df['campaign_low'] = (df['campaign'] <= 2).astype(int)
    
    # Encoding de variáveis categóricas
    cat_columns = df.select_dtypes(include='object').columns
    cat_columns = [col for col in cat_columns if col != 'y']  # Exclui target
    
    for col in cat_columns:
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
        df = pd.concat([df, dummies], axis=1)
        df = df.drop(col, axis=1)
    
    # Converte target para binário
    if 'y' in df.columns:
        df['y'] = df['y'].map({'yes': 1, 'no': 0})
    
    # Converte yes/no restantes
    yes_no_cols = [col for col in df.columns if df[col].dtype == 'object']
    for col in yes_no_cols:
        if col != 'y':
            df[col] = df[col].map({'yes': 1, 'no': 0})
    
    return df


def distribuir_dados_entre_clientes(X, y, num_clientes=3, validacao_global_size=0.2):
    """
    Distribui dados de forma realista entre clientes
    
    Estratégia:
    1. Separa conjunto de VALIDAÇÃO GLOBAL (servidor central)
    2. Divide dados restantes entre clientes de forma ESTRATIFICADA
    3. Cada cliente recebe distribuição diferente mas balanceada
    
    Returns:
        dados_clientes: Lista de tuplas (X_cliente, y_cliente)
        dados_validacao_global: Tupla (X_val_global, y_val_global)
    """
    print("\n[DISTRIBUIÇÃO] Separando dados entre clientes e servidor...")
    
    # 1. SEPARA VALIDAÇÃO GLOBAL (para o servidor central avaliar)
    X_train_total, X_val_global, y_train_total, y_val_global = train_test_split(
        X, y, test_size=validacao_global_size, random_state=42, stratify=y
    )
    
    print(f"  ✓ Validação Global (Servidor): {len(X_val_global)} amostras")
    print(f"    Classe 0: {(y_val_global == 0).sum()}, Classe 1: {(y_val_global == 1).sum()}")
    
    # 2. DIVIDE DADOS DE TREINO ENTRE CLIENTES (estratificado)
    dados_clientes = []
    
    # Cliente 1: 40% dos dados de treino
    X_c1, X_temp, y_c1, y_temp = train_test_split(
        X_train_total, y_train_total, 
        test_size=0.6, 
        random_state=42, 
        stratify=y_train_total
    )
    dados_clientes.append((X_c1, y_c1))
    
    # Cliente 2: 30% dos dados de treino (50% do restante)
    X_c2, X_c3, y_c2, y_c3 = train_test_split(
        X_temp, y_temp,
        test_size=0.5,
        random_state=42,
        stratify=y_temp
    )
    dados_clientes.append((X_c2, y_c2))
    
    # Cliente 3: 30% dos dados de treino
    dados_clientes.append((X_c3, y_c3))
    
    # Mostra distribuição
    for idx, (X_cli, y_cli) in enumerate(dados_clientes, 1):
        classe_0 = (y_cli == 0).sum()
        classe_1 = (y_cli == 1).sum()
        print(f"  ✓ Cliente {idx}: {len(X_cli)} amostras " +
              f"(Classe 0: {classe_0}, Classe 1: {classe_1}, " +
              f"Taxa positiva: {classe_1/len(y_cli)*100:.1f}%)")
    
    return dados_clientes, (X_val_global, y_val_global)


def treinar_cliente_minibatch(X, y, pesos_globais, rodada, idx_cliente):
    """
    Treina um cliente local usando mini-batch por rodada.
    40% dos dados, seed variável por rodada e cliente → variação real nas métricas.
    Iterações crescem gradualmente (10 + rodada*5, máx 80) para simular convergência.
    """
    seed = 42 + idx_cliente + rodada * 7
    rng = np.random.RandomState(seed)
    n = max(200, int(len(X) * 0.40))
    idx_batch = rng.choice(len(X), size=n, replace=False)
    X_batch = X[idx_batch]
    y_batch = y[idx_batch]

    iters_por_rodada = min(10 + rodada * 5, 80)

    modelo = LogisticRegression(
        max_iter=iters_por_rodada,
        random_state=seed,
        solver='saga',
        warm_start=True,
        class_weight='balanced',
        penalty='l2',
        C=0.5,
        tol=1e-3
    )
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_batch)

    if pesos_globais is not None:
        try:
            modelo.fit(X_scaled, y_batch)   # inicializa classes_
            modelo.coef_ = deepcopy(pesos_globais['coef'])
            modelo.intercept_ = deepcopy(pesos_globais['intercept'])
            modelo.classes_ = deepcopy(pesos_globais['classes'])
            modelo.fit(X_scaled, y_batch)   # fine-tune
        except Exception:
            modelo.fit(X_scaled, y_batch)
    else:
        modelo.fit(X_scaled, y_batch)

    return modelo, scaler


def executar_rodada_federada(dados_clientes, dados_val_global, pesos_globais,
                             rodada, envenenado=False, cliente_malicioso=3):
    """
    Executa UMA rodada federada completa

    Fases:
    1. Cada cliente treina localmente com mini-batch (40%, seed variável)
    2. Cliente malicioso corrompe seus pesos
    3. Cada cliente avalia seu modelo LOCAL
    4. Servidor agrega pesos (FedAvg)
    5. Servidor avalia modelo GLOBAL
    """
    pesos_locais = []
    avaliacoes_locais = []

    X_val_global, y_val_global = dados_val_global

    # FASE 1 e 2: Treinamento Local com mini-batch + Corrupção (se aplicável)
    for idx, (X_cliente, y_cliente) in enumerate(dados_clientes, 1):
        modelo_local, scaler_local = treinar_cliente_minibatch(
            X_cliente, y_cliente, pesos_globais, rodada, idx
        )
        pesos_local = {
            'coef': deepcopy(modelo_local.coef_),
            'intercept': deepcopy(modelo_local.intercept_),
            'classes': deepcopy(modelo_local.classes_)
        }

        # Corrompe pesos se for cliente malicioso
        if envenenado and idx == cliente_malicioso:
            pesos_local = envenenar_pesos(pesos_local, taxa=0.9)

        pesos_locais.append(pesos_local)

        # FASE 3: Avaliação LOCAL (nos dados completos do cliente)
        X_full_scaled = scaler_local.transform(X_cliente)
        y_pred_local = modelo_local.predict(X_full_scaled)
        y_proba_local = modelo_local.predict_proba(X_full_scaled)
        aval_local = {
            'acuracia': accuracy_score(y_cliente, y_pred_local),
            'f1_score': f1_score(y_cliente, y_pred_local, average='binary'),
            'precisao': precision_score(y_cliente, y_pred_local, average='binary', zero_division=0),
            'recall': recall_score(y_cliente, y_pred_local, average='binary', zero_division=0),
            'loss': log_loss(y_cliente, y_proba_local),
            'auc': roc_auc_score(y_cliente, y_proba_local[:, 1]),
            'confusion_matrix': confusion_matrix(y_cliente, y_pred_local),
            'cliente': idx,
            'envenenado': (envenenado and idx == cliente_malicioso)
        }
        avaliacoes_locais.append(aval_local)

    # FASE 4: Agregação FedAvg
    pesos_globais_novos = {
        'coef': np.mean([p['coef'] for p in pesos_locais], axis=0),
        'intercept': np.mean([p['intercept'] for p in pesos_locais], axis=0),
        'classes': pesos_locais[0]['classes']
    }

    # FASE 5: Avaliação GLOBAL (servidor avalia no conjunto global com o scaler da rodada)
    # Usa MinMaxScaler fit no conjunto de validação para avaliação global justa
    scaler_global = MinMaxScaler()
    X_val_scaled = scaler_global.fit_transform(X_val_global)

    modelo_global = LogisticRegression(max_iter=1, solver='saga', warm_start=True,
                                        class_weight='balanced', C=0.5)
    # Fit inicial para criar estrutura interna
    modelo_global.fit(X_val_scaled, y_val_global)
    modelo_global.coef_ = deepcopy(pesos_globais_novos['coef'])
    modelo_global.intercept_ = deepcopy(pesos_globais_novos['intercept'])
    modelo_global.classes_ = deepcopy(pesos_globais_novos['classes'])

    y_pred_global = modelo_global.predict(X_val_scaled)
    y_proba_global = modelo_global.predict_proba(X_val_scaled)

    avaliacao_global = {
        'rodada': rodada,
        'acuracia': accuracy_score(y_val_global, y_pred_global),
        'f1_score': f1_score(y_val_global, y_pred_global, average='binary'),
        'precisao': precision_score(y_val_global, y_pred_global, average='binary', zero_division=0),
        'recall': recall_score(y_val_global, y_pred_global, average='binary', zero_division=0),
        'loss': log_loss(y_val_global, y_proba_global),
        'auc': roc_auc_score(y_val_global, y_proba_global[:, 1]),
        'confusion_matrix': confusion_matrix(y_val_global, y_pred_global)
    }

    return pesos_globais_novos, avaliacao_global, avaliacoes_locais


def executar_cenario_completo(dados_clientes, dados_val_global, num_rodadas=12, envenenado=False):
    """Executa cenário federado completo com múltiplas rodadas"""
    historico_global = []
    historico_clientes = []

    # Inicializa pesos globais com mini-batch da rodada 0 (cold start)
    print(f"\n[INICIALIZAÇÃO] Criando modelo global inicial (mini-batch)...")
    X_init = dados_clientes[0][0]
    y_init = dados_clientes[0][1]
    rng0 = np.random.RandomState(42)
    idx0 = rng0.choice(len(X_init), size=min(300, len(X_init)), replace=False)
    scaler0 = MinMaxScaler()
    X0_scaled = scaler0.fit_transform(X_init[idx0])
    modelo0 = LogisticRegression(max_iter=10, solver='saga', class_weight='balanced',
                                  C=0.5, random_state=42)
    modelo0.fit(X0_scaled, y_init[idx0])
    pesos_globais = {
        'coef': deepcopy(modelo0.coef_),
        'intercept': deepcopy(modelo0.intercept_),
        'classes': deepcopy(modelo0.classes_)
    }
    print(f"  ✓ Modelo global inicializado com mini-batch (300 amostras, 10 iter)")
    print(f"\n[TREINAMENTO] Executando {num_rodadas} rodadas federadas...")

    for rodada in range(1, num_rodadas + 1):
        pesos_globais, aval_global, aval_clientes = executar_rodada_federada(
            dados_clientes, dados_val_global, pesos_globais, rodada, envenenado
        )

        historico_global.append(aval_global)
        historico_clientes.extend(aval_clientes)

        if rodada % 3 == 0 or rodada == 1 or rodada == num_rodadas:
            print(f"  Rodada {rodada:2d}: " +
                  f"Acurácia Global={aval_global['acuracia']*100:5.2f}%, " +
                  f"F1={aval_global['f1_score']*100:5.2f}%, " +
                  f"AUC={aval_global['auc']:.4f}")

    return historico_global, historico_clientes


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: Matriz de Confusão com contagens + percentuais legíveis
# ─────────────────────────────────────────────────────────────────────────────
def _plotar_matriz_confusao(ax, cm, cmap, title,
                             xticklabels=None, yticklabels=None,
                             show_cbar=True, fontsize=13):
    """
    Plota uma matriz de confusão com:
      - valor absoluto (n)
      - percentual em relação ao total da linha (recall por classe)
    A escala de cor usa a versão normalizada → evita que a célula TN domine.
    """
    if xticklabels is None:
        xticklabels = ['Não', 'Sim']
    if yticklabels is None:
        yticklabels = ['Não', 'Sim']

    # Normaliza por linha (% de cada classe real que foi prevista em cada coluna)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.where(row_sums > 0, cm / row_sums, 0.0)

    # Texto: "n\n(xx.x%)"
    annot = np.empty(cm.shape, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            pct = cm_norm[i, j] * 100
            annot[i, j] = f"{cm[i, j]:,}\n({pct:.1f}%)"

    sns.heatmap(
        cm_norm, annot=annot, fmt='', cmap=cmap, ax=ax,
        xticklabels=xticklabels, yticklabels=yticklabels,
        cbar=show_cbar,
        annot_kws={"size": fontsize, "weight": "bold"},
        vmin=0, vmax=1,
        linewidths=0.5, linecolor='gray'
    )
    ax.set_title(title, fontweight='bold', fontsize=fontsize + 1, pad=8)


def gerar_visualizacao_comparativa(hist_normal, hist_envenenado):
    """Gera visualização comparativa completa"""
    print("\n[VISUALIZAÇÃO] Gerando gráficos comparativos...")
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()
    
    metricas = ['acuracia', 'f1_score', 'precisao', 'recall', 'auc', 'loss']
    titulos = ['Acurácia Global', 'F1-Score', 'Precisão', 'Recall', 'AUC-ROC', 'Loss']
    
    for idx, (metrica, titulo) in enumerate(zip(metricas, titulos)):
        ax = axes[idx]
        
        rodadas_n = [h['rodada'] for h in hist_normal]
        valores_n = [h[metrica] * 100 if metrica not in ['loss', 'auc'] else h[metrica] 
                     for h in hist_normal]
        
        rodadas_e = [h['rodada'] for h in hist_envenenado]
        valores_e = [h[metrica] * 100 if metrica not in ['loss', 'auc'] else h[metrica]
                     for h in hist_envenenado]
        
        ax.plot(rodadas_n, valores_n, marker='o', linewidth=4, markersize=10,
               color='#2E86AB', label='Normal (sem ataque)', alpha=0.9,
               markeredgecolor='white', markeredgewidth=2)
        ax.plot(rodadas_e, valores_e, marker='s', linewidth=4, markersize=10,
               color='#D62828', label='Envenenado (com ataque)', alpha=0.9,
               markeredgecolor='white', markeredgewidth=2)
        
        ax.fill_between(rodadas_n, valores_n, alpha=0.2, color='#2E86AB')
        ax.fill_between(rodadas_e, valores_e, alpha=0.2, color='#D62828')
        
        # Destaca diferença
        diff_final = abs(valores_n[-1] - valores_e[-1])
        if diff_final > 2:
            sufixo = '%' if metrica not in ['loss', 'auc'] else ''
            ax.annotate(f'Δ = {diff_final:.2f}{sufixo}',
                       xy=(rodadas_e[-1], valores_e[-1]),
                       xytext=(rodadas_e[-1] - 2, (valores_n[-1] + valores_e[-1])/2),
                       fontsize=11, fontweight='bold', color='red',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8),
                       arrowprops=dict(arrowstyle='->', color='red', lw=2))
        
        ax.set_title(f'{titulo}', fontsize=14, fontweight='bold', pad=12)
        ax.set_xlabel('Rodada Federada', fontsize=11, fontweight='bold')
        sufixo_y = '%' if metrica not in ['loss', 'auc'] else ''
        ax.set_ylabel(f'{titulo} {sufixo_y}', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.4, linestyle='--')
        ax.legend(fontsize=10, loc='best', framealpha=0.9)
    
    plt.suptitle('Aprendizado Federado com Dados Distribuídos\n' +
                'Bank Marketing Dataset - Avaliação Global (Servidor Central)',
                fontsize=17, fontweight='bold')
    plt.tight_layout()
    
    import os
    caminho_img = '../../modelagem/supervisionado_x_nao_supervisionado/resultados/bank_fl_distribuido_global.png'
    os.makedirs(os.path.dirname(caminho_img), exist_ok=True)
    plt.savefig(caminho_img, dpi=300, bbox_inches='tight')
    print("  ✓ Salvo: bank_fl_distribuido_global.png")
    plt.close()


def gerar_grafico_convergencia(hist_normal, hist_envenenado):
    """Gera gráfico detalhado de convergência por rodada"""
    print("  Gerando: Convergência por Rodada...")
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    metricas = ['acuracia', 'f1_score', 'auc', 'loss']
    titulos = ['Acurácia (%)', 'F1-Score (%)', 'AUC-ROC', 'Loss']
    
    for idx, (metrica, titulo) in enumerate(zip(metricas, titulos)):
        ax = axes[idx // 2, idx % 2]
        
        rodadas_n = [h['rodada'] for h in hist_normal]
        valores_n = [h[metrica] * 100 if metrica not in ['loss', 'auc'] else h[metrica] 
                     for h in hist_normal]
        
        rodadas_e = [h['rodada'] for h in hist_envenenado]
        valores_e = [h[metrica] * 100 if metrica not in ['loss', 'auc'] else h[metrica]
                     for h in hist_envenenado]
        
        # Normal
        ax.plot(rodadas_n, valores_n, marker='o', linewidth=3, markersize=8,
               color='#06D6A0', label='Cenário Normal', alpha=0.9)
        ax.fill_between(rodadas_n, valores_n, alpha=0.15, color='#06D6A0')
        
        # Envenenado
        ax.plot(rodadas_e, valores_e, marker='s', linewidth=3, markersize=8,
               color='#EF476F', label='Cenário com Ataque', alpha=0.9)
        ax.fill_between(rodadas_e, valores_e, alpha=0.15, color='#EF476F')
        
        ax.set_title(f'{titulo}', fontsize=13, fontweight='bold', pad=10)
        ax.set_xlabel('Rodada', fontsize=11, fontweight='bold')
        ax.set_ylabel(titulo, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=10, loc='best')
    
    plt.suptitle('Bank Marketing - Convergência de Métricas por Rodada\nAprendizado Federado Distribuído',
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    caminho_img = '../../modelagem/supervisionado_x_nao_supervisionado/resultados/bank_convergencia_por_rodada.png'
    plt.savefig(caminho_img, dpi=300, bbox_inches='tight')
    print("    ✓ Salvo: bank_convergencia_por_rodada.png")
    plt.close()


def gerar_grafico_classificacao_por_classe(hist_normal, hist_envenenado):
    """Gera análise de classificação por classe (0=Não, 1=Sim)"""
    print("  Gerando: Análise por Classe...")
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    # Calcula métricas por classe para a última rodada
    cm_normal = hist_normal[-1]['confusion_matrix']
    cm_envenenado = hist_envenenado[-1]['confusion_matrix']
    
    # Acurácia por classe
    ax = axes[0, 0]
    acc_classe_0_normal = cm_normal[0, 0] / cm_normal[0, :].sum() * 100
    acc_classe_1_normal = cm_normal[1, 1] / cm_normal[1, :].sum() * 100
    acc_classe_0_env = cm_envenenado[0, 0] / cm_envenenado[0, :].sum() * 100
    acc_classe_1_env = cm_envenenado[1, 1] / cm_envenenado[1, :].sum() * 100
    
    x = np.arange(2)
    width = 0.35
    ax.bar(x - width/2, [acc_classe_0_normal, acc_classe_1_normal], width, 
           label='Normal', color='#06D6A0', alpha=0.8)
    ax.bar(x + width/2, [acc_classe_0_env, acc_classe_1_env], width,
           label='Envenenado', color='#EF476F', alpha=0.8)
    ax.set_ylabel('Acurácia (%)', fontweight='bold')
    ax.set_title('Acurácia por Classe', fontweight='bold', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(['Classe 0 (Não)', 'Classe 1 (Sim)'])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Precisão por classe
    ax = axes[0, 1]
    prec_classe_0_normal = cm_normal[0, 0] / cm_normal[:, 0].sum() * 100 if cm_normal[:, 0].sum() > 0 else 0
    prec_classe_1_normal = cm_normal[1, 1] / cm_normal[:, 1].sum() * 100 if cm_normal[:, 1].sum() > 0 else 0
    prec_classe_0_env = cm_envenenado[0, 0] / cm_envenenado[:, 0].sum() * 100 if cm_envenenado[:, 0].sum() > 0 else 0
    prec_classe_1_env = cm_envenenado[1, 1] / cm_envenenado[:, 1].sum() * 100 if cm_envenenado[:, 1].sum() > 0 else 0
    
    ax.bar(x - width/2, [prec_classe_0_normal, prec_classe_1_normal], width,
           label='Normal', color='#06D6A0', alpha=0.8)
    ax.bar(x + width/2, [prec_classe_0_env, prec_classe_1_env], width,
           label='Envenenado', color='#EF476F', alpha=0.8)
    ax.set_ylabel('Precisão (%)', fontweight='bold')
    ax.set_title('Precisão por Classe', fontweight='bold', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(['Classe 0 (Não)', 'Classe 1 (Sim)'])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Matriz de Confusão - Normal (com percentuais por linha)
    ax = axes[1, 0]
    _plotar_matriz_confusao(ax, cm_normal, cmap='Greens',
                            title='Matriz de Confusão - Normal',
                            xticklabels=['Pred: Não', 'Pred: Sim'],
                            yticklabels=['Real: Não', 'Real: Sim'])

    # Matriz de Confusão - Envenenado (com percentuais por linha)
    ax = axes[1, 1]
    _plotar_matriz_confusao(ax, cm_envenenado, cmap='Reds',
                            title='Matriz de Confusão - Envenenado',
                            xticklabels=['Pred: Não', 'Pred: Sim'],
                            yticklabels=['Real: Não', 'Real: Sim'])
    
    plt.suptitle('Bank Marketing - Análise por Classe\nRodada Final do Aprendizado Federado',
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    caminho_img = '../../modelagem/supervisionado_x_nao_supervisionado/resultados/bank_analise_por_classe.png'
    plt.savefig(caminho_img, dpi=300, bbox_inches='tight')
    print("    ✓ Salvo: bank_analise_por_classe.png")
    plt.close()


def gerar_matriz_confusao_evolutiva(hist_normal, hist_envenenado):
    """Gera evolução das matrizes de confusão"""
    print("  Gerando: Evolução das Matrizes de Confusão...")
    
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    
    rodadas_analisar = [1, 4, 8, 12]  # Rodadas 1, 4, 8 e final
    
    for idx, rodada in enumerate(rodadas_analisar):
        # Normal
        ax = axes[0, idx]
        cm_normal = hist_normal[rodada - 1]['confusion_matrix']
        _plotar_matriz_confusao(ax, cm_normal, cmap='Blues',
                                title=f'Normal - Rodada {rodada}',
                                xticklabels=['Não', 'Sim'],
                                yticklabels=['Não', 'Sim'],
                                show_cbar=False, fontsize=10)
        if idx == 0:
            ax.set_ylabel('Real', fontweight='bold')

        # Envenenado
        ax = axes[1, idx]
        cm_env = hist_envenenado[rodada - 1]['confusion_matrix']
        _plotar_matriz_confusao(ax, cm_env, cmap='Reds',
                                title=f'Envenenado - Rodada {rodada}',
                                xticklabels=['Não', 'Sim'],
                                yticklabels=['Não', 'Sim'],
                                show_cbar=False, fontsize=10)
        ax.set_xlabel('Predito', fontweight='bold')
        if idx == 0:
            ax.set_ylabel('Real', fontweight='bold')
    
    plt.suptitle('Bank Marketing - Evolução das Matrizes de Confusão\nAprendizado Federado Distribuído',
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    caminho_img = '../../modelagem/supervisionado_x_nao_supervisionado/resultados/bank_matriz_confusao_evolutiva.png'
    plt.savefig(caminho_img, dpi=300, bbox_inches='tight')
    print("    ✓ Salvo: bank_matriz_confusao_evolutiva.png")
    plt.close()


def gerar_tabela_comparativa(hist_normal, hist_envenenado):
    """Gera tabela comparativa de métricas"""
    print("  Gerando: Tabela Comparativa Final...")
    
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')
    
    # Dados da última rodada
    final_normal = hist_normal[-1]
    final_env = hist_envenenado[-1]
    
    metricas = [
        'Acurácia (%)',
        'F1-Score (%)',
        'Precisão (%)',
        'Recall (%)',
        'AUC-ROC',
        'Loss'
    ]
    
    valores_normal = [
        f"{final_normal['acuracia'] * 100:.2f}",
        f"{final_normal['f1_score'] * 100:.2f}",
        f"{final_normal['precisao'] * 100:.2f}",
        f"{final_normal['recall'] * 100:.2f}",
        f"{final_normal['auc']:.4f}",
        f"{final_normal['loss']:.4f}"
    ]
    
    valores_env = [
        f"{final_env['acuracia'] * 100:.2f}",
        f"{final_env['f1_score'] * 100:.2f}",
        f"{final_env['precisao'] * 100:.2f}",
        f"{final_env['recall'] * 100:.2f}",
        f"{final_env['auc']:.4f}",
        f"{final_env['loss']:.4f}"
    ]
    
    degradacoes = [
        f"{(final_normal['acuracia'] - final_env['acuracia']) * 100:.2f}%",
        f"{(final_normal['f1_score'] - final_env['f1_score']) * 100:.2f}%",
        f"{(final_normal['precisao'] - final_env['precisao']) * 100:.2f}%",
        f"{(final_normal['recall'] - final_env['recall']) * 100:.2f}%",
        f"{(final_normal['auc'] - final_env['auc']):.4f}",
        f"{(final_env['loss'] - final_normal['loss']):.4f}"
    ]
    
    table_data = []
    for m, vn, ve, d in zip(metricas, valores_normal, valores_env, degradacoes):
        table_data.append([m, vn, ve, d])
    
    table = ax.table(cellText=table_data,
                     colLabels=['Métrica', 'Normal', 'Envenenado', 'Degradação'],
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.3, 0.2, 0.2, 0.2])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 3)
    
    # Estiliza header
    for i in range(4):
        table[(0, i)].set_facecolor('#2E86AB')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Estiliza células
    for i in range(1, len(table_data) + 1):
        for j in range(4):
            if j == 0:
                table[(i, j)].set_facecolor('#E8F4F8')
                table[(i, j)].set_text_props(weight='bold')
            elif j == 1:
                table[(i, j)].set_facecolor('#D4EDDA')
            elif j == 2:
                table[(i, j)].set_facecolor('#F8D7DA')
            else:
                table[(i, j)].set_facecolor('#FFF3CD')
    
    plt.title('Bank Marketing - Tabela Comparativa de Métricas\nAprendizado Federado Distribuído (Rodada Final)',
             fontsize=15, fontweight='bold', pad=20)
    
    caminho_img = '../../modelagem/supervisionado_x_nao_supervisionado/resultados/bank_tabela_comparativa.png'
    plt.savefig(caminho_img, dpi=300, bbox_inches='tight')
    print("    ✓ Salvo: bank_tabela_comparativa.png")
    plt.close()


def gerar_grafico_impacto_ataque(hist_normal, hist_envenenado):
    """Gera visualização do impacto do ataque"""
    print("  Gerando: Impacto do Ataque...")
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    # Degradação Absoluta
    ax = axes[0]
    final_normal = hist_normal[-1]
    final_env = hist_envenenado[-1]
    
    metricas = ['Acurácia', 'F1-Score', 'Precisão', 'Recall']
    degradacoes = [
        (final_normal['acuracia'] - final_env['acuracia']) * 100,
        (final_normal['f1_score'] - final_env['f1_score']) * 100,
        (final_normal['precisao'] - final_env['precisao']) * 100,
        (final_normal['recall'] - final_env['recall']) * 100
    ]
    
    cores = ['#D62828' if d > 0 else '#06D6A0' for d in degradacoes]
    bars = ax.barh(metricas, degradacoes, color=cores, alpha=0.8)
    
    # Adiciona valores nas barras
    for i, (bar, val) in enumerate(zip(bars, degradacoes)):
        ax.text(val + 1 if val > 0 else val - 1, i, f'{val:.2f}%',
               ha='left' if val > 0 else 'right', va='center',
               fontweight='bold', fontsize=11)
    
    ax.set_xlabel('Degradação (%)', fontweight='bold', fontsize=12)
    ax.set_title('Degradação de Métricas pelo Ataque', fontweight='bold', fontsize=14)
    ax.axvline(x=0, color='black', linestyle='--', linewidth=2)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Severidade do Ataque
    ax = axes[1]
    degradacao_acc = degradacoes[0]
    
    if degradacao_acc > 30:
        severidade = 'CRÍTICA'
        cor_sev = '#8B0000'
    elif degradacao_acc > 20:
        severidade = 'ALTA'
        cor_sev = '#D62828'
    elif degradacao_acc > 10:
        severidade = 'MODERADA'
        cor_sev = '#FF9F1C'
    else:
        severidade = 'BAIXA'
        cor_sev = '#06D6A0'
    
    # Gauge de severidade
    ax.text(0.5, 0.7, 'SEVERIDADE DO ATAQUE', ha='center', fontsize=16,
           fontweight='bold', transform=ax.transAxes)
    ax.text(0.5, 0.45, severidade, ha='center', fontsize=42,
           fontweight='bold', color=cor_sev, transform=ax.transAxes)
    ax.text(0.5, 0.25, f'Degradação de Acurácia: {degradacao_acc:.2f}%',
           ha='center', fontsize=14, transform=ax.transAxes)
    
    # Informações adicionais
    info_text = f"""
    Cliente Malicioso: 1 de 3 (33.3%)
    Método: Sign Flipping Attack
    Taxa de Corrupção: 90%
    Rodadas: {NUM_RODADAS}
    """
    ax.text(0.5, 0.05, info_text, ha='center', fontsize=10,
           transform=ax.transAxes, bbox=dict(boxstyle='round', 
           facecolor='wheat', alpha=0.5))
    
    ax.axis('off')
    
    plt.suptitle('Bank Marketing - Análise de Impacto do Poisoning Attack\nAprendizado Federado Distribuído',
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    caminho_img = '../../modelagem/supervisionado_x_nao_supervisionado/resultados/bank_impacto_ataque.png'
    plt.savefig(caminho_img, dpi=300, bbox_inches='tight')
    print("    ✓ Salvo: bank_impacto_ataque.png")
    plt.close()


def gerar_grafico_desempenho_clientes(hist_cli_normal, hist_cli_env, dados_clientes=None):
    """Gera análise de desempenho por cliente"""
    print("  Gerando: Desempenho por Cliente...")
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    # Filtra última rodada
    rodada_final = max([h['cliente'] for h in hist_cli_normal if 'cliente' in h])
    
    # Separa por cliente
    clientes = [1, 2, 3]
    
    # Acurácia por cliente
    ax = axes[0, 0]
    acc_normal = []
    acc_env = []
    for cli in clientes:
        # Pega avaliações locais da última rodada de cada cliente
        aval_n = [h for h in hist_cli_normal if h.get('cliente') == cli]
        aval_e = [h for h in hist_cli_env if h.get('cliente') == cli]
        if aval_n:
            acc_normal.append(aval_n[-1]['acuracia'] * 100)
        if aval_e:
            acc_env.append(aval_e[-1]['acuracia'] * 100)
    
    x = np.arange(len(clientes))
    width = 0.35
    ax.bar(x - width/2, acc_normal, width, label='Normal', color='#06D6A0', alpha=0.8)
    ax.bar(x + width/2, acc_env, width, label='Envenenado', color='#EF476F', alpha=0.8)
    ax.set_ylabel('Acurácia (%)', fontweight='bold')
    ax.set_title('Acurácia Local por Cliente', fontweight='bold', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Cliente {i}' for i in clientes])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Marca cliente malicioso
    ax.axvline(x=2, color='red', linestyle='--', linewidth=2, alpha=0.5)
    ax.text(2, max(acc_normal + acc_env) * 0.95, 'MALICIOSO', ha='center',
           fontweight='bold', color='red', fontsize=10,
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    # F1-Score por cliente
    ax = axes[0, 1]
    f1_normal = []
    f1_env = []
    for cli in clientes:
        aval_n = [h for h in hist_cli_normal if h.get('cliente') == cli]
        aval_e = [h for h in hist_cli_env if h.get('cliente') == cli]
        if aval_n:
            f1_normal.append(aval_n[-1]['f1_score'] * 100)
        if aval_e:
            f1_env.append(aval_e[-1]['f1_score'] * 100)
    
    ax.bar(x - width/2, f1_normal, width, label='Normal', color='#06D6A0', alpha=0.8)
    ax.bar(x + width/2, f1_env, width, label='Envenenado', color='#EF476F', alpha=0.8)
    ax.set_ylabel('F1-Score (%)', fontweight='bold')
    ax.set_title('F1-Score Local por Cliente', fontweight='bold', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Cliente {i}' for i in clientes])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.axvline(x=2, color='red', linestyle='--', linewidth=2, alpha=0.5)
    
    # Distribuição de dados por cliente
    ax = axes[1, 0]
    if dados_clientes is not None:
        tamanhos = [len(dados_clientes[i][0]) for i in range(len(dados_clientes))]
    else:
        tamanhos = [12352, 9265, 9265]  # Aproximado para 41k dataset
    ax.pie(tamanhos, labels=[f'Cliente {i}\n{t:,} amostras' for i, t in enumerate(tamanhos, 1)],
           autopct='%1.1f%%', startangle=90, colors=['#06D6A0', '#118AB2', '#EF476F'])
    ax.set_title('Distribuição de Dados entre Clientes', fontweight='bold', fontsize=13)
    
    # Comparação de Loss
    ax = axes[1, 1]
    loss_normal = []
    loss_env = []
    for cli in clientes:
        aval_n = [h for h in hist_cli_normal if h.get('cliente') == cli]
        aval_e = [h for h in hist_cli_env if h.get('cliente') == cli]
        if aval_n:
            loss_normal.append(aval_n[-1]['loss'])
        if aval_e:
            loss_env.append(aval_e[-1]['loss'])
    
    ax.bar(x - width/2, loss_normal, width, label='Normal', color='#06D6A0', alpha=0.8)
    ax.bar(x + width/2, loss_env, width, label='Envenenado', color='#EF476F', alpha=0.8)
    ax.set_ylabel('Loss', fontweight='bold')
    ax.set_title('Loss Local por Cliente', fontweight='bold', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Cliente {i}' for i in clientes])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.axvline(x=2, color='red', linestyle='--', linewidth=2, alpha=0.5)
    
    plt.suptitle('Bank Marketing - Análise de Desempenho por Cliente\nAprendizado Federado Distribuído',
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    caminho_img = '../../modelagem/supervisionado_x_nao_supervisionado/resultados/bank_desempenho_clientes.png'
    plt.savefig(caminho_img, dpi=300, bbox_inches='tight')
    print("    ✓ Salvo: bank_desempenho_clientes.png")
    plt.close()


def main():
    """Função principal"""
    print("="*70)
    print("APRENDIZADO FEDERADO COM DADOS DISTRIBUÍDOS")
    print("Bank Marketing Dataset - Poisoning Attack")
    print("="*70)
    
    # 1. Carrega e preprocessa dataset
    df = carregar_e_preprocessar_dataset()
    
    X = df.drop('y', axis=1).values
    y = df['y'].values
    
    print(f"\n[DATASET] Informações:")
    print(f"  ✓ Total de amostras: {len(X)}")
    print(f"  ✓ Total de features: {X.shape[1]}")
    print(f"  ✓ Classe 0 (Não): {(y == 0).sum()} ({(y == 0).sum()/len(y)*100:.1f}%)")
    print(f"  ✓ Classe 1 (Sim): {(y == 1).sum()} ({(y == 1).sum()/len(y)*100:.1f}%)")
    
    # 2. Distribui dados entre clientes e servidor
    dados_clientes, dados_val_global = distribuir_dados_entre_clientes(
        X, y, num_clientes=NUM_CLIENTES, validacao_global_size=0.25
    )
    
    # 3. Executa cenário NORMAL
    print("\n" + "="*70)
    print("CENÁRIO 1: APRENDIZADO FEDERADO NORMAL (sem ataque)")
    print("="*70)
    hist_normal, hist_cli_normal = executar_cenario_completo(
        dados_clientes, dados_val_global, num_rodadas=NUM_RODADAS, envenenado=False
    )
    print(f"\n✅ Resultados Finais (Avaliação Global):")
    print(f"  - Acurácia: {hist_normal[-1]['acuracia']*100:.2f}%")
    print(f"  - F1-Score: {hist_normal[-1]['f1_score']*100:.2f}%")
    print(f"  - AUC-ROC: {hist_normal[-1]['auc']:.4f}")
    
    # 4. Executa cenário ENVENENADO
    print("\n" + "="*70)
    print("CENÁRIO 2: APRENDIZADO FEDERADO COM ATAQUE (Cliente 3 malicioso)")
    print("="*70)
    hist_envenenado, hist_cli_env = executar_cenario_completo(
        dados_clientes, dados_val_global, num_rodadas=NUM_RODADAS, envenenado=True
    )
    print(f"\n⚠️ Resultados Finais (Avaliação Global):")
    print(f"  - Acurácia: {hist_envenenado[-1]['acuracia']*100:.2f}%")
    print(f"  - F1-Score: {hist_envenenado[-1]['f1_score']*100:.2f}%")
    print(f"  - AUC-ROC: {hist_envenenado[-1]['auc']:.4f}")
    
    # 5. Calcula impacto
    degradacao_acc = (hist_normal[-1]['acuracia'] - hist_envenenado[-1]['acuracia']) * 100
    degradacao_f1 = (hist_normal[-1]['f1_score'] - hist_envenenado[-1]['f1_score']) * 100
    degradacao_auc = hist_normal[-1]['auc'] - hist_envenenado[-1]['auc']
    
    print(f"\n" + "="*70)
    print("ANÁLISE DE IMPACTO DO ATAQUE")
    print("="*70)
    print(f"  📉 Degradação de Acurácia: {degradacao_acc:.2f}%")
    print(f"  📉 Degradação de F1-Score: {degradacao_f1:.2f}%")
    print(f"  📉 Degradação de AUC: {degradacao_auc:.4f}")
    
    severidade = 'CRÍTICA' if degradacao_acc > 20 else 'ALTA' if degradacao_acc > 10 else 'MODERADA'
    print(f"  ⚠️ Severidade: {severidade}")
    
    # 6. Gera visualizações
    print("\n[VISUALIZAÇÕES] Gerando todos os gráficos...")
    gerar_visualizacao_comparativa(hist_normal, hist_envenenado)
    gerar_grafico_convergencia(hist_normal, hist_envenenado)
    gerar_grafico_classificacao_por_classe(hist_normal, hist_envenenado)
    gerar_matriz_confusao_evolutiva(hist_normal, hist_envenenado)
    gerar_tabela_comparativa(hist_normal, hist_envenenado)
    gerar_grafico_impacto_ataque(hist_normal, hist_envenenado)
    gerar_grafico_desempenho_clientes(hist_cli_normal, hist_cli_env, dados_clientes)
    
    print("\n✅ Total de visualizações geradas: 7 gráficos")
    
    # 7. Resumo final
    print("\n" + "="*70)
    print("RESUMO FINAL")
    print("="*70)
    print("\n✅ Arquitetura Implementada:")
    print("  1. ✓ Conjunto de validação GLOBAL (servidor central)")
    print("  2. ✓ Conjuntos de dados DIFERENTES para cada cliente")
    print("  3. ✓ Distribuição ESTRATIFICADA de classes")
    print("  4. ✓ Avaliação local (cada cliente) + global (servidor)")
    print("  5. ✓ Agregação FedAvg")
    print("  6. ✓ Poisoning attack (sign flipping)")
    
    print("\n📊 Visualizações geradas:")
    print("  - bank_fl_distribuido_global.png → Comparação de 6 métricas globais")
    print("  - bank_convergencia_por_rodada.png → Convergência detalhada")
    print("  - bank_analise_por_classe.png → Análise por classe (Não/Sim)")
    print("  - bank_matriz_confusao_evolutiva.png → Evolução das matrizes")
    print("  - bank_tabela_comparativa.png → Tabela de métricas finais")
    print("  - bank_impacto_ataque.png → Análise de severidade")
    print("  - bank_desempenho_clientes.png → Desempenho por cliente")
    
    print("\n🎯 Conclusões:")
    print(f"  - Cliente malicioso (33% dos dados) causou impacto {severidade}")
    print(f"  - Dados distribuídos de forma realista entre {NUM_CLIENTES} clientes")
    print(f"  - Validação global independente no servidor central")
    print("="*70)


if __name__ == "__main__":
    main()
