"""
Poisoning Attack em Aprendizado Federado - Bank Marketing Dataset
Ataque de Envenenamento de Modelo usando dados bancários

Dataset: Bank Marketing Campaign
Target: Predizer se cliente vai fazer depósito a prazo (deposit: yes/no)
Features: 40+ features após encoding
Classes: Binário (0 = não depositou, 1 = depositou)

Pipeline: Treina Local → Corrompe Pesos → Avalia Local → Agrega (FedAvg)

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
    log_loss, confusion_matrix, roc_auc_score, roc_curve
)
from sklearn.model_selection import train_test_split
from copy import deepcopy
import warnings
warnings.filterwarnings('ignore')

# Configuração
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

NUM_RODADAS = 10
NUM_CLIENTES = 3


class ModeloFederado:
    """Modelo de Regressão Logística para Aprendizado Federado"""
    
    def __init__(self):
        self._modelo = LogisticRegression(
            max_iter=15,
            random_state=42,
            solver='lbfgs',
            warm_start=True
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


def envenenar_pesos(pesos, taxa=0.8):
    """
    Corrompe pesos do modelo (Sign Flipping Attack)
    Inverte e amplifica os pesos
    """
    pesos_corrompidos = deepcopy(pesos)
    pesos_corrompidos['coef'] = -pesos['coef'] * (1 + taxa)
    pesos_corrompidos['intercept'] = -pesos['intercept'] * (1 + taxa)
    return pesos_corrompidos


def baixar_ou_gerar_dataset():
    """
    Baixa ou gera dataset sintético Bank Marketing
    """
    caminho = r"c:\Users\Administrador\Faculdade\Iniciação-cientifica\project\data\bank-marketing\bank.csv"
    
    # Tenta carregar se já existe
    try:
        df = pd.read_csv(caminho)
        return df
    except FileNotFoundError:
        print("  ⚠️ Dataset não encontrado. Gerando dataset sintético...")
        
        # Gera dataset sintético similar ao Bank Marketing
        np.random.seed(42)
        n_samples = 4521  # Tamanho do dataset original
        
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
            'poutcome': np.random.choice(['unknown', 'failure', 'success', 'other'], n_samples)
        })
        
        # Target: 11.7% de sucesso (como no dataset original)
        df['y'] = np.random.choice(['yes', 'no'], n_samples, p=[0.117, 0.883])
        
        # Cria diretório se não existir
        import os
        os.makedirs(os.path.dirname(caminho), exist_ok=True)
        df.to_csv(caminho, index=False)
        print(f"  ✓ Dataset sintético gerado e salvo em: {caminho}")
        
        return df


def preprocessar_bank_dataset(caminho=None):
    """
    Preprocessa o dataset Bank Marketing seguindo o notebook
    """
    print("\n[PREPROCESSAMENTO] Carregando dataset...")
    
    # Carrega dados
    df = baixar_ou_gerar_dataset()
    
    print(f"  ✓ Dataset carregado: {df.shape[0]} amostras, {df.shape[1]} features")
    
    # Renomeia colunas para padrão do notebook
    df.columns = ['age', 'job', 'marital', 'education', 'default_credit', 'balance', 
                  'house_loan', 'personal_loan', 'contact_type', 'day_ofweek', 'month', 
                  'duration', 'current_campaign_contact_count', 'days_passed',
                  'previous_coutact_count', 'previous_campaing_outcome', 'deposit']
    
    # Imputa valores unknown em job
    def impute_job(row):
        job, edu = row['job'], row['education']
        if job == 'unknown':
            return 'management' if edu == 'tertiary' else 'blue-collar'
        return job
    
    df['job'] = df.apply(impute_job, axis=1)
    
    # Imputa valores unknown em education
    def impute_education(row):
        edu, job = row['education'], row['job']
        if edu == 'unknown':
            if job == 'housemaid':
                return 'primary'
            elif job in ['entrepreneur', 'self-employed', 'management']:
                return 'tertiary'
            return 'secondary'
        return edu
    
    df['education'] = df.apply(impute_education, axis=1)
    
    # Imputa previous_campaing_outcome
    df['previous_campaing_outcome'].replace('unknown', 'nonexistent', inplace=True)
    
    # Imputa contact_type
    df['contact_type'].replace('unknown', 'non-call', inplace=True)
    
    # Remove balance (conforme notebook)
    df.drop('balance', axis=1, inplace=True)
    
    # Encoding das variáveis categóricas
    df = df.join(pd.get_dummies(df['job'], drop_first=True, prefix='job'))
    df = df.join(pd.get_dummies(df['marital'], drop_first=True, prefix='marital'))
    df = df.join(pd.get_dummies(df['education'], drop_first=True, prefix='edu'))
    df = df.join(pd.get_dummies(df['contact_type'], drop_first=True, prefix='contact'))
    df = df.join(pd.get_dummies(df['month'], drop_first=True, prefix='month'))
    df = df.join(pd.get_dummies(df['previous_campaing_outcome'], drop_first=True, prefix='prev'))
    
    # Converte target para binário
    df['deposit'].replace({'yes': 1, 'no': 0}, inplace=True)
    df.replace({'yes': 1, 'no': 0}, inplace=True)
    
    # Remove colunas originais categóricas
    df.drop(['job', 'marital', 'education', 'contact_type', 'month', 
             'previous_campaing_outcome'], axis=1, inplace=True)
    
    # Remove duplicatas
    df = df.drop_duplicates()
    
    print(f"  ✓ Após preprocessamento: {df.shape[0]} amostras, {df.shape[1]} features")
    
    # Separa X e y
    X = df.drop('deposit', axis=1).values
    y = df['deposit'].values
    
    distribuicao = np.bincount(y)
    print(f"  ✓ Distribuição de classes: Não depositou={distribuicao[0]}, Depositou={distribuicao[1]}")
    print(f"  ✓ Taxa de deposito: {distribuicao[1]/len(y)*100:.2f}%")
    
    return X, y


def executar_cenario_federado(dados_clientes, dados_validacao, num_rodadas=10, envenenado=False):
    """Executa cenário federado completo"""
    historico = []
    X_val, y_val = dados_validacao
    
    # Inicializa modelo global com todas as classes
    print("\n[INICIALIZAÇÃO] Criando modelo global...")
    X_todos = np.vstack([dados_clientes[i][0] for i in range(NUM_CLIENTES)])
    y_todos = np.hstack([dados_clientes[i][1] for i in range(NUM_CLIENTES)])
    
    modelo_inicial = ModeloFederado()
    modelo_inicial.treinar_incremental(X_todos[:30], y_todos[:30], None)
    pesos_globais = modelo_inicial.obter_pesos()
    
    # Degrada pesos iniciais
    pesos_globais['coef'] = pesos_globais['coef'] * 0.1
    pesos_globais['intercept'] = pesos_globais['intercept'] * 0.1
    
    print(f"  ✓ Modelo global inicializado")
    print(f"\n[TREINAMENTO] Executando {num_rodadas} rodadas federadas...")
    
    for rodada in range(1, num_rodadas + 1):
        pesos_locais = []
        
        # Fase 1: Treinamento local
        for idx, (X_cliente, y_cliente) in enumerate(dados_clientes, 1):
            modelo_local = ModeloFederado()
            modelo_local.treinar_incremental(X_cliente, y_cliente, pesos_globais)
            pesos_local = modelo_local.obter_pesos()
            
            # Cliente 3 envenena (se cenário envenenado)
            if envenenado and idx == 3:
                pesos_local = envenenar_pesos(pesos_local, taxa=0.8)
            
            pesos_locais.append(pesos_local)
        
        # Fase 2: Agregação FedAvg
        pesos_globais = {
            'coef': np.mean([p['coef'] for p in pesos_locais], axis=0),
            'intercept': np.mean([p['intercept'] for p in pesos_locais], axis=0),
            'classes': pesos_locais[0]['classes']
        }
        
        # Fase 3: Avaliação global
        modelo_global = ModeloFederado()
        modelo_global._carregar_pesos(pesos_globais)
        metricas = modelo_global.avaliar(X_val, y_val)
        metricas['rodada'] = rodada
        
        historico.append(metricas)
        
        if rodada % 2 == 0 or rodada == num_rodadas:
            print(f"  Rodada {rodada:2d}: Acurácia={metricas['acuracia']*100:5.2f}%, "
                  f"F1={metricas['f1_score']*100:5.2f}%, AUC={metricas['auc']:.4f}")
    
    return historico


def gerar_visualizacao_comparativa(hist_normal, hist_envenenado):
    """Gera gráfico comparativo completo"""
    print("\n[VISUALIZAÇÃO] Gerando gráficos...")
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()
    
    metricas = ['acuracia', 'f1_score', 'precisao', 'recall', 'auc', 'loss']
    titulos = ['Acurácia', 'F1-Score', 'Precisão', 'Recall', 'AUC-ROC', 'Loss']
    
    for idx, (metrica, titulo) in enumerate(zip(metricas, titulos)):
        ax = axes[idx]
        
        rodadas_n = [h['rodada'] for h in hist_normal]
        valores_n = [h[metrica] * 100 if metrica != 'loss' else h[metrica] 
                     for h in hist_normal]
        
        rodadas_e = [h['rodada'] for h in hist_envenenado]
        valores_e = [h[metrica] * 100 if metrica != 'loss' else h[metrica]
                     for h in hist_envenenado]
        
        ax.plot(rodadas_n, valores_n, marker='o', linewidth=4, markersize=10,
               color='#2E86AB', label='Normal (sem ataque)', alpha=0.9,
               markeredgecolor='white', markeredgewidth=2)
        ax.plot(rodadas_e, valores_e, marker='s', linewidth=4, markersize=10,
               color='#D62828', label='Envenenado (com ataque)', alpha=0.9,
               markeredgecolor='white', markeredgewidth=2)
        
        ax.fill_between(rodadas_n, valores_n, alpha=0.2, color='#2E86AB')
        ax.fill_between(rodadas_e, valores_e, alpha=0.2, color='#D62828')
        
        # Destaca diferença final
        diff_final = abs(valores_n[-1] - valores_e[-1])
        if diff_final > 2:
            ax.annotate(f'Δ = {diff_final:.1f}{"%" if metrica != "loss" else ""}',
                       xy=(rodadas_e[-1], valores_e[-1]),
                       xytext=(rodadas_e[-1] - 1.5, (valores_n[-1] + valores_e[-1])/2),
                       fontsize=11, fontweight='bold', color='red',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8),
                       arrowprops=dict(arrowstyle='->', color='red', lw=2))
        
        ax.set_title(f'{titulo} por Rodada', fontsize=14, fontweight='bold', pad=12)
        ax.set_xlabel('Rodada Federada', fontsize=11, fontweight='bold')
        ax.set_ylabel(f'{titulo} {"(%)" if metrica not in ["loss", "auc"] else ""}',
                     fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.4, linestyle='--')
        ax.legend(fontsize=10, loc='best', framealpha=0.9)
    
    plt.suptitle('Poisoning Attack - Bank Marketing Dataset\n' +
                'Comparação Normal vs Envenenado',
                fontsize=17, fontweight='bold')
    plt.tight_layout()
    
    # Cria diretório se não existir
    import os
    caminho_img = '../../modelagem/apresentação/bank_comparacao_poisoning.png'
    os.makedirs(os.path.dirname(caminho_img), exist_ok=True)
    
    plt.savefig(caminho_img, dpi=300, bbox_inches='tight')
    print("  ✓ Salvo: bank_comparacao_poisoning.png")
    plt.close()


def gerar_matriz_confusao(hist_normal, hist_envenenado):
    """Gera matriz de confusão comparativa"""
    print("  [Gerando] Matriz de confusão...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    classes = ['Não Depositou', 'Depositou']
    
    # Normal - Rodada 1
    cm_n1 = hist_normal[0]['confusion_matrix']
    cm_n1_pct = cm_n1.astype('float') / cm_n1.sum(axis=1)[:, np.newaxis] * 100
    
    ax = axes[0, 0]
    sns.heatmap(cm_n1_pct, annot=True, fmt='.1f', cmap='Blues',
               xticklabels=classes, yticklabels=classes, ax=ax,
               cbar_kws={'label': 'Percentual (%)'}, vmin=0, vmax=100)
    ax.set_title(f'Normal - Rodada 1\nAcurácia: {hist_normal[0]["acuracia"]*100:.1f}%',
                fontsize=13, fontweight='bold')
    ax.set_ylabel('Classe Real', fontsize=11, fontweight='bold')
    ax.set_xlabel('Classe Predita', fontsize=11, fontweight='bold')
    
    # Normal - Final
    cm_nf = hist_normal[-1]['confusion_matrix']
    cm_nf_pct = cm_nf.astype('float') / cm_nf.sum(axis=1)[:, np.newaxis] * 100
    
    ax = axes[0, 1]
    sns.heatmap(cm_nf_pct, annot=True, fmt='.1f', cmap='Blues',
               xticklabels=classes, yticklabels=classes, ax=ax,
               cbar_kws={'label': 'Percentual (%)'}, vmin=0, vmax=100)
    ax.set_title(f'Normal - Rodada Final\nAcurácia: {hist_normal[-1]["acuracia"]*100:.1f}%',
                fontsize=13, fontweight='bold')
    ax.set_ylabel('Classe Real', fontsize=11, fontweight='bold')
    ax.set_xlabel('Classe Predita', fontsize=11, fontweight='bold')
    
    # Envenenado - Rodada 1
    cm_e1 = hist_envenenado[0]['confusion_matrix']
    cm_e1_pct = cm_e1.astype('float') / cm_e1.sum(axis=1)[:, np.newaxis] * 100
    
    ax = axes[1, 0]
    sns.heatmap(cm_e1_pct, annot=True, fmt='.1f', cmap='Reds',
               xticklabels=classes, yticklabels=classes, ax=ax,
               cbar_kws={'label': 'Percentual (%)'}, vmin=0, vmax=100)
    ax.set_title(f'Envenenado - Rodada 1\nAcurácia: {hist_envenenado[0]["acuracia"]*100:.1f}%',
                fontsize=13, fontweight='bold')
    ax.set_ylabel('Classe Real', fontsize=11, fontweight='bold')
    ax.set_xlabel('Classe Predita', fontsize=11, fontweight='bold')
    
    # Envenenado - Final
    cm_ef = hist_envenenado[-1]['confusion_matrix']
    cm_ef_pct = cm_ef.astype('float') / cm_ef.sum(axis=1)[:, np.newaxis] * 100
    
    ax = axes[1, 1]
    sns.heatmap(cm_ef_pct, annot=True, fmt='.1f', cmap='Reds',
               xticklabels=classes, yticklabels=classes, ax=ax,
               cbar_kws={'label': 'Percentual (%)'}, vmin=0, vmax=100)
    ax.set_title(f'Envenenado - Rodada Final\nAcurácia: {hist_envenenado[-1]["acuracia"]*100:.1f}%',
                fontsize=13, fontweight='bold')
    ax.set_ylabel('Classe Real', fontsize=11, fontweight='bold')
    ax.set_xlabel('Classe Predita', fontsize=11, fontweight='bold')
    
    plt.suptitle('Matriz de Confusão: Bank Marketing Dataset\n' +
                'Evolução Normal vs Envenenado',
                fontsize=17, fontweight='bold')
    plt.tight_layout()
    
    import os
    caminho_img = '../../modelagem/apresentação/bank_matriz_confusao.png'
    os.makedirs(os.path.dirname(caminho_img), exist_ok=True)
    
    plt.savefig(caminho_img, dpi=300, bbox_inches='tight')
    print("  ✓ Salvo: bank_matriz_confusao.png")
    plt.close()


def main():
    """Função principal"""
    print("="*70)
    print("POISONING ATTACK - BANK MARKETING DATASET")
    print("Aprendizado Federado com Ataque de Envenenamento")
    print("="*70)
    
    # 1. Carrega e preprocessa dataset
    X, y = preprocessar_bank_dataset()
    
    # 2. Divide dados
    print("\n[DIVISÃO] Preparando dados para FL...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Divide treino entre 3 clientes
    X_c1, X_temp, y_c1, y_temp = train_test_split(
        X_train, y_train, test_size=0.66, random_state=42, stratify=y_train
    )
    X_c2, X_c3, y_c2, y_c3 = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    dados_clientes = [(X_c1, y_c1), (X_c2, y_c2), (X_c3, y_c3)]
    dados_validacao = (X_val, y_val)
    
    print(f"  ✓ Cliente 1: {len(X_c1)} amostras (Honesto)")
    print(f"  ✓ Cliente 2: {len(X_c2)} amostras (Honesto)")
    print(f"  ✓ Cliente 3: {len(X_c3)} amostras (⚠️ MALICIOSO)")
    print(f"  ✓ Validação: {len(X_val)} amostras")
    
    # 3. Executa cenário NORMAL
    print("\n" + "="*70)
    print("CENÁRIO 1: TREINAMENTO NORMAL (sem ataque)")
    print("="*70)
    hist_normal = executar_cenario_federado(
        dados_clientes, dados_validacao, num_rodadas=NUM_RODADAS, envenenado=False
    )
    print(f"\n✓ Acurácia final: {hist_normal[-1]['acuracia']*100:.2f}%")
    print(f"✓ F1-Score final: {hist_normal[-1]['f1_score']*100:.2f}%")
    print(f"✓ AUC-ROC final: {hist_normal[-1]['auc']:.4f}")
    
    # 4. Executa cenário ENVENENADO
    print("\n" + "="*70)
    print("CENÁRIO 2: TREINAMENTO ENVENENADO (com ataque no Cliente 3)")
    print("="*70)
    hist_envenenado = executar_cenario_federado(
        dados_clientes, dados_validacao, num_rodadas=NUM_RODADAS, envenenado=True
    )
    print(f"\n✓ Acurácia final: {hist_envenenado[-1]['acuracia']*100:.2f}%")
    print(f"✓ F1-Score final: {hist_envenenado[-1]['f1_score']*100:.2f}%")
    print(f"✓ AUC-ROC final: {hist_envenenado[-1]['auc']:.4f}")
    
    degradacao_acc = (hist_normal[-1]['acuracia'] - hist_envenenado[-1]['acuracia']) * 100
    degradacao_f1 = (hist_normal[-1]['f1_score'] - hist_envenenado[-1]['f1_score']) * 100
    
    print(f"\n⚠️ Degradação de Acurácia: {degradacao_acc:.2f}%")
    print(f"⚠️ Degradação de F1-Score: {degradacao_f1:.2f}%")
    
    # 5. Gera visualizações
    print("\n" + "="*70)
    print("GERANDO VISUALIZAÇÕES")
    print("="*70)
    gerar_visualizacao_comparativa(hist_normal, hist_envenenado)
    gerar_matriz_confusao(hist_normal, hist_envenenado)
    
    # 6. Resumo final
    print("\n" + "="*70)
    print("ANÁLISE CONCLUÍDA")
    print("="*70)
    print("\n📊 Visualizações geradas:")
    print("  1. bank_comparacao_poisoning.png")
    print("     → 6 métricas comparadas (Acurácia, F1, Precisão, Recall, AUC, Loss)")
    print("  2. bank_matriz_confusao.png")
    print("     → Evolução da matriz de confusão")
    
    print("\n📈 Conclusões:")
    severidade = 'CRÍTICA' if degradacao_acc > 20 else 'ALTA' if degradacao_acc > 10 else 'MODERADA'
    print(f"  - Severidade do ataque: {severidade}")
    print(f"  - Dataset: Bank Marketing ({len(X)} amostras processadas)")
    print(f"  - Cliente malicioso (33%) impactou o modelo global")
    print(f"  - Pipeline: Treinar → Corromper → Avaliar → Agregar (FedAvg)")
    print("="*70)


if __name__ == "__main__":
    main()
