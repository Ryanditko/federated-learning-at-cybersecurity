"""
Demonstração com Amostra Reduzida - Dataset Iris
Poisoning Attack em Aprendizado Federado

Usa apenas uma AMOSTRA PEQUENA do dataset para demonstração didática:
- 30 amostras de treino (10 por cliente)
- 15 amostras de validação
- Total: 45 amostras (30% do Iris completo)

Pipeline: Treina Local → Corrompe Pesos → Avalia Local → Agrega (FedAvg)

Autor: Projeto de Iniciação Científica
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, 
    log_loss, confusion_matrix
)
from sklearn.model_selection import train_test_split
from copy import deepcopy
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Configuração de estilo
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Configurações
ESPECIES = ['Setosa', 'Versicolor', 'Virginica']
CORES_ESPECIES = {
    'Setosa': '#FF6B6B',
    'Versicolor': '#4ECDC4',
    'Virginica': '#FFD93D'
}

# Tamanho da amostra (30% do dataset original)
TAMANHO_AMOSTRA = 45  # 30 treino + 15 validação


class ModeloSimples:
    """Modelo de classificação simples"""
    
    def __init__(self):
        self._modelo = LogisticRegression(
            max_iter=20,
            random_state=42,
            multi_class='multinomial',
            solver='lbfgs',
            warm_start=True
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def treinar(self, X, y, pesos_iniciais=None):
        """Treina o modelo a partir de pesos iniciais (se fornecidos)"""
        X_scaled = self.scaler.fit_transform(X)
        
        # Se tem pesos iniciais e as dimensões são compatíveis, carrega
        if pesos_iniciais is not None:
            try:
                # Verifica se o modelo já foi inicializado com as classes corretas
                if not hasattr(self._modelo, 'classes_') or not np.array_equal(self._modelo.classes_, pesos_iniciais['classes']):
                    # Primeira vez: faz um fit dummy para inicializar o modelo
                    self._modelo.fit(X_scaled, y)
                
                # Agora carrega os pesos
                self._carregar_pesos(pesos_iniciais)
                self.is_fitted = True
                
                # Treina mais a partir desses pesos
                self._modelo.fit(X_scaled, y)
            except (ValueError, AttributeError) as e:
                # Se falhar, treina normalmente do zero
                self._modelo.fit(X_scaled, y)
        else:
            self._modelo.fit(X_scaled, y)
        
        self.is_fitted = True
    
    def avaliar(self, X, y):
        """Avalia o modelo e retorna métricas"""
        if not hasattr(self.scaler, 'mean_'):
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        y_pred = self._modelo.predict(X_scaled)
        y_proba = self._modelo.predict_proba(X_scaled)
        
        return {
            'acuracia': accuracy_score(y, y_pred),
            'f1_score': f1_score(y, y_pred, average='weighted'),
            'precisao': precision_score(y, y_pred, average='weighted'),
            'recall': recall_score(y, y_pred, average='weighted'),
            'loss': log_loss(y, y_proba),
            'y_pred': y_pred,
            'y_true': y,
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
    Corrompe os pesos do modelo (Sign Flipping Attack)
    
    Inverte e amplifica os pesos para forçar predições incorretas
    """
    pesos_corrompidos = deepcopy(pesos)
    pesos_corrompidos['coef'] = -pesos['coef'] * (1 + taxa)
    pesos_corrompidos['intercept'] = -pesos['intercept'] * (1 + taxa)
    return pesos_corrompidos


def executar_rodada_federada(dados_clientes, dados_validacao, pesos_globais, cliente_malicioso=3):
    """
    Executa UMA rodada federada completa
    
    Pipeline por cliente:
    1. Treina modelo local (partir dos pesos globais)
    2. Corrompe pesos (se for cliente malicioso)
    3. Avalia modelo local
    4. Envia pesos para servidor
    
    Servidor:
    5. Agrega pesos (FedAvg)
    6. Avalia modelo global
    """
    pesos_locais = []
    avaliacoes_locais = []
    
    # FASE 1: TREINAMENTO LOCAL
    for idx, (X_cliente, y_cliente) in enumerate(dados_clientes, 1):
        modelo_local = ModeloSimples()
        
        # 1. TREINA MODELO LOCAL (a partir do modelo global)
        modelo_local.treinar(X_cliente, y_cliente, pesos_globais)
        pesos_local = modelo_local.obter_pesos()
        
        # 2. CORROMPE PESOS (se for cliente malicioso)
        if idx == cliente_malicioso:
            pesos_local = envenenar_pesos(pesos_local, taxa=0.8)
        
        # 3. AVALIA MODELO LOCAL
        X_val, y_val = dados_validacao
        avaliacao_local = modelo_local.avaliar(X_val, y_val)
        avaliacao_local['cliente'] = idx
        avaliacao_local['envenenado'] = (idx == cliente_malicioso)
        
        pesos_locais.append(pesos_local)
        avaliacoes_locais.append(avaliacao_local)
    
    # FASE 2: AGREGAÇÃO FEDAVG
    pesos_globais_novos = {
        'coef': np.mean([p['coef'] for p in pesos_locais], axis=0),
        'intercept': np.mean([p['intercept'] for p in pesos_locais], axis=0),
        'classes': pesos_locais[0]['classes']
    }
    
    # FASE 3: AVALIAÇÃO GLOBAL
    modelo_global = ModeloSimples()
    modelo_global._carregar_pesos(pesos_globais_novos)
    X_val, y_val = dados_validacao
    avaliacao_global = modelo_global.avaliar(X_val, y_val)
    
    return pesos_globais_novos, avaliacao_global, avaliacoes_locais


def executar_cenario(dados_clientes, dados_validacao, num_rodadas=8, envenenado=False):
    """Executa múltiplas rodadas federadas"""
    historico_global = []
    historico_clientes = []
    
    # Inicializa modelo global com TODAS as classes desde o início
    # Treina com dados de todos os clientes para garantir que vê todas as 3 classes
    X_todos = np.vstack([dados_clientes[i][0] for i in range(3)])
    y_todos = np.hstack([dados_clientes[i][1] for i in range(3)])
    
    modelo_inicial = ModeloSimples()
    modelo_inicial.treinar(X_todos[:9], y_todos[:9], None)  # Treina com 9 amostras (3 de cada classe)
    pesos_globais = modelo_inicial.obter_pesos()
    
    # Degrada pesos iniciais para forçar evolução gradual
    pesos_globais['coef'] = pesos_globais['coef'] * 0.05
    pesos_globais['intercept'] = pesos_globais['intercept'] * 0.05
    
    for rodada in range(1, num_rodadas + 1):
        cliente_malicioso = 3 if envenenado else None
        
        pesos_globais, aval_global, aval_clientes = executar_rodada_federada(
            dados_clientes, dados_validacao, pesos_globais, cliente_malicioso
        )
        
        aval_global['rodada'] = rodada
        historico_global.append(aval_global)
        
        for aval in aval_clientes:
            aval['rodada'] = rodada
            historico_clientes.append(aval)
    
    return historico_global, historico_clientes


def gerar_grafico_evolucao_metricas(hist_normal, hist_envenenado):
    """Visualização 1: Evolução das métricas principais"""
    print("\n[GERANDO] Gráfico de Evolução de Métricas")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    metricas = ['acuracia', 'f1_score', 'precisao', 'loss']
    titulos = ['Acurácia', 'F1-Score', 'Precisão', 'Loss']
    
    for idx, (metrica, titulo) in enumerate(zip(metricas, titulos)):
        ax = axes[idx]
        
        rodadas_normal = [h['rodada'] for h in hist_normal]
        valores_normal = [h[metrica] * 100 if metrica != 'loss' else h[metrica] 
                         for h in hist_normal]
        
        rodadas_env = [h['rodada'] for h in hist_envenenado]
        valores_env = [h[metrica] * 100 if metrica != 'loss' else h[metrica] 
                      for h in hist_envenenado]
        
        # Plota com linhas grossas e marcadores grandes
        ax.plot(rodadas_normal, valores_normal, marker='o', linewidth=4, markersize=12,
               color='#2E86AB', label='Normal (sem ataque)', alpha=0.9,
               markeredgecolor='white', markeredgewidth=2)
        ax.plot(rodadas_env, valores_env, marker='s', linewidth=4, markersize=12,
               color='#D62828', label='Envenenado (com ataque)', alpha=0.9,
               markeredgecolor='white', markeredgewidth=2)
        
        # Área sombreada
        ax.fill_between(rodadas_normal, valores_normal, alpha=0.2, color='#2E86AB')
        ax.fill_between(rodadas_env, valores_env, alpha=0.2, color='#D62828')
        
        # Destaca diferença final
        diff_final = abs(valores_normal[-1] - valores_env[-1])
        if diff_final > 3:
            ax.annotate(f'Δ = {diff_final:.1f}{"%" if metrica != "loss" else ""}',
                       xy=(rodadas_env[-1], valores_env[-1]),
                       xytext=(rodadas_env[-1] - 1, (valores_normal[-1] + valores_env[-1])/2),
                       fontsize=12, fontweight='bold', color='red',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8),
                       arrowprops=dict(arrowstyle='->', color='red', lw=2))
        
        ax.set_title(f'{titulo} por Rodada', fontsize=15, fontweight='bold', pad=15)
        ax.set_xlabel('Rodada Federada', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'{titulo} {"(%)" if metrica != "loss" else ""}', 
                     fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.4, linestyle='--')
        ax.legend(fontsize=11, loc='best', framealpha=0.9)
    
    plt.suptitle('Evolução de Métricas - Comparação Normal vs Envenenado\n' +
                f'Dataset Iris (Amostra: {TAMANHO_AMOSTRA} amostras)',
                fontsize=17, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../../modelagem/apresentation/amostra_evolucao_metricas.png', 
                dpi=300, bbox_inches='tight')
    print("  ✓ Salvo: amostra_evolucao_metricas.png")
    plt.close()


def gerar_grafico_clientes_locais(hist_clientes_env):
    """Visualização 2: Desempenho de cada cliente (envenenado)"""
    print("\n[GERANDO] Gráfico de Desempenho por Cliente")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    metricas = ['acuracia', 'f1_score', 'precisao', 'recall']
    titulos = ['Acurácia', 'F1-Score', 'Precisão', 'Recall']
    
    for idx, (metrica, titulo) in enumerate(zip(metricas, titulos)):
        ax = axes[idx]
        
        for cliente_id in [1, 2, 3]:
            dados_cliente = [h for h in hist_clientes_env if h['cliente'] == cliente_id]
            rodadas = [h['rodada'] for h in dados_cliente]
            valores = [h[metrica] * 100 for h in dados_cliente]
            
            cor = '#D62828' if cliente_id == 3 else '#2E86AB'
            estilo = 's' if cliente_id == 3 else 'o'
            label = f'Cliente {cliente_id} {"⚠️ MALICIOSO" if cliente_id == 3 else "✓ Honesto"}'
            lw = 5 if cliente_id == 3 else 3
            ms = 14 if cliente_id == 3 else 10
            
            ax.plot(rodadas, valores, marker=estilo, linewidth=lw, markersize=ms,
                   color=cor, label=label, alpha=0.9,
                   markeredgecolor='white', markeredgewidth=2)
        
        ax.set_title(f'{titulo} dos Clientes Locais', fontsize=14, fontweight='bold', pad=12)
        ax.set_xlabel('Rodada Federada', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'{titulo} (%)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.4, linestyle='--')
        ax.legend(fontsize=10, loc='best', framealpha=0.9)
        
        # Destaca cliente malicioso
        ax.axhline(y=50, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
    
    plt.suptitle('Desempenho Local dos Clientes - Cenário Envenenado\n' +
                f'Dataset Iris (Amostra: {TAMANHO_AMOSTRA} amostras)',
                fontsize=17, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../../modelagem/apresentation/amostra_desempenho_clientes.png', 
                dpi=300, bbox_inches='tight')
    print("  ✓ Salvo: amostra_desempenho_clientes.png")
    plt.close()


def gerar_matriz_confusao_comparativa(hist_normal, hist_envenenado):
    """Visualização 3: Matrizes de confusão (inicial vs final)"""
    print("\n[GERANDO] Matriz de Confusão Comparativa")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # Normal - Rodada 1
    cm_normal_1 = hist_normal[0]['confusion_matrix']
    cm_normal_1_pct = cm_normal_1.astype('float') / cm_normal_1.sum(axis=1)[:, np.newaxis] * 100
    
    ax = axes[0, 0]
    sns.heatmap(cm_normal_1_pct, annot=True, fmt='.1f', cmap='Blues',
               xticklabels=ESPECIES, yticklabels=ESPECIES, ax=ax,
               cbar_kws={'label': 'Acurácia (%)'}, vmin=0, vmax=100)
    ax.set_title(f'Normal - Rodada 1\nAcurácia: {hist_normal[0]["acuracia"]*100:.1f}%',
                fontsize=13, fontweight='bold')
    ax.set_ylabel('Classe Real', fontsize=11, fontweight='bold')
    ax.set_xlabel('Classe Predita', fontsize=11, fontweight='bold')
    
    # Normal - Rodada Final
    cm_normal_f = hist_normal[-1]['confusion_matrix']
    cm_normal_f_pct = cm_normal_f.astype('float') / cm_normal_f.sum(axis=1)[:, np.newaxis] * 100
    
    ax = axes[0, 1]
    sns.heatmap(cm_normal_f_pct, annot=True, fmt='.1f', cmap='Blues',
               xticklabels=ESPECIES, yticklabels=ESPECIES, ax=ax,
               cbar_kws={'label': 'Acurácia (%)'}, vmin=0, vmax=100)
    ax.set_title(f'Normal - Rodada Final\nAcurácia: {hist_normal[-1]["acuracia"]*100:.1f}%',
                fontsize=13, fontweight='bold')
    ax.set_ylabel('Classe Real', fontsize=11, fontweight='bold')
    ax.set_xlabel('Classe Predita', fontsize=11, fontweight='bold')
    
    # Envenenado - Rodada 1
    cm_env_1 = hist_envenenado[0]['confusion_matrix']
    cm_env_1_pct = cm_env_1.astype('float') / cm_env_1.sum(axis=1)[:, np.newaxis] * 100
    
    ax = axes[1, 0]
    sns.heatmap(cm_env_1_pct, annot=True, fmt='.1f', cmap='Reds',
               xticklabels=ESPECIES, yticklabels=ESPECIES, ax=ax,
               cbar_kws={'label': 'Acurácia (%)'}, vmin=0, vmax=100)
    ax.set_title(f'Envenenado - Rodada 1\nAcurácia: {hist_envenenado[0]["acuracia"]*100:.1f}%',
                fontsize=13, fontweight='bold')
    ax.set_ylabel('Classe Real', fontsize=11, fontweight='bold')
    ax.set_xlabel('Classe Predita', fontsize=11, fontweight='bold')
    
    # Envenenado - Rodada Final
    cm_env_f = hist_envenenado[-1]['confusion_matrix']
    cm_env_f_pct = cm_env_f.astype('float') / cm_env_f.sum(axis=1)[:, np.newaxis] * 100
    
    ax = axes[1, 1]
    sns.heatmap(cm_env_f_pct, annot=True, fmt='.1f', cmap='Reds',
               xticklabels=ESPECIES, yticklabels=ESPECIES, ax=ax,
               cbar_kws={'label': 'Acurácia (%)'}, vmin=0, vmax=100)
    ax.set_title(f'Envenenado - Rodada Final\nAcurácia: {hist_envenenado[-1]["acuracia"]*100:.1f}%',
                fontsize=13, fontweight='bold')
    ax.set_ylabel('Classe Real', fontsize=11, fontweight='bold')
    ax.set_xlabel('Classe Predita', fontsize=11, fontweight='bold')
    
    plt.suptitle('Matriz de Confusão: Evolução Normal vs Envenenado\n' +
                f'Dataset Iris (Amostra: {TAMANHO_AMOSTRA} amostras)',
                fontsize=17, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../../modelagem/apresentation/amostra_matriz_confusao_comparativa.png', 
                dpi=300, bbox_inches='tight')
    print("  ✓ Salvo: amostra_matriz_confusao_comparativa.png")
    plt.close()


def gerar_tabela_resumo(hist_normal, hist_envenenado):
    """Visualização 4: Tabela resumo com estatísticas"""
    print("\n[GERANDO] Tabela Resumo")
    
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')
    
    # Dados finais
    normal_final = hist_normal[-1]
    env_final = hist_envenenado[-1]
    
    # Calcula estatísticas
    dados_tabela = [
        ['Métrica', 'Normal', 'Envenenado', 'Degradação', 'Status'],
        ['', '', '', '', ''],
        ['Acurácia', 
         f'{normal_final["acuracia"]*100:.2f}%',
         f'{env_final["acuracia"]*100:.2f}%',
         f'{(normal_final["acuracia"] - env_final["acuracia"])*100:.2f}%',
         '⚠️ CRÍTICO' if (normal_final["acuracia"] - env_final["acuracia"])*100 > 20 else '⚠️ ALTO' if (normal_final["acuracia"] - env_final["acuracia"])*100 > 10 else '✓ MODERADO'],
        
        ['Precisão',
         f'{normal_final["precisao"]*100:.2f}%',
         f'{env_final["precisao"]*100:.2f}%',
         f'{(normal_final["precisao"] - env_final["precisao"])*100:.2f}%',
         '⚠️ ALTO' if (normal_final["precisao"] - env_final["precisao"])*100 > 10 else '✓ MODERADO'],
        
        ['Recall',
         f'{normal_final["recall"]*100:.2f}%',
         f'{env_final["recall"]*100:.2f}%',
         f'{(normal_final["recall"] - env_final["recall"])*100:.2f}%',
         '⚠️ ALTO' if (normal_final["recall"] - env_final["recall"])*100 > 10 else '✓ MODERADO'],
        
        ['F1-Score',
         f'{normal_final["f1_score"]*100:.2f}%',
         f'{env_final["f1_score"]*100:.2f}%',
         f'{(normal_final["f1_score"] - env_final["f1_score"])*100:.2f}%',
         '⚠️ ALTO' if (normal_final["f1_score"] - env_final["f1_score"])*100 > 10 else '✓ MODERADO'],
        
        ['Loss',
         f'{normal_final["loss"]:.4f}',
         f'{env_final["loss"]:.4f}',
         f'{env_final["loss"] - normal_final["loss"]:.4f}',
         '⚠️ ALTO' if env_final["loss"] - normal_final["loss"] > 0.5 else '✓ MODERADO'],
    ]
    
    table = ax.table(cellText=dados_tabela, cellLoc='center', loc='center',
                    bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 3)
    
    # Estiliza cabeçalho
    for i in range(5):
        cell = table[(0, i)]
        cell.set_facecolor('#4ECDC4')
        cell.set_text_props(weight='bold', color='white', fontsize=12)
    
    # Linha separadora
    for i in range(5):
        cell = table[(1, i)]
        cell.set_facecolor('#E8E8E8')
        cell.set_height(0.02)
    
    # Estiliza primeira coluna
    for i in range(2, len(dados_tabela)):
        cell = table[(i, 0)]
        cell.set_facecolor('#F0F0F0')
        cell.set_text_props(weight='bold')
    
    ax.set_title('Resumo Comparativo: Normal vs Envenenado (Rodada Final)\n' +
                f'Dataset Iris (Amostra: {TAMANHO_AMOSTRA} amostras)',
                fontsize=15, fontweight='bold', pad=20)
    
    plt.savefig('../../modelagem/apresentation/amostra_tabela_resumo.png', 
                dpi=300, bbox_inches='tight')
    print("  ✓ Salvo: amostra_tabela_resumo.png")
    plt.close()


def gerar_grafico_impacto_por_rodada(hist_normal, hist_envenenado):
    """Visualização 5: Impacto acumulado por rodada"""
    print("\n[GERANDO] Gráfico de Impacto por Rodada")
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    rodadas = [h['rodada'] for h in hist_normal]
    
    # Calcula degradação acumulada em cada rodada
    degradacao_acc = [(hist_normal[i]['acuracia'] - hist_envenenado[i]['acuracia']) * 100 
                      for i in range(len(hist_normal))]
    degradacao_f1 = [(hist_normal[i]['f1_score'] - hist_envenenado[i]['f1_score']) * 100 
                     for i in range(len(hist_normal))]
    
    # Gráfico de barras empilhadas
    x = np.arange(len(rodadas))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, degradacao_acc, width, label='Degradação Acurácia',
                   color='#D62828', alpha=0.8, edgecolor='black', linewidth=2)
    bars2 = ax.bar(x + width/2, degradacao_f1, width, label='Degradação F1-Score',
                   color='#F77F00', alpha=0.8, edgecolor='black', linewidth=2)
    
    # Adiciona valores nas barras
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 1:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%', ha='center', va='bottom', 
                       fontsize=10, fontweight='bold')
    
    # Linha de referência
    ax.axhline(y=10, color='orange', linestyle='--', linewidth=2, 
              label='Severidade Moderada (10%)', alpha=0.7)
    ax.axhline(y=20, color='red', linestyle='--', linewidth=2, 
              label='Severidade Alta (20%)', alpha=0.7)
    
    ax.set_xlabel('Rodada Federada', fontsize=13, fontweight='bold')
    ax.set_ylabel('Degradação (%)', fontsize=13, fontweight='bold')
    ax.set_title('Impacto do Envenenamento por Rodada\n' +
                f'Dataset Iris (Amostra: {TAMANHO_AMOSTRA} amostras)',
                fontsize=16, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(rodadas)
    ax.legend(fontsize=11, loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('../../modelagem/apresentation/amostra_impacto_por_rodada.png', 
                dpi=300, bbox_inches='tight')
    print("  ✓ Salvo: amostra_impacto_por_rodada.png")
    plt.close()


def carregar_amostra_iris():
    """Carrega uma amostra reduzida do dataset Iris"""
    caminho = r"c:\Users\Administrador\Faculdade\Iniciação-cientifica\project\data\iris\iris.csv"
    
    try:
        df = pd.read_csv(caminho)
    except:
        from sklearn.datasets import load_iris
        iris = load_iris()
        df = pd.DataFrame(iris.data, columns=iris.feature_names)
        df['species'] = iris.target
    
    if 'species' in df.columns:
        X = df.drop('species', axis=1).values
        if df['species'].dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(df['species'])
        else:
            y = df['species'].values
    else:
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
    
    # Seleciona apenas AMOSTRA_SIZE amostras (estratificadas)
    from sklearn.model_selection import train_test_split
    X_amostra, _, y_amostra, _ = train_test_split(
        X, y, train_size=TAMANHO_AMOSTRA, random_state=42, stratify=y
    )
    
    return X_amostra, y_amostra


def main():
    """Função principal"""
    print("="*70)
    print("DEMONSTRAÇÃO COM AMOSTRA REDUZIDA - DATASET IRIS")
    print("Poisoning Attack em Aprendizado Federado")
    print("="*70)
    
    # 1. Carrega amostra
    X, y = carregar_amostra_iris()
    print(f"\n✓ Amostra carregada: {len(X)} amostras (30% do dataset completo)")
    print(f"  Classes: {ESPECIES}")
    print(f"  Distribuição: {np.bincount(y)} amostras por classe")
    
    # 2. Divide dados
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.33, random_state=42, stratify=y
    )
    
    # Divide treino entre 3 clientes (distribuição igual)
    X_c1, X_temp, y_c1, y_temp = train_test_split(
        X_train, y_train, test_size=0.66, random_state=42, stratify=y_train
    )
    X_c2, X_c3, y_c2, y_c3 = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    dados_clientes = [(X_c1, y_c1), (X_c2, y_c2), (X_c3, y_c3)]
    dados_validacao = (X_val, y_val)
    
    print(f"\n✓ Sistema configurado:")
    print(f"  - Cliente 1: {len(X_c1)} amostras (Honesto)")
    print(f"  - Cliente 2: {len(X_c2)} amostras (Honesto)")
    print(f"  - Cliente 3: {len(X_c3)} amostras (⚠️ MALICIOSO)")
    print(f"  - Validação: {len(X_val)} amostras")
    
    # 3. Executa cenário NORMAL
    print("\n" + "="*70)
    print("EXECUTANDO CENÁRIO NORMAL (sem ataque)")
    print("="*70)
    hist_normal_global, hist_normal_clientes = executar_cenario(
        dados_clientes, dados_validacao, num_rodadas=8, envenenado=False
    )
    print(f"✓ Concluído: {len(hist_normal_global)} rodadas")
    print(f"  Acurácia inicial: {hist_normal_global[0]['acuracia']*100:.2f}%")
    print(f"  Acurácia final:   {hist_normal_global[-1]['acuracia']*100:.2f}%")
    
    # 4. Executa cenário ENVENENADO
    print("\n" + "="*70)
    print("EXECUTANDO CENÁRIO ENVENENADO (com ataque no Cliente 3)")
    print("="*70)
    hist_env_global, hist_env_clientes = executar_cenario(
        dados_clientes, dados_validacao, num_rodadas=8, envenenado=True
    )
    print(f"✓ Concluído: {len(hist_env_global)} rodadas")
    print(f"  Acurácia inicial: {hist_env_global[0]['acuracia']*100:.2f}%")
    print(f"  Acurácia final:   {hist_env_global[-1]['acuracia']*100:.2f}%")
    print(f"  Degradação:       {(hist_normal_global[-1]['acuracia'] - hist_env_global[-1]['acuracia'])*100:.2f}%")
    
    # 5. Gera visualizações
    print("\n" + "="*70)
    print("GERANDO VISUALIZAÇÕES")
    print("="*70)
    
    gerar_grafico_evolucao_metricas(hist_normal_global, hist_env_global)
    gerar_grafico_clientes_locais(hist_env_clientes)
    gerar_matriz_confusao_comparativa(hist_normal_global, hist_env_global)
    gerar_tabela_resumo(hist_normal_global, hist_env_global)
    gerar_grafico_impacto_por_rodada(hist_normal_global, hist_env_global)
    
    # 6. Resumo final
    print("\n" + "="*70)
    print("ANÁLISE COM AMOSTRA CONCLUÍDA")
    print("="*70)
    print("\n📊 Visualizações geradas (pasta: modelagem/apresentation):")
    print("  1. amostra_evolucao_metricas.png")
    print("     → Evolução de Acurácia, F1, Precisão e Loss")
    print("\n  2. amostra_desempenho_clientes.png")
    print("     → Desempenho individual de cada cliente local")
    print("\n  3. amostra_matriz_confusao_comparativa.png")
    print("     → Matrizes de confusão (inicial vs final)")
    print("\n  4. amostra_tabela_resumo.png")
    print("     → Tabela com estatísticas comparativas")
    print("\n  5. amostra_impacto_por_rodada.png")
    print("     → Degradação acumulada por rodada")
    
    print("\n📈 Conclusões da Amostra:")
    degradacao = (hist_normal_global[-1]['acuracia'] - hist_env_global[-1]['acuracia']) * 100
    print(f"  - Degradação de acurácia: {degradacao:.2f}%")
    print(f"  - Severidade: {'CRÍTICA' if degradacao > 20 else 'ALTA' if degradacao > 10 else 'MODERADA'}")
    print(f"  - Pipeline completo: Treinar → Corromper → Avaliar → Agregar")
    print(f"  - Cliente malicioso (33% dos clientes) impactou significativamente")
    print("="*70)


if __name__ == "__main__":
    main()
