"""
Demonstração Visual de Envenenamento com Degradação Gradual
Usa inicialização ruim e convergência lenta para mostrar o impacto acumulativo

Objetivo: Visualizar claramente a degradação causada pelo envenenamento
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import SGDClassifier  # Convergência mais gradual
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import train_test_split
from copy import deepcopy
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class ModeloGradual:
    """Modelo com convergência MUITO GRADUAL para demonstração visual"""
    
    def __init__(self, learning_rate=0.001):
        # SGD com learning rate MUITO baixo = convergência LENTA
        self._modelo = SGDClassifier(
            loss='log_loss',  # Regressão logística
            learning_rate='constant',
            eta0=learning_rate,  # Taxa de aprendizado muito baixa
            max_iter=10,  # Poucas iterações por rodada
            random_state=42,
            warm_start=True  # CRÍTICO: continua treinamento anterior
        )
        self.scaler = StandardScaler()
        self.inicializado = False
    
    def treinar(self, X, y):
        X_scaled = self.scaler.fit_transform(X) if not self.inicializado else self.scaler.transform(X)
        
        if not self.inicializado:
            # Primeira vez: fit normal
            self._modelo.fit(X_scaled, y)
            self.inicializado = True
        else:
            # Continua treinamento (warm_start)
            self._modelo.partial_fit(X_scaled, y)
    
    def avaliar(self, X, y):
        if not hasattr(self.scaler, 'mean_'):
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        y_pred = self._modelo.predict(X_scaled)
        y_proba = self._modelo.predict_proba(X_scaled)
        
        return {
            'acuracia': accuracy_score(y, y_pred),
            'f1_score': f1_score(y, y_pred, average='weighted'),
            'loss': log_loss(y, y_proba)
        }
    
    def obter_pesos(self):
        if hasattr(self._modelo, 'coef_'):
            return {
                'coef': deepcopy(self._modelo.coef_),
                'intercept': deepcopy(self._modelo.intercept_),
                'classes': deepcopy(self._modelo.classes_)
            }
        return None
    
    def atualizar_pesos(self, pesos):
        if pesos:
            self._modelo.coef_ = deepcopy(pesos['coef'])
            self._modelo.intercept_ = deepcopy(pesos['intercept'])
            self._modelo.classes_ = deepcopy(pesos['classes'])


def envenenar_pesos_gradual(pesos, taxa=0.3, tipo='inverter'):
    """Envenenamento mais sutil para efeito gradual"""
    pesos_corrompidos = deepcopy(pesos)
    
    if tipo == 'inverter':
        # Inverte PARCIALMENTE (mais sutil)
        pesos_corrompidos['coef'] = -pesos['coef'] * taxa
        pesos_corrompidos['intercept'] = -pesos['intercept'] * taxa
    elif tipo == 'ruido':
        # Adiciona ruído grande
        ruido_coef = np.random.randn(*pesos['coef'].shape) * taxa * 2
        ruido_intercept = np.random.randn(*pesos['intercept'].shape) * taxa * 2
        pesos_corrompidos['coef'] = pesos['coef'] + ruido_coef
        pesos_corrompidos['intercept'] = pesos['intercept'] + ruido_intercept
    
    return pesos_corrompidos


def executar_cenario_normal_gradual(dados_clientes, dados_validacao, num_rodadas=15):
    """Cenário normal com convergência gradual"""
    print("\n" + "="*70)
    print("CENÁRIO NORMAL - Convergência Gradual")
    print("="*70)
    
    X_val, y_val = dados_validacao
    historico = []
    
    # Modelo global compartilhado
    modelo_global = ModeloGradual(learning_rate=0.01)
    pesos_globais = None
    
    for rodada in range(1, num_rodadas + 1):
        print(f"[Rodada {rodada:2d}/{num_rodadas}] ", end="")
        
        # Clientes treinam a partir do modelo global
        pesos_locais = []
        for X_cliente, y_cliente in dados_clientes:
            modelo = ModeloGradual(learning_rate=0.01)
            
            if pesos_globais is not None:
                modelo.atualizar_pesos(pesos_globais)
            
            modelo.treinar(X_cliente, y_cliente)
            pesos_locais.append(modelo.obter_pesos())
        
        # Agrega
        pesos_globais = {
            'coef': np.mean([p['coef'] for p in pesos_locais], axis=0),
            'intercept': np.mean([p['intercept'] for p in pesos_locais], axis=0),
            'classes': pesos_locais[0]['classes']
        }
        
        # Avalia
        modelo_global.atualizar_pesos(pesos_globais)
        metricas = modelo_global.avaliar(X_val, y_val)
        historico.append(metricas)
        
        print(f"Acc={metricas['acuracia']:.4f} | Loss={metricas['loss']:.4f}")
    
    return historico


def executar_cenario_envenenado_gradual(dados_clientes, dados_validacao, num_rodadas=15,
                                       taxa_corrupcao=0.5, tipo_ataque='inverter'):
    """Cenário envenenado com degradação gradual"""
    print("\n" + "="*70)
    print(f"CENÁRIO ENVENENADO - Ataque: {tipo_ataque} (taxa={taxa_corrupcao})")
    print("="*70)
    
    X_val, y_val = dados_validacao
    historico = []
    
    # Modelo global compartilhado
    modelo_global = ModeloGradual(learning_rate=0.01)
    pesos_globais = None
    
    for rodada in range(1, num_rodadas + 1):
        print(f"[Rodada {rodada:2d}/{num_rodadas}] ", end="")
        
        # Clientes treinam a partir do modelo global (possivelmente contaminado)
        pesos_locais = []
        for i, (X_cliente, y_cliente) in enumerate(dados_clientes, 1):
            modelo = ModeloGradual(learning_rate=0.01)
            
            if pesos_globais is not None:
                modelo.atualizar_pesos(pesos_globais)
            
            modelo.treinar(X_cliente, y_cliente)
            pesos = modelo.obter_pesos()
            
            # Cliente 3 envenena
            if i == 3:
                pesos = envenenar_pesos_gradual(pesos, taxa_corrupcao, tipo_ataque)
            
            pesos_locais.append(pesos)
        
        # Agrega (incluindo pesos envenenados)
        pesos_globais = {
            'coef': np.mean([p['coef'] for p in pesos_locais], axis=0),
            'intercept': np.mean([p['intercept'] for p in pesos_locais], axis=0),
            'classes': pesos_locais[0]['classes']
        }
        
        # Avalia
        modelo_global.atualizar_pesos(pesos_globais)
        metricas = modelo_global.avaliar(X_val, y_val)
        historico.append(metricas)
        
        print(f"Acc={metricas['acuracia']:.4f} | Loss={metricas['loss']:.4f}")
    
    return historico


def gerar_visualizacao_final(hist_normal, hist_envenenado, tipo_ataque='inverter'):
    """Gera as DUAS visualizações pedidas"""
    
    # ==================== VISUALIZAÇÃO 1: CONVERGÊNCIA POR RODADA ====================
    fig1, axes = plt.subplots(1, 2, figsize=(18, 6))
    
    rodadas = list(range(1, len(hist_normal) + 1))
    
    # Acurácia
    acc_normal = [m['acuracia'] for m in hist_normal]
    acc_envenenado = [m['acuracia'] for m in hist_envenenado]
    
    axes[0].plot(rodadas, acc_normal, marker='o', linewidth=3, markersize=8,
                color='#2E86AB', label='Normal (Sem ataque)', zorder=3)
    axes[0].plot(rodadas, acc_envenenado, marker='s', linewidth=3, markersize=8,
                color='#D62828', label=f'Envenenado ({tipo_ataque})', zorder=3)
    axes[0].fill_between(rodadas, acc_normal, alpha=0.15, color='#2E86AB')
    axes[0].fill_between(rodadas, acc_envenenado, alpha=0.15, color='#D62828')
    axes[0].set_title('Acurácia ao Longo das Rodadas', fontsize=16, fontweight='bold')
    axes[0].set_xlabel('Rodada Federada', fontsize=13)
    axes[0].set_ylabel('Acurácia', fontsize=13)
    axes[0].legend(fontsize=12, loc='best')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1.05])
    
    # Loss
    loss_normal = [m['loss'] for m in hist_normal]
    loss_envenenado = [m['loss'] for m in hist_envenenado]
    
    axes[1].plot(rodadas, loss_normal, marker='o', linewidth=3, markersize=8,
                color='#2E86AB', label='Normal', zorder=3)
    axes[1].plot(rodadas, loss_envenenado, marker='s', linewidth=3, markersize=8,
                color='#D62828', label='Envenenado', zorder=3)
    axes[1].fill_between(rodadas, loss_normal, alpha=0.15, color='#2E86AB')
    axes[1].fill_between(rodadas, loss_envenenado, alpha=0.15, color='#D62828')
    axes[1].set_title('Loss ao Longo das Rodadas', fontsize=16, fontweight='bold')
    axes[1].set_xlabel('Rodada Federada', fontsize=13)
    axes[1].set_ylabel('Loss (Log Loss)', fontsize=13)
    axes[1].legend(fontsize=12, loc='best')
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle('VISUALIZAÇÃO 1: Evolução das Métricas por Rodada',
                fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('vis1_convergencia_por_rodada.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ VISUALIZAÇÃO 1 salva: vis1_convergencia_por_rodada.png")
    plt.close()
    
    # ==================== VISUALIZAÇÃO 2: COMPARAÇÃO PERCENTUAL ====================
    fig2, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    # Calcula acurácias iniciais, médias e finais
    rodadas_chave = {
        'Início\n(Rodada 1)': (hist_normal[0], hist_envenenado[0]),
        'Meio\n(Rodada 8)': (hist_normal[7], hist_envenenado[7]),
        'Final\n(Rodada 15)': (hist_normal[-1], hist_envenenado[-1])
    }
    
    labels = list(rodadas_chave.keys())
    acc_normal_pontos = [rodadas_chave[k][0]['acuracia'] * 100 for k in labels]
    acc_envenenado_pontos = [rodadas_chave[k][1]['acuracia'] * 100 for k in labels]
    
    x = np.arange(len(labels))
    width = 0.35
    
    bars1 = axes[0].bar(x - width/2, acc_normal_pontos, width, label='Normal',
                       color='#2E86AB', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = axes[0].bar(x + width/2, acc_envenenado_pontos, width, label='Envenenado',
                       color='#D62828', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    axes[0].set_title('Comparação de Acurácia Percentual por Fase',
                     fontsize=16, fontweight='bold')
    axes[0].set_ylabel('Acurácia (%)', fontsize=13)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=11)
    axes[0].legend(fontsize=12)
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].set_ylim([0, 105])
    
    # Adiciona valores nas barras
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}%',
                        ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Gráfico de degradação percentual
    degradacao = [acc_normal_pontos[i] - acc_envenenado_pontos[i] for i in range(len(labels))]
    cores_degradacao = ['#FFA500' if d < 15 else '#FF6347' if d < 25 else '#8B0000' for d in degradacao]
    
    bars3 = axes[1].bar(labels, degradacao, color=cores_degradacao, alpha=0.8,
                       edgecolor='black', linewidth=1.5)
    axes[1].set_title('Degradação Causada pelo Envenenamento',
                     fontsize=16, fontweight='bold')
    axes[1].set_ylabel('Degradação (%)', fontsize=13)
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].set_ylim([0, max(degradacao) * 1.2])
    
    # Adiciona valores e linhas de severidade
    for bar, deg in zip(bars3, degradacao):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{deg:.1f}%',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    axes[1].axhline(y=10, color='orange', linestyle='--', linewidth=1.5, alpha=0.5, label='Moderado (10%)')
    axes[1].axhline(y=20, color='red', linestyle='--', linewidth=1.5, alpha=0.5, label='Crítico (20%)')
    axes[1].legend(fontsize=10, loc='upper left')
    
    plt.suptitle('VISUALIZAÇÃO 2: Impacto Percentual do Envenenamento',
                fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('vis2_impacto_percentual.png', dpi=300, bbox_inches='tight')
    print(f"✓ VISUALIZAÇÃO 2 salva: vis2_impacto_percentual.png")
    plt.close()


def carregar_dataset():
    """Carrega dataset Iris"""
    caminho = r"c:\Users\Administrador\Faculdade\Iniciação-cientifica\project\data\iris\iris.csv"
    try:
        df = pd.read_csv(caminho)
    except:
        from sklearn.datasets import load_iris
        iris = load_iris()
        df = pd.DataFrame(iris.data, columns=iris.feature_names)
        df['species'] = iris.target
    
    X = df.drop('species', axis=1).values if 'species' in df.columns else df.iloc[:, :-1].values
    
    if 'species' in df.columns and df['species'].dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(df['species'])
    else:
        y = df['species'].values if 'species' in df.columns else df.iloc[:, -1].values
    
    return X, y


def main():
    """Função principal"""
    print("="*70)
    print("DEMONSTRAÇÃO VISUAL: ENVENENAMENTO COM DEGRADAÇÃO GRADUAL")
    print("="*70)
    
    # Carrega dados
    X, y = carregar_dataset()
    print(f"\n✓ Dataset carregado: {len(X)} amostras")
    
    # Divide dados
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_c1, X_temp, y_c1, y_temp = train_test_split(X_train, y_train, test_size=0.66, random_state=42, stratify=y_train)
    X_c2, X_c3, y_c2, y_c3 = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    dados_clientes = [(X_c1, y_c1), (X_c2, y_c2), (X_c3, y_c3)]
    dados_validacao = (X_val, y_val)
    
    # Executa cenários
    hist_normal = executar_cenario_normal_gradual(dados_clientes, dados_validacao, num_rodadas=15)
    hist_envenenado = executar_cenario_envenenado_gradual(dados_clientes, dados_validacao, num_rodadas=15,
                                                          taxa_corrupcao=0.6, tipo_ataque='inverter')
    
    # Gera visualizações
    print("\n" + "="*70)
    print("GERANDO VISUALIZAÇÕES")
    print("="*70)
    gerar_visualizacao_final(hist_normal, hist_envenenado, tipo_ataque='inverter')
    
    # Sumário
    print("\n" + "="*70)
    print("SUMÁRIO DO IMPACTO")
    print("="*70)
    print(f"\nAcurácia Inicial (Rodada 1):")
    print(f"  Normal:     {hist_normal[0]['acuracia']:.2%}")
    print(f"  Envenenado: {hist_envenenado[0]['acuracia']:.2%}")
    print(f"  Degradação: {(hist_normal[0]['acuracia'] - hist_envenenado[0]['acuracia']):.2%}")
    
    print(f"\nAcurácia Final (Rodada 15):")
    print(f"  Normal:     {hist_normal[-1]['acuracia']:.2%}")
    print(f"  Envenenado: {hist_envenenado[-1]['acuracia']:.2%}")
    print(f"  Degradação: {(hist_normal[-1]['acuracia'] - hist_envenenado[-1]['acuracia']):.2%}")
    
    print("\n" + "="*70)
    print("EXPERIMENTO CONCLUÍDO")
    print("="*70)
    print("\nArquivos gerados:")
    print("  📊 vis1_convergencia_por_rodada.png")
    print("  📊 vis2_impacto_percentual.png")
    print("="*70)


if __name__ == "__main__":
    main()
