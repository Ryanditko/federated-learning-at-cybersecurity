"""
Script de Comparação: Cenário Normal vs Cenário Envenenado
Executa ambos os cenários e gera visualizações comparativas

Objetivo: Demonstrar o impacto do envenenamento de dados através de 
         comparação direta entre sistema federado normal e atacado.

Autor: Projeto de Iniciação Científica
Dataset: Iris (Classificação de Espécies)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, log_loss
from sklearn.model_selection import train_test_split
from copy import deepcopy
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Configuração de estilo
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class ModeloSimples:
    """Modelo de classificação simplificado"""
    
    def __init__(self):
        self._modelo = LogisticRegression(max_iter=1000, random_state=42,
                                          multi_class='multinomial', solver='lbfgs')
        self.scaler = StandardScaler()
    
    def treinar(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self._modelo.fit(X_scaled, y)
    
    def avaliar(self, X, y):
        # Se scaler ainda não foi fitado, fita primeiro
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


def envenenar_pesos(pesos, taxa=0.8, tipo='inverter'):
    """Corrompe pesos do modelo"""
    pesos_corrompidos = deepcopy(pesos)
    
    if tipo == 'inverter':
        pesos_corrompidos['coef'] = -pesos['coef'] * (1 + taxa)
        pesos_corrompidos['intercept'] = -pesos['intercept'] * (1 + taxa)
    elif tipo == 'aleatorio':
        ruido_coef = np.random.randn(*pesos['coef'].shape) * taxa
        ruido_intercept = np.random.randn(*pesos['intercept'].shape) * taxa
        pesos_corrompidos['coef'] = pesos['coef'] + ruido_coef
        pesos_corrompidos['intercept'] = pesos['intercept'] + ruido_intercept
    elif tipo == 'amplificar':
        pesos_corrompidos['coef'] = pesos['coef'] * (1 + taxa * 10)
        pesos_corrompidos['intercept'] = pesos['intercept'] * (1 + taxa * 10)
    
    return pesos_corrompidos


def executar_cenario_normal(dados_clientes, dados_validacao, num_rodadas=5):
    """Executa cenário normal (todos os clientes honestos)"""
    print("\n" + "="*70)
    print("CENÁRIO 1: APRENDIZADO FEDERADO NORMAL (Sem Envenenamento)")
    print("="*70)
    
    X_val, y_val = dados_validacao
    historico_metricas = []
    
    # Modelo global inicial (compartilhado entre rodadas)
    modelo_global = ModeloSimples()
    pesos_globais = None
    
    for rodada in range(1, num_rodadas + 1):
        print(f"\n[Rodada {rodada}/{num_rodadas}]")
        
        # 1. Cada cliente RECEBE o modelo global e treina A PARTIR DELE
        pesos_locais = []
        for i, (X_cliente, y_cliente) in enumerate(dados_clientes, 1):
            modelo = ModeloSimples()
            
            # IMPORTANTE: Se existe modelo global, cliente começa a partir dele
            if pesos_globais is not None:
                modelo.atualizar_pesos(pesos_globais)
            
            # Treina a partir do modelo recebido (fine-tuning)
            modelo.treinar(X_cliente, y_cliente)
            metricas = modelo.avaliar(X_cliente, y_cliente)
            pesos_locais.append(modelo.obter_pesos())
            print(f"  Cliente {i}: Acc={metricas['acuracia']:.4f}")
        
        # 2. Agrega modelos (FedAvg)
        pesos_globais = {
            'coef': np.mean([p['coef'] for p in pesos_locais], axis=0),
            'intercept': np.mean([p['intercept'] for p in pesos_locais], axis=0),
            'classes': pesos_locais[0]['classes']
        }
        
        # 3. Atualiza e avalia modelo global
        modelo_global.atualizar_pesos(pesos_globais)
        metricas_globais = modelo_global.avaliar(X_val, y_val)
        
        print(f"  Modelo Global: Acc={metricas_globais['acuracia']:.4f} | "
              f"F1={metricas_globais['f1_score']:.4f} | "
              f"Loss={metricas_globais['loss']:.4f}")
        
        historico_metricas.append(metricas_globais)
    
    return historico_metricas


def executar_cenario_envenenado(dados_clientes, dados_validacao, num_rodadas=5,
                                taxa_corrupcao=0.8, tipo_ataque='inverter'):
    """Executa cenário envenenado (1 cliente malicioso)"""
    print("\n" + "="*70)
    print(f"CENÁRIO 2: APRENDIZADO FEDERADO COM ENVENENAMENTO")
    print(f"Ataque: {tipo_ataque} | Taxa: {taxa_corrupcao}")
    print("="*70)
    
    X_val, y_val = dados_validacao
    historico_metricas = []
    
    # Modelo global inicial (compartilhado entre rodadas)
    modelo_global = ModeloSimples()
    pesos_globais = None
    
    for rodada in range(1, num_rodadas + 1):
        print(f"\n[Rodada {rodada}/{num_rodadas}]")
        
        # 1. Cada cliente RECEBE o modelo global (potencialmente corrompido) e treina A PARTIR DELE
        pesos_locais = []
        for i, (X_cliente, y_cliente) in enumerate(dados_clientes, 1):
            modelo = ModeloSimples()
            
            # IMPORTANTE: Cliente recebe modelo global corrompido da rodada anterior
            if pesos_globais is not None:
                modelo.atualizar_pesos(pesos_globais)
            
            # Treina a partir do modelo recebido (que pode estar envenenado)
            modelo.treinar(X_cliente, y_cliente)
            metricas = modelo.avaliar(X_cliente, y_cliente)
            pesos = modelo.obter_pesos()
            
            # Cliente 3 é malicioso - corrompe pesos APÓS treinar
            if i == 3:
                pesos_antes = deepcopy(pesos)
                pesos = envenenar_pesos(pesos, taxa_corrupcao, tipo_ataque)
                print(f"  Cliente {i} (ENVENENADO): Acc={metricas['acuracia']:.4f} → Pesos CORROMPIDOS")
            else:
                print(f"  Cliente {i} (Honesto): Acc={metricas['acuracia']:.4f}")
            
            pesos_locais.append(pesos)
        
        # 2. Agrega modelos (FedAvg) - INCLUI PESOS CORROMPIDOS
        pesos_globais = {
            'coef': np.mean([p['coef'] for p in pesos_locais], axis=0),
            'intercept': np.mean([p['intercept'] for p in pesos_locais], axis=0),
            'classes': pesos_locais[0]['classes']
        }
        
        # 3. Atualiza e avalia modelo global (CONTAMINADO)
        modelo_global.atualizar_pesos(pesos_globais)
        metricas_globais = modelo_global.avaliar(X_val, y_val)
        
        print(f"  Modelo Global: Acc={metricas_globais['acuracia']:.4f} | "
              f"F1={metricas_globais['f1_score']:.4f} | "
              f"Loss={metricas_globais['loss']:.4f}")
        
        historico_metricas.append(metricas_globais)
    
    return historico_metricas


def gerar_grafico_comparativo(historico_normal, historico_envenenado, tipo_ataque='inverter'):
    """Gera gráfico comparando os dois cenários"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    rodadas = list(range(1, len(historico_normal) + 1))
    
    # Extrai métricas
    metricas = ['acuracia', 'f1_score', 'precisao', 'loss']
    titulos = ['Acurácia', 'F1-Score', 'Precisão', 'Loss']
    cores_normal = '#2E86AB'
    cores_envenenado = '#D62828'
    
    for idx, (metrica, titulo) in enumerate(zip(metricas, titulos)):
        ax = axes[idx // 2, idx % 2]
        
        valores_normal = [m[metrica] for m in historico_normal]
        valores_envenenado = [m[metrica] for m in historico_envenenado]
        
        # Linha do cenário normal
        ax.plot(rodadas, valores_normal, marker='o', linewidth=2.5, markersize=8,
               color=cores_normal, label='Normal (Sem ataque)', zorder=3)
        
        # Linha do cenário envenenado
        ax.plot(rodadas, valores_envenenado, marker='s', linewidth=2.5, markersize=8,
               color=cores_envenenado, label=f'Envenenado ({tipo_ataque})', zorder=3)
        
        # Área sombreada
        ax.fill_between(rodadas, valores_normal, alpha=0.1, color=cores_normal)
        ax.fill_between(rodadas, valores_envenenado, alpha=0.1, color=cores_envenenado)
        
        ax.set_title(f'{titulo} - Comparação', fontsize=14, fontweight='bold')
        ax.set_xlabel('Rodada Federada', fontsize=12)
        ax.set_ylabel(titulo, fontsize=12)
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
        
        # Ajusta limites do eixo Y
        if metrica != 'loss':
            ax.set_ylim([0, 1.05])
    
    plt.suptitle('Comparação: Cenário Normal vs Envenenado\nDataset Iris - Regressão Logística',
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('comparacao_normal_vs_envenenado.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Gráfico comparativo salvo: comparacao_normal_vs_envenenado.png")
    plt.close()


def gerar_tabela_comparativa(historico_normal, historico_envenenado):
    """Gera tabela comparativa das métricas"""
    
    print("\n" + "="*70)
    print("TABELA COMPARATIVA - MÉTRICAS FINAIS")
    print("="*70)
    
    # Última rodada
    metricas_normal_final = historico_normal[-1]
    metricas_envenenado_final = historico_envenenado[-1]
    
    # Calcula degradação
    print(f"\n{'Métrica':<15} {'Normal':<12} {'Envenenado':<12} {'Degradação':<12} {'Impacto'}")
    print("-" * 70)
    
    metricas = [
        ('Acurácia', 'acuracia'),
        ('F1-Score', 'f1_score'),
        ('Precisão', 'precisao'),
        ('Recall', 'recall'),
        ('Loss', 'loss')
    ]
    
    for nome, chave in metricas:
        valor_normal = metricas_normal_final[chave]
        valor_envenenado = metricas_envenenado_final[chave]
        
        if chave == 'loss':
            # Para loss, menor é melhor
            degradacao = valor_envenenado - valor_normal
            impacto = "↑ PIOR" if degradacao > 0 else "↓ melhor"
        else:
            # Para outras métricas, maior é melhor
            degradacao = valor_normal - valor_envenenado
            impacto = "↓ PIOR" if degradacao > 0 else "↑ melhor"
        
        print(f"{nome:<15} {valor_normal:<12.4f} {valor_envenenado:<12.4f} "
              f"{abs(degradacao):<12.4f} {impacto}")
    
    # Estatísticas adicionais
    print("\n" + "="*70)
    print("ANÁLISE DE IMPACTO")
    print("="*70)
    
    acc_normal = metricas_normal_final['acuracia']
    acc_envenenado = metricas_envenenado_final['acuracia']
    perda_percentual = ((acc_normal - acc_envenenado) / acc_normal) * 100
    
    print(f"\nImpacto na Acurácia:")
    print(f"  - Cenário Normal:     {acc_normal:.2%} ({acc_normal*100:.1f}%)")
    print(f"  - Cenário Envenenado: {acc_envenenado:.2%} ({acc_envenenado*100:.1f}%)")
    print(f"  - Perda Absoluta:     {acc_normal - acc_envenenado:.2%}")
    print(f"  - Perda Percentual:   {perda_percentual:.1f}% de degradação")
    
    if perda_percentual > 20:
        print(f"  - Severidade:         ⚠️  CRÍTICA (>{20}%)")
    elif perda_percentual > 10:
        print(f"  - Severidade:         ⚠️  ALTA (>{10}%)")
    elif perda_percentual > 5:
        print(f"  - Severidade:         ⚠️  MODERADA (>{5}%)")
    else:
        print(f"  - Severidade:         ✓ BAIXA (<{5}%)")


def carregar_dataset_iris():
    """Carrega e prepara o dataset Iris"""
    caminho_dataset = r"c:\Users\Administrador\Faculdade\Iniciação-cientifica\project\data\iris\iris.csv"
    
    try:
        df = pd.read_csv(caminho_dataset)
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
    
    return X, y


def main():
    """Função principal"""
    print("="*70)
    print("COMPARAÇÃO: CENÁRIO NORMAL vs ENVENENADO")
    print("Dataset: Iris | Modelo: Regressão Logística")
    print("="*70)
    
    # 1. Carrega dataset
    X, y = carregar_dataset_iris()
    print(f"\n✓ Dataset carregado: {len(X)} amostras, {X.shape[1]} features")
    
    # 2. Divide dados
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
    
    print(f"✓ Dados divididos: 3 clientes + 1 conjunto de validação")
    
    # 3. Executa cenário normal
    historico_normal = executar_cenario_normal(dados_clientes, dados_validacao, num_rodadas=5)
    
    # 4. Executa cenário envenenado
    historico_envenenado = executar_cenario_envenenado(
        dados_clientes, dados_validacao, num_rodadas=5,
        taxa_corrupcao=0.8, tipo_ataque='inverter'
    )
    
    # 5. Gera visualizações
    print("\n" + "="*70)
    print("GERANDO VISUALIZAÇÕES COMPARATIVAS")
    print("="*70)
    gerar_grafico_comparativo(historico_normal, historico_envenenado, tipo_ataque='inverter')
    
    # 6. Gera tabela comparativa
    gerar_tabela_comparativa(historico_normal, historico_envenenado)
    
    print("\n" + "="*70)
    print("EXPERIMENTO COMPARATIVO CONCLUÍDO")
    print("="*70)
    print("\nArquivos gerados:")
    print("  - comparacao_normal_vs_envenenado.png")
    print("\nConclusão:")
    print("  O envenenamento de 1 cliente em 3 (33%) causa degradação")
    print("  significativa no modelo global, demonstrando a vulnerabilidade")
    print("  do algoritmo FedAvg padrão contra ataques de envenenamento.")
    print("="*70)


if __name__ == "__main__":
    main()
