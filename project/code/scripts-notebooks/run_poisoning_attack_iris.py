"""
Sistema de Aprendizado Federado com Envenenamento de Dados
Cenário: Model Poisoning Attack no Dataset Iris
Fluxo: Treina Modelo Local > Corrompe os Pesos > Avalia o Modelo

Autor: Projeto de Iniciação Científica
Dataset: Iris (Classificação de Espécies)
Modelo: Regressão Logística (Multi-classe)

Objetivo: Demonstrar o impacto de envenenamento de modelo através da 
         corrupção de pesos após treinamento local.

Padrão UML seguido:
- Abstração: Classes base abstratas
- Encapsulamento: Getters/Setters
- Herança: Cliente herda comportamentos
- Composição: Servidor contém múltiplos clientes
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, 
    log_loss, confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split
from copy import deepcopy
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple
import warnings
import os
warnings.filterwarnings('ignore')

# Configuração de estilo
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class ModeloClassificacao:
    """
    Classe que encapsula o modelo de classificação
    Padrão UML: Encapsulamento de modelo interno
    """
    
    def __init__(self, max_iter: int = 1000, random_state: int = 42):
        self._modelo_interno = LogisticRegression(
            max_iter=max_iter, 
            random_state=random_state,
            multi_class='multinomial',
            solver='lbfgs'
        )
    
    def obter_pesos(self) -> Dict:
        """Retorna os pesos do modelo (coeficientes e intercept)"""
        if hasattr(self._modelo_interno, 'coef_'):
            return {
                'coef': deepcopy(self._modelo_interno.coef_),
                'intercept': deepcopy(self._modelo_interno.intercept_),
                'classes': deepcopy(self._modelo_interno.classes_)
            }
        return None
    
    def atualizar_pesos(self, pesos: Dict):
        """Atualiza os pesos do modelo"""
        if pesos and 'coef' in pesos and 'intercept' in pesos:
            self._modelo_interno.coef_ = deepcopy(pesos['coef'])
            self._modelo_interno.intercept_ = deepcopy(pesos['intercept'])
            if 'classes' in pesos:
                self._modelo_interno.classes_ = deepcopy(pesos['classes'])
    
    def fit(self, X, y):
        """Treina o modelo"""
        return self._modelo_interno.fit(X, y)
    
    def predict(self, X):
        """Faz predições"""
        return self._modelo_interno.predict(X)
    
    def predict_proba(self, X):
        """Retorna probabilidades"""
        return self._modelo_interno.predict_proba(X)


class ClienteBase(ABC):
    """
    Classe abstrata base para clientes federados
    Padrão UML: Classe abstrata com métodos abstratos
    """
    
    def __init__(self, id_cliente: str, dados_locais: Tuple[np.ndarray, np.ndarray]):
        self.id_cliente = id_cliente
        self.X_local, self.y_local = dados_locais
        self.modelo_local = ModeloClassificacao()
        self.historico_treinamento = []
        self.scaler = StandardScaler()
    
    @abstractmethod
    def treinar_modelo(self) -> Dict:
        """Método abstrato para treinamento"""
        pass
    
    def set_modelo_local(self, pesos: Dict):
        """Atualiza modelo local com pesos recebidos"""
        self.modelo_local.atualizar_pesos(pesos)
    
    def avaliar_modelo_local(self, X_test, y_test) -> Dict:
        """Avalia o modelo local em dados de teste"""
        X_test_scaled = self.scaler.transform(X_test)
        y_pred = self.modelo_local.predict(X_test_scaled)
        y_proba = self.modelo_local.predict_proba(X_test_scaled)
        
        return {
            'cliente_id': self.id_cliente,
            'acuracia': accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred, average='weighted'),
            'precisao': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'loss': log_loss(y_test, y_proba),
            'y_pred': y_pred,
            'y_true': y_test
        }


class ClienteHonesto(ClienteBase):
    """
    Cliente honesto que treina modelo corretamente
    Padrão UML: Herança de ClienteBase
    """
    
    def treinar_modelo(self) -> Dict:
        """Treina o modelo local com dados não corrompidos"""
        X_scaled = self.scaler.fit_transform(self.X_local)
        self.modelo_local.fit(X_scaled, self.y_local)
        
        # Avalia no próprio conjunto de treinamento
        y_pred = self.modelo_local.predict(X_scaled)
        acuracia = accuracy_score(self.y_local, y_pred)
        
        historico = {
            'cliente_id': self.id_cliente,
            'tipo': 'honesto',
            'acuracia_treino': acuracia
        }
        self.historico_treinamento.append(historico)
        
        return historico


class ClienteEnvenenado(ClienteBase):
    """
    Cliente malicioso que corrompe pesos após treinamento
    Padrão UML: Herança de ClienteBase + Comportamento específico
    """
    
    def __init__(self, id_cliente: str, dados_locais: Tuple[np.ndarray, np.ndarray], 
                 taxa_corrupcao: float = 0.5, tipo_ataque: str = 'inverter'):
        super().__init__(id_cliente, dados_locais)
        self.taxa_corrupcao = taxa_corrupcao
        self.tipo_ataque = tipo_ataque
    
    def treinar_modelo(self) -> Dict:
        """
        Treina o modelo local e depois CORROMPE OS PESOS
        
        Tipos de ataque:
        - 'inverter': Inverte sinal dos pesos
        - 'aleatorio': Adiciona ruído aleatório
        - 'amplificar': Amplifica pesos drasticamente
        - 'zerar': Zera parcialmente os pesos
        """
        # 1. TREINA NORMALMENTE
        X_scaled = self.scaler.fit_transform(self.X_local)
        self.modelo_local.fit(X_scaled, self.y_local)
        
        # Avalia ANTES da corrupção
        y_pred_antes = self.modelo_local.predict(X_scaled)
        acuracia_antes = accuracy_score(self.y_local, y_pred_antes)
        
        # 2. CORROMPE OS PESOS
        pesos_originais = self.modelo_local.obter_pesos()
        pesos_corrompidos = self._corromper_pesos(pesos_originais)
        self.modelo_local.atualizar_pesos(pesos_corrompidos)
        
        # Avalia DEPOIS da corrupção
        y_pred_depois = self.modelo_local.predict(X_scaled)
        acuracia_depois = accuracy_score(self.y_local, y_pred_depois)
        
        historico = {
            'cliente_id': self.id_cliente,
            'tipo': 'envenenado',
            'tipo_ataque': self.tipo_ataque,
            'taxa_corrupcao': self.taxa_corrupcao,
            'acuracia_antes': acuracia_antes,
            'acuracia_depois': acuracia_depois,
            'degradacao': acuracia_antes - acuracia_depois
        }
        self.historico_treinamento.append(historico)
        
        return historico
    
    def _corromper_pesos(self, pesos: Dict) -> Dict:
        """Aplica corrupção aos pesos do modelo"""
        pesos_corrompidos = deepcopy(pesos)
        
        if self.tipo_ataque == 'inverter':
            # Inverte o sinal dos coeficientes
            pesos_corrompidos['coef'] = -pesos['coef'] * (1 + self.taxa_corrupcao)
            pesos_corrompidos['intercept'] = -pesos['intercept'] * (1 + self.taxa_corrupcao)
        
        elif self.tipo_ataque == 'aleatorio':
            # Adiciona ruído gaussiano
            ruido_coef = np.random.randn(*pesos['coef'].shape) * self.taxa_corrupcao
            ruido_intercept = np.random.randn(*pesos['intercept'].shape) * self.taxa_corrupcao
            pesos_corrompidos['coef'] = pesos['coef'] + ruido_coef
            pesos_corrompidos['intercept'] = pesos['intercept'] + ruido_intercept
        
        elif self.tipo_ataque == 'amplificar':
            # Amplifica pesos drasticamente
            pesos_corrompidos['coef'] = pesos['coef'] * (1 + self.taxa_corrupcao * 10)
            pesos_corrompidos['intercept'] = pesos['intercept'] * (1 + self.taxa_corrupcao * 10)
        
        elif self.tipo_ataque == 'zerar':
            # Zera percentual dos pesos
            mascara_coef = np.random.rand(*pesos['coef'].shape) > self.taxa_corrupcao
            mascara_intercept = np.random.rand(*pesos['intercept'].shape) > self.taxa_corrupcao
            pesos_corrompidos['coef'] = pesos['coef'] * mascara_coef
            pesos_corrompidos['intercept'] = pesos['intercept'] * mascara_intercept
        
        return pesos_corrompidos


class ServidorFederado:
    """
    Servidor central do aprendizado federado
    Padrão UML: Composição (contém múltiplos clientes)
    """
    
    def __init__(self, dados_validacao: Tuple[np.ndarray, np.ndarray]):
        self.clientes = []
        self.modelo_global = ModeloClassificacao()
        self.X_val, self.y_val = dados_validacao
        self.scaler_global = StandardScaler()
        
        # Históricos para análise
        self.historico_agregacao = []
        self.historico_metricas_globais = []
        self.historico_metricas_por_classe = []
    
    def adicionar_cliente(self, cliente: ClienteBase):
        """Adiciona um cliente ao sistema federado"""
        self.clientes.append(cliente)
    
    def executar_rodada_federada(self, num_rodada: int) -> Dict:
        """
        Executa uma rodada completa de aprendizado federado
        1. Compartilha modelo global atual com os clientes
        2. Clientes treinam A PARTIR do modelo global (fine-tuning)
        3. Clientes honestos ou envenenados enviam pesos
        4. Servidor agrega os modelos
        5. Avalia modelo global
        """
        print(f"\n{'='*70}")
        print(f"RODADA FEDERADA {num_rodada}")
        print(f"{'='*70}")
        
        # 0. COMPARTILHA MODELO GLOBAL COM CLIENTES (se não é primeira rodada)
        if num_rodada > 1:
            pesos_globais = self.modelo_global.obter_pesos()
            if pesos_globais:
                print("\n[FASE 0] Compartilhamento do Modelo Global:")
                print("-" * 70)
                print(f"  ✓ Modelo global distribuído para {len(self.clientes)} clientes")
                for cliente in self.clientes:
                    cliente.set_modelo_local(pesos_globais)
        
        # 1. TREINAMENTO LOCAL (A PARTIR DO MODELO GLOBAL RECEBIDO)
        print("\n[FASE 1] Treinamento Local dos Clientes:")
        print("-" * 70)
        resultados_clientes = []
        
        for cliente in self.clientes:
            resultado = cliente.treinar_modelo()
            resultados_clientes.append(resultado)
            
            if resultado['tipo'] == 'honesto':
                print(f"  ✓ {cliente.id_cliente}: Honesto | Acc={resultado['acuracia_treino']:.4f}")
            else:
                print(f"  ⚠ {cliente.id_cliente}: ENVENENADO ({resultado['tipo_ataque']}) | "
                      f"Antes={resultado['acuracia_antes']:.4f} → "
                      f"Depois={resultado['acuracia_depois']:.4f} | "
                      f"Degradação={resultado['degradacao']:.4f}")
        
        # 2. AGREGAÇÃO DOS MODELOS (FedAvg simples)
        print("\n[FASE 2] Agregação de Modelos:")
        print("-" * 70)
        pesos_agregados = self._agregar_modelos()
        self.modelo_global.atualizar_pesos(pesos_agregados)
        print(f"  ✓ Agregação FedAvg concluída ({len(self.clientes)} clientes)")
        
        # 3. AVALIAÇÃO GLOBAL
        print("\n[FASE 3] Avaliação do Modelo Global:")
        print("-" * 70)
        metricas_globais = self._avaliar_modelo_global()
        
        print(f"  Acurácia Global: {metricas_globais['acuracia_global']:.4f} "
              f"({metricas_globais['acuracia_global']*100:.2f}%)")
        print(f"  F1-Score: {metricas_globais['f1_global']:.4f}")
        print(f"  Loss: {metricas_globais['loss_global']:.4f}")
        
        # 4. AVALIAÇÃO POR CLASSE
        metricas_classes = self._avaliar_por_classe()
        print(f"\n  Acurácia por Espécie:")
        for classe, acc in metricas_classes.items():
            print(f"    - {classe}: {acc:.4f} ({acc*100:.1f}%)")
        
        # Armazena histórico
        resultado_rodada = {
            'num_rodada': num_rodada,
            'metricas_globais': metricas_globais,
            'metricas_classes': metricas_classes,
            'resultados_clientes': resultados_clientes
        }
        self.historico_agregacao.append(resultado_rodada)
        
        return resultado_rodada
    
    def _agregar_modelos(self) -> Dict:
        """Agrega os modelos locais usando FedAvg"""
        pesos_clientes = []
        
        for cliente in self.clientes:
            pesos = cliente.modelo_local.obter_pesos()
            if pesos:
                pesos_clientes.append(pesos)
        
        if not pesos_clientes:
            return None
        
        # FedAvg: Média simples dos pesos
        pesos_agregados = {
            'coef': np.mean([p['coef'] for p in pesos_clientes], axis=0),
            'intercept': np.mean([p['intercept'] for p in pesos_clientes], axis=0),
            'classes': pesos_clientes[0]['classes']
        }
        
        return pesos_agregados
    
    def _avaliar_modelo_global(self) -> Dict:
        """Avalia o modelo global no conjunto de validação"""
        X_val_scaled = self.scaler_global.fit_transform(self.X_val)
        y_pred = self.modelo_global.predict(X_val_scaled)
        y_proba = self.modelo_global.predict_proba(X_val_scaled)
        
        metricas = {
            'acuracia_global': accuracy_score(self.y_val, y_pred),
            'f1_global': f1_score(self.y_val, y_pred, average='weighted'),
            'precisao_global': precision_score(self.y_val, y_pred, average='weighted'),
            'recall_global': recall_score(self.y_val, y_pred, average='weighted'),
            'loss_global': log_loss(self.y_val, y_proba),
            'y_pred': y_pred,
            'y_true': self.y_val,
            'confusion_matrix': confusion_matrix(self.y_val, y_pred)
        }
        
        self.historico_metricas_globais.append(metricas)
        return metricas
    
    def _avaliar_por_classe(self) -> Dict:
        """Avalia acurácia por classe/espécie"""
        if not self.historico_metricas_globais:
            return {}
        
        ultima_metrica = self.historico_metricas_globais[-1]
        y_true = ultima_metrica['y_true']
        y_pred = ultima_metrica['y_pred']
        
        # Nomes das espécies
        especies = ['Setosa', 'Versicolor', 'Virginica']
        
        metricas_por_classe = {}
        for i, especie in enumerate(especies):
            # Acurácia binária para cada classe
            mascara = y_true == i
            if mascara.sum() > 0:
                acc_classe = accuracy_score(y_true[mascara], y_pred[mascara])
                metricas_por_classe[especie] = acc_classe
        
        self.historico_metricas_por_classe.append(metricas_por_classe)
        return metricas_por_classe
    
    def gerar_relatorio_completo(self):
        """Gera relatório final com todas as métricas"""
        print(f"\n{'='*70}")
        print("RELATÓRIO FINAL - CENÁRIO DE ENVENENAMENTO")
        print(f"{'='*70}")
        
        # Estatísticas dos clientes
        print(f"\nClientes:")
        honestos = sum(1 for c in self.clientes if isinstance(c, ClienteHonesto))
        envenenados = sum(1 for c in self.clientes if isinstance(c, ClienteEnvenenado))
        print(f"  - Honestos: {honestos}")
        print(f"  - Envenenados: {envenenados}")
        
        # Evolução das métricas
        print(f"\nEvolução das Métricas Globais:")
        print(f"  Rodada | Acurácia | F1-Score | Loss")
        print(f"  {'-'*40}")
        for i, metricas in enumerate(self.historico_metricas_globais, 1):
            print(f"  {i:^6} | {metricas['acuracia_global']:^8.4f} | "
                  f"{metricas['f1_global']:^8.4f} | {metricas['loss_global']:^8.4f}")
        
        # Matriz de confusão final
        if self.historico_metricas_globais:
            print(f"\nMatriz de Confusão (Última Rodada):")
            cm = self.historico_metricas_globais[-1]['confusion_matrix']
            especies = ['Setosa', 'Versicolor', 'Virginica']
            print("\n           ", "  ".join([f"{e:>11}" for e in especies]))
            for i, especie in enumerate(especies):
                print(f"{especie:>11}:", "  ".join([f"{cm[i,j]:>11}" for j in range(3)]))
    
    def gerar_graficos(self):
        """Gera visualizações dos resultados"""
        if not self.historico_metricas_globais:
            print("Nenhuma métrica disponível para gerar gráficos")
            return
        
        # 1. GRÁFICO DE CONVERGÊNCIA (Métricas por Rodada)
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        rodadas = list(range(1, len(self.historico_metricas_globais) + 1))
        acuracias = [m['acuracia_global'] for m in self.historico_metricas_globais]
        f1_scores = [m['f1_global'] for m in self.historico_metricas_globais]
        losses = [m['loss_global'] for m in self.historico_metricas_globais]
        
        # Acurácia
        axes[0, 0].plot(rodadas, acuracias, marker='o', linewidth=2.5, markersize=8, color='#2E86AB')
        axes[0, 0].set_title('Acurácia Global por Rodada', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Rodada Federada', fontsize=12)
        axes[0, 0].set_ylabel('Acurácia', fontsize=12)
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim([0, 1.05])
        
        # F1-Score
        axes[0, 1].plot(rodadas, f1_scores, marker='s', linewidth=2.5, markersize=8, color='#A23B72')
        axes[0, 1].set_title('F1-Score Global por Rodada', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Rodada Federada', fontsize=12)
        axes[0, 1].set_ylabel('F1-Score', fontsize=12)
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim([0, 1.05])
        
        # Loss
        axes[1, 0].plot(rodadas, losses, marker='^', linewidth=2.5, markersize=8, color='#F18F01')
        axes[1, 0].set_title('Loss Global por Rodada', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Rodada Federada', fontsize=12)
        axes[1, 0].set_ylabel('Loss (Log Loss)', fontsize=12)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Comparação de métricas
        axes[1, 1].plot(rodadas, acuracias, marker='o', label='Acurácia', linewidth=2.5, markersize=8)
        axes[1, 1].plot(rodadas, f1_scores, marker='s', label='F1-Score', linewidth=2.5, markersize=8)
        axes[1, 1].set_title('Comparação de Métricas', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Rodada Federada', fontsize=12)
        axes[1, 1].set_ylabel('Valor', fontsize=12)
        axes[1, 1].legend(fontsize=11)
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_ylim([0, 1.05])
        
        plt.tight_layout()
        plt.savefig('poisoning_attack_convergencia.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Gráfico salvo: poisoning_attack_convergencia.png")
        plt.close()
        
        # 2. GRÁFICO DE ACURÁCIA POR CLASSE/ESPÉCIE
        self._gerar_grafico_especies()
    
    def _gerar_grafico_especies(self):
        """Gera gráfico de acurácia por espécie ao longo das rodadas"""
        if not self.historico_metricas_por_classe:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))
        
        especies = ['Setosa', 'Versicolor', 'Virginica']
        rodadas = list(range(1, len(self.historico_metricas_por_classe) + 1))
        cores = ['#06D6A0', '#118AB2', '#EF476F']
        
        # Gráfico 1: Linhas com evolução por espécie
        for i, especie in enumerate(especies):
            acuracias_especie = [m.get(especie, 0) for m in self.historico_metricas_por_classe]
            axes[0].plot(rodadas, acuracias_especie, marker='o', linewidth=2.5, 
                        markersize=8, label=especie, color=cores[i])
        
        axes[0].set_title('Acurácia por Espécie ao Longo das Rodadas', 
                         fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Rodada Federada', fontsize=12)
        axes[0].set_ylabel('Acurácia', fontsize=12)
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim([0, 1.05])
        
        # Gráfico 2: Barras com acurácia final por espécie
        if self.historico_metricas_por_classe:
            ultima_rodada = self.historico_metricas_por_classe[-1]
            acuracias_finais = [ultima_rodada.get(e, 0) for e in especies]
            
            bars = axes[1].bar(especies, acuracias_finais, color=cores, alpha=0.8, edgecolor='black')
            axes[1].set_title('Acurácia Final por Espécie', fontsize=14, fontweight='bold')
            axes[1].set_ylabel('Acurácia', fontsize=12)
            axes[1].set_ylim([0, 1.05])
            axes[1].grid(True, alpha=0.3, axis='y')
            
            # Adiciona valores nas barras
            for bar, acc in zip(bars, acuracias_finais):
                height = bar.get_height()
                axes[1].text(bar.get_x() + bar.get_width()/2., height,
                           f'{acc:.3f}\n({acc*100:.1f}%)',
                           ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('poisoning_attack_especies.png', dpi=300, bbox_inches='tight')
        print(f"✓ Gráfico salvo: poisoning_attack_especies.png")
        plt.close()


def carregar_dataset_iris():
    """Carrega e prepara o dataset Iris"""
    print("\n[CARREGANDO DATASET IRIS]")
    print("-" * 70)
    
    # Tenta carregar do arquivo local primeiro
    caminho_dataset = r"c:\Users\Administrador\Faculdade\Iniciação-cientifica\project\data\iris\iris.csv"
    
    try:
        df = pd.read_csv(caminho_dataset)
        print(f"✓ Dataset carregado: {caminho_dataset}")
    except:
        # Fallback: usa sklearn
        from sklearn.datasets import load_iris
        iris = load_iris()
        df = pd.DataFrame(iris.data, columns=iris.feature_names)
        df['species'] = iris.target
        print(f"✓ Dataset carregado via sklearn.datasets")
    
    # Preparação
    if 'species' in df.columns:
        X = df.drop('species', axis=1).values
        
        # Se species é string, converte para numérico
        if df['species'].dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(df['species'])
        else:
            y = df['species'].values
    else:
        # Assume que a última coluna é o target
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
    
    print(f"  - Amostras: {len(X)}")
    print(f"  - Features: {X.shape[1]}")
    print(f"  - Classes: {len(np.unique(y))}")
    
    return X, y


def main():
    """Função principal do experimento"""
    print("="*70)
    print("EXPERIMENTO: ENVENENAMENTO DE MODELO - DATASET IRIS")
    print("Cenário: Treina Local > Corrompe Pesos > Avalia")
    print("="*70)
    
    # 1. CARREGA DATASET
    X, y = carregar_dataset_iris()
    
    # 2. DIVIDE DADOS
    print("\n[DIVISÃO DOS DADOS]")
    print("-" * 70)
    
    # Split principal: treino + validação
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Divide treino entre clientes (3 clientes: 2 honestos, 1 envenenado)
    X_c1, X_temp, y_c1, y_temp = train_test_split(
        X_train, y_train, test_size=0.66, random_state=42, stratify=y_train
    )
    X_c2, X_c3, y_c2, y_c3 = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"✓ Cliente 1 (Honesto): {len(X_c1)} amostras")
    print(f"✓ Cliente 2 (Honesto): {len(X_c2)} amostras")
    print(f"✓ Cliente 3 (ENVENENADO): {len(X_c3)} amostras")
    print(f"✓ Validação: {len(X_val)} amostras")
    
    # 3. CRIA SERVIDOR E CLIENTES
    print("\n[CONFIGURAÇÃO DO SISTEMA FEDERADO]")
    print("-" * 70)
    
    servidor = ServidorFederado(dados_validacao=(X_val, y_val))
    
    # Clientes honestos
    cliente1 = ClienteHonesto(id_cliente="Cliente_1_Honesto", dados_locais=(X_c1, y_c1))
    cliente2 = ClienteHonesto(id_cliente="Cliente_2_Honesto", dados_locais=(X_c2, y_c2))
    
    # Cliente envenenado (vamos testar diferentes tipos de ataque)
    cliente3 = ClienteEnvenenado(
        id_cliente="Cliente_3_Envenenado", 
        dados_locais=(X_c3, y_c3),
        taxa_corrupcao=0.8,  # 80% de corrupção
        tipo_ataque='inverter'  # Inverte sinais dos pesos
    )
    
    servidor.adicionar_cliente(cliente1)
    servidor.adicionar_cliente(cliente2)
    servidor.adicionar_cliente(cliente3)
    
    print(f"✓ Servidor configurado com 3 clientes")
    print(f"  - 2 clientes honestos")
    print(f"  - 1 cliente envenenado (taxa=0.8, ataque='inverter')")
    
    # 4. EXECUTA MÚLTIPLAS RODADAS FEDERADAS
    print("\n[EXECUÇÃO DAS RODADAS FEDERADAS]")
    print("="*70)
    
    num_rodadas = 5
    for rodada in range(1, num_rodadas + 1):
        servidor.executar_rodada_federada(rodada)
    
    # 5. GERA RELATÓRIO E GRÁFICOS
    servidor.gerar_relatorio_completo()
    servidor.gerar_graficos()
    
    print("\n" + "="*70)
    print("EXPERIMENTO CONCLUÍDO COM SUCESSO")
    print("="*70)
    print("\nArquivos gerados:")
    print("  - poisoning_attack_convergencia.png")
    print("  - poisoning_attack_especies.png")
    print("\nObservações:")
    print("  O envenenamento degrada significativamente a acurácia do cliente")
    print("  atacante, e ao agregar com FedAvg, contamina o modelo global.")
    print("="*70)


if __name__ == "__main__":
    main()
