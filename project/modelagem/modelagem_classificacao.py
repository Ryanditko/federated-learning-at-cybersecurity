"""
Sistema de Aprendizado Federado com Classificação
Modelo: LogisticRegression (Classificação)
Dataset: Iris (project/data/iris/iris.csv)
Objetivo: Classificar espécies de flores (setosa, versicolor, virginica)
Feature: Rounds locais (épocas) + Convergência adaptativa
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
from abc import ABC, abstractmethod
from typing import Dict, List
import warnings
import os
warnings.filterwarnings('ignore')

class ModeloClassificacao:
    """Classe que encapsula o modelo de classificação"""
    
    def __init__(self, max_iter=1000, random_state=42):
        self._modelo_interno = LogisticRegression(
            max_iter=max_iter, 
            random_state=random_state,
            multi_class='multinomial',
            solver='lbfgs'  # Suporta multinomial
        )
    
    def obter_pesos(self):
        """Retorna os pesos do modelo (coeficientes e intercept)"""
        if hasattr(self._modelo_interno, 'coef_'):
            return {
                'coef': deepcopy(self._modelo_interno.coef_),
                'intercept': deepcopy(self._modelo_interno.intercept_),
                'classes': deepcopy(self._modelo_interno.classes_)
            }
        return None
    
    def atualizar_pesos(self, pesos):
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
    
    def get_coef(self):
        """Retorna os coeficientes"""
        return self._modelo_interno.coef_ if hasattr(self._modelo_interno, 'coef_') else None
    
    def get_intercept(self):
        """Retorna o intercept"""
        return self._modelo_interno.intercept_ if hasattr(self._modelo_interno, 'intercept_') else None


class ServidorFederadoClassificacao:
    """Servidor central do aprendizado federado para classificação"""
    
    def __init__(self, max_rodadas: int = 20, criterio_convergencia: float = 0.005, 
                 dados_validacao=None, threshold_acuracia: float = 0.95, inicializar_aleatorio: bool = True):
        self.rodada_atual = 0
        self.max_rodadas = max_rodadas
        self.criterio_convergencia = criterio_convergencia
        self.threshold_acuracia = threshold_acuracia  # Acurácia alvo
        self.clientes = []
        self.modelo_global = ModeloClassificacao()
        self.metricas_avaliacao = {}
        self.historico_metricas = []
        self.outliers_detectados = []
        self.desempenho_anterior = None
        self.dados_validacao = dados_validacao
        self.inicializar_aleatorio = inicializar_aleatorio
        
        # Históricos para gráficos
        self.historico_acuracia_global = []
        self.historico_f1_global = []
        self.historico_loss_global = []
        self.historico_clientes_aceitos = []
        self.historico_clientes_rejeitados = []
        
        # Convergência
        self.convergiu = False
        self.rodada_convergencia = None
    
    def adicionar_cliente(self, cliente):
        """Adiciona um cliente"""
        self.clientes.append(cliente)
    
    def inicializar_modelo_global_aleatorio(self):
        """Inicializa modelo global com pesos ZERO (não treinado - chute aleatório)"""
        if self.dados_validacao and self.inicializar_aleatorio:
            X_val, y_val = self.dados_validacao
            
            # Cria modelo vazio mas com estrutura correta
            classes_unicas = np.unique(y_val)
            n_classes = len(classes_unicas)
            n_features = X_val.shape[1]
            
            # PESOS COMPLETAMENTE ZERO (modelo não treinado = chute aleatório ~33%)
            pesos_zero = {
                'coef': np.zeros((n_classes, n_features)),  # Matriz zero
                'intercept': np.zeros(n_classes),  # Intercepts zero
                'classes': classes_unicas
            }
            
            self.modelo_global.atualizar_pesos(pesos_zero)
            print(f"[INFO] Modelo global inicializado com pesos ZERO (esperado ~33% acuracia)")
    
    def compartilhar_modelo_global(self):
        """Distribui pesos do modelo global para todos os clientes"""
        pesos_globais = self.modelo_global.obter_pesos()
        if pesos_globais:
            for cliente in self.clientes:
                cliente.set_modelo_local(pesos_globais)
    
    def avaliar_modelo(self):
        """Avalia o desempenho do modelo global no conjunto de validação"""
        if self.dados_validacao is not None and self.modelo_global.obter_pesos() is not None:
            X_val, y_val = self.dados_validacao
            scaler = StandardScaler()
            X_val_scaled = scaler.fit_transform(X_val)
            
            # Predições
            y_pred = self.modelo_global.predict(X_val_scaled)
            y_proba = self.modelo_global.predict_proba(X_val_scaled)
            
            # Métricas
            acuracia_global = accuracy_score(y_val, y_pred)
            f1_global = f1_score(y_val, y_pred, average='weighted')
            precisao_global = precision_score(y_val, y_pred, average='weighted')
            recall_global = recall_score(y_val, y_pred, average='weighted')
            loss_global = log_loss(y_val, y_proba)
            
            self.historico_acuracia_global.append(acuracia_global)
            self.historico_f1_global.append(f1_global)
            self.historico_loss_global.append(loss_global)
            
            self.metricas_avaliacao = {
                'rodada': self.rodada_atual,
                'acuracia_global': acuracia_global,
                'f1_global': f1_global,
                'precisao_global': precisao_global,
                'recall_global': recall_global,
                'loss_global': loss_global,
                'num_clientes': len(self.clientes)
            }
            self.historico_metricas.append(self.metricas_avaliacao.copy())
            
            return acuracia_global
        return 0
    
    def set_modelo_global(self):
        """Atualiza o modelo global com agregação dos modelos locais"""
        num_aceitos, num_rejeitados = self._agregar_modelos()
        self.historico_clientes_aceitos.append(num_aceitos)
        self.historico_clientes_rejeitados.append(num_rejeitados)
    
    def avaliar_convergencia(self, acuracia_atual: float) -> bool:
        """Verifica se o treinamento convergiu"""
        # Critério 1: Atingiu número máximo de rodadas
        if self.rodada_atual >= self.max_rodadas:
            print(f"   [CONVERGENCIA] Numero maximo de rodadas atingido ({self.max_rodadas})")
            return True
        
        # Critério 2: Acurácia atingiu threshold desejado
        if acuracia_atual >= self.threshold_acuracia:
            print(f"   [CONVERGENCIA] Acuracia alvo atingida: {acuracia_atual:.4f} >= {self.threshold_acuracia}")
            self.convergiu = True
            self.rodada_convergencia = self.rodada_atual
            return True
        
        # Critério 3: Estabilidade do modelo (variação pequena)
        if self.desempenho_anterior is not None:
            variacao = abs(acuracia_atual - self.desempenho_anterior)
            
            if variacao < self.criterio_convergencia and self.rodada_atual >= 5:
                print(f"   [CONVERGENCIA] Estabilizacao detectada: variacao={variacao:.5f} < {self.criterio_convergencia}")
                self.convergiu = True
                self.rodada_convergencia = self.rodada_atual
                return True
            
            self.desempenho_anterior = acuracia_atual
        else:
            self.desempenho_anterior = acuracia_atual
        
        return False
    
    def executar_aprendizado_federado(self):
        """Executa o loop principal do aprendizado federado"""
        print("="*70)
        print("SISTEMA DE APRENDIZADO FEDERADO - CLASSIFICACAO IRIS")
        print("="*70)
        print(f"Configuracao:")
        print(f"  - Modelo: Regressao Logistica (Multinomial)")
        print(f"  - Clientes: {len(self.clientes)}")
        print(f"  - Rodadas maximas: {self.max_rodadas}")
        print(f"  - Acuracia alvo: {self.threshold_acuracia:.2%}")
        print(f"  - Criterio convergencia: {self.criterio_convergencia}")
        print("="*70)
        
        # Inicializa modelo global com pesos ruins (para convergência gradual)
        if self.inicializar_aleatorio:
            self.inicializar_modelo_global_aleatorio()
        
        while not self.avaliar_convergencia(
            self.historico_acuracia_global[-1] if self.historico_acuracia_global else 0
        ):
            self.rodada_atual += 1
            print(f"\n{'='*70}")
            print(f"RODADA FEDERADA {self.rodada_atual}/{self.max_rodadas}")
            print(f"{'='*70}")
            
            # 1. Compartilhar modelo global com todos os clientes
            self.compartilhar_modelo_global()
            
            # 2. Cada cliente treina LOCALMENTE (múltiplos rounds/épocas)
            print("\n[FASE 1] Treinamento Local (com rounds internos):")
            print("-" * 70)
            for cliente in self.clientes:
                # Modelo local com EXTREMAMENTE POUCAS iterações (convergência lentíssima)
                cliente.modelo_local = ModeloClassificacao(max_iter=5)  # APENAS 5 iterações!
                historico_cliente = cliente.treinar_modelo()
                
                # Mostra progresso simplificado
                if historico_cliente:
                    acc_final = historico_cliente[-1]['acuracia']
                    print(f"  {cliente.id_cliente}: {len(historico_cliente)} rounds | Acc={acc_final:.3f}")
            
            # 3. Agregar modelos locais no servidor (com detecção de outliers)
            print(f"\n[FASE 2] Agregacao de Modelos (FedAvg + MAD):")
            print("-" * 70)
            self.set_modelo_global()
            
            # 4. Avaliar modelo global no conjunto de validação
            print(f"\n[FASE 3] Avaliacao do Modelo Global:")
            print("-" * 70)
            acuracia_global = self.avaliar_modelo()
            
            # 5. Mostrar métricas da rodada (simplificado)
            if self.metricas_avaliacao:
                acc = self.metricas_avaliacao['acuracia_global']
                f1 = self.metricas_avaliacao['f1_global']
                loss = self.metricas_avaliacao['loss_global']
                
                print(f"  Metricas Globais: Acc={acc:.3f} ({acc*100:.1f}%) | F1={f1:.3f} | Loss={loss:.3f}")
                
                # Barra de progresso compacta
                progresso = min(100, (acc / self.threshold_acuracia) * 100)
                barra_len = 30
                preenchido = int(barra_len * acc / self.threshold_acuracia)
                barra = '█' * min(preenchido, barra_len) + '░' * max(0, barra_len - preenchido)
                print(f"  Progresso: [{barra}] {progresso:.0f}%")
        
        print(f"\n{'='*70}")
        print("APRENDIZADO FEDERADO CONCLUIDO")
        print(f"{'='*70}")
        
        if self.convergiu:
            print(f"\n✓ Modelo convergiu na rodada {self.rodada_convergencia}")
            print(f"✓ Acuracia final: {self.historico_acuracia_global[-1]:.4f} ({self.historico_acuracia_global[-1]*100:.2f}%)")
        else:
            print(f"\n⚠ Numero maximo de rodadas atingido")
            print(f"  Acuracia final: {self.historico_acuracia_global[-1]:.4f} ({self.historico_acuracia_global[-1]*100:.2f}%)")
        
        # Gerar visualizações
        print(f"\n[GERANDO GRAFICOS] Salvando graficos de convergencia...")
        self.gerar_graficos()
        
        # Gerar gráfico de classificação por espécie
        print(f"\n[GERANDO GRAFICOS] Salvando analise de especies...")
        self.gerar_grafico_classificacao_especies()
        
        # Relatório de detecção de outliers
        if self.outliers_detectados:
            print("\n" + "="*70)
            print("RELATORIO DE DETECCAO DE OUTLIERS (MAD)")
            print("="*70)
            for deteccao in self.outliers_detectados:
                rodada = deteccao['rodada']
                clientes = ', '.join(deteccao['clientes'])
                print(f"  Rodada {rodada}: {clientes}")
            
            # Estatísticas
            total_deteccoes = sum(len(d['clientes']) for d in self.outliers_detectados)
            print(f"\nEstatisticas:")
            print(f"  Total de deteccoes: {total_deteccoes}")
            print(f"  Rodadas com deteccao: {len(self.outliers_detectados)}/{self.rodada_atual}")
            print(f"  Taxa de deteccao: {len(self.outliers_detectados)/self.rodada_atual*100:.1f}%")
        else:
            print("\n✓ Nenhum outlier detectado durante o treinamento.")
    
    def _agregar_modelos(self):
        """Agrega modelos usando FedAvg com detecção MAD de outliers"""
        coefs = []
        intercepts = []
        clientes_aceitos = []
        clientes_rejeitados = []
        
        # Coleta todos os coeficientes
        todos_coefs = []
        for cliente in self.clientes:
            coef = cliente.modelo_local.get_coef()
            if coef is not None:
                # Achatar matriz de coeficientes para cálculo de distância
                todos_coefs.append(coef.flatten())
        
        if not todos_coefs:
            return 0, 0
        
        # Calcula mediana e MAD (Median Absolute Deviation)
        mediana_coefs = np.median(todos_coefs, axis=0)
        
        # Para cada cliente, calcula distância da mediana
        for cliente in self.clientes:
            coef = cliente.modelo_local.get_coef()
            if coef is not None:
                coef_flat = coef.flatten()
                
                # Distância euclidiana dos coeficientes para a mediana
                distancia = np.linalg.norm(coef_flat - mediana_coefs)
                
                # Calcula threshold baseado em MAD
                desvios = [np.linalg.norm(c - mediana_coefs) for c in todos_coefs]
                mad = np.median(np.abs(desvios - np.median(desvios)))
                threshold = np.median(desvios) + 3 * mad  # 3 MADs = outlier
                
                # Detecta outlier
                if distancia > threshold and mad > 0:
                    clientes_rejeitados.append(cliente.id_cliente)
                    print(f"  [OUTLIER] {cliente.id_cliente:20s} | Distancia: {distancia:.4f} > Threshold: {threshold:.4f}")
                else:
                    coefs.append(cliente.modelo_local.get_coef())
                    intercepts.append(cliente.modelo_local.get_intercept())
                    clientes_aceitos.append(cliente.id_cliente)
        
        # Agrega apenas clientes não-outliers (FedAvg)
        if coefs:
            pesos_agregados = {
                'coef': np.mean(coefs, axis=0),
                'intercept': np.mean(intercepts, axis=0)
            }
            
            # Preserva classes (importante para multinomial)
            pesos_exemplo = self.clientes[0].modelo_local.obter_pesos()
            if pesos_exemplo and 'classes' in pesos_exemplo:
                pesos_agregados['classes'] = pesos_exemplo['classes']
            
            self.modelo_global.atualizar_pesos(pesos_agregados)
            
            print(f"  [AGREGACAO] Aceitos: {len(clientes_aceitos)} | Rejeitados: {len(clientes_rejeitados)}")
            
            if clientes_rejeitados:
                self.outliers_detectados.append({
                    'rodada': self.rodada_atual,
                    'clientes': clientes_rejeitados
                })
        else:
            print(f"  [ALERTA] Todos os clientes foram detectados como outliers!")
        
        return len(clientes_aceitos), len(clientes_rejeitados)
    
    def gerar_graficos(self):
        """Gera gráfico de convergência do aprendizado federado"""
        if not self.historico_metricas:
            print("  Nenhuma metrica para visualizar.")
            return
        
        n_rodadas = len(self.historico_metricas)
        rodadas = list(range(1, n_rodadas + 1))
        
        # Criar figura com 6 subplots
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.25)
        
        fig.suptitle('Convergência do Aprendizado Federado - Classificação Iris', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        # Subplot 1: Acurácia ao longo das rodadas (PRINCIPAL - LIMPO E CLARO)
        ax1 = fig.add_subplot(gs[0, :])  # Ocupa toda primeira linha
        
        # Linha principal com marcadores
        ax1.plot(rodadas, self.historico_acuracia_global, marker='o', linewidth=2.5, 
                color='#2e7d32', markersize=7, label='Acurácia Global', zorder=3,
                markeredgecolor='white', markeredgewidth=1)
        
        # Área sombreada sob a curva
        ax1.fill_between(rodadas, 0, self.historico_acuracia_global, 
                         alpha=0.15, color='#4caf50', zorder=1)
        
        # Linha de threshold (meta)
        ax1.axhline(y=self.threshold_acuracia, color='#d32f2f', linestyle='--', 
                   linewidth=2, label=f'Meta ({self.threshold_acuracia:.0%})', zorder=2)
        
        # Marca ponto de convergência
        if self.convergiu and self.rodada_convergencia:
            idx = self.rodada_convergencia - 1
            acc_conv = self.historico_acuracia_global[idx]
            ax1.scatter(self.rodada_convergencia, acc_conv, 
                       s=400, color='#ffc107', marker='*', edgecolors='#ff6f00', linewidths=2.5,
                       label=f'Convergência (R{self.rodada_convergencia})', zorder=5)
            
            # Linha vertical
            ax1.axvline(x=self.rodada_convergencia, color='#ffc107', linestyle=':', 
                       linewidth=1.5, alpha=0.5, zorder=1)
        
        ax1.set_xlabel('Rodada Federada', fontsize=13, fontweight='bold')
        ax1.set_ylabel('Acurácia', fontsize=13, fontweight='bold')
        ax1.set_title('Convergência do Aprendizado Federado - Classificação Iris', 
                     fontsize=15, fontweight='bold', pad=15)
        
        # Grid limpo
        ax1.grid(True, alpha=0.2, linestyle='-', linewidth=0.7)
        ax1.set_axisbelow(True)
        
        ax1.legend(fontsize=10, loc='lower right', framealpha=0.9)
        ax1.set_ylim([0, 1.05])  # Começa do ZERO para mostrar evolução completa
        ax1.set_xlim([0.5, n_rodadas + 0.5])
        
        # Anotações apenas em pontos-chave (início e fim)
        for i in [0, -1]:
            r = rodadas[i]
            acc = self.historico_acuracia_global[i]
            label = 'Início' if i == 0 else 'Final'
            ax1.annotate(f'{label}\n{acc:.1%}', xy=(r, acc), xytext=(0, 15),
                       textcoords='offset points', ha='center', fontsize=9,
                       fontweight='bold', color='#1b5e20',
                       bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                                edgecolor='#4caf50', linewidth=1.5, alpha=0.95))
        
        # Subplot 2: F1-Score
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(rodadas, self.historico_f1_global, marker='s', linewidth=2, 
                color='#3498db', markersize=7, label='F1-Score Global')
        ax2.set_xlabel('Rodada', fontsize=11)
        ax2.set_ylabel('F1-Score', fontsize=11)
        ax2.set_title('F1-Score por Rodada', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=10)
        ax2.set_ylim([0, 1.05])
        
        # Subplot 3: Loss (Log Loss)
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.plot(rodadas, self.historico_loss_global, marker='^', linewidth=2, 
                color='#e74c3c', markersize=7, label='Log Loss')
        ax3.set_xlabel('Rodada', fontsize=11)
        ax3.set_ylabel('Log Loss', fontsize=11)
        ax3.set_title('Loss por Rodada (menor = melhor)', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=10)
        
        # Subplot 4: Clientes Aceitos vs Rejeitados (Outliers)
        ax4 = fig.add_subplot(gs[2, 0])
        x_pos = np.arange(1, n_rodadas + 1)
        ax4.bar(x_pos, self.historico_clientes_aceitos, label='Aceitos', 
               color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=1)
        ax4.bar(x_pos, self.historico_clientes_rejeitados, 
               bottom=self.historico_clientes_aceitos,
               label='Outliers (Rejeitados)', color='#e74c3c', alpha=0.8,
               edgecolor='black', linewidth=1)
        ax4.set_xlabel('Rodada', fontsize=11)
        ax4.set_ylabel('Número de Clientes', fontsize=11)
        ax4.set_title('Clientes Aceitos vs Outliers por Rodada', fontsize=12, fontweight='bold')
        ax4.legend(fontsize=10)
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(rodadas)
        
        # Subplot 5: Métricas Combinadas
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.plot(rodadas, self.historico_acuracia_global, marker='o', linewidth=2, 
                label='Acurácia', color='#2ecc71')
        ax5.plot(rodadas, self.historico_f1_global, marker='s', linewidth=2, 
                label='F1-Score', color='#3498db')
        
        # Adiciona precisão e recall se disponíveis
        if self.historico_metricas:
            precisao = [m['precisao_global'] for m in self.historico_metricas]
            recall = [m['recall_global'] for m in self.historico_metricas]
            ax5.plot(rodadas, precisao, marker='d', linewidth=2, 
                    label='Precisão', color='#9b59b6', linestyle='--')
            ax5.plot(rodadas, recall, marker='v', linewidth=2, 
                    label='Recall', color='#f39c12', linestyle='--')
        
        ax5.set_xlabel('Rodada', fontsize=11)
        ax5.set_ylabel('Score', fontsize=11)
        ax5.set_title('Todas as Métricas de Classificação', fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        ax5.legend(fontsize=9, loc='lower right')
        ax5.set_ylim([0, 1.05])
        
        # Salvar figura
        plt.savefig('resultados_fl_classificacao.png', dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"  ✓ Graficos salvos em: resultados_fl_classificacao.png")
        plt.close()
    
    def gerar_grafico_classificacao_especies(self):
        """Gera gráfico mostrando classificação por espécie (Real vs Predito)"""
        if self.dados_validacao is None:
            return
        
        X_val, y_val = self.dados_validacao
        scaler = StandardScaler()
        X_val_scaled = scaler.fit_transform(X_val)
        
        # Predições do modelo final
        y_pred = self.modelo_global.predict(X_val_scaled)
        y_proba = self.modelo_global.predict_proba(X_val_scaled)
        
        # Nomes das classes
        label_encoder = LabelEncoder()
        especies = ['setosa', 'versicolor', 'virginica']
        
        # Criar figura com 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle('Análise de Classificação por Espécie - Dataset Iris', 
                    fontsize=16, fontweight='bold', y=1.02)
        
        # Subplot 1: Distribuição Real vs Predita
        ax1 = axes[0]
        real_counts = np.bincount(y_val, minlength=3)
        pred_counts = np.bincount(y_pred, minlength=3)
        
        x_pos = np.arange(len(especies))
        width = 0.35
        
        bars1 = ax1.bar(x_pos - width/2, real_counts, width, label='Real', 
                       color='#2196F3', alpha=0.8, edgecolor='black', linewidth=1.5)
        bars2 = ax1.bar(x_pos + width/2, pred_counts, width, label='Predito', 
                       color='#4CAF50', alpha=0.8, edgecolor='black', linewidth=1.5)
        
        ax1.set_xlabel('Espécie', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Quantidade', fontsize=12, fontweight='bold')
        ax1.set_title('Distribuição Real vs Predição', fontsize=13, fontweight='bold')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(especies, fontsize=11)
        ax1.legend(fontsize=11)
        ax1.grid(axis='y', alpha=0.3)
        
        # Adicionar valores nas barras
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}',
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Subplot 2: Percentual por Espécie
        ax2 = axes[1]
        percentuais = (pred_counts / pred_counts.sum()) * 100
        colors = ['#FF6B6B', '#4ECDC4', '#FFE66D']
        wedges, texts, autotexts = ax2.pie(percentuais, labels=especies, autopct='%1.1f%%',
                                            colors=colors, startangle=90,
                                            textprops={'fontsize': 11, 'fontweight': 'bold'},
                                            wedgeprops={'edgecolor': 'black', 'linewidth': 1.5})
        ax2.set_title('Percentual de Classificação\n(Dados de Validação)', 
                     fontsize=13, fontweight='bold')
        
        # Subplot 3: Confiança Média por Espécie
        ax3 = axes[2]
        confiancas_por_especie = []
        for i in range(3):
            indices = np.where(y_pred == i)[0]
            if len(indices) > 0:
                conf_media = y_proba[indices, i].mean()
                confiancas_por_especie.append(conf_media * 100)
            else:
                confiancas_por_especie.append(0)
        
        bars = ax3.bar(especies, confiancas_por_especie, color=colors, 
                      alpha=0.8, edgecolor='black', linewidth=1.5)
        ax3.set_xlabel('Espécie', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Confiança Média (%)', fontsize=12, fontweight='bold')
        ax3.set_title('Confiança do Modelo\npor Espécie', fontsize=13, fontweight='bold')
        ax3.set_ylim([0, 105])
        ax3.grid(axis='y', alpha=0.3)
        
        # Adicionar valores nas barras
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('classificacao_especies_iris.png', dpi=300, bbox_inches='tight')
        print(f"  ✓ Grafico de especies salvo em: classificacao_especies_iris.png")
        plt.close()
        
        # Imprimir estatísticas detalhadas
        print("\n" + "="*70)
        print("ESTATISTICAS DE CLASSIFICACAO POR ESPECIE")
        print("="*70)
        for i, especie in enumerate(especies):
            real = real_counts[i]
            pred = pred_counts[i]
            perc = percentuais[i]
            conf = confiancas_por_especie[i]
            print(f"\n{especie.upper()}:")
            print(f"  Amostras reais:      {real}")
            print(f"  Amostras preditas:   {pred}")
            print(f"  Percentual predito:  {perc:.1f}%")
            print(f"  Confiança media:     {conf:.1f}%")


class ClienteFederadoClassificacao(ABC):
    """Classe abstrata para cliente federado (classificação)"""
    
    def __init__(self, id_cliente: str, dados: pd.DataFrame, target_col: str,
                 max_rounds_locais: int = 10, criterio_convergencia_local: float = 0.01):
        self.id_cliente = id_cliente
        self.dados = dados
        self.target_col = target_col
        self.modelo_local = ModeloClassificacao()
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.metricas_avaliacao = {}
        
        # Configuração de rounds locais
        self.max_rounds_locais = max_rounds_locais
        self.criterio_convergencia_local = criterio_convergencia_local
        self.historico_rounds_locais = []
    
    def set_modelo_local(self, pesos_globais):
        """Atualiza modelo local com pesos do servidor"""
        self.modelo_local.atualizar_pesos(pesos_globais)
    
    def get_metricas_avaliacao(self):
        """Retorna métricas de avaliação"""
        return self.metricas_avaliacao
    
    @abstractmethod
    def treinar_modelo(self):
        """Método abstrato para treinamento (deve ser implementado)"""
        pass


class ClienteMaliciosoClassificacao(ClienteFederadoClassificacao):
    """Cliente que pode simular ataques de envenenamento"""
    
    def __init__(self, id_cliente: str, dados: pd.DataFrame, target_col: str, 
                 tipo_ataque: str = "nenhum", max_rounds_locais: int = 10):
        super().__init__(id_cliente, dados, target_col, max_rounds_locais)
        self.tipo_ataque = tipo_ataque
        self.dados_originais = dados.copy()
    
    def treinar_modelo(self):
        """Treina modelo local com MÚLTIPLOS ROUNDS (épocas)"""
        # APLICAR ENVENENAMENTO DE DADOS ANTES (se configurado)
        if self.tipo_ataque != "nenhum" and "dados" in self.tipo_ataque:
            self._envenenar_dados()
        
        # Preparar dados
        X = self.dados.drop(columns=[self.target_col]).values
        y_labels = self.dados[self.target_col].values
        
        # Codificar labels
        y = self.label_encoder.fit_transform(y_labels)
        
        # Normalizar features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split treino/validação local (80/20)
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # TREINAMENTO COM MÚLTIPLOS ROUNDS LOCAIS
        self.historico_rounds_locais = []
        acuracia_anterior = 0
        
        for round_local in range(1, self.max_rounds_locais + 1):
            # Treinar modelo
            self.modelo_local.fit(X_train, y_train)
            
            # Avaliar no conjunto de validação local
            y_pred = self.modelo_local.predict(X_val)
            acuracia_local = accuracy_score(y_val, y_pred)
            
            # Registrar histórico
            self.historico_rounds_locais.append({
                'round': round_local,
                'acuracia': acuracia_local
            })
            
            # Verificar convergência local (early stop)
            if round_local > 1:
                melhoria = acuracia_local - acuracia_anterior
                if abs(melhoria) < self.criterio_convergencia_local:
                    # print(f"    [{self.id_cliente}] Convergiu localmente no round {round_local}")
                    break
            
            acuracia_anterior = acuracia_local
        
        # Avaliar métricas finais
        y_pred_final = self.modelo_local.predict(X_val)
        y_proba_final = self.modelo_local.predict_proba(X_val)
        
        self.metricas_avaliacao = {
            'acuracia': accuracy_score(y_val, y_pred_final),
            'f1': f1_score(y_val, y_pred_final, average='weighted'),
            'rounds_executados': len(self.historico_rounds_locais)
        }
        
        # APLICAR ATAQUES (se configurado)
        if self.tipo_ataque != "nenhum":
            self._aplicar_ataque()
        
        return self.historico_rounds_locais
    
    def _aplicar_ataque(self):
        """Aplica ataques de envenenamento"""
        if "modelo" in self.tipo_ataque:
            self._envenenar_modelo()
    
    def _envenenar_dados(self):
        """Envenena dados: adiciona ruído gaussiano a 30% das amostras"""
        self.dados = self.dados_originais.copy().reset_index(drop=True)
        
        cols = [c for c in self.dados.columns if c != self.target_col]
        n_envenenadas = int(len(self.dados) * 0.3)
        indices = np.random.choice(len(self.dados), n_envenenadas, replace=False)
        
        for col in cols:
            ruido = np.random.normal(0, self.dados[col].std() * 3, n_envenenadas)
            self.dados.loc[indices, col] += ruido
    
    def _envenenar_modelo(self):
        """Envenena modelo: inverte ou randomiza coeficientes"""
        pesos = self.modelo_local.obter_pesos()
        
        if "invertidos" in self.tipo_ataque:
            pesos['coef'] = -pesos['coef']
        elif "randomizados" in self.tipo_ataque:
            pesos['coef'] = np.random.randn(*pesos['coef'].shape) * 0.5
        
        self.modelo_local.atualizar_pesos(pesos)


def carregar_dataset_iris():
    """Carrega dataset Iris"""
    caminho = "../data/iris/iris.csv"
    
    if not os.path.exists(caminho):
        print(f"[ERRO] Dataset não encontrado: {caminho}")
        return None
    
    dados = pd.read_csv(caminho)
    print(f"[INFO] Dataset carregado: {dados.shape[0]} amostras, {dados.shape[1]} colunas")
    print(f"[INFO] Classes: {dados['species'].value_counts().to_dict()}")
    
    return dados


def dividir_dados_clientes(dados, num_clientes=4):
    """Divide dados entre clientes (IID)"""
    dados_embaralhados = dados.sample(frac=1, random_state=42).reset_index(drop=True)
    chunk_size = len(dados_embaralhados) // num_clientes
    chunks = []
    
    for i in range(num_clientes):
        inicio = i * chunk_size
        fim = inicio + chunk_size if i < num_clientes - 1 else len(dados_embaralhados)
        chunks.append(dados_embaralhados.iloc[inicio:fim].copy())
        print(f"[INFO] Cliente {i+1}: {len(chunks[i])} amostras")
    
    return chunks


def main():
    """Função principal"""
    print("\n" + "="*70)
    print("INICIALIZANDO SISTEMA DE APRENDIZADO FEDERADO")
    print("="*70)
    
    # 1. Carregar dataset
    print("\n[PASSO 1] Carregando dataset Iris...")
    dados = carregar_dataset_iris()
    if dados is None:
        return
    
    # 2. Preparar conjunto de validação global (20% dos dados)
    print("\n[PASSO 2] Separando conjunto de validação global...")
    dados_treino, dados_validacao = train_test_split(
        dados, test_size=0.2, random_state=42, stratify=dados['species']
    )
    
    X_val = dados_validacao.drop(columns=['species']).values
    label_encoder = LabelEncoder()
    y_val = label_encoder.fit_transform(dados_validacao['species'].values)
    
    print(f"  Treino: {len(dados_treino)} amostras")
    print(f"  Validação: {len(dados_validacao)} amostras")
    
    # 3. Dividir dados de treino entre clientes
    print("\n[PASSO 3] Dividindo dados entre clientes...")
    num_clientes = 4
    chunks = dividir_dados_clientes(dados_treino, num_clientes)
    
    # 4. Criar servidor federado
    print("\n[PASSO 4] Criando servidor federado...")
    servidor = ServidorFederadoClassificacao(
        max_rodadas=50,  # Aumentado para 50 rodadas (convergência longa)
        criterio_convergencia=0.001,  # Muito rigoroso (0.1% variação)
        dados_validacao=(X_val, y_val),
        threshold_acuracia=0.96,  # 96% (mais alcançável que 97%)
        inicializar_aleatorio=True  # Força início com modelo ruim
    )
    
    # 5. Criar clientes (3 honestos + 1 atacante)
    print("\n[PASSO 5] Criando clientes...")
    
    # Cliente 1 - Honesto (25 rounds locais)
    servidor.adicionar_cliente(
        ClienteMaliciosoClassificacao(
            "Cliente_1_Honesto", chunks[0], "species", "nenhum", max_rounds_locais=25
        )
    )
    
    # Cliente 2 - Honesto (25 rounds locais)
    servidor.adicionar_cliente(
        ClienteMaliciosoClassificacao(
            "Cliente_2_Honesto", chunks[1], "species", "nenhum", max_rounds_locais=25
        )
    )
    
    # Cliente 3 - ATACANTE (data poisoning com 25 rounds)
    servidor.adicionar_cliente(
        ClienteMaliciosoClassificacao(
            "Cliente_3_MALICIOSO", chunks[2], "species", "dados", max_rounds_locais=25
        )
    )
    
    # Cliente 4 - Honesto (25 rounds locais)
    servidor.adicionar_cliente(
        ClienteMaliciosoClassificacao(
            "Cliente_4_Honesto", chunks[3], "species", "nenhum", max_rounds_locais=25
        )
    )
    
    print(f"  Total de clientes: {len(servidor.clientes)}")
    print(f"  Clientes honestos: 3")
    print(f"  Clientes maliciosos: 1 (data poisoning)")
    
    # 6. Executar aprendizado federado
    print("\n[PASSO 6] Iniciando aprendizado federado...")
    servidor.executar_aprendizado_federado()
    
    print("\n" + "="*70)
    print("SISTEMA FINALIZADO COM SUCESSO")
    print("="*70)


if __name__ == "__main__":
    main()
