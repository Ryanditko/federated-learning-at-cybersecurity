"""
Sistema de Aprendizado Federado com Ataques de Envenenamento
Modelo: LinearRegression
Dataset: Iris (project/data/iris/iris.csv)
Objetivo: Predizer petal width baseado em sepal length, sepal width, petal length
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from copy import deepcopy
from abc import ABC, abstractmethod
from typing import Dict, List
import warnings
import os
warnings.filterwarnings('ignore')

class Modelo:
    """Classe que encapsula o modelo de machine learning"""
    
    def __init__(self):
        self._modelo_interno = LinearRegression()
    
    def obter_pesos(self):
        """Retorna os pesos do modelo (coeficientes e intercept)"""
        if hasattr(self._modelo_interno, 'coef_'):
            return {
                'coef': deepcopy(self._modelo_interno.coef_),
                'intercept': deepcopy(self._modelo_interno.intercept_)
            }
        return None
    
    def atualizar_pesos(self, pesos):
        """Atualiza os pesos do modelo"""
        if pesos and 'coef' in pesos and 'intercept' in pesos:
            self._modelo_interno.coef_ = deepcopy(pesos['coef'])
            self._modelo_interno.intercept_ = deepcopy(pesos['intercept'])
    
    def fit(self, X, y):
        """Treina o modelo"""
        return self._modelo_interno.fit(X, y)
    
    def predict(self, X):
        """Faz predições"""
        return self._modelo_interno.predict(X)
    
    def get_coef(self):
        """Retorna os coeficientes"""
        return self._modelo_interno.coef_ if hasattr(self._modelo_interno, 'coef_') else None
    
    def get_intercept(self):
        """Retorna o intercept"""
        return self._modelo_interno.intercept_ if hasattr(self._modelo_interno, 'intercept_') else None


class ServidorFederado:
    """Servidor central do aprendizado federado"""
    
    def __init__(self, max_rodadas: int = 5, criterio_convergencia: float = 0.01, dados_validacao=None):
        self.rodada_atual = 0
        self.max_rodadas = max_rodadas
        self.criterio_convergencia = criterio_convergencia
        self.clientes = []
        self.modelo_global = Modelo()
        self.metricas_avaliacao = {}
        self.historico_metricas = []
        self.outliers_detectados = []
        self.desempenho_anterior = None
        self.dados_validacao = dados_validacao
        self.historico_r2_global = []
        self.historico_mse_global = []
        self.historico_mae_global = []
    
    def adicionar_cliente(self, cliente):
        """Adiciona um cliente"""
        self.clientes.append(cliente)
    
    def compartilhar_modelo_global(self):
        """Distribui pesos do modelo global para todos os clientes"""
        pesos_globais = self.modelo_global.obter_pesos()
        if pesos_globais:
            for cliente in self.clientes:
                cliente.set_modelo_local(pesos_globais)
    
    def avaliar_modelo(self):
        """Avalia o desempenho do modelo global"""
        if not self.clientes:
            return
        
        r2_scores = []
        for cliente in self.clientes:
            metricas = cliente.get_metricas_avaliacao()
            if metricas:
                r2_scores.append(metricas.get('r2', 0))
        
        # Avalia no conjunto de validação global apenas se o modelo foi treinado
        if self.dados_validacao is not None and self.modelo_global.obter_pesos() is not None:
            X_val, y_val = self.dados_validacao
            scaler = StandardScaler()
            X_val_scaled = scaler.fit_transform(X_val)
            
            y_pred = self.modelo_global.predict(X_val_scaled)
            r2_global = r2_score(y_val, y_pred)
            mse_global = mean_squared_error(y_val, y_pred)
            mae_global = mean_absolute_error(y_val, y_pred)
            
            self.historico_r2_global.append(r2_global)
            self.historico_mse_global.append(mse_global)
            self.historico_mae_global.append(mae_global)
        else:
            r2_global = np.mean(r2_scores) if r2_scores else 0
            mse_global = 0
            mae_global = 0
        
        if r2_scores:
            self.metricas_avaliacao = {
                'rodada': self.rodada_atual,
                'r2_medio_clientes': np.mean(r2_scores),
                'r2_global': r2_global,
                'mse_global': mse_global,
                'mae_global': mae_global,
                'num_clientes': len(self.clientes)
            }
            self.historico_metricas.append(self.metricas_avaliacao.copy())
    
    def set_modelo_global(self):
        """Atualiza o modelo global com agregação dos modelos locais"""
        self._agregar_modelos()
    
    def avaliar_convergencia(self) -> bool:
        """Verifica se o treinamento convergiu"""
        # Critério 1: Atingiu número máximo de rodadas
        if self.rodada_atual >= self.max_rodadas:
            return True
        
        # Critério 2: Estabilidade do modelo (early stop)
        if self.desempenho_anterior is not None and self.metricas_avaliacao:
            desempenho_atual = self.metricas_avaliacao.get('r2_medio', 0)
            variacao = abs(desempenho_atual - self.desempenho_anterior)
            
            if variacao < self.criterio_convergencia:
                print(f"   Convergencia antecipada: variacao={variacao:.4f} < {self.criterio_convergencia}")
                return True
            
            self.desempenho_anterior = desempenho_atual
        elif self.metricas_avaliacao:
            self.desempenho_anterior = self.metricas_avaliacao.get('r2_medio', 0)
        
        return False
    
    def executar_aprendizado_federado(self):
        """Executa o loop principal do aprendizado federado"""
        print("Iniciando Aprendizado Federado\n")
        
        while not self.avaliar_convergencia():
            self.rodada_atual += 1
            print(f"\n{'='*50}")
            print(f"Rodada {self.rodada_atual}/{self.max_rodadas}")
            print(f"{'='*50}")
            
            # 1. Compartilhar modelo global
            self.compartilhar_modelo_global()
            
            # 2. Cada cliente treina localmente
            print("\nTreinamento Local:")
            for cliente in self.clientes:
                cliente.modelo_local = Modelo()
                cliente.treinar_modelo()
            
            # 3. Avaliar o modelo global
            print("\nAvaliacao do Modelo:")
            self.avaliar_modelo()
            
            # 4. Atualizar modelo global com agregação
            print("\nAgregacao de Modelos")
            self.set_modelo_global()
            
            # 5. Mostrar métricas
            if self.metricas_avaliacao:
                r2_global = self.metricas_avaliacao.get('r2_global', 0)
                mse_global = self.metricas_avaliacao.get('mse_global', 0)
                mae_global = self.metricas_avaliacao.get('mae_global', 0)
                print(f"\nModelo Global Atualizado:")
                print(f"  R2: {r2_global:.4f} | MSE: {mse_global:.4f} | MAE: {mae_global:.4f}\n")
        
        print(f"\n{'='*50}")
        print("Aprendizado Federado Concluido")
        print(f"{'='*50}")
        
        # Gerar visualizações
        self.gerar_graficos()
        
        # Relatório de detecção de outliers
        if self.outliers_detectados:
            print("\nRelatorio de Deteccao de Outliers:")
            print("-" * 50)
            for deteccao in self.outliers_detectados:
                rodada = deteccao['rodada']
                clientes = ', '.join(deteccao['clientes'])
                print(f"  Rodada {rodada}: {clientes}")
            
            # Estatísticas
            total_deteccoes = sum(len(d['clientes']) for d in self.outliers_detectados)
            print(f"\nTotal de deteccoes: {total_deteccoes}")
            print(f"Rodadas com deteccao: {len(self.outliers_detectados)}/{self.max_rodadas}")
        else:
            print("\nNenhum outlier detectado durante o treinamento.")
    
    def _agregar_modelos(self):
        """Agrega modelos usando FedAvg com detecção de outliers"""
        coefs = []
        intercepts = []
        clientes_aceitos = []
        clientes_rejeitados = []
        
        # Coleta todos os coeficientes primeiro
        todos_coefs = []
        for cliente in self.clientes:
            coef = cliente.modelo_local.get_coef()
            if coef is not None:
                todos_coefs.append(coef)
        
        if not todos_coefs:
            return
        
        # Calcula mediana e MAD (Median Absolute Deviation)
        mediana_coefs = np.median(todos_coefs, axis=0)
        
        # Para cada cliente, calcula distância da mediana
        for i, cliente in enumerate(self.clientes):
            coef = cliente.modelo_local.get_coef()
            if coef is not None:
                # Distância euclidiana dos coeficientes para a mediana
                distancia = np.linalg.norm(coef - mediana_coefs)
                
                # Calcula threshold baseado em MAD
                desvios = [np.linalg.norm(c - mediana_coefs) for c in todos_coefs]
                mad = np.median(np.abs(desvios - np.median(desvios)))
                threshold = np.median(desvios) + 3 * mad  # 3 MADs = outlier    
                
                # Detecta outlier
                if distancia > threshold:
                    clientes_rejeitados.append(cliente.id_cliente)
                    print(f"  [OUTLIER DETECTADO] {cliente.id_cliente} - Distancia: {distancia:.4f} > Threshold: {threshold:.4f}")
                else:
                    coefs.append(cliente.modelo_local.get_coef())
                    intercepts.append(cliente.modelo_local.get_intercept())
                    clientes_aceitos.append(cliente.id_cliente)
        
        # Agrega apenas clientes não-outliers
        if coefs:
            pesos_agregados = {
                'coef': np.mean(coefs, axis=0),
                'intercept': np.mean(intercepts)
            }
            self.modelo_global.atualizar_pesos(pesos_agregados)
            print(f"\n  Clientes aceitos na agregacao: {clientes_aceitos}")
            if clientes_rejeitados:
                print(f"  Clientes rejeitados (outliers): {clientes_rejeitados}")
                self.outliers_detectados.append({
                    'rodada': self.rodada_atual,
                    'clientes': clientes_rejeitados
                })
        else:
            print(f"\n  [ALERTA] Todos os clientes foram detectados como outliers!")
    
    def gerar_graficos(self):
        """Gera gráficos de evolução do modelo global"""
        if not self.historico_metricas:
            print("\nNenhuma metrica para visualizar.")
            return
        
        # Garante que todos os históricos tenham o mesmo tamanho
        n_rodadas = len(self.historico_metricas)
        rodadas = [m['rodada'] for m in self.historico_metricas]
        
        # Se não há histórico global, preenche com zeros
        if len(self.historico_r2_global) < n_rodadas:
            self.historico_r2_global.extend([0] * (n_rodadas - len(self.historico_r2_global)))
        if len(self.historico_mse_global) < n_rodadas:
            self.historico_mse_global.extend([0] * (n_rodadas - len(self.historico_mse_global)))
        if len(self.historico_mae_global) < n_rodadas:
            self.historico_mae_global.extend([0] * (n_rodadas - len(self.historico_mae_global)))
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Evolucao do Modelo Global - Aprendizado Federado (Iris Dataset)', fontsize=16, fontweight='bold')
        
        # 1. R² Score ao longo das rodadas
        if self.historico_r2_global and len(self.historico_r2_global) == len(rodadas):
            axes[0, 0].plot(rodadas, self.historico_r2_global[:n_rodadas], marker='o', linewidth=2, 
                           color='#2ecc71', markersize=8, label='R² Global')
            axes[0, 0].set_xlabel('Rodada', fontsize=11)
            axes[0, 0].set_ylabel('R² Score', fontsize=11)
            axes[0, 0].set_title('R² Score por Rodada', fontsize=12, fontweight='bold')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].legend()
            axes[0, 0].set_ylim([0, 1])
            
            # Adicionar anotações
            for i, (x, y) in enumerate(zip(rodadas, self.historico_r2_global[:n_rodadas])):
                axes[0, 0].annotate(f'{y:.3f}', (x, y), textcoords="offset points", 
                                   xytext=(0,10), ha='center', fontsize=9)
        
        # 2. MSE (Mean Squared Error) ao longo das rodadas
        if self.historico_mse_global and len(self.historico_mse_global) == len(rodadas):
            axes[0, 1].plot(rodadas, self.historico_mse_global[:n_rodadas], marker='s', linewidth=2, 
                           color='#e74c3c', markersize=8, label='MSE Global')
            axes[0, 1].set_xlabel('Rodada', fontsize=11)
            axes[0, 1].set_ylabel('MSE', fontsize=11)
            axes[0, 1].set_title('Mean Squared Error por Rodada', fontsize=12, fontweight='bold')
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].legend()
        
        # 3. MAE (Mean Absolute Error) ao longo das rodadas
        if self.historico_mae_global and len(self.historico_mae_global) == len(rodadas):
            axes[1, 0].plot(rodadas, self.historico_mae_global[:n_rodadas], marker='^', linewidth=2, 
                           color='#3498db', markersize=8, label='MAE Global')
            axes[1, 0].set_xlabel('Rodada', fontsize=11)
            axes[1, 0].set_ylabel('MAE', fontsize=11)
            axes[1, 0].set_title('Mean Absolute Error por Rodada', fontsize=12, fontweight='bold')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].legend()
        
        # 4. Número de clientes aceitos/rejeitados por rodada
        clientes_aceitos = []
        clientes_rejeitados = []
        
        for rodada_num in rodadas:
            # Conta clientes rejeitados nesta rodada
            rejeitados = 0
            for deteccao in self.outliers_detectados:
                if deteccao['rodada'] == rodada_num:
                    rejeitados = len(deteccao['clientes'])
                    break
            
            total_clientes = len(self.clientes)
            aceitos = total_clientes - rejeitados
            
            clientes_aceitos.append(aceitos)
            clientes_rejeitados.append(rejeitados)
        
        x_pos = np.arange(len(rodadas))
        axes[1, 1].bar(x_pos, clientes_aceitos, label='Clientes Aceitos', color='#2ecc71', alpha=0.8)
        axes[1, 1].bar(x_pos, clientes_rejeitados, bottom=clientes_aceitos, 
                      label='Outliers Detectados', color='#e74c3c', alpha=0.8)
        
        axes[1, 1].set_xlabel('Rodada', fontsize=11)
        axes[1, 1].set_ylabel('Número de Clientes', fontsize=11)
        axes[1, 1].set_title('Clientes Aceitos vs Outliers por Rodada', fontsize=12, fontweight='bold')
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels(rodadas)
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        # Salvar gráfico
        output_path = 'c:/Users/Administrador/Faculdade-Impacta/Iniciação-cientifica/project/modelagem/resultados_fl.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nGraficos salvos em: {output_path}")
        plt.show()
     
    def gerar_relatorio_estatistico(self):
        """Gera relatório estatístico completo"""
        if not self.historico_metricas:
            return
        
        print(f"\n{'='*70}")
        print("RELATORIO ESTATISTICO DO APRENDIZADO FEDERADO")
        print(f"{'='*70}")
        
        # Estatísticas de R²
        if self.historico_r2_global:
            r2_valores = self.historico_r2_global
            print(f"\nR² Score:")
            print(f"  Inicial: {r2_valores[0]:.4f}")
            print(f"  Final: {r2_valores[-1]:.4f}")
            print(f"  Melhoria: {(r2_valores[-1] - r2_valores[0]):.4f}")
            print(f"  Media: {np.mean(r2_valores):.4f}")
            print(f"  Desvio Padrao: {np.std(r2_valores):.4f}")
            print(f"  Maximo: {np.max(r2_valores):.4f}")
            print(f"  Minimo: {np.min(r2_valores):.4f}")
        
        # Estatísticas de MSE
        if self.historico_mse_global:
            mse_valores = self.historico_mse_global
            print(f"\nMean Squared Error:")
            print(f"  Inicial: {mse_valores[0]:.4f}")
            print(f"  Final: {mse_valores[-1]:.4f}")
            print(f"  Reducao: {(mse_valores[0] - mse_valores[-1]):.4f}")
            print(f"  Media: {np.mean(mse_valores):.4f}")
            print(f"  Desvio Padrao: {np.std(mse_valores):.4f}")
        
        # Estatísticas de MAE
        if self.historico_mae_global:
            mae_valores = self.historico_mae_global
            print(f"\nMean Absolute Error:")
            print(f"  Inicial: {mae_valores[0]:.4f}")
            print(f"  Final: {mae_valores[-1]:.4f}")
            print(f"  Reducao: {(mae_valores[0] - mae_valores[-1]):.4f}")
            print(f"  Media: {np.mean(mae_valores):.4f}")
            print(f"  Desvio Padrao: {np.std(mae_valores):.4f}")
        
        # Informações de detecção
        print(f"\nDeteccao de Outliers:")
        print(f"  Total de rodadas: {len(self.historico_metricas)}")
        print(f"  Rodadas com deteccao: {len(self.outliers_detectados)}")
        if self.outliers_detectados:
            total_outliers = sum(len(d['clientes']) for d in self.outliers_detectados)
            print(f"  Total de clientes rejeitados: {total_outliers}")
            print(f"  Taxa de rejeicao: {(total_outliers / (len(self.clientes) * len(self.historico_metricas))):.2%}")
        
        print(f"\n{'='*70}\n")


class ClienteFederado(ABC):
    """Classe abstrata base para clientes federados"""
    
    def __init__(self, id_cliente: str, dados: pd.DataFrame, target_col: str):
        self.id_cliente = id_cliente
        self.dados = dados.copy()
        self.dados_originais = dados.copy()
        self.target_col = target_col
        self.modelo_local = None
        self.metricas_avaliacao = {}
        self.scaler = StandardScaler()
    
    @abstractmethod
    def treinar_modelo(self):
        """Treina o modelo local"""
        pass
    
    def get_modelo_local(self) -> Modelo:
        """Retorna o modelo local treinado"""
        return self.modelo_local
    
    def get_metricas_avaliacao(self) -> Dict:
        """Retorna as métricas de avaliação do treinamento"""
        return self.metricas_avaliacao
    
    def avaliar_modelo(self):
        """Avalia o modelo local"""
        if self.metricas_avaliacao:
            print(f"    Avaliando {self.id_cliente}: R2={self.metricas_avaliacao.get('r2', 0):.4f}")
    
    def set_modelo_local(self, pesos_globais):
        """Atualiza modelo local com pesos globais"""
        if self.modelo_local and pesos_globais:
            self.modelo_local.atualizar_pesos(pesos_globais)
    
    def compartilhar_modelo_local(self):
        """Compartilha o modelo local"""
        print(f"    {self.id_cliente}: Compartilhando modelo local")
    
    def obter_pesos(self) -> Dict:
        """Retorna os pesos do modelo local"""
        if self.modelo_local:
            return self.modelo_local.obter_pesos()
        return None


class ClienteMalicioso(ClienteFederado):
    """Cliente malicioso com ataques de envenenamento"""
    
    def __init__(self, id_cliente: str, dados: pd.DataFrame, target_col: str, tipo_ataque: str):
        super().__init__(id_cliente, dados, target_col)
        self.tipo_ataque = tipo_ataque
    
    def treinar_modelo(self):
        """Treina o modelo com ataque"""
        # Determina se é honesto ou malicioso
        if self.tipo_ataque == "nenhum":
            print(f"  {self.id_cliente}: Treinamento honesto")
        else:
            print(f"  {self.id_cliente} - Ataque: {self.tipo_ataque}")
        
        # Aplica ataque nos dados (se aplicável)
        if "dados" in self.tipo_ataque:
            self.envenenar_dados()
        
        # Prepara dados
        X = self.dados.drop(columns=[self.target_col]).values
        y = self.dados[self.target_col].values
        X_scaled = self.scaler.fit_transform(X)
        
        # Treina modelo
        if self.modelo_local:
            self.modelo_local.fit(X_scaled, y)
            
            # Avalia
            y_pred = self.modelo_local.predict(X_scaled)
            r2 = r2_score(y, y_pred)
            
            # Aplica ataque no modelo (se aplicável)
            if "modelo" in self.tipo_ataque:
                self.envenenar_modelo()
            
            # Define métricas
            tipo_cliente = 'honesto' if self.tipo_ataque == "nenhum" else 'malicioso'
            self.metricas_avaliacao = {
                'r2': r2,
                'tipo': tipo_cliente,
                'ataque': self.tipo_ataque
            }
            
            print(f"  R2: {r2:.4f}")
    
    def envenenar_dados(self):
        """Envenenamento de dados (DataFrame)"""
        self.dados = self.dados_originais.copy()
        
        # Seleciona colunas numericas (exceto target)
        cols = [c for c in self.dados.columns if c != self.target_col]
        
        # Adiciona ruido em 30% dos dados
        n_envenenadas = int(len(self.dados) * 0.3)
        indices = np.random.choice(len(self.dados), n_envenenadas, replace=False)
        
        for col in cols:
            ruido = np.random.normal(0, self.dados[col].std() * 3, n_envenenadas)
            self.dados.loc[indices, col] += ruido
        
        print(f"  Dados envenenados: {n_envenenadas}/{len(self.dados)}")
    
    def envenenar_modelo(self):
        """Envenenamento do modelo (pesos)"""
        pesos = self.modelo_local.obter_pesos()
        if not pesos:
            return
        
        if "invertidos" in self.tipo_ataque:
            # Inverte os coeficientes
            pesos['coef'] = -pesos['coef']
            self.modelo_local.atualizar_pesos(pesos)
            print(f"  Coeficientes invertidos")
        
        elif "randomizados" in self.tipo_ataque:
            # Randomiza os coeficientes
            pesos['coef'] = np.random.randn(*pesos['coef'].shape)
            self.modelo_local.atualizar_pesos(pesos)
            print(f"  Coeficientes randomizados")


def carregar_dataset_iris():
    """Carrega e prepara o dataset Iris do CSV"""
    print("\nCarregando Iris Dataset...")
    
    # Caminho relativo ao arquivo
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'iris', 'iris.csv')
    
    # Carrega o CSV
    df = pd.read_csv(csv_path)
    
    # Features: sepal length, sepal width, petal length
    # Target: petal width (regressão)
    X = df[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)']]
    y = df['petal width (cm)']
    
    print(f"  Amostras: {len(X)}")
    print(f"  Features: {X.shape[1]} (sepal length, sepal width, petal length)")
    print(f"  Target: petal width (regressao linear)")
    print(f"  Range do target: [{y.min():.2f}, {y.max():.2f}]")
    
    return X, y


def dividir_dados_clientes(X, y, n_clientes=3, validacao_size=0.2):
    """Divide dados entre clientes e conjunto de validação"""
    # Separar conjunto de validação global
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=validacao_size, random_state=42
    )
    
    # Dividir dados de treino entre clientes
    indices = np.arange(len(X_train))
    np.random.shuffle(indices)
    
    clientes_indices = np.array_split(indices, n_clientes)
    
    dados_clientes = []
    for i, idx in enumerate(clientes_indices):
        X_cliente = X_train.iloc[idx].reset_index(drop=True)
        y_cliente = y_train.iloc[idx].reset_index(drop=True)
        
        df_cliente = X_cliente.copy()
        df_cliente['target'] = y_cliente
        
        dados_clientes.append(df_cliente)
        print(f"  Cliente {i+1}: {len(df_cliente)} amostras")
    
    return dados_clientes, (X_val.values, y_val.values)


# Exemplo de uso
if __name__ == "__main__":
    print("="*70)
    print("SISTEMA DE APRENDIZADO FEDERADO")
    print("Modelo: LinearRegression")
    print("Dataset: Iris (petal width prediction)")
    print("Com Deteccao de Outliers (MAD)")
    print("="*70)
    
    # Carrega dataset Iris
    X, y = carregar_dataset_iris()
    
    # Divide dados entre clientes
    print("\nDividindo dados entre clientes...")
    dados_clientes, dados_validacao = dividir_dados_clientes(X, y, n_clientes=4)
    
    # Cria servidor com dados de validação
    servidor = ServidorFederado(max_rodadas=10, dados_validacao=dados_validacao)
    
    # Adiciona clientes com diferentes perfis
    print("\nConfigurando clientes:")
    servidor.adicionar_cliente(
        ClienteMalicioso("Cliente_1_Honesto", dados_clientes[0], "target", "nenhum")
    )
    servidor.adicionar_cliente(
        ClienteMalicioso("Cliente_2_Malicioso_Dados", dados_clientes[1], "target", "dados")
    )
    servidor.adicionar_cliente(
        ClienteMalicioso("Cliente_3_Honesto", dados_clientes[2], "target", "nenhum")
    )
    servidor.adicionar_cliente(
        ClienteMalicioso("Cliente_4_Malicioso_Modelo", dados_clientes[3], "target", "modelo_invertidos")
    )
    
    print(f"\nTotal de clientes: {len(servidor.clientes)}")
    print(f"  - 2 clientes honestos")
    print(f"  - 2 clientes maliciosos (1 ataque em dados, 1 ataque em modelo)")
    print(f"  - Criterio de convergencia: {servidor.criterio_convergencia}")
    print(f"  - Conjunto de validacao: {len(dados_validacao[0])} amostras")
    
    # Executa treinamento
    servidor.executar_aprendizado_federado()
    
    # Gera relatório estatístico
    servidor.gerar_relatorio_estatistico()