"""
Sistema de Aprendizado Federado com Ataques de Envenenamento
Modelo: LinearRegression
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from copy import deepcopy
from abc import ABC, abstractmethod
from typing import Dict, List

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
    
    def __init__(self, max_rodadas: int = 5, criterio_convergencia: float = 0.01):
        self.rodada_atual = 0
        self.max_rodadas = max_rodadas
        self.criterio_convergencia = criterio_convergencia
        self.clientes = []
        self.modelo_global = Modelo()
        self.metricas_avaliacao = {}
        self.historico_metricas = []
        self.outliers_detectados = []
        self.desempenho_anterior = None
    
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
        
        if r2_scores:
            self.metricas_avaliacao = {
                'rodada': self.rodada_atual,
                'r2_medio': np.mean(r2_scores),
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
                r2_medio = self.metricas_avaliacao.get('r2_medio', 0)
                print(f"\nModelo Global Atualizado - R2 Medio: {r2_medio:.4f}\n")
        
        print(f"\n{'='*50}")
        print("Aprendizado Federado Concluido")
        print(f"{'='*50}")
        
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


# Exemplo de uso
if __name__ == "__main__":
    print("="*50)
    print("SISTEMA DE APRENDIZADO FEDERADO")
    print("Modelo: LinearRegression")
    print("Com Deteccao de Outliers")
    print("="*50)
    
    # Gera dados sinteticos
    np.random.seed(42)
    n_samples = 1000
    n_features = 4
    
    # Cliente 1: Ataque em dados
    dados1 = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'x{i}' for i in range(n_features)]
    )
    dados1['y'] = 2*dados1['x0'] + 3*dados1['x1'] - dados1['x2'] + np.random.randn(n_samples)*0.5
    
    # Cliente 2: Ataque em modelo
    dados2 = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'x{i}' for i in range(n_features)]
    )
    dados2['y'] = -dados2['x0'] + 2.5*dados2['x1'] + 0.8*dados2['x3'] + np.random.randn(n_samples)*0.5
    
    # Cliente 3: Honesto (sem ataques) - para testar detecção
    dados3 = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'x{i}' for i in range(n_features)]
    )
    dados3['y'] = 1.5*dados3['x0'] + 2*dados3['x1'] + 0.5*dados3['x2'] + np.random.randn(n_samples)*0.5
    
    # Cria servidor
    servidor = ServidorFederado(max_rodadas=5)
    
    # Adiciona clientes
    servidor.adicionar_cliente(
        ClienteMalicioso("Cliente_1_Malicioso", dados1, "y", "dados")
    )
    servidor.adicionar_cliente(
        ClienteMalicioso("Cliente_2_Malicioso", dados2, "y", "modelo_invertidos")
    )
    servidor.adicionar_cliente(
        ClienteMalicioso("Cliente_3_Honesto", dados3, "y", "nenhum")
    )
    
    print(f"\nTotal de clientes: {len(servidor.clientes)}")
    print(f"  - 2 clientes maliciosos (com ataques)")
    print(f"  - 1 cliente honesto (tipo_ataque='nenhum')")
    print(f"  - Criterio de convergencia: {servidor.criterio_convergencia}")
    
    # Executa treinamento
    servidor.executar_aprendizado_federado()