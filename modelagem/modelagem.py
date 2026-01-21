"""
Sistema Simplificado de Aprendizado Federado com Ataques de Envenenamento
Modelo: LinearRegression
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

class ServidorFederado:
    """Servidor central do aprendizado federado"""
    
    def __init__(self, max_rodadas: int = 5):
        self.rodada_atual = 0
        self.max_rodadas = max_rodadas
        self.clientes = []
        self.modelo_global = LinearRegression()
        self.outliers_detectados = []  # Histórico de detecções
    
    def adicionar_cliente(self, cliente):
        """Adiciona um cliente"""
        self.clientes.append(cliente)
    
    def treinar(self):
        """Executa o aprendizado federado"""
        print("Iniciando Aprendizado Federado\n")
        
        for rodada in range(1, self.max_rodadas + 1):
            self.rodada_atual = rodada
            print(f"\n{'='*50}")
            print(f"Rodada {rodada}/{self.max_rodadas}")
            print(f"{'='*50}")
            
            # Cada cliente treina localmente
            r2_scores = []
            for cliente in self.clientes:
                cliente.modelo_local = LinearRegression()
                cliente.treinar()
                r2_scores.append(cliente.r2)
            
            # Agrega modelos (média dos coeficientes)
            self._agregar_modelos()
            
            # Mostra resultados
            print(f"\nR2 medio da rodada: {np.mean(r2_scores):.4f}")
        
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
            if hasattr(cliente.modelo_local, 'coef_'):
                todos_coefs.append(cliente.modelo_local.coef_)
        
        if not todos_coefs:
            return
        
        # Calcula mediana e MAD (Median Absolute Deviation)
        mediana_coefs = np.median(todos_coefs, axis=0)
        
        # Para cada cliente, calcula distância da mediana
        for i, cliente in enumerate(self.clientes):
            if hasattr(cliente.modelo_local, 'coef_'):
                # Distância euclidiana dos coeficientes para a mediana
                distancia = np.linalg.norm(cliente.modelo_local.coef_ - mediana_coefs)
                
                # Calcula threshold baseado em MAD
                desvios = [np.linalg.norm(c - mediana_coefs) for c in todos_coefs]
                mad = np.median(np.abs(desvios - np.median(desvios)))
                threshold = np.median(desvios) + 3 * mad  # 3 MADs = outlier
                
                # Detecta outlier
                if distancia > threshold:
                    clientes_rejeitados.append(cliente.nome)
                    print(f"  [OUTLIER DETECTADO] {cliente.nome} - Distancia: {distancia:.4f} > Threshold: {threshold:.4f}")
                else:
                    coefs.append(cliente.modelo_local.coef_)
                    intercepts.append(cliente.modelo_local.intercept_)
                    clientes_aceitos.append(cliente.nome)
        
        # Agrega apenas clientes não-outliers
        if coefs:
            self.modelo_global.coef_ = np.mean(coefs, axis=0)
            self.modelo_global.intercept_ = np.mean(intercepts)
            print(f"\n  Clientes aceitos na agregacao: {clientes_aceitos}")
            if clientes_rejeitados:
                print(f"  Clientes rejeitados (outliers): {clientes_rejeitados}")
                self.outliers_detectados.append({
                    'rodada': self.rodada_atual,
                    'clientes': clientes_rejeitados
                })
        else:
            print(f"\n  [ALERTA] Todos os clientes foram detectados como outliers!")


class ClienteMalicioso:
    """Cliente malicioso com ataques de envenenamento"""
    
    def __init__(self, nome: str, dados: pd.DataFrame, target_col: str, tipo_ataque: str):
        self.nome = nome
        self.dados = dados.copy()
        self.dados_originais = dados.copy()
        self.target_col = target_col
        self.tipo_ataque = tipo_ataque
        self.modelo_local = None
        self.scaler = StandardScaler()
        self.r2 = 0
    
    def treinar(self):
        """Treina o modelo com ataque"""
        tipo_display = self.tipo_ataque if self.tipo_ataque != "nenhum" else "SEM ATAQUE"
        print(f"\n{self.nome} - Ataque: {tipo_display}")
        
        # Aplica ataque nos dados
        if "dados" in self.tipo_ataque:
            self._envenenar_dados()
        
        # Prepara dados
        X = self.dados.drop(columns=[self.target_col]).values
        y = self.dados[self.target_col].values
        X_scaled = self.scaler.fit_transform(X)
        
        # Treina modelo
        self.modelo_local.fit(X_scaled, y)
        
        # Avalia
        y_pred = self.modelo_local.predict(X_scaled)
        self.r2 = r2_score(y, y_pred)
        
        # Aplica ataque no modelo
        if "modelo" in self.tipo_ataque:
            self._envenenar_modelo()
        
        print(f"  R2: {self.r2:.4f}")
    
    def _envenenar_dados(self):
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
    
    def _envenenar_modelo(self):
        """Envenenamento do modelo (pesos)"""
        if not hasattr(self.modelo_local, 'coef_'):
            return
        
        if "invertidos" in self.tipo_ataque:
            # Inverte os coeficientes
            self.modelo_local.coef_ = -self.modelo_local.coef_
            print(f"  Coeficientes invertidos")
        
        elif "randomizados" in self.tipo_ataque:
            # Randomiza os coeficientes
            self.modelo_local.coef_ = np.random.randn(*self.modelo_local.coef_.shape)
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
    print("  - 2 clientes maliciosos (com ataques)")
    print("  - 1 cliente honesto (sem ataques)")
    
    # Executa treinamento
    servidor.treinar()