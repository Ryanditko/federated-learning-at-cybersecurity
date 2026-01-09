import pandas as pd
import numpy as np
from typing import List, Dict
from copy import deepcopy
from abc import ABC, abstractmethod


class Modelo:
    """Classe base que representa um modelo de machine learning"""
    def __init__(self):
        self.pesos: List = []


class ModeloLocal(Modelo):
    """Modelo local de um cliente"""
    pass


class ModeloGlobal(Modelo):
    """Modelo global do servidor"""
    pass


class ServidorFederado:
    """Servidor central do aprendizado federado"""
    
    def __init__(self, max_rodadas: int, criterio_convergencia: float):
        self.rodada_atual: int = 0
        self.max_rodadas: int = max_rodadas
        self.criterio_convergencia: float = criterio_convergencia
        self.lista_clientes: List["ClienteFederado"] = []
        self.modelo_global: ModeloGlobal = ModeloGlobal()
        self.metricas_avaliacao: Dict = {}
    
    def adicionar_cliente(self, cliente: "ClienteFederado") -> None:
        """Adiciona um cliente à lista"""
        self.lista_clientes.append(cliente)
    
    def avaliar_modelo(self) -> None:
        """Avalia o modelo global"""
        pass
    
    def comparar_modelos_global(self) -> bool:
        """Compara modelos para verificar convergência"""
        return False
    
    def set_modelo_global(self) -> None:
        """Atualiza o modelo global (agregação)"""
        pass
    
    def avaliar_convergencia(self) -> bool:
        """Verifica convergência"""
        return False


class ClienteFederado(ABC):
    """Classe abstrata base para clientes federados"""
    
    def __init__(self, id_cliente: str):
        self.id_cliente: str = id_cliente
        self.dados: pd.DataFrame = pd.DataFrame()
        self.modelo_local: ModeloLocal = ModeloLocal()
        self.metricas_avaliacao: Dict = {}
    
    @abstractmethod
    def treinar_modelo(self) -> None:
        """Treina o modelo local (método abstrato)"""
        pass
    
    def get_modelo_local(self) -> ModeloLocal:
        """Retorna o modelo local"""
        return self.modelo_local
    
    def get_metricas_avaliacao(self) -> Dict:
        """Retorna as métricas de avaliação"""
        return self.metricas_avaliacao
    
    def set_modelo_local(self, modelo_global: ModeloGlobal) -> None:
        """Atualiza modelo local com pesos globais"""
        self.modelo_local.pesos = deepcopy(modelo_global.pesos)
    
    def comparar_modelo_local(self) -> None:
        """Compara modelo local atual com anterior"""
        pass
    
    def other_pessoa(self) -> List:
        """Retorna lista de outras pessoas"""
        return []


class ClienteHonesto(ClienteFederado):
    """Cliente honesto - treina normalmente"""
    
    def treinar_modelo(self) -> None:
        """Treina o modelo de forma honesta"""
        print(f"🟢 {self.id_cliente}: Treinando...")
        # Simula treinamento bem-sucedido
        self.metricas_avaliacao = {
            'acuracia': np.random.uniform(0.85, 0.95),
            'perda': np.random.uniform(0.05, 0.15)
        }


class ClienteMalicioso(ClienteFederado):
    """Cliente malicioso - executa ataques"""
    
    def __init__(self, id_cliente: str, tipo_ataque: str):
        super().__init__(id_cliente)
        self.tipo_ataque: str = tipo_ataque
    
    def envenenar_dados(self) -> None:
        """Envenena os dados de treinamento"""
        print(f"   🔴 Envenenando dados...")
        # Inverte labels, adiciona ruído, etc.
        pass
    
    def envenenar_modelo(self) -> None:
        """Envenena o modelo (pesos)"""
        print(f"   🔴 Envenenando modelo...")
        # Adiciona ruído nos pesos
        pass
    
    def treinar_modelo(self) -> None:
        """Treina e aplica ataque"""
        print(f"🔴 {self.id_cliente}: Ataque '{self.tipo_ataque}'")
        
        if self.tipo_ataque == "dados":
            self.envenenar_dados()
        
        # Simula treinamento (degradado)
        self.metricas_avaliacao = {
            'acuracia': np.random.uniform(0.40, 0.60),
            'perda': np.random.uniform(0.30, 0.50)
        }
        
        if self.tipo_ataque == "modelo":
            self.envenenar_modelo()


# Exemplo de uso
if __name__ == "__main__":
    print("="*60)
    print("SISTEMA DE APRENDIZADO FEDERADO COM ATAQUES")
    print("="*60)
    
    # Criar servidor
    servidor = ServidorFederado(max_rodadas=3, criterio_convergencia=0.01)
    
    # Adicionar clientes
    servidor.adicionar_cliente(ClienteHonesto("cliente_1"))
    servidor.adicionar_cliente(ClienteHonesto("cliente_2"))
    servidor.adicionar_cliente(ClienteMalicioso("cliente_malicioso", "dados"))
    
    print(f"\n📊 Configuração:")
    print(f"  • Clientes: {len(servidor.lista_clientes)}")
    print(f"  • Rodadas: {servidor.max_rodadas}\n")
    
    # Simular rodadas
    for rodada in range(servidor.max_rodadas):
        print(f"--- Rodada {rodada + 1} ---")
        
        # Cada cliente treina
        for cliente in servidor.lista_clientes:
            cliente.treinar_modelo()
        
        servidor.rodada_atual += 1
        print()
    
    print("="*60)
    print("✓ Concluído!")
    print("="*60)

