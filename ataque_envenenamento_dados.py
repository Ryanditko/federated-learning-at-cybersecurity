"""
Sistema de Aprendizado Federado com Detecção de Ataques de Envenenamento de Dados e Modelos.
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from copy import deepcopy
from abc import ABC, abstractmethod


class Modelo:
    """Classe base que representa um modelo de machine learning"""
    
    def __init__(self):
        # Atributo público conforme diagrama UML (sem underscore)
        self.pesos: List = []
    
    def obter_pesos(self) -> List:
        """Retorna os pesos do modelo"""
        return deepcopy(self.pesos)
    
    def atualizar_pesos(self, novos_pesos: List) -> None:
        """Atualiza os pesos do modelo"""
        self.pesos = deepcopy(novos_pesos)


class ServidorFederado:
    """Servidor central do aprendizado federado"""
    
    def __init__(self, max_rodadas: int, criterio_convergencia: float):
        # Atributos privados conforme diagrama UML
        self._rodada_atual: int = 0
        self._max_rodadas: int = max_rodadas
        self._criterio_convergencia: float = criterio_convergencia
        self._lista_clientes: List["ClienteFederado"] = []
        self._modelo_global: Modelo = Modelo()
        self._metricas_avaliacao: Dict = {}
        self._historico_metricas: List[Dict] = []
        # Atributo auxiliar para controle de convergência
        self._desempenho_anterior: Optional[float] = None
    
    def adicionar_cliente(self, cliente: "ClienteFederado") -> None:
        """Adiciona um cliente à lista de participantes (conforme diagrama UML)"""
        self._lista_clientes.append(cliente)
    
    def avaliar_modelo(self) -> None:
        """Avalia o desempenho do modelo global (conforme diagrama UML)"""
        if not self._lista_clientes:
            return
        
        # Coleta métricas de todos os clientes
        acuracias = []
        perdas = []
        
        for cliente in self._lista_clientes:
            metricas = cliente.get_metricas_avaliacao()
            if metricas:
                acuracias.append(metricas.get('acuracia', 0))
                perdas.append(metricas.get('perda', 0))
        
        # Calcula métricas médias
        if acuracias:
            self._metricas_avaliacao = {
                'rodada': self._rodada_atual,
                'acuracia_media': np.mean(acuracias),
                'perda_media': np.mean(perdas) if perdas else 0,
                'num_clientes': len(self._lista_clientes)
            }
            
            # Armazena no histórico
            self._historico_metricas.append(deepcopy(self._metricas_avaliacao))
    
    def compartilhar_modelo_global(self) -> None:
        """Distribui pesos do modelo global para todos os clientes (conforme diagrama UML)"""
        pesos_globais = self._modelo_global.obter_pesos()
        for cliente in self._lista_clientes:
            cliente.set_modelo_local(pesos_globais)
    
    def set_modelo_global(self) -> None:
        """
        Atualiza o modelo global com agregação dos modelos locais (conforme diagrama UML)
        Este método implementa a agregação usando média ponderada baseada no desempenho
        """
        if not self._lista_clientes:
            return
        
        # Calcula pesos de agregação baseados no desempenho dos clientes
        pesos_agregacao = self._calcular_pesos_agregacao()
        
        # Inicializa estrutura para armazenar pesos agregados
        pesos_agregados = None
        
        # Agrega os modelos locais
        for cliente in self._lista_clientes:
            pesos_cliente = cliente.get_modelo_local().obter_pesos()
            peso_agregacao = pesos_agregacao.get(cliente._id_cliente, 0)
            
            if pesos_agregados is None:
                # Primeira iteração: inicializa estrutura
                if isinstance(pesos_cliente, list) and len(pesos_cliente) > 0:
                    pesos_agregados = [p * peso_agregacao for p in pesos_cliente]
                else:
                    pesos_agregados = []
            else:
                # Soma ponderada dos pesos
                if len(pesos_cliente) == len(pesos_agregados):
                    pesos_agregados = [
                        pa + (pc * peso_agregacao) 
                        for pa, pc in zip(pesos_agregados, pesos_cliente)
                    ]
        
        # Atualiza o modelo global com os pesos agregados
        if pesos_agregados:
            self._modelo_global.atualizar_pesos(pesos_agregados)
    
    def avaliar_convergencia(self) -> bool:
        """
        Verifica se o treinamento convergiu (conforme diagrama UML)
        
        Implementa dois critérios de convergência:
        1. Número máximo de rodadas (max_rodadas) - critério de parada forçada
        2. Estabilidade do modelo global (criterio_convergencia) - early stop
        
        Returns:
            bool: True se convergiu, False caso contrário
        """
        # Critério 1: Atingiu número máximo de rodadas
        if self._rodada_atual >= self._max_rodadas:
            print(f"   ⚠️ Convergência: Atingiu max_rodadas ({self._max_rodadas})")
            return True
        
        # Critério 2: Estabilidade do modelo (early stop)
        if self._desempenho_anterior is not None:
            desempenho_atual = self._metricas_avaliacao.get('acuracia_media', 0)
            variacao = abs(desempenho_atual - self._desempenho_anterior)
            
            if variacao < self._criterio_convergencia:
                print(f"   ✓ Convergência antecipada: variação={variacao:.4f} < {self._criterio_convergencia}")
                return True
            
            self._desempenho_anterior = desempenho_atual
        else:
            # Primeira rodada: armazena desempenho inicial
            self._desempenho_anterior = self._metricas_avaliacao.get('acuracia_media', 0)
        
        return False
    
    # Métodos auxiliares (não estão no diagrama UML, mas são necessários para implementação)
    
    def _calcular_pesos_agregacao(self) -> Dict[str, float]:
        """
        Método auxiliar privado: Calcula pesos de agregação baseados no desempenho
        Não está no diagrama UML, mas é usado internamente por set_modelo_global()
        """
        metricas_clientes = {}
        for cliente in self._lista_clientes:
            metricas = cliente.get_metricas_avaliacao()
            metricas_clientes[cliente._id_cliente] = metricas
        
        pesos_agregacao = {}
        
        # Calcula pesos baseados em acurácia
        total_acuracia = sum(m.get('acuracia', 0) for m in metricas_clientes.values())
        
        if total_acuracia > 0:
            for id_cliente, metrica in metricas_clientes.items():
                acuracia = metrica.get('acuracia', 0)
                pesos_agregacao[id_cliente] = acuracia / total_acuracia
        else:
            # Pesos uniformes se não houver métricas
            peso_uniforme = 1.0 / len(self._lista_clientes) if self._lista_clientes else 0
            for id_cliente in metricas_clientes.keys():
                pesos_agregacao[id_cliente] = peso_uniforme
        
        return pesos_agregacao
    
    def executar_aprendizado_federado(self) -> None:
        """
        Método auxiliar: Executa o loop principal do aprendizado federado
        Não está no diagrama UML, mas orquestra os métodos do diagrama
        """
        print("🚀 Iniciando Aprendizado Federado\n")
        
        while not self.avaliar_convergencia():
            self._rodada_atual += 1
            print(f"{'='*60}")
            print(f"📍 Rodada {self._rodada_atual}/{self._max_rodadas}")
            print(f"{'='*60}")
            
            # 1. Compartilhar modelo global (método do diagrama UML)
            self.compartilhar_modelo_global()
            
            # 2. Cada cliente treina localmente
            print("\n🔄 Treinamento Local:")
            for cliente in self._lista_clientes:
                cliente.treinar_modelo()
            
            # 3. Avaliar o modelo global (método do diagrama UML)
            print("\n📊 Avaliação do Modelo:")
            self.avaliar_modelo()
            
            # 4. Atualizar modelo global com agregação (método do diagrama UML)
            print("\n🔀 Agregação de Modelos")
            self.set_modelo_global()
            
            # 5. Mostrar métricas
            acuracia_media = self._metricas_avaliacao.get('acuracia_media', 0)
            print(f"\n✓ Modelo Global Atualizado - Acurácia Média: {acuracia_media:.4f}\n")
        
        print(f"{'='*60}")
        print("✅ Aprendizado Federado Concluído!")
        print(f"{'='*60}")
        print(f"Total de rodadas: {self._rodada_atual}")
        print(f"Acurácia final: {self._metricas_avaliacao.get('acuracia_media', 0):.4f}")



class ClienteFederado(ABC):
    """Classe abstrata base para clientes federados (conforme diagrama UML)"""
    
    def __init__(self, id_cliente: str):
        # Atributos privados conforme diagrama UML
        self._id_cliente: str = id_cliente
        self._dados: pd.DataFrame = pd.DataFrame()
        self._modelo_local: Modelo = Modelo()
        self._metricas_avaliacao: Dict = {}
    
    @abstractmethod
    def treinar_modelo(self) -> None:
        """Treina o modelo local (método abstrato conforme diagrama UML)"""
        pass
    
    def get_modelo_local(self) -> Modelo:
        """Retorna o modelo local treinado (conforme diagrama UML)"""
        return self._modelo_local
    
    def get_metricas_avaliacao(self) -> Dict:
        """Retorna as métricas de avaliação do treinamento (conforme diagrama UML)"""
        return self._metricas_avaliacao
    
    def avaliar_modelo(self) -> None:
        """Avalia o modelo local (conforme diagrama UML)"""
        # Implementação simulada de avaliação
        if self._metricas_avaliacao:
            print(f"    📊 Avaliando {self._id_cliente}: Acurácia={self._metricas_avaliacao.get('acuracia', 0):.2f}")
    
    def set_modelo_local(self, pesos_globais: List) -> None:
        """
        Atualiza modelo local com pesos globais (conforme diagrama UML)
        Nota: No diagrama era set_modelo_local(modelo_global: ModeloGlobal), 
        mas como ModeloGlobal não existe mais, recebe os pesos diretamente
        """
        self._modelo_local.atualizar_pesos(pesos_globais)
    
    def compartilhar_modelo_local(self) -> None:
        """Compartilha o modelo local (conforme diagrama UML)"""
        # Simula compartilhamento do modelo
        print(f"    📤 {self._id_cliente}: Compartilhando modelo local")
    
    def obter_pesos(self) -> List:
        """Retorna os pesos do modelo local (conforme diagrama UML)"""
        return self._modelo_local.obter_pesos()


class ClienteHonesto(ClienteFederado):
    """Cliente honesto que treina normalmente sem ataques (conforme diagrama UML)"""
    
    def treinar_modelo(self) -> None:
        """Treina o modelo de forma honesta e legítima (conforme diagrama UML)"""
        print(f"  🟢 {self._id_cliente}: Treinamento honesto")
        
        # Simula treinamento bem-sucedido
        self._metricas_avaliacao = {
            'acuracia': np.random.uniform(0.85, 0.95),
            'perda': np.random.uniform(0.05, 0.15),
            'tipo': 'honesto'
        }
        
        # Simula atualização dos pesos após treinamento
        pesos_atuais = self._modelo_local.obter_pesos()
        if pesos_atuais:
            novos_pesos = [p + np.random.uniform(-0.01, 0.01) for p in pesos_atuais]
        else:
            novos_pesos = [np.random.uniform(-1, 1) for _ in range(10)]
        
        self._modelo_local.atualizar_pesos(novos_pesos)


class ClienteMalicioso(ClienteFederado):
    """Cliente malicioso que executa ataques de envenenamento (conforme diagrama UML)"""
    
    def __init__(self, id_cliente: str, tipo_ataque: str):
        super().__init__(id_cliente)
        # Atributo específico conforme diagrama UML
        self._tipo_ataque: str = tipo_ataque
    
    def envenenar_dados(self) -> None:
        """
        Aplica envenenamento nos dados de treinamento (conforme diagrama UML)
        Nota: No diagrama é envenenar_dados(), sem parâmetros nem retorno
        """
        print(f"    🔴 Aplicando envenenamento de dados...")
        # Simula corrupção dos dados (inverter labels, adicionar ruído, etc.)
        if not self._dados.empty:
            # Implementação simulada de envenenamento
            self._dados = self._dados.copy()  # Simula modificação
    
    def envenenar_modelo(self) -> None:
        """
        Aplica envenenamento no modelo/pesos (conforme diagrama UML)
        Nota: No diagrama é envenenar_modelo(), sem parâmetros nem retorno
        """
        print(f"    🔴 Aplicando envenenamento no modelo...")
        pesos_originais = self._modelo_local.obter_pesos()
        
        if pesos_originais:
            # Adiciona ruído significativo nos pesos
            pesos_envenenados = [p + np.random.uniform(-0.5, 0.5) for p in pesos_originais]
            self._modelo_local.atualizar_pesos(pesos_envenenados)
    
    def treinar_modelo(self) -> None:
        """Treina o modelo aplicando o ataque especificado (conforme diagrama UML)"""
        print(f"  🔴 {self._id_cliente}: Ataque tipo '{self._tipo_ataque}'")
        
        # Aplica ataque conforme tipo
        if self._tipo_ataque == "dados":
            self.envenenar_dados()
        
        # Simula treinamento com desempenho degradado
        self._metricas_avaliacao = {
            'acuracia': np.random.uniform(0.40, 0.60),
            'perda': np.random.uniform(0.30, 0.50),
            'tipo': 'malicioso',
            'ataque': self._tipo_ataque
        }
        
        # Atualiza pesos
        pesos_atuais = self._modelo_local.obter_pesos()
        if pesos_atuais:
            novos_pesos = [p + np.random.uniform(-0.1, 0.1) for p in pesos_atuais]
        else:
            novos_pesos = [np.random.uniform(-1, 1) for _ in range(10)]
        
        self._modelo_local.atualizar_pesos(novos_pesos)
        
        # Aplica envenenamento no modelo se for o tipo de ataque
        if self._tipo_ataque == "modelo":
            self.envenenar_modelo()


# Exemplo de uso
if __name__ == "__main__":
    print("="*60)
    print("SISTEMA DE APRENDIZADO FEDERADO COM ATAQUES")
    print("="*60)
    
    # Criar servidor com ambos os critérios de convergência
    # max_rodadas=5: critério de parada por número de iterações
    # criterio_convergencia=0.01: early stop por estabilidade do modelo
    servidor = ServidorFederado(max_rodadas=5, criterio_convergencia=0.01)
    
    # Adicionar clientes
    servidor.adicionar_cliente(ClienteHonesto("cliente_1"))
    servidor.adicionar_cliente(ClienteHonesto("cliente_2"))
    servidor.adicionar_cliente(ClienteHonesto("cliente_3"))
    servidor.adicionar_cliente(ClienteMalicioso("cliente_malicioso_1", "dados"))
    servidor.adicionar_cliente(ClienteMalicioso("cliente_malicioso_2", "modelo"))
    
    print(f"\n📊 Configuração:")
    print(f"  • Total de clientes: {len(servidor._lista_clientes)}")
    print(f"  • Clientes honestos: 3")
    print(f"  • Clientes maliciosos: 2")
    print(f"  • Max rodadas: {servidor._max_rodadas}")
    print(f"  • Critério convergência (early stop): {servidor._criterio_convergencia}")
    print()
    
    # Executar aprendizado federado
    servidor.executar_aprendizado_federado()