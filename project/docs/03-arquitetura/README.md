# 03 - Arquitetura do Sistema

## Visão Geral da Arquitetura

O sistema implementado segue uma arquitetura cliente-servidor clássica de Aprendizado Federado, com componentes adicionais para detecção de ataques e visualização de resultados.

## Diagrama de Componentes

```
┌──────────────────────────────────────────────────────────────┐
│                MODELAGEM DO SISTEMA                          │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌────────────────────────────────────────────────────┐     │
│  │          SERVIDOR FEDERADO                         │     │
│  │  ┌──────────────────────────────────────────┐      │     │
│  │  │  Modelo Global (LinearRegression)        │      │     │
│  │  └──────────────────────────────────────────┘      │     │
│  │  ┌──────────────────────────────────────────┐      │     │
│  │  │  Agregador FedAvg + MAD                  │      │     │
│  │  └──────────────────────────────────────────┘      │     │
│  │  ┌──────────────────────────────────────────┐      │     │
│  │  │  Detector de Outliers (MAD)              │      │     │
│  │  └──────────────────────────────────────────┘      │     │
│  │  ┌──────────────────────────────────────────┐      │     │
│  │  │  Gerador de Visualizações                │      │     │
│  │  └──────────────────────────────────────────┘      │     │
│  └────────────────────────────────────────────────────┘     │
│                         │                                    │
│         ┌───────────────┼───────────────┐                   │
│         │               │               │                    │
│    ┌────▼───┐      ┌────▼───┐     ┌────▼───┐              │
│    │Cliente1│      │Cliente2│     │Cliente3│              │
│    │Honesto │      │MALICIOSO│     │Honesto │              │
│    └────────┘      └────────┘     └────────┘              │
│         │               │               │                    │
│    ┌────▼──────────────▼───────────────▼────┐              │
│    │  Dataset Distribuído (Iris/NSL-KDD)    │              │
│    └─────────────────────────────────────────┘              │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

## Componentes Principais

### 1. Classe `Modelo`

**Responsabilidade**: Encapsular o modelo de machine learning

**Atributos:**
```python
_modelo_interno: LinearRegression  # Modelo scikit-learn
```

**Métodos Principais:**
```python
obter_pesos() -> Dict                # Retorna coef e intercept
atualizar_pesos(pesos: Dict)         # Atualiza modelo
fit(X, y)                             # Treina modelo
predict(X) -> array                   # Faz predições
get_coef() -> array                   # Retorna coeficientes
get_intercept() -> float              # Retorna intercept
```

**Diagrama de Classes:**
```
┌───────────────────────┐
│      Modelo           │
├───────────────────────┤
│ - _modelo_interno     │
├───────────────────────┤
│ + obter_pesos()       │
│ + atualizar_pesos()   │
│ + fit()               │
│ + predict()           │
│ + get_coef()          │
│ + get_intercept()     │
└───────────────────────┘
```

### 2. Classe `ServidorFederado`

**Responsabilidade**: Coordenar o treinamento federado

**Atributos:**
```python
rodada_atual: int                      # Contador de rodadas
max_rodadas: int                       # Limite de rodadas
criterio_convergencia: float           # Threshold para early stop
clientes: List[ClienteMalicioso]       # Lista de clientes
modelo_global: Modelo                  # Modelo compartilhado
dados_validacao: Tuple                 # (X_val, y_val)
historico_r2_global: List[float]       # Histórico de R²
historico_mse_global: List[float]      # Histórico de MSE
historico_mae_global: List[float]      # Histórico de MAE
outliers_detectados: List[Dict]        # Detecções por rodada
```

**Métodos Principais:**
```python
adicionar_cliente(cliente)             # Registra novo cliente
compartilhar_modelo_global()           # Envia modelo aos clientes
avaliar_modelo()                       # Avalia no conjunto de validação
set_modelo_global()                    # Agrega modelos locais
_agregar_modelos()                     # FedAvg + MAD
avaliar_convergencia() -> bool         # Verifica critério de parada
executar_aprendizado_federado()        # Loop principal
gerar_graficos()                       # Cria visualizações
gerar_relatorio_estatistico()          # Imprime estatísticas
```

**Fluxo de Execução:**
```
Início
  │
  ▼
┌─────────────────────┐
│ Inicialização       │
│ - Cria modelo global│
│ - Registra clientes │
└──────────┬──────────┘
           │
           ▼
     ┌─────────────┐
     │   Rodada t  │◄──────┐
     └──────┬──────┘       │
            │              │
            ▼              │
   ┌────────────────┐      │
   │ 1. Compartilha │      │
   │    modelo      │      │
   └────────┬───────┘      │
            │              │
            ▼              │
   ┌────────────────┐      │
   │ 2. Clientes    │      │
   │    treinam     │      │
   └────────┬───────┘      │
            │              │
            ▼              │
   ┌────────────────┐      │
   │ 3. Avalia      │      │
   │    modelo      │      │
   └────────┬───────┘      │
            │              │
            ▼              │
   ┌────────────────┐      │
   │ 4. Agrega      │      │
   │   (FedAvg+MAD) │      │
   └────────┬───────┘      │
            │              │
            ▼              │
     ┌──────────────┐      │
     │ Convergiu?   │      │
     └──┬───────┬───┘      │
   Não  │       │ Sim      │
        └───────┘          │
            │              │
            ▼              │
   ┌────────────────┐      │
   │ 5. Gera        │      │
   │    gráficos    │      │
   └────────┬───────┘      │
            │              │
            ▼              │
          Fim
```

### 3. Classe `ClienteFederado` (Abstract)

**Responsabilidade**: Classe base para clientes

**Atributos:**
```python
id_cliente: str                        # Identificador único
dados: DataFrame                       # Dataset local
dados_originais: DataFrame             # Backup (para ataques)
target_col: str                        # Nome da coluna target
modelo_local: Modelo                   # Modelo treinado localmente
metricas_avaliacao: Dict               # R², MSE, etc.
scaler: StandardScaler                 # Normalizador
```

**Métodos Abstratos:**
```python
@abstractmethod
treinar_modelo()                       # Implementado por subclasses
```

**Métodos Concretos:**
```python
get_modelo_local() -> Modelo
get_metricas_avaliacao() -> Dict
set_modelo_local(pesos)
obter_pesos() -> Dict
```

### 4. Classe `ClienteMalicioso` (extends ClienteFederado)

**Responsabilidade**: Representar clientes honestos e maliciosos

**Atributos Adicionais:**
```python
tipo_ataque: str                       # "nenhum", "dados", "modelo_*"
```

**Métodos Específicos:**
```python
treinar_modelo()                       # Implementação com suporte a ataques
envenenar_dados()                      # Corrompe dados locais
envenenar_modelo()                     # Manipula pesos do modelo
```

**Tipos de Ataque:**
```
"nenhum"             → Cliente honesto
"dados"              → Data poisoning (ruído gaussiano)
"modelo_invertidos"  → Model poisoning (inverte coeficientes)
"modelo_randomizados"→ Model poisoning (coeficientes aleatórios)
```

## Fluxo de Dados

### Rodada de Treinamento Completa

```
PASSO 1: Distribuição do Modelo Global
────────────────────────────────────────
Servidor                        Clientes
   │                               │
   ├──── pesos globais ────────────┤
   │                               │
   │                               ▼
   │                          Cliente 1: set_modelo(pesos)
   │                          Cliente 2: set_modelo(pesos)
   │                          Cliente 3: set_modelo(pesos)
   │                          Cliente 4: set_modelo(pesos)

PASSO 2: Treinamento Local
────────────────────────────
Servidor                        Clientes
   │                               │
   │                          Cliente 1:
   │                            - Carrega dados locais
   │                            - Treina modelo
   │                            - R² = 0.85
   │                               │
   │                          Cliente 2 (MALICIOSO):
   │                            - Envenenar dados (30% ruído)
   │                            - Treina modelo
   │                            - R² = 0.23
   │                               │
   │                          Cliente 3:
   │                            - Treina normalmente
   │                            - R² = 0.87
   │                               │
   │                          Cliente 4:
   │                            - Treina normalmente
   │                            - R² = 0.84

PASSO 3: Agregação com Detecção
────────────────────────────────
Servidor                        Análise
   │                               │
   │◄────── pesos locais ──────────┤
   │                               │
   ▼                               │
Cálculo MAD:                       │
  - mediana_coefs = [...]          │
  - distâncias:                    │
    * Cliente 1: 0.15              │
    * Cliente 2: 0.92 ← OUTLIER!   │
    * Cliente 3: 0.18              │
    * Cliente 4: 0.14              │
  - threshold = 0.58               │
   │                               │
   ▼                               │
Agregação FedAvg:                  │
  - Aceitos: [1, 3, 4]             │
  - Rejeitados: [2]                │
  - modelo_global = mean([1,3,4])  │
   │                               │
   ▼                               │
Salva detecção:                    │
  - rodada: 1                      │
  - clientes: ["Cliente_2"]        │

PASSO 4: Avaliação
────────────────────
Servidor
   │
   ▼
Conjunto de Validação (30 amostras):
  - R² global: 0.915
  - MSE: 0.0539
  - MAE: 0.1801
   │
   ▼
Armazena histórico:
  - historico_r2_global.append(0.915)
  - historico_mse_global.append(0.0539)
  - historico_mae_global.append(0.1801)
```

## Algoritmo de Detecção MAD

### Pseudocódigo Detalhado

```python
def _agregar_modelos(self):
    # FASE 1: Coleta de Modelos
    coefs = []
    intercepts = []
    clientes_aceitos = []
    clientes_rejeitados = []
    
    todos_coefs = []
    for cliente in self.clientes:
        coef = cliente.modelo_local.get_coef()
        if coef is not None:
            todos_coefs.append(coef)
    
    if not todos_coefs:
        return  # Nenhum modelo válido
    
    # FASE 2: Cálculo da Mediana
    mediana_coefs = np.median(todos_coefs, axis=0)
    
    # FASE 3: Detecção de Outliers
    for i, cliente in enumerate(self.clientes):
        coef = cliente.modelo_local.get_coef()
        
        if coef is not None:
            # Distância euclidiana à mediana
            distancia = np.linalg.norm(coef - mediana_coefs)
            
            # Cálculo do threshold MAD
            desvios = [np.linalg.norm(c - mediana_coefs) for c in todos_coefs]
            mad = np.median(np.abs(desvios - np.median(desvios)))
            threshold = np.median(desvios) + 3 * mad
            
            # Decisão: Aceitar ou Rejeitar
            if distancia > threshold:
                clientes_rejeitados.append(cliente.id_cliente)
                print(f"[OUTLIER] {cliente.id_cliente} - dist: {distancia:.4f} > {threshold:.4f}")
            else:
                coefs.append(coef)
                intercepts.append(cliente.modelo_local.get_intercept())
                clientes_aceitos.append(cliente.id_cliente)
    
    # FASE 4: Agregação (FedAvg)
    if coefs:
        pesos_agregados = {
            'coef': np.mean(coefs, axis=0),
            'intercept': np.mean(intercepts)
        }
        self.modelo_global.atualizar_pesos(pesos_agregados)
        
        # Registra detecções
        if clientes_rejeitados:
            self.outliers_detectados.append({
                'rodada': self.rodada_atual,
                'clientes': clientes_rejeitados
            })
```

### Complexidade Computacional

```
Operação                           Complexidade    Observação
─────────────────────────────────────────────────────────────
Coleta de coeficientes            O(n×d)          n=clientes, d=dimensões
Cálculo da mediana                O(n×d×log(n))   Por dimensão
Cálculo de distâncias             O(n×d)          Norma euclidiana
Cálculo do MAD                    O(n×log(n))     Uma mediana
Decisão outlier (por cliente)     O(d)            Comparação escalar
Agregação FedAvg                  O(m×d)          m=aceitos
─────────────────────────────────────────────────────────────
TOTAL                             O(n×d×log(n))   Dominado pela mediana
```

Para **n=100 clientes, d=10 dimensões**: ~10,000 operações (milliseconds)

## Padrões de Design Utilizados

### 1. Strategy Pattern

**Aplicação**: Tipos de ataque diferentes

```python
class ClienteMalicioso:
    def treinar_modelo(self):
        if self.tipo_ataque == "dados":
            self.envenenar_dados()
        elif self.tipo_ataque == "modelo_invertidos":
            # ... treina ...
            self.envenenar_modelo()
```

### 2. Template Method Pattern

**Aplicação**: Classe abstrata `ClienteFederado`

```python
class ClienteFederado(ABC):
    @abstractmethod
    def treinar_modelo(self):
        pass  # Subclasses implementam
```

### 3. Observer Pattern (Implícito)

**Aplicação**: Histórico de métricas

```python
self.historico_r2_global.append(r2_global)  # Observadores
self.historico_mse_global.append(mse_global)
```

### 4. Facade Pattern

**Aplicação**: `executar_aprendizado_federado()`

```python
def executar_aprendizado_federado(self):
    # Fachada que orquestra:
    while not self.avaliar_convergencia():
        self.compartilhar_modelo_global()
        # ... treinar clientes ...
        self.avaliar_modelo()
        self.set_modelo_global()
    self.gerar_graficos()
```

## Configuração e Parâmetros

### Arquivo de Configuração (Conceitual)

```python
# Configurações do Servidor
CONFIG_SERVIDOR = {
    'max_rodadas': 10,
    'criterio_convergencia': 0.01,
    'threshold_mad_multiplier': 3,  # k×MAD
}

# Configurações de Clientes
CONFIG_CLIENTES = {
    'n_clientes': 4,
    'fracao_maliciosos': 0.25,  # 25%
    'tipos_ataque': ['dados', 'modelo_invertidos'],
}

# Configurações de Dados
CONFIG_DADOS = {
    'dataset': 'iris',
    'validacao_size': 0.2,
    'normalizacao': True,
    'random_state': 42,
}

# Configurações de Visualização
CONFIG_VIZ = {
    'dpi': 300,
    'figsize': (14, 10),
    'formato': 'png',
}
```

## Escalabilidade e Performance

### Limitações Atuais

```
Componente                 Limite Atual    Bottleneck
────────────────────────────────────────────────────────
Número de clientes         ~100            Cálculo MAD
Dimensões do modelo        ~1000           Memória
Tamanho do dataset         ~100k amostras  I/O disco
Rodadas de treinamento     ~1000           Tempo total
```

### Otimizações Futuras

1. **Paralelização**: Treinar clientes em paralelo
2. **Sampling**: Selecionar apenas fração de clientes por rodada
3. **Compressão**: Comprimir comunicação cliente-servidor
4. **Caching**: Cachear cálculos de mediana

## Diagrama de Sequência

```
Cliente1    Cliente2    Cliente3    Servidor
   │            │            │          │
   │            │            │    ┌─────▼─────┐
   │            │            │    │ Inicializa│
   │            │            │    └─────┬─────┘
   │◄───────────┴────────────┴──────────┤ Envia modelo
   │                                     │
   ├──────────────────────────────────┐  │
   │ Treina localmente                │  │
   └──────────────────────────────────┘  │
   │                                     │
   ├─────────────────────────────────────► Envia pesos
   │                                     │
   │                              ┌──────▼──────┐
   │                              │ Detecta MAD │
   │                              └──────┬──────┘
   │                              ┌──────▼──────┐
   │                              │ Agrega FedAvg│
   │                              └──────┬──────┘
   │                              ┌──────▼──────┐
   │                              │   Avalia    │
   │                              └──────┬──────┘
   │                                     │
   │◄────────────────────────────────────┤ Repete
```

## Estrutura de Arquivos

```
project/
├── modelagem/
│   ├── modelagem.py              # Implementação completa
│   ├── teste_iris_simples.py     # Testes automatizados
│   ├── testes_estatisticos.py    # Análises estatísticas
│   ├── modelo_exemplo.py         # Exemplo didático
│   ├── resultados_fl.png         # Gráficos gerados
│   └── README.md                 # Documentação técnica
├── data/
│   ├── iris/
│   │   └── iris.csv              # Dataset Iris
│   └── nsl-kdd/
│       └── *.arff                # Dataset NSL-KDD
└── docs/
    └── 03-arquitetura/
        └── README.md             # Este arquivo
```

---

**Última atualização**: Fevereiro 2026
