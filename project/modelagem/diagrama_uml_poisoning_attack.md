# Diagrama UML - Sistema de Detecção de Envenenamento em Aprendizado Federado

## Visão Geral da Arquitetura

```
┌─────────────────────────────────────────────────────────────────────┐
│                    SISTEMA DE APRENDIZADO FEDERADO                  │
│                    COM DETECÇÃO DE ENVENENAMENTO                     │
└─────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│                         «abstract»                                    │
│                        ClienteBase                                    │
├──────────────────────────────────────────────────────────────────────┤
│ # id_cliente: str                                                     │
│ # X_local: np.ndarray                                                 │
│ # y_local: np.ndarray                                                 │
│ # modelo_local: ModeloClassificacao                                   │
│ # historico_treinamento: List[Dict]                                   │
│ # scaler: StandardScaler                                              │
├──────────────────────────────────────────────────────────────────────┤
│ + __init__(id_cliente, dados_locais)                                 │
│ + «abstract» treinar_modelo(): Dict                                  │
│ + set_modelo_local(pesos: Dict): void                                │
│ + avaliar_modelo_local(X_test, y_test): Dict                         │
└──────────────────────────────────────────────────────────────────────┘
                              △
                              │
                              │ «inherits»
              ┌───────────────┴───────────────┐
              │                               │
              │                               │
┌─────────────▼─────────────┐   ┌────────────▼──────────────┐
│    ClienteHonesto          │   │   ClienteEnvenenado       │
├────────────────────────────┤   ├───────────────────────────┤
│                            │   │ - taxa_corrupcao: float   │
│                            │   │ - tipo_ataque: str        │
├────────────────────────────┤   ├───────────────────────────┤
│ + treinar_modelo(): Dict   │   │ + treinar_modelo(): Dict  │
│                            │   │ - _corromper_pesos(): Dict│
└────────────────────────────┘   └───────────────────────────┘


┌──────────────────────────────────────────────────────────────────────┐
│                      ModeloClassificacao                              │
├──────────────────────────────────────────────────────────────────────┤
│ - _modelo_interno: LogisticRegression                                 │
├──────────────────────────────────────────────────────────────────────┤
│ + __init__(max_iter, random_state)                                   │
│ + obter_pesos(): Dict                                                 │
│ + atualizar_pesos(pesos: Dict): void                                 │
│ + fit(X, y): ModeloClassificacao                                      │
│ + predict(X): np.ndarray                                              │
│ + predict_proba(X): np.ndarray                                        │
└──────────────────────────────────────────────────────────────────────┘


┌──────────────────────────────────────────────────────────────────────┐
│                      ServidorFederado                                 │
├──────────────────────────────────────────────────────────────────────┤
│ - clientes: List[ClienteBase]                                         │
│ - modelo_global: ModeloClassificacao                                  │
│ - X_val: np.ndarray                                                   │
│ - y_val: np.ndarray                                                   │
│ - scaler_global: StandardScaler                                       │
│ - historico_agregacao: List[Dict]                                     │
│ - historico_metricas_globais: List[Dict]                              │
│ - historico_metricas_por_classe: List[Dict]                           │
├──────────────────────────────────────────────────────────────────────┤
│ + __init__(dados_validacao)                                           │
│ + adicionar_cliente(cliente: ClienteBase): void                       │
│ + executar_rodada_federada(num_rodada: int): Dict                    │
│ - _agregar_modelos(): Dict                                            │
│ - _avaliar_modelo_global(): Dict                                      │
│ - _avaliar_por_classe(): Dict                                         │
│ + gerar_relatorio_completo(): void                                    │
│ + gerar_graficos(): void                                              │
│ - _gerar_grafico_especies(): void                                     │
└──────────────────────────────────────────────────────────────────────┘
                              ◆ «composes»
                              │
                              │ 1..*
                              │
                              ▼
                    ┌─────────────────┐
                    │  ClienteBase    │
                    └─────────────────┘
```

## Fluxo de Execução

```
┌─────────────┐
│   INÍCIO    │
└──────┬──────┘
       │
       ▼
┌──────────────────────────────────┐
│ 1. Carrega Dataset Iris          │
│    - X, y = carregar_dataset()   │
└──────┬───────────────────────────┘
       │
       ▼
┌──────────────────────────────────┐
│ 2. Divide Dados                  │
│    - Treino (clientes)           │
│    - Validação (servidor)        │
└──────┬───────────────────────────┘
       │
       ▼
┌──────────────────────────────────┐
│ 3. Cria Servidor e Clientes      │
│    - 2 Honestos                  │
│    - 1 Envenenado                │
└──────┬───────────────────────────┘
       │
       ▼
┌──────────────────────────────────┐
│ 4. Loop de Rodadas Federadas     │
│    ┌──────────────────────────┐  │
│    │ a) Treinamento Local     │  │
│    │    - Honesto: treina OK  │  │
│    │    - Envenenado:         │  │
│    │      1. Treina           │  │
│    │      2. CORROMPE PESOS   │  │
│    └──────────────────────────┘  │
│    ┌──────────────────────────┐  │
│    │ b) Agregação (FedAvg)    │  │
│    │    - Média dos pesos     │  │
│    └──────────────────────────┘  │
│    ┌──────────────────────────┐  │
│    │ c) Avaliação Global      │  │
│    │    - Métricas gerais     │  │
│    │    - Métricas por classe │  │
│    └──────────────────────────┘  │
└──────┬───────────────────────────┘
       │
       ▼
┌──────────────────────────────────┐
│ 5. Gera Relatório e Gráficos     │
│    - convergencia.png            │
│    - especies.png                │
└──────┬───────────────────────────┘
       │
       ▼
┌──────────────┐
│     FIM      │
└──────────────┘
```

## Tipos de Ataques Implementados

### 1. Inverter (Sign Flipping)
```python
coef_corrompido = -coef_original * (1 + taxa)
```
- **Impacto**: Inverte completamente a direção das predições
- **Severidade**: ALTA
- **Detectabilidade**: MÉDIA

### 2. Aleatório (Gaussian Noise)
```python
coef_corrompido = coef_original + ruído_gaussiano
```
- **Impacto**: Adiciona incerteza às predições
- **Severidade**: MÉDIA
- **Detectabilidade**: BAIXA

### 3. Amplificar (Scale Attack)
```python
coef_corrompido = coef_original * (1 + taxa * 10)
```
- **Impacto**: Domina a agregação FedAvg
- **Severidade**: MUITO ALTA
- **Detectabilidade**: ALTA

### 4. Zerar (Weight Zeroing)
```python
coef_corrompido = coef_original * mascara_aleatoria
```
- **Impacto**: Remove informação aprendida
- **Severidade**: MÉDIA
- **Detectabilidade**: MÉDIA

## Métricas de Avaliação

### Métricas Globais
- **Acurácia**: Proporção de predições corretas
- **F1-Score**: Média harmônica entre precisão e recall
- **Precisão**: Proporção de verdadeiros positivos
- **Recall**: Taxa de detecção
- **Loss**: Log loss (entropia cruzada)

### Métricas por Espécie (Iris)
- **Setosa**: Acurácia específica
- **Versicolor**: Acurácia específica
- **Virginica**: Acurácia específica

## Padrões de Projeto Utilizados

1. **Abstract Factory**: ClienteBase como interface abstrata
2. **Strategy Pattern**: Diferentes tipos de ataque (tipo_ataque)
3. **Template Method**: Fluxo de treinar_modelo() padronizado
4. **Composite**: ServidorFederado compõe múltiplos ClienteBase
5. **Encapsulation**: ModeloClassificacao encapsula LogisticRegression
6. **Observer**: Históricos para monitoramento de métricas

## Diagrama de Sequência - Rodada Federada

```
Cliente1     Cliente2     Cliente3       Servidor
(Honesto)    (Honesto)   (Envenenado)    (Central)
   │            │            │               │
   │◄───────────┴────────────┴───────────────┤ 1. compartilhar_modelo_global()
   │            │            │               │
   ├────────────┼────────────┼──────────────►│ 2. treinar_modelo()
   │  treina    │  treina    │ treina +      │
   │  normal    │  normal    │ CORROMPE      │
   │            │            │               │
   ├────────────┴────────────┴──────────────►│ 3. enviar_pesos_locais()
   │            │            │               │
   │            │            │  FedAvg       │ 4. agregar_modelos()
   │            │            │  (média)      │
   │            │            │               │
   │            │            │  avaliar      │ 5. avaliar_modelo_global()
   │            │            │               │
   │            │            │  ┌─────────┐  │
   │            │            │  │Métricas │  │
   │            │            │  │Degradadas│  │
   │            │            │  └─────────┘  │
```

## Observações Importantes

1. **Envenenamento pós-treinamento**: O ataque ocorre APÓS o treinamento local
2. **FedAvg vulnerável**: Média simples não detecta modelos maliciosos
3. **Impacto gradual**: 1 cliente envenenado em 3 reduz acurácia de ~95% para ~67%
4. **Persistência**: Sem mecanismos de defesa, o impacto persiste em todas as rodadas
5. **Visualização**: Gráficos mostram claramente a degradação do modelo

## Possíveis Defesas (Não Implementadas)

- **Krum**: Seleciona modelo mais próximo da mediana
- **Trimmed Mean**: Remove outliers antes da agregação
- **Median Aggregation**: Usa mediana ao invés de média
- **Byzantine-Robust Aggregation**: Detecta e remove modelos bizantinos
- **Differential Privacy**: Adiciona ruído para proteção

## Referências

- Dataset: Iris (sklearn.datasets)
- Modelo: Logistic Regression (multinomial)
- Agregação: FedAvg (Federated Averaging)
- Framework: Scikit-learn + NumPy + Matplotlib
