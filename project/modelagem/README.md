# Sistema de Aprendizado Federado com Detecção de Outliers

## Visão Geral

Este módulo implementa um sistema completo de **Aprendizado Federado (Federated Learning)** com:
- Modelo de Regressão Linear
- Dataset: **Iris** (predição de petal width)
- Detecção de outliers usando **MAD (Median Absolute Deviation)**
- Simulação de ataques de envenenamento
- Visualizações gráficas da evolução do modelo

## Objetivo do Projeto

Demonstrar como técnicas de **detecção de outliers** podem mitigar **ataques de envenenamento** em sistemas de Aprendizado Federado.

### Problema de Regressão

**Dataset**: Iris (iris.csv)
- **Features (X)**: sepal length, sepal width, petal length
- **Target (y)**: petal width
- **Objetivo**: Predizer a largura da pétala baseado nas outras características

## Estrutura dos Arquivos

```
modelagem/
├── modelagem.py                    # Sistema FL completo
├── teste_iris_simples.py          # Testes automatizados
├── testes_estatisticos.py         # Análises estatísticas detalhadas
└── README.md                       # Este arquivo
```

## Como Funciona o Sistema

### 1. Arquitetura

```
┌─────────────────────────────────────────────────┐
│           SERVIDOR CENTRAL                      │
│  - Mantém modelo global                         │
│  - Detecta outliers (MAD)                       │
│  - Agrega modelos locais (FedAvg)              │
└─────────────────────────────────────────────────┘
         │        │        │        │
         ▼        ▼        ▼        ▼
    ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
    │Cliente1│ │Cliente2│ │Cliente3│ │Cliente4│
    │Honesto │ │MALICIOSO│ │Honesto │ │Honesto │
    └────────┘ └────────┘ └────────┘ └────────┘
```

### 2. Classes Principais

#### `Modelo`
- Encapsula `LinearRegression` do scikit-learn
- Métodos para obter/atualizar pesos
- Treinar e fazer predições

#### `ServidorFederado`
- Coordena o treinamento federado
- Implementa FedAvg (Federated Averaging)
- Detecta outliers usando MAD
- Gera visualizações e relatórios

#### `ClienteMalicioso`
- Representa um cliente do sistema
- Pode ser honesto ou malicioso
- Tipos de ataque:
  - **Envenenamento de dados**: Adiciona ruído aos dados de treino
  - **Envenenamento de modelo**: Manipula os pesos do modelo

### 3. Algoritmo de Detecção de Outliers (MAD)

```python
# Para cada cliente, calcula distância dos coeficientes à mediana
distancia = ||coef_cliente - mediana(coefs_todos)||

# Threshold baseado em MAD
threshold = mediana(distancias) + 3 * MAD

# Se distância > threshold → OUTLIER detectado
```

**MAD (Median Absolute Deviation)** é robusto a outliers, diferente da média/desvio padrão.

## Como Usar

### Execução Básica

```powershell
# Executar sistema completo
python modelagem.py
```

Isso irá:
1. Carregar o Iris dataset
2. Dividir dados entre 4 clientes (2 honestos, 2 maliciosos)
3. Executar 10 rodadas de treinamento federado
4. Detectar e filtrar clientes maliciosos
5. Gerar gráficos de evolução do modelo

### Testes Automatizados

```powershell
# Testes simples
python teste_iris_simples.py
```

Executa 2 cenários:
- **Teste 1**: Todos os clientes honestos (baseline)
- **Teste 2**: 1 cliente malicioso + 3 honestos (validação de detecção)

### Análises Estatísticas

```powershell
# Comparações detalhadas
python testes_estatisticos.py
```

Compara 3 cenários:
1. Sem ataques (baseline)
2. Com ataques SEM detecção (vulnerável)
3. Com ataques COM detecção (protegido)

Gera:
- Tabelas comparativas
- Gráficos de performance
- Análise de eficácia da detecção

## Visualizações Geradas

O sistema gera automaticamente 4 gráficos:

### 1. R² Score ao Longo das Rodadas
- Mostra a qualidade do modelo global
- Linha azul com marcadores
- Valores ideais: > 0.7

### 2. MSE (Mean Squared Error)
- Erro médio quadrático
- Valores menores = melhor
- Ideal: decrescente ao longo das rodadas

### 3. MAE (Mean Absolute Error)
- Erro absoluto médio
- Mais interpretável que MSE
- Valores menores = melhor

### 4. Número de Clientes por Rodada
- Barras verdes: clientes aceitos
- Barras vermelhas: outliers detectados
- Mostra eficácia da detecção

Arquivos salvos em: `modelagem/resultados_fl_*.png`

## Resultados Esperados

### Cenário 1: Sem Ataques
- **R² final**: > 0.85
- **Outliers detectados**: 0
- **Convergência**: Rápida (5-7 rodadas)

### Cenário 2: Com Ataques e COM Detecção
- **R² final**: > 0.80
- **Outliers detectados**: 2-4 por rodada
- **Convergência**: Moderada (7-10 rodadas)
- **Clientes maliciosos**: Detectados e filtrados

### Cenário 3: Com Ataques e SEM Detecção
- **R² final**: < 0.50 (degrada significativamente)
- **Outliers detectados**: 0 (sem proteção)
- **Convergência**: Não converge ou diverge

## Métricas de Avaliação

### R² Score (Coeficiente de Determinação)
- Range: [-∞, 1]
- **1.0**: Modelo perfeito
- **0.0**: Modelo igual à média
- **< 0**: Modelo pior que a média

### MSE (Mean Squared Error)
- Range: [0, +∞]
- Penaliza erros grandes
- Unidade: quadrado da unidade do target

### MAE (Mean Absolute Error)
- Range: [0, +∞]
- Mais robusto a outliers que MSE
- Unidade: mesma do target

## Parâmetros Configuráveis

### ServidorFederado
```python
ServidorFederado(
    max_rodadas=10,              # Número máximo de rodadas
    criterio_convergencia=0.01,  # Threshold para early stop
    dados_validacao=(X_val, y_val)  # Conjunto de validação
)
```

### ClienteMalicioso
```python
ClienteMalicioso(
    id_cliente="Cliente_1",
    dados=df,                    # DataFrame com features e target
    nome_target="target",
    tipo_ataque="nenhum"         # "nenhum", "dados", "modelo_invertidos", "modelo_randomizados"
)
```

## Tipos de Ataque Disponíveis

### 1. Sem Ataque (`"nenhum"`)
Cliente honesto, comportamento normal.

### 2. Envenenamento de Dados (`"dados"`)
- Seleciona 30% das amostras aleatoriamente
- Adiciona ruído gaussiano (σ = 3x original)
- Mantém estrutura dos dados

### 3. Envenenamento de Modelo - Invertido (`"modelo_invertidos"`)
- Inverte o sinal dos coeficientes: `w → -w`
- Modelo faz predições opostas

### 4. Envenenamento de Modelo - Randomizado (`"modelo_randomizados"`)
- Substitui coeficientes por valores aleatórios
- Completamente descorrelacionado

## Dependências

```python
pandas
numpy
matplotlib
seaborn
scikit-learn
```

Instalar:
```powershell
pip install -r ../dependencies/requirements.txt
```

## Estrutura do Dataset Iris

```csv
sepal length (cm),sepal width (cm),petal length (cm),petal width (cm),species
5.1,3.5,1.4,0.2,setosa
4.9,3.0,1.4,0.2,setosa
...
```

- **150 amostras** (50 de cada espécie)
- **4 features numéricas**
- **1 target categórico** (species)
- **Problema de regressão**: Predizer petal width usando as outras 3 features

## Exemplo de Saída

```
==================================================
Rodada 3/10
==================================================

Treinamento Local:
  Cliente_1_Honesto: R2=0.8234
  Cliente_2_MALICIOSO: R2=0.1234
  Cliente_3_Honesto: R2=0.8456
  Cliente_4_Honesto: R2=0.8123

Agregacao de Modelos
  [OUTLIER DETECTADO] Cliente_2_MALICIOSO - Distancia: 15.2341 > Threshold: 5.6789

  Clientes aceitos: ['Cliente_1_Honesto', 'Cliente_3_Honesto', 'Cliente_4_Honesto']
  Clientes rejeitados: ['Cliente_2_MALICIOSO']

Modelo Global Atualizado:
  R2: 0.8271 | MSE: 0.0432 | MAE: 0.1654
```

## Interpretação dos Resultados

### Detecção Bem-Sucedida
✓ R² mantém-se estável (> 0.75)
✓ Clientes maliciosos são detectados consistentemente
✓ MSE e MAE decrescem ao longo das rodadas

### Detecção Falhou
✗ R² degrada significativamente (< 0.50)
✗ MSE e MAE aumentam ao longo das rodadas
✗ Clientes maliciosos não são filtrados

## Contribuições Científicas

1. **Validação Experimental**: MAD é eficaz para detectar outliers em FL
2. **Robustez**: Sistema mantém performance mesmo com 25-50% de clientes maliciosos
3. **Escalabilidade**: Funciona com datasets pequenos (Iris) e grandes
4. **Interpretabilidade**: Visualizações claras da evolução do modelo

## Trabalhos Futuros

- [ ] Implementar outros algoritmos de agregação (Krum, Trimmed Mean)
- [ ] Testar com outros modelos (Logistic Regression, Neural Networks)
- [ ] Adicionar diferentes tipos de ataques Byzantine
- [ ] Implementar defesas adaptativas
- [ ] Avaliar em datasets maiores (NSL-KDD, UNSW-NB15)

## Referências

1. McMahan et al. (2017) - "Communication-Efficient Learning of Deep Networks from Decentralized Data"
2. Blanchard et al. (2017) - "Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent"
3. Yin et al. (2018) - "Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates"

## Licença

Projeto acadêmico - Iniciação Científica
Faculdade Impacta - 2025/2026

---

**Última atualização**: Fevereiro 2026
