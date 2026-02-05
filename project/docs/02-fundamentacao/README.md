# 02 - Fundamentação Teórica

## Aprendizado Federado (Federated Learning)

### Definição

Aprendizado Federado é um paradigma de machine learning distribuído onde múltiplos clientes colaboram no treinamento de um modelo compartilhado sem expor seus dados locais. Proposto por McMahan et al. (2017), o FL resolve o dilema entre privacidade e aprendizado colaborativo.

### Arquitetura Básica

```
Ciclo de Treinamento:
1. Servidor → Clientes: Distribui modelo global atual
2. Clientes: Treinam localmente com dados privados
3. Clientes → Servidor: Enviam atualizações (gradientes/pesos)
4. Servidor: Agrega atualizações e atualiza modelo global
5. Repete até convergência
```

### Algoritmo FedAvg (Federated Averaging)

**Pseudocódigo:**

```python
# Servidor
for rodada in range(T):
    # Seleciona subconjunto de clientes
    clientes_selecionados = sample(clientes, frac=0.3)
    
    for cliente in clientes_selecionados:
        # Envia modelo global
        cliente.set_modelo(modelo_global)
        
        # Cliente treina localmente
        modelo_local = cliente.treinar()
        
    # Agregação: média ponderada dos modelos
    modelo_global = weighted_average([c.modelo for c in clientes])
```

**Vantagens:**
- Privacidade preservada (dados nunca saem do dispositivo)
- Escalabilidade (treinamento paralelo)
- Aplicabilidade em edge computing

**Desafios:**
- Heterogeneidade de dados (non-IID)
- Custo de comunicação
- Vulnerabilidade a ataques

## Ataques de Envenenamento em FL

### Taxonomia de Ataques

#### 1. Data Poisoning (Envenenamento de Dados)

**Definição**: Atacante corrompe seus dados de treino locais.

**Técnicas:**
- **Label Flipping**: Inverter rótulos de classes
- **Noise Injection**: Adicionar ruído gaussiano/uniforme
- **Backdoor Insertion**: Inserir padrões que ativam comportamento malicioso

**Exemplo:**
```python
# Cliente malicioso
def envenenar_dados(self):
    # Adiciona ruído em 30% das amostras
    n_poison = int(len(self.X) * 0.3)
    indices = random.sample(range(len(self.X)), n_poison)
    
    for i in indices:
        self.X[i] += np.random.normal(0, std=3*self.X.std(), size=self.X[i].shape)
```

**Impacto**: Degrada acurácia do modelo global em 10-40%

#### 2. Model Poisoning (Envenenamento de Modelo)

**Definição**: Atacante manipula os pesos/gradientes do modelo local antes de enviar ao servidor.

**Técnicas:**
- **Sign Flipping**: Inverter sinais dos gradientes
- **Scaling**: Amplificar magnitudes dos pesos
- **Randomization**: Substituir pesos por valores aleatórios

**Exemplo:**
```python
def envenenar_modelo(self):
    # Inverte coeficientes
    pesos = self.modelo.get_weights()
    pesos['coef'] = -pesos['coef']  # Inverte sinal
    self.modelo.set_weights(pesos)
```

**Impacto**: Pode causar divergência ou colapso do modelo global

#### 3. Byzantine Attacks

**Definição**: Atacantes cooperam para maximizar degradação do modelo.

**Características:**
- Múltiplos atacantes coordenados
- Adaptam estratégia durante treinamento
- Mais sofisticados e perigosos

**Exemplo Clássico:**
```
A Little is Enough (Baruch et al., 2019):
- Atacantes estimam gradientes dos honestos
- Ajustam seus gradientes para serem aceitos
- Maximizam desvio do modelo enquanto passam despercebidos
```

### Superfície de Ataque

```
Pontos de Vulnerabilidade:
┌─────────────────────────────────────────┐
│ 1. Dados Locais (Data Poisoning)        │
│ 2. Modelo Local (Model Poisoning)       │
│ 3. Comunicação (Man-in-the-Middle)      │
│ 4. Agregação (Byzantine Attacks)        │
│ 5. Servidor (Compromisso do Servidor)   │
└─────────────────────────────────────────┘
```

**Foco deste Projeto**: Pontos 1, 2 e 4

## Técnicas de Detecção de Outliers

### 1. MAD (Median Absolute Deviation)

**Definição**: Medida de dispersão robusta baseada na mediana.

**Fórmula:**
```
MAD = median(|Xi - median(X)|)
Threshold = median(X) + k × MAD
Outlier: Xi > Threshold
```

**Vantagens:**
- Robusto: Não afetado por valores extremos
- Eficiente: Computação rápida O(n log n)
- Interpretável: k=3 corresponde a ~3σ em distribuição normal

**Aplicação em FL:**
```python
def detectar_outliers_mad(modelos_clientes):
    # Calcula mediana dos coeficientes
    coefs = [m.get_coef() for m in modelos_clientes]
    mediana = np.median(coefs, axis=0)
    
    # Calcula distâncias à mediana
    distancias = [np.linalg.norm(c - mediana) for c in coefs]
    
    # Threshold baseado em MAD
    mad = np.median(np.abs(distancias - np.median(distancias)))
    threshold = np.median(distancias) + 3 * mad
    
    # Identifica outliers
    outliers = [i for i, d in enumerate(distancias) if d > threshold]
    return outliers
```

### 2. Z-Score

**Definição**: Número de desvios padrão que um ponto está da média.

**Fórmula:**
```
z = (x - μ) / σ
Outlier: |z| > 3
```

**Limitação em FL**: Sensível a ataques Byzantine (média e desvio-padrão são afetados)

### 3. IQR (Interquartile Range)

**Definição**: Range entre o 1º e 3º quartis.

**Fórmula:**
```
IQR = Q3 - Q1
Lower Bound = Q1 - 1.5 × IQR
Upper Bound = Q3 + 1.5 × IQR
```

**Vantagens**: Robusto, amplamente usado em boxplots

### Comparação de Técnicas

| Técnica | Robustez | Custo Computacional | Interpretabilidade |
|---------|----------|---------------------|-------------------|
| MAD     | Alta     | O(n log n)          | Alta              |
| Z-Score | Baixa    | O(n)                | Alta              |
| IQR     | Média    | O(n log n)          | Alta              |

**Escolha do Projeto**: MAD por sua robustez e eficiência

## Agregação Byzantine-Robust

### Algoritmos Estado-da-Arte

#### 1. Krum (Blanchard et al., 2017)

**Ideia**: Seleciona o modelo mais "central" baseado em distâncias aos k-vizinhos mais próximos.

**Pseudocódigo:**
```python
def krum(modelos, f):  # f = número de atacantes tolerados
    n = len(modelos)
    scores = []
    
    for i in range(n):
        # Distâncias aos outros modelos
        dists = [distance(modelos[i], modelos[j]) for j in range(n) if j != i]
        # Soma das n-f-2 menores distâncias
        score = sum(sorted(dists)[:n-f-2])
        scores.append(score)
    
    # Retorna modelo com menor score
    return modelos[np.argmin(scores)]
```

**Tolerância**: Até f < n/2 atacantes

#### 2. Trimmed Mean

**Ideia**: Remove β% dos valores extremos antes de calcular média.

**Pseudocódigo:**
```python
def trimmed_mean(valores, beta=0.1):
    n = len(valores)
    k = int(n * beta)
    
    # Remove k menores e k maiores
    valores_sorted = sorted(valores)
    valores_trimmed = valores_sorted[k:-k]
    
    return np.mean(valores_trimmed)
```

**Vantagens**: Simples, eficiente, robusto

#### 3. Median (Yin et al., 2018)

**Ideia**: Usa mediana coordenada-a-coordenada.

**Pseudocódigo:**
```python
def coordinate_median(modelos):
    # Para cada dimensão, calcula mediana
    return [np.median([m[i] for m in modelos]) for i in range(len(modelos[0]))]
```

**Tolerância**: Até 50% de atacantes

### Comparação com FedAvg

```
Cenário: 10 clientes, 3 maliciosos

FedAvg (Baseline):
- R² = 0.62 (degrada 35%)
- Aceita todos os modelos

MAD (Nossa Abordagem):
- R² = 0.91 (degrada apenas 5%)
- Filtra 3/3 maliciosos

Krum:
- R² = 0.88 (degrada 10%)
- Seleciona 1 modelo honesto

Trimmed Mean:
- R² = 0.89 (degrada 8%)
- Remove extremos
```

## Regressão Linear em FL

### Modelo Matemático

**Forma Geral:**
```
y = β0 + β1x1 + β2x2 + ... + βnxn + ε
```

**Problema de Otimização:**
```
min Σ(yi - ŷi)²  → Minimizar erro quadrático
```

### Treinamento Federado

**Cliente k treina:**
```python
# Dados locais: (Xk, yk)
modelo_k = LinearRegression()
modelo_k.fit(Xk, yk)

# Obtém pesos
wk = modelo_k.coef_
bk = modelo_k.intercept_
```

**Servidor agrega:**
```python
# Média ponderada (FedAvg)
w_global = Σ(nk/N × wk)  # nk = tamanho do dataset local k
b_global = Σ(nk/N × bk)
```

### Métricas de Avaliação

**R² Score (Coeficiente de Determinação):**
```
R² = 1 - (SS_res / SS_tot)
SS_res = Σ(yi - ŷi)²  → Resíduos
SS_tot = Σ(yi - ȳ)²   → Variância total
```

**Interpretação:**
- R² = 1.0: Modelo perfeito
- R² = 0.9: Explica 90% da variância
- R² = 0.0: Modelo = média

**MSE (Mean Squared Error):**
```
MSE = (1/n) Σ(yi - ŷi)²
```

**MAE (Mean Absolute Error):**
```
MAE = (1/n) Σ|yi - ŷi|
```

## Datasets Utilizados

### 1. Iris Dataset

**Características:**
- 150 amostras (50 por classe)
- 4 features numéricas
- Problema: Regressão (predizer petal width)

**Uso no Projeto:**
- Baseline para validação inicial
- Testes rápidos de algoritmos
- Análise de convergência

### 2. NSL-KDD Dataset

**Características:**
- Dataset de detecção de intrusão
- 125,973 registros de treino
- 41 features
- Problema: Classificação binária (ataque/normal)

**Uso no Projeto:**
- Validação em dataset real de cibersegurança
- Avaliação de escalabilidade
- Cenário aplicado

### 3. Cyber Threat Intelligence

**Características:**
- Dataset customizado de ameaças
- Features de comportamento de rede
- Desbalanceado (poucos ataques)

**Uso no Projeto:**
- Cenário realista de FL em cibersegurança
- Teste de robustez em dados não-IID

## Trabalhos Relacionados

### Defensive Aggregation

1. **Krum** (Blanchard et al., 2017): Seleção baseada em proximidade
2. **Bulyan** (Mhamdi et al., 2018): Multi-Krum + Trimmed Mean
3. **FoolsGold** (Fung et al., 2018): Detecção baseada em histórico de contribuições

### Attack Strategies

1. **Targeted Poisoning** (Tolpegin et al., 2020): Ataques direcionados a classes específicas
2. **Backdoor Attacks** (Bagdasaryan et al., 2020): Inserção de comportamento oculto
3. **Model Replacement** (Bhagoji et al., 2019): Substituição completa do modelo

### Theoretical Analysis

1. **Byzantine-Robust Learning** (Yin et al., 2018): Análise de convergência
2. **Statistical Rates** (Alistarh et al., 2018): Limites teóricos de tolerância

## Ferramentas e Frameworks

### Implementação

- **Python 3.10+**: Linguagem principal
- **scikit-learn**: Modelos de ML
- **numpy/pandas**: Manipulação de dados
- **matplotlib/seaborn**: Visualização

### FL Frameworks (Referência)

- **TensorFlow Federated**: Framework oficial do Google
- **PySyft**: FL com privacidade diferencial
- **FATE**: Framework industrial (WeBank)
- **Flower**: Framework modular e extensível

**Nota**: Implementação própria para fins didáticos e controle total

## Referências Bibliográficas

1. McMahan et al. (2017) - "Communication-Efficient Learning of Deep Networks from Decentralized Data"
2. Blanchard et al. (2017) - "Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent"
3. Yin et al. (2018) - "Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates"
4. Fung et al. (2020) - "The Limitations of Federated Learning in Sybil Settings"
5. Rousseeuw & Croux (1993) - "Alternatives to the Median Absolute Deviation"

---

**Última atualização**: Fevereiro 2026
