# MITIGAÇÃO DE ATAQUES POR ENVENENAMENTO EM APRENDIZADO FEDERADO

![Federated Learning Architecture](https://upload.wikimedia.org/wikipedia/commons/1/11/Centralized_federated_learning_protocol.png)

<div align="center">

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

![Status](https://img.shields.io/badge/Status-Active-success?style=flat-square)
![License](https://img.shields.io/badge/License-Academic-blue?style=flat-square)
![Python Version](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square)

</div>

---

## Sobre o Projeto

Este projeto de Iniciação Científica investiga estratégias de **detecção e mitigação de ataques por envenenamento** (poisoning attacks) em sistemas de **Aprendizado Federado** (Federated Learning - FL). 

O aprendizado federado é um paradigma emergente de machine learning distribuído que permite treinamento colaborativo de modelos preservando a privacidade dos dados. No entanto, sua natureza descentralizada o torna vulnerável a **ataques maliciosos de envenenamento**, onde agentes adversários manipulam o processo de treinamento enviando atualizações corrompidas.

Este estudo combina **revisão integrativa da literatura** com **implementações práticas** e **simulações computacionais** para avaliar técnicas defensivas baseadas em **detecção de anomalias**, contribuindo para o fortalecimento da segurança em sistemas de ML distribuído.

---

## Objetivos

### Objetivo Geral

Investigar e avaliar estratégias de detecção, prevenção e mitigação de ataques de envenenamento em sistemas de aprendizado federado através de técnicas de detecção de outliers e agregação robusta.

### Objetivos Específicos

1. **Analisar vulnerabilidades** do aprendizado federado a ataques por envenenamento de dados e modelos
2. **Investigar métodos** baseados em detecção de outliers para identificação de agentes maliciosos
3. **Implementar ataques reais** (sign flipping, gradient manipulation) em cenários controlados
4. **Simular arquiteturas federadas** com múltiplos clientes e servidor central
5. **Validar abordagens defensivas** através de métricas quantitativas e análise comparativa
6. **Avaliar impacto** de diferentes taxas de participação maliciosa no modelo global

---

## Estrutura do Repositório

```plaintext
.
├── README.md                              # Documentação principal
│
├── project/
│   ├── code/
│   │   ├── scripts-notebooks/             # Scripts Python principais
│   │   │   ├── run_federated_learning_bank_distribuido.py
│   │   │   ├── run_poisoning_attack_bank.py
│   │   │   ├── run_poisoning_attack_iris.py
│   │   │   ├── run_analise_por_classe_iris.py
│   │   │   ├── run_visualizacao_completa_poisoning.py
│   │   │   └── run_cyber_outlier_detection.py
│   │   │
│   │   └── scripts-datasets/              # Datasets organizados
│   │       ├── iris-dataset/
│   │       ├── penguin-dataset/
│   │       ├── cyber-outlier-detection/
│   │       └── nsl-kdd/
│   │
│   ├── data/                               # Datasets brutos
│   │   ├── bank-marketing/
│   │   ├── iris/
│   │   ├── cyber-outlier-detection/
│   │   └── nsl-kdd/
│   │
│   ├── notebooks/                          # Jupyter Notebooks
│   │   ├── iris/
│   │   ├── penguin/
│   │   ├── cyber-outlier-detection/
│   │   └── nsl-kdd/
│   │
│   ├── modelagem/                          # Modelagem e resultados
│   │   ├── apresentação/                   # Visualizações geradas
│   │   ├── pipeline/
│   │   └── primeiros_resultados/
│   │
│   ├── docs/                               # Documentação técnica
│   │   ├── 01-introducao/
│   │   ├── 02-fundamentacao/
│   │   ├── 03-arquitetura/
│   │   ├── 04-implementacao/
│   │   ├── 05-experimentos/
│   │   ├── 06-guias/
│   │   └── 07-referencias/
│   │
│   └── dependencies/
│       └── requirements.txt               # Dependências Python
```

---

## Metodologia

O projeto adota uma abordagem mista combinando pesquisa teórica, implementação prática e validação experimental:

### 1. Revisão Integrativa da Literatura

Revisão sistemática sobre:
- Ataques de envenenamento em aprendizado federado (data poisoning, model poisoning)
- Técnicas de mitigação baseadas em detecção de outliers
- Métodos de agregação robusta (FedAvg, Krum, Trimmed Mean, Median)
- Frameworks de segurança para sistemas distribuídos

### 2. Implementação de Arquiteturas Federadas

Desenvolvimento de sistemas de aprendizado federado com:
- **Servidor Central**: Coordena treinamento e agrega modelos
- **Múltiplos Clientes**: Treinamento local em dados distribuídos
- **Validação Global**: Conjunto independente no servidor para avaliação
- **Distribuição Estratificada**: Classes balanceadas entre clientes

### 3. Simulações de Ataques

Implementação de ataques reais:
- **Sign Flipping Attack**: Inversão e amplificação de pesos do modelo
- **Gradient Manipulation**: Corrupção de gradientes locais
- **Model Poisoning**: Envenenamento após treinamento local
- **Byzantine Attacks**: Clientes maliciosos com comportamento adversário

### 4. Técnicas de Detecção e Mitigação

Avaliação de múltiplas abordagens:
- **Detecção de Outliers**: Isolation Forest, LOF, One-Class SVM, Elliptic Envelope, DBSCAN
- **Agregação Robusta**: Filtragem de atualizações suspeitas
- **Análise Estatística**: Testes de hipótese e métricas de convergência

### 5. Validação Experimental

Análise através de múltiplos datasets:
- **Iris Dataset**: Classificação multi-classe (3 espécies, 150 amostras)
- **Bank Marketing**: Classificação binária com desbalanceamento (4,521 amostras)
- **Cyber Threat Intelligence**: Detecção de ameaças cibernéticas
- **NSL-KDD**: Detecção de intrusão em redes

**Métricas de Avaliação**:
- Acurácia, Precisão, Recall, F1-Score
- AUC-ROC, Loss
- Matriz de Confusão
- Degradação de Performance
- Análise por Classe

---

## Principais Resultados

### Poisoning Attack no Bank Marketing Dataset

Experimento com **aprendizado federado distribuído** demonstrando impacto de cliente malicioso:

**Configuração**:
- 3 clientes (1 malicioso = 33% de participação)
- Dataset: Bank Marketing (4,521 amostras, 2 classes)
- Ataque: Sign Flipping com taxa 90%
- Rodadas: 12 iterações federadas
- Agregação: FedAvg (média aritmética)

**Resultados**:

| Métrica | Normal | Envenenado | Degradação |
|---------|--------|------------|------------|
| **Acurácia Global** | 88.06% | 53.23% | **-34.84%** |
| **F1-Score** | 0.00% | 17.21% | -17.21% |
| **Precisão** | 0.00% | 11.07% | -11.07% |
| **Recall** | 0.00% | 35.56% | -35.56% |
| **AUC-ROC** | 0.4904 | 0.4661 | -0.0243 |
| **Loss** | 4.2818 | 6.8923 | +2.6105 |

**Severidade do Ataque**: **CRÍTICA** (degradação > 30%)

**Visualizações Geradas**:
- 7 gráficos completos de análise (convergência, matriz de confusão, impacto por classe, desempenho de clientes)

### Detecção de Outliers em Ameaças Cibernéticas

Avaliação de técnicas de detecção de anomalias aplicadas a ameaças cibernéticas:

| Técnica | Acurácia | Precisão | Recall | F1-Score |
|---------|----------|----------|--------|----------|
| **Elliptic Envelope** | **99.52%** | **97.62%** | **99.52%** | **98.56%** |
| Isolation Forest | 97.14% | 85.71% | 97.14% | 91.09% |
| Local Outlier Factor | 95.24% | 80.00% | 95.24% | 86.96% |
| One-Class SVM | 90.48% | 65.52% | 90.48% | 76.00% |
| DBSCAN | 85.71% | 55.56% | 85.71% | 67.57% |

**Conclusão**: Técnicas de detecção de outliers demonstraram alta eficácia (85-99% de acurácia) na identificação de agentes maliciosos, validando sua aplicabilidade como mecanismo de defesa em aprendizado federado.

### Poisoning Attack no Iris Dataset

Experimento clássico demonstrando impacto em classificação multi-classe:

**Configuração**:
- 3 espécies de íris (setosa, versicolor, virginica)
- 150 amostras, 4 features
- Ataque: Sign Flipping Attack

**Visualizações**:
- Análise de acurácia por classe
- Evolução de métricas por rodada
- Matriz de confusão evolutiva
- Tabela comparativa detalhada
- Impacto relativo por classe

---

## Tecnologias e Ferramentas

<div align="center">

### Linguagens e Frameworks

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

### Machine Learning

![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)

### Visualização

![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge&logo=python&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-3776AB?style=for-the-badge&logo=python&logoColor=white)

### Controle de Versão

![Git](https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=git&logoColor=white)
![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)

### Ambiente de Desenvolvimento

![VS Code](https://img.shields.io/badge/VS_Code-007ACC?style=for-the-badge&logo=visual-studio-code&logoColor=white)

</div>

**Bibliotecas Principais**:
- `scikit-learn`: Modelos de ML e detecção de outliers
- `pandas`, `numpy`: Manipulação e análise de dados
- `matplotlib`, `seaborn`: Visualização avançada
- `copy`, `abc`: Padrões de projeto (Deep Copy, Abstract Classes)

---

## Como Executar

### Pré-requisitos

```bash
Python 3.8+
pip
Git
```

### Instalação

```powershell
# Clone o repositório
git clone https://github.com/Ryanditko/IC-aprendizado-federado-e-machine-learning-em-cybersecurity.git
cd IC-aprendizado-federado-e-machine-learning-em-cybersecurity

# Navegue até a pasta de dependências
cd project/dependencies

# Instale as dependências
pip install -r requirements.txt
```

### Execução dos Experimentos

#### Aprendizado Federado com Bank Marketing

```powershell
cd project/code/scripts-notebooks

# Executa experimento completo com dados distribuídos
python run_federated_learning_bank_distribuido.py

# Gera 7 visualizações:
# - Comparação de métricas globais
# - Convergência por rodada
# - Análise por classe
# - Evolução de matrizes de confusão
# - Tabela comparativa
# - Impacto do ataque
# - Desempenho por cliente
```

#### Poisoning Attack no Iris Dataset

```powershell
# Ataque completo com análise detalhada
python run_poisoning_attack_iris.py

# Análise por classe
python run_analise_por_classe_iris.py

# Visualização completa
python run_visualizacao_completa_poisoning.py

# Demonstração com amostra reduzida
python run_amostra_poisoning_iris.py
```

#### Detecção de Outliers em Ameaças Cibernéticas

```powershell
# Avalia 5 técnicas de detecção de outliers
python run_cyber_outlier_detection.py
```

#### Notebooks Jupyter

```powershell
# Inicia Jupyter Notebook
cd project/notebooks
jupyter notebook

# Acesse:
# - iris/ : Experimentos com Iris
# - cyber-outlier-detection/ : Detecção de ameaças
# - nsl-kdd/ : Detecção de intrusão
```

---

## Arquitetura de Aprendizado Federado

### Componentes Principais

**Servidor Central (Agregador)**:
- Inicializa modelo global
- Coordena rodadas de treinamento
- Agrega modelos locais (FedAvg)
- Mantém conjunto de validação global
- Avalia performance do modelo agregado

**Clientes (Participantes)**:
- Recebem modelo global do servidor
- Treinam localmente em dados privados
- Enviam atualizações (pesos/gradientes) ao servidor
- **Cliente Malicioso**: Corrompe atualizações antes de enviar

**Pipeline de Ataque**:

```
1. Servidor → Envia modelo global aos clientes
2. Cliente Honesto → Treina localmente → Envia pesos legítimos
3. Cliente Malicioso → Treina localmente → CORROMPE pesos → Envia pesos envenenados
4. Servidor → Agrega todos os pesos (incluindo maliciosos) → Modelo global corrompido
5. Validação → Avalia degradação de performance
```

**Técnicas de Corrupção Implementadas**:
- **Sign Flipping**: Inverte sinal dos pesos e amplifica (`-w * 1.9`)
- **Gradient Manipulation**: Altera direção do gradiente
- **Random Noise**: Adiciona ruído gaussiano aos pesos

---

## Aplicação em Cibersegurança

### Problema Real

Em sistemas de detecção de ameaças federados (ex: antivírus distribuído, detecção de intrusão em IoT), **dispositivos comprometidos** podem enviar atualizações maliciosas que:
- Fazem o modelo ignorar ataques reais (False Negatives)
- Geram alertas falsos excessivos (False Positives)
- Degradam performance geral do sistema

### Solução Proposta

**Pipeline de Defesa**:

1. **Coleta de Atualizações**: Servidor recebe modelos de todos os clientes
2. **Extração de Features**: Converte pesos em vetores de características
3. **Detecção de Outliers**: Aplica Elliptic Envelope / Isolation Forest
4. **Filtragem**: Remove clientes identificados como outliers
5. **Agregação Robusta**: FedAvg apenas com clientes confiáveis
6. **Validação**: Verifica melhoria na performance global

**Técnicas Validadas**:

| Técnica | Vantagens | Desvantagens | Acurácia |
|---------|-----------|--------------|----------|
| **Elliptic Envelope** | Alta precisão, rápido | Assume distribuição gaussiana | 99.52% |
| **Isolation Forest** | Não paramétrico, escalável | Sensível a hiperparâmetros | 97.14% |
| **LOF** | Detecta outliers locais | Alto custo computacional | 95.24% |
| **One-Class SVM** | Robusto em alta dimensão | Difícil ajuste de parâmetros | 90.48% |
| **DBSCAN** | Sem pré-definir clusters | Sensível a densidade | 85.71% |

**Recomendação**: **Elliptic Envelope** para ambientes controlados, **Isolation Forest** para produção escalável.

---

## Visualizações e Análises

### Bank Marketing - Aprendizado Federado Distribuído

**7 visualizações geradas** (`project/modelagem/apresentação/`):

1. **bank_fl_distribuido_global.png**: Comparação de 6 métricas (Acurácia, F1, Precisão, Recall, AUC, Loss)
2. **bank_convergencia_por_rodada.png**: Evolução detalhada por rodada federada
3. **bank_analise_por_classe.png**: Performance separada por classe (Não/Sim depósito)
4. **bank_matriz_confusao_evolutiva.png**: Matrizes de confusão em 4 momentos (rodadas 1, 4, 8, 12)
5. **bank_tabela_comparativa.png**: Tabela resumo com degradações
6. **bank_impacto_ataque.png**: Análise de severidade e gauge de impacto
7. **bank_desempenho_clientes.png**: Acurácia, F1-Score e Loss por cliente (destaca malicioso)

### Iris Dataset - Poisoning Attack

**5 visualizações principais**:

1. **analise_acuracia_por_classe.png**: Acurácia comparativa por espécie
2. **analise_metricas_completas_por_classe.png**: 4 métricas por espécie
3. **analise_matriz_confusao_evolutiva.png**: Evolução temporal das matrizes
4. **analise_tabela_comparativa_por_classe.png**: Tabela detalhada por classe
5. **analise_impacto_relativo_por_classe.png**: Degradação percentual por espécie

### Amostra Reduzida - Demonstração

**5 visualizações para apresentação**:

1. **amostra_desempenho_clientes.png**: Comparação entre 3 clientes
2. **amostra_evolucao_metricas.png**: Convergência ao longo das rodadas
3. **amostra_impacto_por_rodada.png**: Degradação progressiva
4. **amostra_matriz_confusao_comparativa.png**: Normal vs Envenenado
5. **amostra_tabela_resumo.png**: Resumo executivo

---

## Documentação Técnica Completa

Para detalhes técnicos aprofundados, consulte `project/docs/`:

### Estrutura da Documentação

**01-introducao/**
- Contexto do problema
- Motivação e relevância
- Perguntas de pesquisa

**02-fundamentacao/**
- Aprendizado Federado
- Ataques de Envenenamento
- Detecção de Outliers
- Revisão da Literatura

**03-arquitetura/**
- Diagrama UML do sistema
- Componentes e interações
- Fluxo de dados

**04-implementacao/**
- Código-fonte comentado
- Padrões de projeto utilizados
- Decisões técnicas

**05-experimentos/**
- Metodologia experimental
- Resultados detalhados
- Análise estatística

**06-guias/**
- Guia de apresentação para o professor
- Storytelling dos experimentos
- Resumo rápido de gráficos

**07-referencias/**
- Bibliografia completa
- Papers referenciados
- Recursos adicionais

### Arquivos Especiais

- **EXPERIMENTO_ENVENENAMENTO.md**: Documentação completa do experimento principal
- **GUIA_APRESENTACAO_PROFESSOR.md**: Roteiro para defesa
- **STORYTELLING.md**: Narrativa dos resultados
- **RESUMO_RAPIDO_GRAFICOS.md**: Interpretação das visualizações

---

## Contribuições Científicas

Este projeto contribui para o avanço do conhecimento em:

### Segurança em Aprendizado Federado

- **Mapeamento de vulnerabilidades**: Identificação de pontos de ataque em arquiteturas federadas
- **Técnicas de mitigação**: Avaliação comparativa de métodos defensivos
- **Métricas de impacto**: Quantificação da severidade de ataques

### Detecção de Anomalias

- **Avaliação empírica**: Comparação de 5 técnicas de detecção de outliers
- **Aplicação em contexto federado**: Adaptação de métodos tradicionais para cenário distribuído
- **Análise de trade-offs**: Custo computacional vs. precisão

### Cibersegurança

- **Diretrizes práticas**: Recomendações para implementação segura de ML distribuído
- **Validação em datasets reais**: Ameaças cibernéticas, detecção de intrusão
- **Framework de defesa**: Pipeline completo de detecção e mitigação

### Machine Learning

- **Robustez de modelos**: Avaliação de degradação sob ataque
- **Agregação robusta**: Alternativas ao FedAvg tradicional
- **Análise por classe**: Impacto diferenciado em classes desbalanceadas

---

## Trabalhos Futuros

### Extensões Planejadas

1. **Defesas Avançadas**:
   - Implementação de Byzantine-Robust Aggregation (Krum, Trimmed Mean, Median)
   - Differential Privacy para proteção adicional
   - Reputação de clientes baseada em histórico

2. **Ataques Sofisticados**:
   - Backdoor Attacks (trojans em modelos)
   - Adaptive Attacks (adversários que aprendem com defesas)
   - Sybil Attacks (múltiplos clientes maliciosos coordenados)

3. **Datasets Adicionais**:
   - CICIDS2017/2018 (tráfego de rede)
   - UNSW-NB15 (ataques em rede)
   - Malware datasets (detecção de malware distribuída)

4. **Otimizações**:
   - Paralelização com Ray/Dask
   - Implementação em frameworks federados (TensorFlow Federated, PySyft)
   - GPU acceleration

5. **Validação em Produção**:
   - Deployment em ambiente real (IoT, edge devices)
   - Testes de escalabilidade (100+ clientes)
   - Análise de latência e comunicação

---

## Publicações e Apresentações

**Status**: Projeto em andamento (Iniciação Científica 2024-2026)

**Possíveis Venues**:
- Workshop de Iniciação Científica (Faculdade Impacta)
- Simpósio Brasileiro de Segurança da Informação (SBSeg)
- Workshop de Trabalhos de Iniciação Científica e de Graduação (WTICG - CSBC)

---

## Informações do Projeto

**Tipo**: Iniciação Científica  
**Instituição**: Faculdade Impacta de Tecnologia  
**Área**: Ciência da Computação / Cibersegurança / Machine Learning  
**Período**: 2024-2026

**Repositório**: [github.com/Ryanditko/IC-aprendizado-federado-e-machine-learning-em-cybersecurity](https://github.com/Ryanditko/IC-aprendizado-federado-e-machine-learning-em-cybersecurity)

---

## Licença

Este projeto é de natureza **acadêmica** e destinado a fins **educacionais e de pesquisa científica**.

---

<div align="center">

**Temas**: Segurança | Machine Learning | Federated Learning | Data Science | Cybersecurity

Projeto desenvolvido no âmbito do programa de Iniciação Científica

![GitHub Stars](https://img.shields.io/github/stars/Ryanditko/IC-aprendizado-federado-e-machine-learning-em-cybersecurity?style=social)
![GitHub Forks](https://img.shields.io/github/forks/Ryanditko/IC-aprendizado-federado-e-machine-learning-em-cybersecurity?style=social)

</div>
