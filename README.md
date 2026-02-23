# Mitigação De Ataques Por Envenamento Em Aprendizado Federado

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

## Estrutura do Projeto

O projeto está organizado em módulos que facilitam o desenvolvimento e a experimentação:

- **`code/`** - Implementações de algoritmos e experimentos
- **`data/`** - Datasets utilizados nos experimentos
- **`notebooks/`** - Análises exploratórias e protótipos
- **`modelagem/`** - Modelos desenvolvidos e resultados obtidos
- **`docs/`** - Documentação técnica e referências bibliográficas

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

## Principais Contribuições

Este projeto demonstra que **técnicas de detecção de outliers** são eficazes para identificar **agentes maliciosos** em sistemas de aprendizado federado, contribuindo para a segurança de sistemas de ML distribuído.

### Experimentos Implementados

#### Poisoning Attack em Datasets Reais

**Bank Marketing Dataset**:
- Experimento com aprendizado federado distribuído
- Simulação de cliente malicioso em ambiente multi-cliente
- Ataque: Sign Flipping Attack
- Demonstra impacto crítico na performance do modelo global
- Análise detalhada de degradação de métricas

**Iris Dataset**:
- Experimento clássico em classificação multi-classe
- Avaliação de impacto por classe
- Análise evolutiva ao longo das rodadas federadas
- Visualizações comparativas entre cenários normal e atacado

### Técnicas de Detecção Avaliadas

O projeto implementa e compara múltiplas técnicas de detecção de outliers:

- **Elliptic Envelope**: Modelo gaussiano multivariado
- **Isolation Forest**: Detecção baseada em isolamento aleatório
- **Local Outlier Factor (LOF)**: Análise de densidade local
- **One-Class SVM**: Fronteira de decisão em alta dimensão
- **DBSCAN**: Clustering baseado em densidade

**Aplicação**: Todas as técnicas foram validadas em contexto de detecção de ameaças cibernéticas e identificação de clientes maliciosos em aprendizado federado, demonstrando alta eficácia na proteção de sistemas distribuídos.

### Tipos de Ataques Simulados

- **Sign Flipping Attack**: Inversão e amplificação de pesos do modelo
- **Gradient Manipulation**: Corrupção de gradientes durante treinamento
- **Model Poisoning**: Envenenamento após treinamento local
- **Byzantine Attacks**: Comportamento adversário coordenado

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
- **Sign Flipping**: Inverte sinal dos pesos e amplifica por fator multiplicativo
- **Gradient Manipulation**: Altera direção do gradiente de descida
- **Random Noise**: Adiciona ruído gaussiano aos pesos do modelo

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
3. **Detecção de Outliers**: Aplica técnicas de detecção de anomalias
4. **Filtragem**: Remove clientes identificados como outliers
5. **Agregação Robusta**: FedAvg apenas com clientes confiáveis
6. **Validação**: Avalia performance do modelo global

**Técnicas de Detecção Comparadas**:

| Técnica | Vantagens | Desvantagens | Adequação |
|---------|-----------|--------------|-----------|
| **Elliptic Envelope** | Alta precisão, rápido | Assume distribuição gaussiana | Ambientes controlados |
| **Isolation Forest** | Não paramétrico, escalável | Sensível a hiperparâmetros | Produção escalável |
| **LOF** | Detecta outliers locais | Alto custo computacional | Datasets pequenos |
| **One-Class SVM** | Robusto em alta dimensão | Difícil ajuste de parâmetros | Alta dimensionalidade |
| **DBSCAN** | Sem pré-definir clusters | Sensível a densidade | Dados com clusters naturais |

**Resultados**: Todas as técnicas demonstraram eficácia na detecção de agentes maliciosos, com trade-offs entre precisão e custo computacional.

---

## Visualizações e Análises

O projeto gera visualizações detalhadas para análise dos experimentos, incluindo gráficos de convergência, matrizes de confusão evolutivas, comparações de métricas e análises por classe. Todas as visualizações são salvas automaticamente na pasta `project/modelagem/apresentação/`.

---

## Documentação Técnica

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

**Repositório**: [github.com/Ryanditko/federated-learning-at-cybersecurity](https://github.com/Ryanditko/federated-learning-at-cybersecurity)

---

## Licença

Este projeto é de natureza **acadêmica** e destinado a fins **educacionais e de pesquisa científica**.

---

<div align="center">

**Temas**: Segurança | Machine Learning | Federated Learning | Data Science | Cybersecurity

Projeto desenvolvido no âmbito do programa de Iniciação Científica

![GitHub Stars](https://img.shields.io/github/stars/Ryanditko/federated-learning-at-cybersecurity?style=social)
![GitHub Forks](https://img.shields.io/github/forks/Ryanditko/federated-learning-at-cybersecurity?style=social)

</div>
