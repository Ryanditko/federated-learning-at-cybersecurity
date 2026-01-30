# Documentação do Projeto - Aprendizado Federado e Segurança

> **Navegação Estruturada para Storytelling do Projeto de Iniciação Científica**

---

## Visão Geral do Projeto

Este projeto de Iniciação Científica investiga *------

**Navegação Rápida:**
- [Proposta](tema-projeto/projeto.md) | [Outliers](detecção-de-outliers/outilers.md) | [Notebooks](notebooks-docs/notebooks.md)
- [Supervisionado](aprendizado-supervisionado/) | [Não-Supervisionado](aprendizado-não-supervisionado/) | [Avaliação](avaliações-de-modelos/)

## Licençacnicas de mitigação de ataques por envenenamento em Aprendizado Federado** utilizando detecção de outliers e machine learning para segurança cibernética.

**Área**: Ciência da Computação | Cibersegurança | Machine Learning  
**Instituição**: Faculdade Impacta  
**Tipo**: Pesquisa Científica Aplicada

---

## Guia de Leitura (Storytelling)

Para compreender o projeto de forma narrativa e progressiva, siga esta sequência:

### 1. **Contexto e Motivação**

**[Proposta do Projeto](tema-projeto/projeto.md)**
- Problema científico e relevância
- Objetivos gerais e específicos
- Metodologia de pesquisa
- Cronograma de execução (12 meses)

**Por que começar aqui?** Contextualiza o problema de segurança em aprendizado federado e estabelece a base científica do projeto.

---

### 2. **Fundamentação Teórica**

#### Aprendizado Supervisionado
Técnicas de classificação e predição com rótulos conhecidos:

- **[Supervisionado - Parte 2](aprendizado-supervisionado/supervisionado-2.md)**: Naive Bayes, Regressão Logística, KNN
- **[Supervisionado - Parte 3](aprendizado-supervisionado/supervisionado-3.md)**: Árvores de Decisão, Random Forest
- **[Supervisionado - Parte 4](aprendizado-supervisionado/supervisionado-4.md)**: SVM, Redes Neurais

**Por que ler?** Compreender os algoritmos de ML que serão protegidos no contexto de aprendizado federado.

#### Aprendizado Não-Supervisionado
Técnicas de descoberta de padrões sem rótulos:

- **[Não-Supervisionado - Parte 1](aprendizado-não-supervisionado/não-supervisionado-1.md)**: Clustering (K-Means, Hierárquico)
- **[Não-Supervisionado - Parte 2](aprendizado-não-supervisionado/não-supervisionado-2.md)**: DBSCAN, Mean-Shift
- **[Não-Supervisionado - Parte 3](aprendizado-não-supervisionado/não-supervisionado-3.md)**: Redução de Dimensionalidade (PCA, t-SNE)
- **[Não-Supervisionado - Parte 4](aprendizado-não-supervisionado/não-supervisionado-4.md)**: Análise de Componentes Independentes

**Por que ler?** Base para entender técnicas de detecção de anomalias, fundamentais para segurança.

---

### 3. **Solução Proposta: Detecção de Outliers**

**[Detecção de Outliers em Aprendizado Federado](detecção-de-outliers/outilers.md)** - **DOCUMENTO CENTRAL**

**Conteúdo Completo (534 linhas):**
1. **Fundamentação Teórica**: O que são outliers e por que são críticos em FL
2. **5 Técnicas Avaliadas**:
   - Isolation Forest
   - Local Outlier Factor (LOF)
   - One-Class SVM
   - Elliptic Envelope - **Melhor resultado (53.14% acurácia)**
   - DBSCAN - **Melhor recall (60.99%)**
3. **Metodologia Experimental**: Dataset Kaggle de ameaças cibernéticas
4. **Resultados Comparativos**: Tabelas e análises de performance
5. **Aplicação em FL**: Pipeline de defesa contra envenenamento
6. **Implementação**: Pseudocódigo e diretrizes práticas
7. **Trade-offs**: Quando usar cada técnica
8. **Trabalhos Futuros**: Extensões e melhorias

**Por que este é o documento central?** Apresenta a solução completa para o problema de pesquisa, com teoria, experimentos e aplicação prática.

---

### 4. **Avaliação de Modelos**

**Metodologias de Avaliação**:

- **[Avaliação - Parte 1](avaliações-de-modelos/avaliação-1.md)**: Métricas básicas (Accuracy, Precision, Recall, F1-Score)
- **[Outras avaliações](avaliações-de-modelos/)**: Validação cruzada, curvas ROC, etc.

**Por que ler?** Compreender como os modelos de detecção foram validados cientificamente.

---

### 5. **Experimentos Práticos**

**[Documentação dos Notebooks](notebooks-docs/notebooks.md)** (491 linhas)

**Análises Realizadas:**

#### **Cyber Threat Outlier Detection** (Análise Principal)
- **Dataset**: Kaggle Cyber Threat Intelligence
- **Técnica Vencedora**: Elliptic Envelope
- **Resultado**: **99.52% de acurácia** (97.62% precision, 99.52% recall)
- **Outros métodos testados**: Isolation Forest (97.14%), LOF (95.24%), One-Class SVM (90.48%), DBSCAN (85.71%)
- **Visualizações**: Scatter plots, decision boundaries, confusion matrices

#### **Iris Dataset** (Validação em Problema Clássico)
- **Dataset**: 150 amostras, 3 espécies
- **Modelos avaliados**: 6 algoritmos supervisionados
- **Melhor resultado**: SVM com **93.33% de acurácia**
- **Objetivo**: Validar técnicas em problema benchmark

#### **Penguins Dataset** (Classificação Multiclasse)
- **Dataset**: 333 pinguins, 3 espécies
- **Resultado**: **100% de acurácia** em múltiplos modelos
- **Features**: Medidas físicas (bico, nadadeiras, massa)

#### **Weight-Height Dataset** (Classificação Binária)
- **Objetivo**: Predição de gênero baseada em peso/altura
- **Tipo**: Classificação binária simples
- **Features**: 2 variáveis numéricas

**Por que ler?** Ver a aplicação prática das técnicas teóricas em datasets reais, com resultados mensuráveis.

---

### 6. **Desafios e Lições Aprendidas**

**[Desafios do Projeto](desafios/)**

Obstáculos enfrentados durante a pesquisa e como foram superados.

**Por que ler?** Compreender o processo científico real, incluindo dificuldades e soluções.

---

## Resultados-Chave (Para Apresentação)

### Tabela de Performance - Detecção de Ameaças Cibernéticas

| Técnica | Accuracy | Precision | Recall | F1-Score | **Recomendação** |
|---------|----------|-----------|--------|----------|------------------|
| **Elliptic Envelope** | **99.52%** | **97.62%** | **99.52%** | **98.56%** | Melhor balanço geral |
| Isolation Forest | 97.14% | 85.71% | 97.14% | 91.09% | Bom trade-off |
| LOF | 95.24% | 80.00% | 95.24% | 86.96% | Alternativa viável |
| One-Class SVM | 90.48% | 65.52% | 90.48% | 76.00% | Baixa precisão |
| DBSCAN | 85.71% | 55.56% | 85.71% | 67.57% | Baixa precisão |

**Conclusão Científica**: Técnicas de detecção de outliers demonstraram **alta eficácia (85-99% de acurácia)** na identificação de agentes maliciosos, validando sua aplicabilidade como mecanismo de defesa em aprendizado federado.

---

## Recursos Adicionais

### Estrutura do Repositório
```
project/
├── code/              # Scripts Python implementados
├── data/              # Datasets utilizados nos experimentos
├── notebooks/         # Jupyter Notebooks das análises
├── docs/              # Você está aqui
└── dependencies/      # requirements.txt
```

### Tecnologias Utilizadas
- **Python 3.8+** (Linguagem principal)
- **scikit-learn** (Machine Learning)
- **pandas, numpy** (Manipulação de dados)
- **matplotlib, seaborn** (Visualizações)
- **Jupyter Notebook** (Ambiente de análise)

### Como Executar os Experimentos

**Instalar dependências:**
```powershell
pip install -r project/dependencies/requirements.txt
```

**Executar notebooks:**
```powershell
cd project/notebooks
jupyter notebook
```

**Executar scripts:**
```powershell
python project/code/scripts-notebooks/run_cyber_outlier_detection.py
```

---

## Aplicação em Aprendizado Federado

### **Pipeline de Defesa Proposto:**

1. **Coleta de Atualizações** → Clientes enviam gradientes ao servidor central
2. **Detecção de Outliers** → Aplicação do Elliptic Envelope
3. **Filtragem** → Exclusão de agentes suspeitos (outliers)
4. **Agregação Robusta** → Combinação apenas de gradientes legítimos
5. **Atualização do Modelo Global** → Distribuição da nova versão segura

### **Contribuições Científicas:**

- **Mapeamento de técnicas** de detecção de outliers para segurança em FL  
- **Avaliação comparativa** de 5 métodos em contexto de ameaças cibernéticas  
- **Diretrizes práticas** para implementação de defesas robustas  
- **Validação experimental** com métricas de performance quantitativas

---

## Como Usar Esta Documentação

### Para **Apresentações Acadêmicas**:
1. Comece com [Proposta do Projeto](tema-projeto/projeto.md) (contexto)
2. Explique a [Detecção de Outliers](detecção-de-outliers/outilers.md) (solução)
3. Mostre os [Resultados dos Notebooks](notebooks-docs/notebooks.md) (validação)
4. Conclua com a tabela de performance acima

### Para **Compreensão Técnica Profunda**:
Leia na ordem do **Guia de Leitura** acima (1️⃣ → 6️⃣)

### Para **Implementação Prática**:
1. Vá direto para [Detecção de Outliers - Seção de Implementação](detecção-de-outliers/outilers.md#implementação)
2. Consulte os [Notebooks](notebooks-docs/notebooks.md) para exemplos de código
3. Execute os scripts em `project/code/`

---

## Informações do Projeto

**Pesquisador**: Ryan  
**Instituição**: Faculdade Impacta  
**Programa**: Iniciação Científica  
**Área**: Ciência da Computação / Cibersegurança  
**Período**: 2024-2025 (12 meses)

---

## � Licença

Projeto acadêmico destinado a fins educacionais e de pesquisa científica.

---

**🌟 Navegação Rápida:**
- � [Proposta](tema-projeto/projeto.md) | 🛡️ [Outliers](detecção-de-outliers/outilers.md) | 🧪 [Notebooks](notebooks-docs/notebooks.md)
- 🧠 [Supervisionado](aprendizado-supervisionado/) | 🔍 [Não-Supervisionado](aprendizado-não-supervisionado/) | 📊 [Avaliação](avaliações-de-modelos/)

---

*Documentação organizada para storytelling científico - Iniciação Científica 2024/2025*

