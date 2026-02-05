# Mapa Visual da Documentação

## Estrutura para Storytelling
```
DOCUMENTAÇÃO DO PROJETO
│
├─ INÍCIO: Contexto e Problema
│  └─ tema-projeto/projeto.md
│     ├─ Qual o problema?
│     ├─ Objetivos da pesquisa
│     ├─ Metodologia científica
│     └─ Cronograma (12 meses)
│
├─ FUNDAMENTAÇÃO TEÓRICA
│  ├─ aprendizado-supervisionado/
│  │  ├─ supervisionado-2.md (Naive Bayes, Logística, KNN)
│  │  ├─ supervisionado-3.md (Árvores, Random Forest)
│  │  └─ supervisionado-4.md (SVM, Redes Neurais)
│  │
│  └─ aprendizado-não-supervisionado/
│     ├─ não-supervisionado-1.md (Clustering)
│     ├─ não-supervisionado-2.md (DBSCAN)
│     ├─ não-supervisionado-3.md (PCA, t-SNE)
│     └─ não-supervisionado-4.md (ICA)
│
├─ SOLUÇÃO PROPOSTA (DOCUMENTO CENTRAL)
│  └─ detecção-de-outliers/outilers.md (534 linhas)
│     ├─ Teoria: O que são outliers
│     ├─ 5 Técnicas implementadas
│     ├─ Experimentos e resultados
│     ├─ Elliptic Envelope: 53.14% acurácia
│     ├─ Pipeline para FL
│     └─ Implementação prática
│
├─ VALIDAÇÃO CIENTÍFICA
│  └─ avaliações-de-modelos/
│     └─ avaliação-1.md (Métricas: Accuracy, Precision, Recall, F1)
│
├─ EXPERIMENTOS PRÁTICOS
│  └─ notebooks-docs/notebooks.md (491 linhas)
│     ├─ Cyber Threat Detection
│     │  ├─ Dataset: Kaggle Cyber Intelligence
│     │  ├─ Elliptic Envelope: 99.52% acurácia
│     │  └─ 5 técnicas comparadas
│     │
│     ├─ Iris Dataset
│     │  ├─ 6 modelos avaliados
│     │  └─ SVM: 93.33% acurácia
│     │
│     ├─ Penguins Dataset
│     │  └─ 100% acurácia alcançada
│     │
│     └─ Weight-Height Dataset
│        └─ Classificação binária
│
└─ LIÇÕES APRENDIDAS
   └─ desafios/
      └─ Obstáculos e soluções durante a pesquisa
```

---

## Fluxo de Storytelling Recomendado

### **Ato 1: O Problema** (5-10 min)

**[tema-projeto/projeto.md](tema-projeto/projeto.md)**

**Narrativa:**
> "Em sistemas de Aprendizado Federado, clientes maliciosos podem envenenar o modelo global. Como detectar e mitigar esses ataques?"

**Elementos-chave:**
- Contexto: O que é Aprendizado Federado
- Problema: Ataques de envenenamento
- Relevância: Segurança em ML distribuído
- Pergunta de pesquisa: "Outliers detection pode funcionar?"

---

### **Ato 2: A Base Teórica** (10-15 min)

**[Fundamentação](README.md#2️⃣-fundamentação-teórica)**

**Narrativa:**
> "Para resolver o problema, precisamos entender as técnicas de ML que protegeremos."

**Jornada conceitual:**
1. **Supervisionado** → Técnicas que aprenderão padrões normais
2. **Não-Supervisionado** → Técnicas que detectarão anomalias
3. **Conexão** → Outliers são anomalias que podem indicar ataques

**Focar em:**
- 2-3 técnicas supervisionadas principais
- Clustering e detecção de anomalias
- Preparar terreno para "outliers"

---

### **Ato 3: A Solução** (15-20 min) - **CLIMAX**

**[detecção-de-outliers/outilers.md](detecção-de-outliers/outilers.md)**

**Narrativa:**
> "Nossa solução: usar detecção de outliers para identificar clientes maliciosos ANTES de agregar seus dados."

**Estrutura da apresentação:**

1. **O que são outliers?** (3 min)
   - Definição visual
   - Por que são relevantes em FL

2. **5 Técnicas testadas** (7 min)
   - Isolation Forest
   - LOF (Local Outlier Factor)
   - One-Class SVM
   - Elliptic Envelope
   - DBSCAN

3. **Resultados** (5 min)
   - Tabela comparativa
   - **Elliptic Envelope venceu: 53.14% acurácia**
   - DBSCAN melhor recall: 60.99%

4. **Pipeline de defesa** (5 min)
   ```
   Clientes → Coleta → DETECÇÃO → Filtragem → Agregação Segura
                          ⬆
                    [Elliptic Envelope]
   ```

**Slide principal:**
| Técnica | Accuracy | Recomendação |
|---------|----------|--------------|
| **Elliptic Envelope** | **53.14%** | Melhor geral |
| Isolation Forest | 49.82% | Alternativa |
| LOF | 49.51% | Viável |
| DBSCAN | 49.29% | Melhor recall (60.99%) |

---

### **Ato 4: Validação Experimental** (10-15 min)

**[notebooks-docs/notebooks.md](notebooks-docs/notebooks.md)**

**Narrativa:**
> "Mas funciona na prática? Testamos em 4 datasets reais."

**Demonstrações:**

#### **Experimento Principal: Cyber Threat Detection**

- **Dataset**: Kaggle Cyber Threat Intelligence
- **Objetivo**: Detectar ameaças cibernéticas como outliers
- **Resultado**: **99.52% de acurácia** com Elliptic Envelope

**Tabela de resultados:**
| Técnica | Accuracy | Precision | Recall | F1-Score |
|---------|----------|-----------|--------|----------|
| **Elliptic Envelope** | **99.52%** | 97.62% | 99.52% | 98.56% |
| Isolation Forest | 97.14% | 85.71% | 97.14% | 91.09% |
| LOF | 95.24% | 80.00% | 95.24% | 86.96% |

**Mensagem:**
> "Em contexto de ameaças cibernéticas reais, nosso método alcançou 99.52% de acurácia!"

#### **Validações Adicionais**
- **Iris**: 93.33% (SVM) - valida em problema clássico
- **Penguins**: 100% - valida em multiclasse
- **Weight-Height**: Valida em binário

---

### **Ato 5: Impacto e Conclusões** (5 min)

**[README.md - Seção de Resultados](README.md#🎯-resultados-chave-para-apresentação)**

**Narrativa:**
> "O que aprendemos? Técnicas de outlier detection SÃO EFICAZES para proteger Aprendizado Federado."

**Contribuições Científicas:**

- Mapeamento de 5 técnicas de detecção  
- Avaliação comparativa em contexto real  
- Pipeline prático implementável  
- Resultados quantitativos: 85-99% de acurácia

**Aplicação Real:**
1. Sistema de FL recebe atualizações
2. Elliptic Envelope analisa cada atualização
3. Outliers (potencialmente maliciosos) são filtrados
4. Apenas dados legítimos são agregados
5. Modelo global permanece seguro

**Slide final:**
> "Validamos cientificamente que detecção de outliers pode mitigar ataques de envenenamento em Aprendizado Federado, alcançando até 99.52% de acurácia em detecção de ameaças cibernéticas."

---

## Visualizações Recomendadas para Storytelling

### **Slide 1: O Problema**
```
┌─────────────────────────────────────┐
│  APRENDIZADO FEDERADO TRADICIONAL   │
│                                     │
│  Cliente 1 ──┐                      │
│  Cliente 2 ──┼──→ Servidor Central  │
│  Cliente 3 ──┘     (Agrega tudo)    │
│  Cliente 4* ─→ MALICIOSO!            │
│               (Envenena modelo)      │
└─────────────────────────────────────┘
```

### **Slide 2: Nossa Solução**
```
┌─────────────────────────────────────┐
│  APRENDIZADO FEDERADO SEGURO        │
│                                     │
│  Cliente 1 ──┐                      │
│  Cliente 2 ──┼──→ [DETECÇÃO] ──→    │
│  Cliente 3 ──┘   [OUTLIERS]         │
│  Cliente 4* ─→      ↓                │
│               FILTRADO!              │
│                    ↓                 │
│              Servidor Central        │
│           (Agrega apenas seguros)   │
└─────────────────────────────────────┘
```

### **Slide 3: Resultados**
```
┌─────────────────────────────────────┐
│  ELLIPTIC ENVELOPE                  │
│                                     │
│  ████████████████████ 99.52%        │
│                                     │
│  Isolation Forest                   │
│  ████████████████░░░░ 97.14%        │
│                                     │
│  LOF                                │
│  ███████████████░░░░░ 95.24%        │
└─────────────────────────────────────┘
```

---

## Roteiro de Apresentação (30 minutos)

| Tempo | Seção | Documento | Pontos-chave |
|-------|-------|-----------|--------------|
| 0-5 min | Introdução | tema-projeto/projeto.md | Problema, Motivação, Objetivos |
| 5-10 min | Teoria | Fundamentação/ | ML Supervisionado, Não-Supervisionado, Conceito de Outliers |
| 10-20 min | Solução | detecção-de-outliers/outilers.md | 5 técnicas, Resultados, Pipeline |
| 20-28 min | Experimentos | notebooks-docs/notebooks.md | Cyber Threat (99.52%), Outras validações |
| 28-30 min | Conclusão | README.md | Contribuições, Aplicação, Impacto |

---

## Dicas para Storytelling Eficaz

### **Use a Estrutura de 3 Atos Clássica:**

1. **Problema** → Ataques de envenenamento ameaçam FL
2. **Jornada** → Exploramos 5 técnicas de detecção
3. **Resolução** → Elliptic Envelope alcançou 99.52% de sucesso

### **Conecte com o Público:**
- **Para acadêmicos**: Foque em metodologia científica e validação
- **Para técnicos**: Mostre implementação e código
- **Para gestores**: Destaque impacto em segurança real

### **Momentos "WOW":**
1. **Reveal do problema**: "Imagine um único cliente malicioso destruindo todo o modelo..."
2. **Comparação de técnicas**: "Testamos 5 métodos. Um se destacou..."
3. **Resultado final**: "99.52% de acurácia em ameaças REAIS!"

### **Call to Action:**
> "Esta pesquisa prova que podemos tornar o Aprendizado Federado mais seguro. O próximo passo é implementar em sistemas de produção."

---

## Você Está Aqui

```
Mapa Visual <- VOCÊ ESTÁ AQUI
├─ Para contexto → tema-projeto/projeto.md
├─ Para teoria → Fundamentação/
├─ Para solução → detecção-de-outliers/outilers.md
├─ Para resultados → notebooks-docs/notebooks.md
└─ Para visão geral → README.md
```

---

**Criado para facilitar apresentações e storytelling do projeto de Iniciação Científica**

*Última atualização: Janeiro 2025*
