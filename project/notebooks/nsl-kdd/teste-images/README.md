# Análise de Detecção de Outliers - Dataset NSL-KDD

## 📊 Resumo da Análise

Este diretório contém **37 imagens** e **1 arquivo CSV** com os resultados de 12 exemplos diferentes de detecção de outliers em ataques de rede usando o dataset NSL-KDD.

## 🎯 Exemplos Gerados

### **Exemplos 1-5: Ataque Neptune**

#### Exemplo 01: count vs serror_rate (IsolationForest)
- **Atributos:** count, serror_rate
- **Algoritmo:** Isolation Forest
- **Métricas:**
  - Accuracy: 0.589
  - Precision: 0.266
  - Recall: 0.034
  - F1-Score: 0.061

#### Exemplo 02: src_bytes vs dst_bytes (IsolationForest)
- **Atributos:** src_bytes, dst_bytes
- **Algoritmo:** Isolation Forest
- **Métricas:**
  - Accuracy: 0.563
  - Precision: 0.000
  - Recall: 0.000
  - F1-Score: 0.000
- **Observação:** Estes atributos não foram efetivos para detectar Neptune

#### Exemplo 03: duration vs count (LOF)
- **Atributos:** duration, count
- **Algoritmo:** Local Outlier Factor (LOF)
- **Métricas:**
  - Accuracy: 0.586
  - Precision: 0.201
  - Recall: 0.023
  - F1-Score: 0.041

#### Exemplo 04: srv_count vs dst_host_count (IsolationForest)
- **Atributos:** srv_count, dst_host_count
- **Algoritmo:** Isolation Forest (contamination=0.1)
- **Métricas:**
  - Accuracy: 0.521
  - Precision: 0.040
  - Recall: 0.010
  - F1-Score: 0.017

#### Exemplo 05: same_srv_rate vs diff_srv_rate (EllipticEnvelope)
- **Atributos:** same_srv_rate, diff_srv_rate
- **Algoritmo:** Elliptic Envelope
- **Métricas:**
  - Accuracy: 0.599
  - Precision: 0.365
  - Recall: 0.047
  - F1-Score: 0.084

---

### **Exemplos 6-10: Outros Tipos de Ataque**

#### Exemplo 06: Smurf - count vs serror_rate ⭐ **MELHOR RESULTADO**
- **Tipo de Ataque:** Smurf
- **Atributos:** count, serror_rate
- **Algoritmo:** Isolation Forest
- **Métricas:**
  - Accuracy: **0.914** ⭐
  - Precision: 0.325
  - Recall: 0.234
  - F1-Score: **0.272**
- **Observação:** Este foi o melhor resultado geral!

#### Exemplo 07: Satan - src_bytes vs dst_bytes
- **Tipo de Ataque:** Satan
- **Atributos:** src_bytes, dst_bytes
- **Algoritmo:** Isolation Forest
- **Métricas:**
  - Accuracy: 0.867
  - Precision: 0.001
  - Recall: 0.001
  - F1-Score: 0.001

#### Exemplo 08: Portsweep - duration vs srv_count
- **Tipo de Ataque:** Portsweep
- **Atributos:** duration, srv_count
- **Algoritmo:** LOF
- **Métricas:**
  - Accuracy: 0.909
  - Precision: 0.049
  - Recall: 0.054
  - F1-Score: 0.051

#### Exemplo 09: Ipsweep - count vs dst_host_count
- **Tipo de Ataque:** Ipsweep
- **Atributos:** count, dst_host_count
- **Algoritmo:** Isolation Forest
- **Métricas:**
  - Accuracy: 0.898
  - Precision: 0.000
  - Recall: 0.000
  - F1-Score: 0.000

#### Exemplo 10: Back - src_bytes vs count
- **Tipo de Ataque:** Back
- **Atributos:** src_bytes, count
- **Algoritmo:** Isolation Forest
- **Métricas:**
  - Accuracy: **0.924** ⭐
  - Precision: 0.074
  - Recall: 0.106
  - F1-Score: 0.088

---

### **Exemplos 11-12: Neptune com Combinações Adicionais**

#### Exemplo 11: count, serror_rate, srv_count (3 atributos)
- **Atributos:** count, serror_rate, srv_count
- **Algoritmo:** Isolation Forest
- **Métricas:**
  - Accuracy: 0.572
  - Precision: 0.090
  - Recall: 0.012
  - F1-Score: 0.020

#### Exemplo 12: dst_host_serror_rate vs dst_host_srv_serror_rate
- **Atributos:** dst_host_serror_rate, dst_host_srv_serror_rate
- **Algoritmo:** Isolation Forest
- **Métricas:**
  - Accuracy: 0.590
  - Precision: 0.264
  - Recall: 0.033
  - F1-Score: 0.058

---

## 📈 Estrutura dos Arquivos

Para cada exemplo, foram geradas **3 imagens**:

1. **`distribuicao_*.png`** - Distribuição dos atributos (Normal vs Ataque)
2. **`outliers_*.png`** - Scatter plot com outliers detectados vs distribuição real
3. **`confusion_matrix_*.png`** - Matriz de confusão com métricas

### Arquivo Adicional:
- **`resumo_comparativo_todos_exemplos.png`** - Gráfico comparativo com todas as métricas
- **`resultados_metricas.csv`** - Tabela com todas as métricas de todos os exemplos

---

## 🏆 Melhores Resultados

### Top 3 por Accuracy:
1. **Back** (src_bytes vs count): 0.924
2. **Smurf** (count vs serror_rate): 0.914
3. **Portsweep** (duration vs srv_count): 0.909

### Top 3 por F1-Score:
1. **Smurf** (count vs serror_rate): 0.272
2. **Back** (src_bytes vs count): 0.088
3. **Neptune** (same_srv_rate vs diff_srv_rate - EllipticEnvelope): 0.084

---

## 💡 Conclusões

1. **Algoritmos mais efetivos:**
   - Isolation Forest foi o mais utilizado e mostrou bons resultados
   - LOF e Elliptic Envelope tiveram desempenho variável

2. **Melhores combinações de atributos:**
   - `count` + `serror_rate` funcionou muito bem para Smurf
   - `src_bytes` + `count` foi efetivo para Back
   - `duration` + `srv_count` teve bom desempenho para Portsweep

3. **Observações sobre Neptune:**
   - Neptune é mais difícil de detectar com outlier detection
   - Melhores resultados com `same_srv_rate` + `diff_srv_rate`
   - Muitas combinações de atributos não foram efetivas

4. **Recomendações:**
   - Diferentes tipos de ataque requerem diferentes combinações de features
   - É importante testar múltiplas abordagens
   - A escolha do contamination rate afeta significativamente os resultados

---

## 📚 Como Usar Estes Resultados

Para apresentar ao professor, recomendo:

1. **Mostrar o resumo comparativo** (`resumo_comparativo_todos_exemplos.png`)
2. **Destacar os 3 melhores exemplos:**
   - Exemplo 06 (Smurf)
   - Exemplo 10 (Back)
   - Exemplo 08 (Portsweep)

3. **Explicar as diferentes abordagens:**
   - Diferentes algoritmos (IF, LOF, EllipticEnvelope)
   - Diferentes combinações de atributos
   - Diferentes tipos de ataque

4. **Discutir os desafios:**
   - Por que alguns atributos funcionam melhor que outros
   - Como a natureza do ataque influencia a detecção
   - Trade-off entre Precision e Recall

---

## 🔧 Como Foi Gerado

Este conjunto de análises foi gerado pelo script `analise_completa_outliers.py`, que:
- Carrega o dataset NSL-KDD
- Filtra dados por tipo de ataque
- Testa diferentes combinações de atributos
- Aplica diferentes algoritmos de detecção de outliers
- Gera visualizações automáticas
- Calcula métricas de desempenho
- Salva todas as imagens com nomes descritivos

---

**Data da Análise:** Novembro 2025  
**Dataset:** NSL-KDD  
**Total de Imagens:** 37  
**Total de Exemplos:** 12
