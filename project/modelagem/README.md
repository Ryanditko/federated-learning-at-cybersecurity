# Sistema de Aprendizado Federado com Detecção de Ataques

## Visão Geral

Este sistema implementa um ambiente de aprendizado federado utilizando regressão linear como modelo base. O objetivo principal é simular ataques de envenenamento (data poisoning e model poisoning) e demonstrar técnicas de detecção de clientes maliciosos através de análise estatística de outliers.

## Arquitetura

O sistema é composto por dois componentes principais:

### 1. ServidorFederado

Responsável por coordenar o processo de aprendizado federado:

- Gerencia a lista de clientes participantes
- Mantém o modelo global (LinearRegression)
- Executa rodadas de treinamento
- Agrega modelos locais usando algoritmo FedAvg
- Detecta e filtra clientes outliers
- Gera relatórios de detecção

### 2. ClienteMalicioso

Representa os participantes do aprendizado federado, podendo simular comportamento malicioso:

- Armazena dados locais (nunca compartilhados diretamente)
- Treina modelo local com seus próprios dados
- Pode executar ataques de envenenamento
- Compartilha apenas os parâmetros do modelo treinado

## Tipos de Ataques Implementados

### Envenenamento de Dados

O ataque manipula o conjunto de dados de treinamento:

- Seleciona 30% das amostras aleatoriamente
- Adiciona ruído gaussiano com desvio padrão 3x maior que o original
- Corrompe as features numéricas, mantendo a estrutura dos dados

Este ataque é mais sutil e pode passar despercebido pela detecção de outliers.

### Envenenamento de Modelo

O ataque manipula diretamente os coeficientes do modelo treinado:

**Inversão de pesos:**
- Inverte o sinal de todos os coeficientes
- Simula um modelo que faz predições opostas ao esperado

**Randomização de pesos:**
- Substitui coeficientes por valores aleatórios
- Destrói completamente a capacidade preditiva do modelo

Este ataque é mais agressivo e geralmente é detectado pelo sistema.

## Algoritmo de Detecção de Outliers

O sistema utiliza o método MAD (Median Absolute Deviation) para identificar clientes maliciosos:

### Processo de Detecção

1. **Coleta de Modelos:**
   - Servidor recebe coeficientes de todos os clientes

2. **Cálculo da Mediana:**
   - Calcula mediana dos coeficientes (mais robusta que média)

3. **Cálculo de Distância:**
   - Mede distância euclidiana entre coeficientes de cada cliente e a mediana

4. **Definição de Threshold:**
   - MAD = mediana dos desvios absolutos
   - Threshold = mediana + 3 * MAD
   - Clientes acima do threshold são marcados como outliers

5. **Agregação Seletiva:**
   - Apenas clientes não-outliers participam da agregação
   - Modelo global é protegido de contribuições maliciosas

### Vantagens do MAD

- Resistente a outliers (não é afetado pelos próprios valores anômalos)
- Baseado em mediana ao invés de média
- Threshold adaptativo a cada rodada
- Amplamente utilizado em literatura estatística

## Fluxo de Execução

### Inicialização

```
1. Gera dados sintéticos para cada cliente
2. Cria instância do ServidorFederado
3. Adiciona clientes (maliciosos e honestos)
4. Inicia processo de treinamento
```

### Rodada de Treinamento

```
Para cada rodada (1 até max_rodadas):
    1. Servidor distribui modelo global (implícito)
    2. Cada cliente:
        a. Aplica ataque (se aplicável)
        b. Treina modelo local com seus dados
        c. Calcula métricas (R2)
    3. Servidor:
        a. Coleta modelos locais
        b. Detecta outliers usando MAD
        c. Agrega apenas clientes válidos
        d. Atualiza modelo global
    4. Exibe métricas da rodada
```

### Finalização

```
1. Gera relatório de outliers detectados
2. Calcula estatísticas finais
3. Exibe resumo do treinamento
```

## Métricas e Avaliação

### Métricas de Modelo

- **R2 Score:** Coeficiente de determinação (0 a 1)
  - Valores próximos a 1 indicam bom ajuste
  - Valores baixos indicam modelo comprometido

### Métricas de Detecção

- **Taxa de Detecção:** Percentual de outliers identificados
- **Falsos Positivos:** Clientes honestos marcados como outliers
- **Distância vs Threshold:** Métricas por cliente e rodada

## Configuração de Experimentos

### Parâmetros Principais

- `max_rodadas`: Número de iterações do treinamento federado
- `n_samples`: Quantidade de amostras por cliente
- `n_features`: Número de features do dataset
- `tipo_ataque`: String identificando o tipo de ataque

### Tipos de Ataque Disponíveis

- `"dados"`: Envenenamento de dados
- `"modelo_invertidos"`: Inversão de coeficientes
- `"modelo_randomizados"`: Randomização de coeficientes
- `"nenhum"`: Cliente honesto (controle)

## Resultados Esperados

### Comportamento com Ataques

- Clientes com inversão de pesos são detectados consistentemente
- Clientes com dados envenenados podem não ser detectados
- Clientes honestos nunca são marcados como outliers
- Modelo global melhora ao filtrar contribuições maliciosas

### Comparação de Desempenho

**Sem detecção:**
- R2 médio reduzido pela presença de modelos maliciosos
- Modelo global comprometido

**Com detecção:**
- R2 médio mais alto (melhoria de aproximadamente 15-20%)
- Modelo global mais robusto e confiável

## Limitações

### Limitações Técnicas

- Detecção baseada apenas em distância de coeficientes
- Ataques sutis podem não ser detectados
- Threshold fixo (3 MADs) pode não ser ideal para todos os cenários
- Não implementa validação cruzada

### Simplificações

- Agregação usa média simples (não ponderada por tamanho de dataset)
- Clientes têm acesso direto ao servidor (não simula comunicação real)
- Dados sintéticos com distribuição normal
- Modelo linear pode não capturar complexidades reais

## Extensões Possíveis

### Melhorias na Detecção

- Implementar múltiplos critérios de detecção
- Adicionar análise de histórico temporal
- Utilizar técnicas de machine learning para detecção
- Implementar sistema de reputação de clientes

### Melhorias no Sistema

- Adicionar suporte a outros modelos (redes neurais, árvores)
- Implementar agregação ponderada por tamanho de dataset
- Adicionar validação com dados de teste separados
- Simular latência e falhas de comunicação

### Novos Tipos de Ataque

- Label flipping (inversão de labels)
- Backdoor attacks
- Ataques adaptativos que tentam evitar detecção
- Ataques coordenados entre múltiplos clientes

## Requisitos

### Bibliotecas Python

```
pandas
numpy
scikit-learn
```

### Versões Testadas

- Python 3.8+
- NumPy 1.20+
- Pandas 1.3+
- scikit-learn 0.24+

## Execução

```bash
python modelagem.py
```

O script executa automaticamente um experimento com 3 clientes (2 maliciosos e 1 honesto) durante 5 rodadas de treinamento.

## Interpretação dos Resultados

### Saída do Console

```
[OUTLIER DETECTADO] Cliente_X - Distancia: Y > Threshold: Z
```

Indica que o cliente X foi identificado como outlier com distância Y, excedendo o threshold Z.

```
Clientes aceitos na agregacao: [lista]
Clientes rejeitados (outliers): [lista]
```

Mostra quais clientes participaram ou foram excluídos da agregação.

### Relatório Final

Ao final da execução, o sistema exibe:

- Total de detecções realizadas
- Número de rodadas com detecções
- Lista de clientes detectados por rodada

Estes dados permitem avaliar a eficácia do sistema de detecção e o comportamento dos diferentes tipos de ataque ao longo do treinamento.
