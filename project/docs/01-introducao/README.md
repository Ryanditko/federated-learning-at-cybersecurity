# 01 - Introdução

## Visão Geral do Projeto

Este projeto de Iniciação Científica investiga técnicas de mitigação de ataques por envenenamento em sistemas de Aprendizado Federado utilizando detecção de outliers e machine learning aplicado à segurança cibernética.

## Contexto e Motivação

### O Problema

Sistemas de Aprendizado Federado (Federated Learning - FL) permitem que múltiplos dispositivos ou organizações treinem colaborativamente um modelo de machine learning sem compartilhar dados brutos. Essa abordagem é fundamental para:

- Preservação de privacidade de dados sensíveis
- Conformidade com regulamentações (LGPD, GDPR)
- Treinamento descentralizado em larga escala
- Aplicações em saúde, finanças, IoT

Porém, sistemas FL são vulneráveis a **ataques de envenenamento**, onde participantes maliciosos comprometem o modelo global enviando:
- Dados corrompidos (data poisoning)
- Modelos adulterados (model poisoning)
- Gradientes manipulados (gradient poisoning)

### Relevância Científica

A detecção e mitigação desses ataques é crítica para:
1. **Segurança**: Proteger infraestruturas críticas
2. **Confiabilidade**: Garantir qualidade de modelos em ambientes não-confiáveis
3. **Aplicabilidade**: Viabilizar FL em cenários reais (hospitais, bancos, dispositivos IoT)

## Objetivos do Projeto

### Objetivo Geral

Desenvolver e validar experimentalmente um sistema de detecção de ataques de envenenamento em Aprendizado Federado utilizando técnicas robustas de detecção de outliers.

### Objetivos Específicos

1. **Implementar** um sistema completo de Aprendizado Federado com agregação FedAvg
2. **Desenvolver** mecanismos de detecção baseados em MAD (Median Absolute Deviation)
3. **Simular** diferentes tipos de ataques de envenenamento (dados, modelo, gradientes)
4. **Avaliar** a eficácia da detecção em múltiplos cenários experimentais
5. **Documentar** resultados e contribuições para a área de segurança em FL

## Estrutura do Projeto

### Fases de Desenvolvimento

```
Fase 1: Fundamentação Teórica (Meses 1-2)
├── Revisão bibliográfica sobre FL
├── Estudo de ataques Byzantine
└── Análise de técnicas de detecção

Fase 2: Implementação (Meses 3-5)
├── Sistema FL base
├── Simulação de ataques
├── Implementação de defesas
└── Integração com datasets

Fase 3: Experimentação (Meses 6-8)
├── Testes com Iris dataset
├── Testes com NSL-KDD dataset
├── Validação estatística
└── Análise comparativa

Fase 4: Documentação (Meses 9-12)
├── Redação de relatórios
├── Preparação de artigos
├── Documentação técnica
└── Apresentação de resultados
```

### Componentes Principais

1. **Servidor Federado**: Coordena treinamento, detecta outliers, agrega modelos
2. **Clientes**: Representam participantes (honestos e maliciosos)
3. **Datasets**: Iris (regressão), NSL-KDD (classificação), Cyber Threat Intelligence
4. **Detecção**: Algoritmos MAD, Z-score, IQR para identificar anomalias
5. **Visualizações**: Gráficos de evolução, métricas, detecções

## Contribuições Esperadas

### Científicas

- Validação empírica de MAD em FL com datasets pequenos e grandes
- Análise comparativa de técnicas de detecção de outliers
- Caracterização de ataques e suas assinaturas estatísticas

### Técnicas

- Sistema FL modular e extensível
- Framework para simulação de ataques
- Ferramentas de visualização e análise

### Educacionais

- Documentação completa para reprodutibilidade
- Guias práticos de implementação
- Material didático sobre segurança em FL

## Metodologia

### Abordagem Científica

1. **Revisão Sistemática**: Análise de literatura sobre FL, ataques Byzantine, detecção de outliers
2. **Experimentação Controlada**: Simulações com configurações variadas (número de clientes, taxa de maliciosos)
3. **Validação Estatística**: Testes de hipóteses, intervalos de confiança, análise de significância
4. **Benchmarking**: Comparação com trabalhos relacionados (Krum, Trimmed Mean, Bulyan)

### Ferramentas e Tecnologias

- **Linguagem**: Python 3.10+
- **ML Framework**: scikit-learn, TensorFlow (futuro)
- **Análise de Dados**: pandas, numpy
- **Visualização**: matplotlib, seaborn
- **Controle de Versão**: Git/GitHub
- **Documentação**: Markdown, Jupyter Notebooks

## Premissas e Limitações

### Premissas

1. Maioria dos participantes são honestos (Byzantine Fault Tolerance)
2. Servidor central é confiável (não comprometido)
3. Comunicação entre clientes e servidor é segura
4. Ataques são passivos (não adaptam comportamento durante treinamento)

### Limitações Conhecidas

1. **Escopo**: Foco em ataques de envenenamento (não cobre todos os tipos de ataques)
2. **Simulação**: Ambiente controlado (não testa em produção real)
3. **Datasets**: Limitado a 3-4 datasets por questões de tempo
4. **Escalabilidade**: Testes com até 100 clientes (não avalia centenas de milhares)

## Resultados Preliminares

Baseado nos experimentos iniciais com o dataset Iris:

- **Taxa de Detecção**: 100% (2/2 rodadas)
- **R² Mantido**: > 0.90 (excelente qualidade)
- **Overhead**: < 5% de tempo computacional adicional
- **Robustez**: Sistema funciona com até 25% de clientes maliciosos

## Próximos Passos

1. Expandir testes para NSL-KDD dataset (classificação binária)
2. Implementar defesas adaptativas
3. Comparar com algoritmos estado-da-arte (Krum, Median, Trimmed Mean)
4. Preparar artigo para conferência/workshop
5. Criar dashboard interativo de visualizações

## Referências Iniciais

1. **McMahan et al. (2017)** - Communication-Efficient Learning of Deep Networks from Decentralized Data
2. **Blanchard et al. (2017)** - Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent
3. **Yin et al. (2018)** - Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates
4. **Fung et al. (2020)** - The Limitations of Federated Learning in Sybil Settings

## Contato e Informações

**Instituição**: Faculdade Impacta  
**Área**: Ciência da Computação - Cibersegurança  
**Período**: 2025/2026 (12 meses)  
**Tipo**: Pesquisa Aplicada em Segurança de Machine Learning

---

**Última atualização**: Fevereiro 2026