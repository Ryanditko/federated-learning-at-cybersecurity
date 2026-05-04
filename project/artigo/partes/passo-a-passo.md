# INTRODUÇÃO

## Parágrafo 1: Contexto (Move 1 - Steps 1 e 2)

**Frase de Introdução:** O Aprendizado Federado (Federated Learning - FL) se estabeleceu como um paradigma fundamental para o treinamento distribuído de modelos de machine learning, permitindo que múltiplos participantes colaborem no desenvolvimento de modelos compartilhados sem necessidade de centralizar dados brutos.

**Step 1 – Relevância e Importância:** Este avanço é particularmente relevante em domínios onde a privacidade é crítica—instituições de saúde, sistemas financeiros, IoT industrial e infraestruturas de dados pessoais—onde regulamentações como LGPD e GDPR impõem restrições rigorosas sobre o compartilhamento de informações sensíveis.

**Step 2 – Arquitetura e Benefícios vs. Vulnerabilidades:** Diferentemente dos paradigmas centralizados tradicionais, FL mantém os dados localizados nos clientes enquanto apenas atualizações de modelo ou gradientes são transmitidas para um servidor central coordenador, oferecendo benefícios significativos em termos de privacidade e conformidade regulatória.

**Frase de Conclusão:** Contudo, a natureza descentralizada e sem supervisão direta do FL introduz vulnerabilidades que comprometem a integridade do processo de aprendizado.

---

## Parágrafo 2: Necessidade/Problema (Move 2 - Step 3: Ataques de Envenenamento)

**Frase de Introdução:** Apesar de suas vantagens, sistemas de FL enfrentam uma vulnerabilidade fundamental: sua suscetibilidade a ataques de envenenamento (poisoning attacks), nos quais participantes maliciosos comprometem o processo de aprendizado.

**Step 3 – Caracterização do Problema e Impacto:** Ataques deste tipo já foram extensivamente documentados na literatura acadêmica, incluindo variantes como sign flipping attacks, onde invasores invertem deliberadamente sinais de atualização para degradar o desempenho do modelo global, e gradient manipulation attacks, que distorcem informações de gradiente para induzir comportamentos adversários. A gravidade desta ameaça é amplificada pelo fato de que esquemas de agregação convencionais—particularmente FedAvg—tratam todas as contribuições com peso igual, sem mecanismo algum para validar a autenticidade ou qualidade das atualizações recebidas. Consequentemente, um único agente malicioso ou um pequeno consórcio coordenado pode potencialmente comprometer a integridade de todo o modelo global, especialmente durante estágios iniciais de treinamento quando o modelo ainda é frágil e suscetível a perturbações.

**Frase de Conclusão:** Esta deficiência crítica representa um obstáculo substancial à adoção de FL em aplicações de alta criticidade, onde falhas de segurança podem ter consequências severas.

---

## Parágrafo 3: Solução/Objetivo (Move 3)

**Frase de Introdução:** O presente estudo propõe uma abordagem baseada em detecção de outliers como mecanismo defensivo contra ataques de envenenamento em ambientes de aprendizado federado.

**Apresentação da Solução e Metodologia:** A fundamentação desta estratégia repousa na hipótese de que atualizações legítimas provenientes de clientes honestos exibem coerência estatística, enquanto atualizações adversárias constituem anomalias significativas no espaço de distribuição de gradientes. Para avaliar a viabilidade desta abordagem, implementamos e comparamos cinco técnicas consolidadas de detecção de anomalias: Isolation Forest, Local Outlier Factor (LOF), One-Class SVM, Elliptic Envelope e DBSCAN. O sistema proposto opera através de um pipeline defensivo que intercepta atualizações de clientes no servidor central, aplica detecção de outliers antes da fase de agregação e descarta atualizações classificadas como anômalas. Esta validação foi conduzida através de simulações experimentais em múltiplos cenários de ataque utilizando datasets distintos—Iris, NSL-KDD e Cyber Threat Intelligence—permitindo uma avaliação abrangente da eficácia da defesa sob diversas condições operacionais.

**Frase de Conclusão:** Os resultados demonstram que a abordagem proposta oferece proteção robusta contra envenenamento mantendo compatibilidade com a eficiência computacional requerida por sistemas federados, evidenciando que técnicas de detecção de outliers constituem uma contribuição significativa e viável para fortalecer a segurança de sistemas de aprendizado federado.
