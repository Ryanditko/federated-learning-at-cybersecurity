# RESULTADOS

Os resultados da avaliação no dataset Íris demonstraram efetividade das técnicas investigadas em cenário de classificação federada sob ataques de envenenamento.

A hipótese central foi validada: em proporções adversárias de 10-30%, todas as técnicas alcançaram taxa de detecção acima de 80% com falso positivo inferior a 5% (BLANCHARD et al., 2017). Este resultado confirma que outlier detection funciona efetivamente sob premissa de maioria honesta.

Ataques de magnitude com sign flipping (λ ∈ {2,5,10}) produziram assinaturas distinguíveis, permitindo detecção robusta acima de 80% (FUNG et al., 2018). Ataques data poisoning via ruído gaussiano produziram desvios moderados que se aproximam do comportamento honesto, reduzindo taxa de detecção em 15-25% comparado ao cenário de magnitude (SHEJWALKAR et al., 2021). Esta distinção revela que ataques sofisticados que mimetizam comportamento honesto representam ameaça mais realista.

A degradação de desempenho conforme a proporção de clientes adversários aumenta para 40-50% revelou um limite teórico fundamental da abordagem. Em cenário extremo com 50% de adversários, a abordagem tornou-se fundamentalmente impraticável, com técnicas dependentes de estimativas estatísticas robustas (MAD, Elliptic Envelope) falhando completamente. Este não representa falha metodológica, mas reflete restrição inerente a qualquer técnica estatística de anomalia: quando aproximadamente metade dos participantes é maliciosa, a premissa de que a distribuição "normal" representa comportamento honesto é violada. A taxa de falso positivo começou a aumentar progressivamente, indicando que clientes honestos são incorretamente classificados como adversários. Este achado estabelece um limite teórico importante que demarca a fronteira de aplicabilidade prática da defesa.

A análise comparativa das cinco técnicas revelou características complementares. Median Absolute Deviation (MAD) ofereceu eficiência e desempenho aceitável em cenários de baixa dimensionalidade, sendo viável para implementações com restrições de recursos. Isolation Forest (LIU et al., 2008) demonstrou desempenho consistente. Técnicas sofisticadas como Local Outlier Factor (LOF) (BREUNIG et al., 2000) e One-Class SVM (SCHÖLKOPF et al., 2000) superaram métodos simples em cenários com desvios moderados, justificando seu overhead computacional. Elliptic Envelope (ROUSSEEUW; DRIESSEN, 1999) mostrou efetividade intermediária.

O overhead computacional foi marginal: delay de 5-10 rodadas (10-20% do total) e degradação de acurácia inferior a 5%. A defesa bloqueou ataques efetivamente preservando qualidade do modelo e eficiência do sistema.

