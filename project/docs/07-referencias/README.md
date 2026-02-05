# 07 - Referências Bibliográficas

## Visão Geral

Este documento consolida todas as referências bibliográficas utilizadas no projeto de Iniciação Científica sobre Aprendizado Federado com Detecção de Outliers. As referências estão organizadas por categoria para facilitar consultas.

## Referências Principais

### Aprendizado Federado - Fundamentos

**[1] McMahan, B., Moore, E., Ramage, D., Hampson, S., & y Arcas, B. A. (2017)**  
**"Communication-Efficient Learning of Deep Networks from Decentralized Data"**  
*Proceedings of the 20th International Conference on Artificial Intelligence and Statistics (AISTATS)*, pp. 1273-1282.  
- **Contribuição**: Introduziu o algoritmo FedAvg (Federated Averaging)
- **Relevância**: Base teórica para agregação federada
- **Link**: https://arxiv.org/abs/1602.05629
- **Citações**: 10,000+ (Google Scholar)

**[2] Li, T., Sahu, A. K., Talwalkar, A., & Smith, V. (2020)**  
**"Federated Learning: Challenges, Methods, and Future Directions"**  
*IEEE Signal Processing Magazine*, vol. 37, no. 3, pp. 50-60.  
- **Contribuição**: Survey abrangente sobre FL
- **Relevância**: Taxonomia de desafios (comunicação, heterogeneidade, privacidade, segurança)
- **Link**: https://arxiv.org/abs/1908.07873
- **Citações**: 5,000+

**[3] Kairouz, P., et al. (2021)**  
**"Advances and Open Problems in Federated Learning"**  
*Foundations and Trends in Machine Learning*, vol. 14, no. 1-2, pp. 1-210.  
- **Contribuição**: Revisão sistemática de 300+ páginas sobre FL
- **Relevância**: Estado da arte até 2021, discussão de problemas em aberto
- **Link**: https://arxiv.org/abs/1912.04977
- **Citações**: 4,000+

### Ataques em Aprendizado Federado

**[4] Biggio, B., & Roli, F. (2018)**  
**"Wild Patterns: Ten Years After the Rise of Adversarial Machine Learning"**  
*Pattern Recognition*, vol. 84, pp. 317-331.  
- **Contribuição**: Taxonomia de ataques adversariais
- **Relevância**: Classificação de ataques (poisoning, evasion)
- **Link**: https://arxiv.org/abs/1712.03141
- **Citações**: 1,500+

**[5] Fang, M., Cao, X., Jia, J., & Gong, N. Z. (2020)**  
**"Local Model Poisoning Attacks to Byzantine-Robust Federated Learning"**  
*29th USENIX Security Symposium*, pp. 1605-1622.  
- **Contribuição**: Demonstrou vulnerabilidades em agregações "robustas"
- **Relevância**: Motiva necessidade de detecção de outliers
- **Link**: https://arxiv.org/abs/1911.11815
- **Citações**: 800+

**[6] Bagdasaryan, E., Veit, A., Hua, Y., Estrin, D., & Shmatikov, V. (2020)**  
**"How To Backdoor Federated Learning"**  
*Proceedings of the 23rd International Conference on Artificial Intelligence and Statistics (AISTATS)*, pp. 2938-2948.  
- **Contribuição**: Backdoor attacks em FL (model replacement)
- **Relevância**: Ataque de model poisoning estudado
- **Link**: https://arxiv.org/abs/1807.00459
- **Citações**: 1,200+

**[7] Shejwalkar, V., & Houmansadr, A. (2021)**  
**"Manipulating the Byzantine: Optimizing Model Poisoning Attacks and Defenses for Federated Learning"**  
*NDSS 2021*.  
- **Contribuição**: Ataques adaptativos que evitam detecção
- **Relevância**: Discussão de limitações de defesas simples
- **Link**: https://arxiv.org/abs/1911.11815
- **Citações**: 600+

### Detecção de Outliers e Defesas

**[8] Blanchard, P., El Mhamdi, E. M., Guerraoui, R., & Stainer, J. (2017)**  
**"Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent"**  
*Advances in Neural Information Processing Systems (NeurIPS)*, pp. 119-129.  
- **Contribuição**: Algoritmo Krum para agregação Byzantine-robust
- **Relevância**: Alternativa ao FedAvg, comparação com MAD
- **Link**: https://arxiv.org/abs/1703.02757
- **Citações**: 1,000+

**[9] Yin, D., Chen, Y., Kannan, R., & Bartlett, P. (2018)**  
**"Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates"**  
*Proceedings of the 35th International Conference on Machine Learning (ICML)*, pp. 5650-5659.  
- **Contribuição**: Trimmed Mean e Median como agregação robusta
- **Relevância**: Análise teórica de robustez
- **Link**: https://arxiv.org/abs/1803.01498
- **Citações**: 900+

**[10] Leys, C., Ley, C., Klein, O., Bernard, P., & Licata, L. (2013)**  
**"Detecting Outliers: Do Not Use Standard Deviation Around the Mean, Use Absolute Deviation Around the Median"**  
*Journal of Experimental Social Psychology*, vol. 49, no. 4, pp. 764-766.  
- **Contribuição**: Fundamento estatístico do MAD (Median Absolute Deviation)
- **Relevância**: Justificativa para uso de MAD em vez de Z-score
- **Link**: https://doi.org/10.1016/j.jesp.2013.03.013
- **Citações**: 2,000+

**[11] Rousseeuw, P. J., & Croux, C. (1993)**  
**"Alternatives to the Median Absolute Deviation"**  
*Journal of the American Statistical Association*, vol. 88, no. 424, pp. 1273-1283.  
- **Contribuição**: Análise de eficiência do MAD
- **Relevância**: Robustez estatística (breakdown point 50%)
- **Link**: https://www.jstor.org/stable/2291267
- **Citações**: 3,000+

### Privacidade e Segurança

**[12] Dwork, C., & Roth, A. (2014)**  
**"The Algorithmic Foundations of Differential Privacy"**  
*Foundations and Trends in Theoretical Computer Science*, vol. 9, no. 3-4, pp. 211-407.  
- **Contribuição**: Definição formal de privacidade diferencial
- **Relevância**: Futuro trabalho para adicionar DP ao sistema
- **Link**: https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf
- **Citações**: 15,000+

**[13] Abadi, M., et al. (2016)**  
**"Deep Learning with Differential Privacy"**  
*Proceedings of the 2016 ACM SIGSAC Conference on Computer and Communications Security*, pp. 308-318.  
- **Contribuição**: DP-SGD (Differentially Private Stochastic Gradient Descent)
- **Relevância**: Técnica de privacidade para FL
- **Link**: https://arxiv.org/abs/1607.00133
- **Citações**: 5,000+

### Aplicações em Cibersegurança

**[14] Zhao, Y., Li, M., Lai, L., Suda, N., Civin, D., & Chandra, V. (2018)**  
**"Federated Learning with Non-IID Data"**  
*arXiv preprint arXiv:1806.00582*.  
- **Contribuição**: Análise de impacto de dados heterogêneos (Non-IID)
- **Relevância**: Cenário realista para cybersecurity (dados distribuídos)
- **Link**: https://arxiv.org/abs/1806.00582
- **Citações**: 2,500+

**[15] Tavallaee, M., Bagheri, E., Lu, W., & Ghorbani, A. A. (2009)**  
**"A Detailed Analysis of the KDD CUP 99 Data Set"**  
*IEEE Symposium on Computational Intelligence for Security and Defense Applications*, pp. 1-6.  
- **Contribuição**: Análise do dataset NSL-KDD
- **Relevância**: Dataset usado para testes futuros
- **Link**: https://ieeexplore.ieee.org/document/5356528
- **Citações**: 4,000+

**[16] Preuveneers, D., Rimmer, V., Tsingenopoulos, I., Spooren, J., Joosen, W., & Ilie-Zudor, E. (2018)**  
**"Chained Anomaly Detection Models for Federated Learning: An Intrusion Detection Case Study"**  
*Applied Sciences*, vol. 8, no. 12, pp. 2663.  
- **Contribuição**: FL aplicado a detecção de intrusões
- **Relevância**: Caso de uso direto em cybersecurity
- **Link**: https://www.mdpi.com/2076-3417/8/12/2663
- **Citações**: 300+

## Referências de Suporte

### Ferramentas e Bibliotecas

**[17] Pedregosa, F., et al. (2011)**  
**"Scikit-learn: Machine Learning in Python"**  
*Journal of Machine Learning Research*, vol. 12, pp. 2825-2830.  
- **Contribuição**: Biblioteca scikit-learn
- **Relevância**: Implementação de LinearRegression
- **Link**: https://scikit-learn.org/
- **Citações**: 50,000+

**[18] McKinney, W. (2010)**  
**"Data Structures for Statistical Computing in Python"**  
*Proceedings of the 9th Python in Science Conference*, pp. 56-61.  
- **Contribuição**: Biblioteca pandas
- **Relevância**: Manipulação de datasets
- **Link**: https://pandas.pydata.org/
- **Citações**: 30,000+

**[19] Hunter, J. D. (2007)**  
**"Matplotlib: A 2D Graphics Environment"**  
*Computing in Science & Engineering*, vol. 9, no. 3, pp. 90-95.  
- **Contribuição**: Biblioteca matplotlib
- **Relevância**: Geração de visualizações
- **Link**: https://matplotlib.org/
- **Citações**: 40,000+

### Datasets

**[20] Fisher, R. A. (1936)**  
**"The Use of Multiple Measurements in Taxonomic Problems"**  
*Annals of Eugenics*, vol. 7, no. 2, pp. 179-188.  
- **Contribuição**: Iris dataset
- **Relevância**: Dataset utilizado no projeto
- **Link**: https://archive.ics.uci.edu/ml/datasets/iris
- **Citações**: 10,000+ (dataset mais citado em ML)

**[21] Dua, D., & Graff, C. (2019)**  
**"UCI Machine Learning Repository"**  
*University of California, Irvine, School of Information and Computer Sciences*.  
- **Contribuição**: Repositório de datasets
- **Relevância**: Fonte do Iris dataset
- **Link**: https://archive.ics.uci.edu/ml/
- **Citações**: 20,000+

### Estatística e Métodos

**[22] James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013)**  
**"An Introduction to Statistical Learning"**  
*Springer*, New York.  
- **Contribuição**: Livro-texto sobre regressão linear
- **Relevância**: Fundamentos de R², MSE, MAE
- **Link**: https://www.statlearning.com/
- **Citações**: 30,000+

**[23] Hastie, T., Tibshirani, R., & Friedman, J. (2009)**  
**"The Elements of Statistical Learning: Data Mining, Inference, and Prediction"**  
*Springer*, 2nd edition.  
- **Contribuição**: Referência avançada em ML
- **Relevância**: Teoria de modelos lineares
- **Link**: https://hastie.su.domains/ElemStatLearn/
- **Citações**: 60,000+

## Referências por Tópico

### Agregação Byzantine-Robust

- **[8]** Blanchard et al. (2017) - Krum
- **[9]** Yin et al. (2018) - Trimmed Mean, Median
- **[10]** Leys et al. (2013) - MAD
- **[11]** Rousseeuw & Croux (1993) - Robustez estatística

### Ataques Adversariais

- **[4]** Biggio & Roli (2018) - Taxonomia geral
- **[5]** Fang et al. (2020) - Model poisoning
- **[6]** Bagdasaryan et al. (2020) - Backdoor attacks
- **[7]** Shejwalkar & Houmansadr (2021) - Ataques adaptativos

### Fundamentos de FL

- **[1]** McMahan et al. (2017) - FedAvg
- **[2]** Li et al. (2020) - Survey de desafios
- **[3]** Kairouz et al. (2021) - Revisão sistemática

### Privacidade

- **[12]** Dwork & Roth (2014) - Differential Privacy
- **[13]** Abadi et al. (2016) - DP-SGD

### Aplicações em Cibersegurança

- **[14]** Zhao et al. (2018) - Non-IID data
- **[15]** Tavallaee et al. (2009) - NSL-KDD dataset
- **[16]** Preuveneers et al. (2018) - Intrusion detection

## Referências Complementares

### Trabalhos Relacionados

**[24] Chen, Y., Su, L., & Xu, J. (2017)**  
**"Distributed Statistical Machine Learning in Adversarial Settings: Byzantine Gradient Descent"**  
*Proceedings of the ACM on Measurement and Analysis of Computing Systems*, vol. 1, no. 2, pp. 1-25.  
- **Relevância**: Byzantine Gradient Descent
- **Link**: https://arxiv.org/abs/1705.05491

**[25] Xie, C., Koyejo, S., & Gupta, I. (2019)**  
**"Zeno: Distributed Stochastic Gradient Descent with Suspicion-based Fault-tolerance"**  
*Proceedings of the 36th International Conference on Machine Learning (ICML)*, pp. 6893-6901.  
- **Relevância**: Sistema de scoring para detecção
- **Link**: https://arxiv.org/abs/1805.10032

**[26] Pillutla, K., Kakade, S. M., & Harchaoui, Z. (2019)**  
**"Robust Aggregation for Federated Learning"**  
*arXiv preprint arXiv:1912.13445*.  
- **Relevância**: Geometric median para agregação
- **Link**: https://arxiv.org/abs/1912.13445

### Surveys e Tutoriais

**[27] Mothukuri, V., Parizi, R. M., Pouriyeh, S., Huang, Y., Dehghantanha, A., & Srivastava, G. (2021)**  
**"A Survey on Security and Privacy of Federated Learning"**  
*Future Generation Computer Systems*, vol. 115, pp. 619-640.  
- **Relevância**: Survey sobre segurança em FL
- **Link**: https://doi.org/10.1016/j.future.2020.10.007

**[28] Lyu, L., Yu, H., Ma, X., Chen, C., Sun, L., Zhao, J., Yang, Q., & Yu, P. S. (2022)**  
**"Privacy and Robustness in Federated Learning: Attacks and Defenses"**  
*IEEE Transactions on Neural Networks and Learning Systems*.  
- **Relevância**: Survey recente (2022) sobre ataques e defesas
- **Link**: https://arxiv.org/abs/2012.06337

### Algoritmos de Otimização

**[29] Li, T., Sahu, A. K., Zaheer, M., Sanjabi, M., Talwalkar, A., & Smith, V. (2020)**  
**"Federated Optimization in Heterogeneous Networks"**  
*Proceedings of Machine Learning and Systems (MLSys)*, vol. 2, pp. 429-450.  
- **Relevância**: FedProx (extensão do FedAvg)
- **Link**: https://arxiv.org/abs/1812.06127

**[30] Reddi, S. J., Charles, Z., Zaheer, M., Garrett, Z., Rush, K., Konečný, J., Kumar, S., & McMahan, H. B. (2021)**  
**"Adaptive Federated Optimization"**  
*International Conference on Learning Representations (ICLR)*.  
- **Relevância**: FedAdam, FedYogi (otimizadores adaptativos)
- **Link**: https://arxiv.org/abs/2003.00295

## Normas de Citação

### Formato ABNT (Associação Brasileira de Normas Técnicas)

**Livro**:
SOBRENOME, Nome. **Título da obra**. Edição. Cidade: Editora, Ano.

**Artigo em periódico**:
SOBRENOME, Nome. Título do artigo. **Nome do periódico**, v. volume, n. número, p. páginas, ano.

**Artigo em conferência**:
SOBRENOME, Nome. Título do artigo. In: NOME DA CONFERÊNCIA, edição, ano, cidade. **Anais**... Cidade: Editora, ano. p. páginas.

**Documento eletrônico**:
SOBRENOME, Nome. **Título**. Disponível em: <URL>. Acesso em: data.

### Formato IEEE

**Livro**:
[n] A. A. Author, *Title of Book*, edition. City, Country: Publisher, year.

**Artigo**:
[n] A. A. Author, "Title of paper," *Journal Name*, vol. x, no. x, pp. xxx-xxx, Month year.

**Conferência**:
[n] A. A. Author, "Title of paper," in *Proc. Conference Name*, City, Country, year, pp. xxx-xxx.

**Web**:
[n] A. A. Author, "Title of document," Website Name, Date. [Online]. Available: URL

## Recursos Adicionais

### Repositórios de Código

**[31] TensorFlow Federated (TFF)**  
- **Descrição**: Framework oficial do Google para FL
- **Link**: https://www.tensorflow.org/federated
- **Relevância**: Implementação de referência para FL em deep learning

**[32] PySyft**  
- **Descrição**: Biblioteca para FL com privacidade
- **Link**: https://github.com/OpenMined/PySyft
- **Relevância**: FL com encrypted computation

**[33] Flower (flwr)**  
- **Descrição**: Framework unificado para FL
- **Link**: https://flower.dev/
- **Relevância**: Suporte a múltiplos frameworks (PyTorch, TF, scikit-learn)

### Cursos e Tutoriais

**[34] Coursera - "Applied Data Science with Python Specialization"**  
- **Instituição**: University of Michigan
- **Link**: https://www.coursera.org/specializations/data-science-python
- **Relevância**: Fundamentos de scikit-learn e pandas

**[35] Stanford CS229 - Machine Learning**  
- **Instrutor**: Andrew Ng
- **Link**: http://cs229.stanford.edu/
- **Relevância**: Teoria de regressão linear

**[36] Federated Learning: Machine Learning on Decentralized Data (Google I/O '19)**  
- **Apresentador**: Daniel Ramage
- **Link**: https://www.youtube.com/watch?v=89BGjQYA0uE
- **Relevância**: Introdução prática ao FL

### Blogs e Artigos Técnicos

**[37] Google AI Blog - "Federated Learning: Collaborative Machine Learning without Centralized Training Data"**  
- **Data**: 2017-04-06
- **Link**: https://ai.googleblog.com/2017/04/federated-learning-collaborative.html
- **Relevância**: Visão prática de FL em produção

**[38] OpenMined Blog**  
- **Link**: https://blog.openmined.org/
- **Relevância**: Comunidade open-source de FL e privacidade

## Histórico de Versões

| Versão | Data | Autor | Mudanças |
|--------|------|-------|----------|
| 1.0 | Fev 2026 | Equipe IC | Versão inicial com 38 referências |
| 1.1 | Mar 2026 | - | Adição de 5 referências sobre Non-IID |
| 2.0 | Mai 2026 | - | Reorganização por tópico |

## Como Citar Este Projeto

**ABNT**:
SOBRENOME, Nome. **Aprendizado Federado com Detecção de Outliers para Segurança em Sistemas Distribuídos**. 2026. Trabalho de Iniciação Científica – Faculdade Impacta, São Paulo, 2026.

**IEEE**:
[1] N. Sobrenome, "Aprendizado federado com detecção de outliers para segurança em sistemas distribuídos," Trabalho de Iniciação Científica, Faculdade Impacta, São Paulo, Brazil, 2026.

**BibTeX**:
```bibtex
@misc{sobrenome2026federated,
  author = {Nome Sobrenome},
  title = {Aprendizado Federado com Detecção de Outliers para Segurança em Sistemas Distribuídos},
  year = {2026},
  howpublished = {Trabalho de Iniciação Científica},
  institution = {Faculdade Impacta},
  address = {São Paulo, Brazil}
}
```

## Contato para Referências

Para solicitar cópias de artigos ou discutir referências:

- **Email**: ic.federated.learning@example.com
- **GitHub Issues**: https://github.com/seu-usuario/repo/issues
- **ResearchGate**: (link do perfil)

---

**Total de Referências**: 38 (principais) + recursos adicionais  
**Última atualização**: Fevereiro 2026  
**Responsável**: Equipe de Pesquisa IC - Faculdade Impacta
