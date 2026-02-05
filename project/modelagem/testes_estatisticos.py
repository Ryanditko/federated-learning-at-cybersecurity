"""
Testes Estatísticos para Validação do Sistema de Aprendizado Federado
Análises de Regressão Linear e Detecção de Outliers
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


class TestadorEstatistico:
    """Classe para realizar testes estatísticos no modelo de Aprendizado Federado"""
    
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.modelo = LinearRegression()
        self.scaler = StandardScaler()
        self.resultados = {}
    
    def teste_normalidade_residuos(self, X_test, y_test):
        """
        Teste de Normalidade dos Resíduos (Shapiro-Wilk)
        H0: Os resíduos seguem distribuição normal
        """
        print("\n" + "="*70)
        print("TESTE 1: NORMALIDADE DOS RESIDUOS (Shapiro-Wilk)")
        print("="*70)
        
        # Treinar modelo
        X_scaled = self.scaler.fit_transform(X_test)
        self.modelo.fit(X_scaled, y_test)
        
        # Calcular resíduos
        y_pred = self.modelo.predict(X_scaled)
        residuos = y_test - y_pred
        
        # Teste de Shapiro-Wilk
        statistic, p_value = stats.shapiro(residuos[:5000])  # Limitado para performance
        
        print(f"\nEstatistica do teste: {statistic:.6f}")
        print(f"P-valor: {p_value:.6f}")
        print(f"\nInterpretacao (alpha=0.05):")
        if p_value > 0.05:
            print("  Nao rejeitamos H0: Residuos sao aproximadamente normais")
            print("  O modelo atende a suposicao de normalidade")
        else:
            print("  Rejeitamos H0: Residuos nao sao normais")
            print("  O modelo pode ter problemas de especificacao")
        
        # Visualização
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histograma
        axes[0].hist(residuos, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='black')
        axes[0].set_xlabel('Residuos')
        axes[0].set_ylabel('Densidade')
        axes[0].set_title('Distribuicao dos Residuos')
        axes[0].grid(True, alpha=0.3)
        
        # Adicionar curva normal teórica
        mu, std = residuos.mean(), residuos.std()
        x = np.linspace(residuos.min(), residuos.max(), 100)
        axes[0].plot(x, stats.norm.pdf(x, mu, std), 'r-', linewidth=2, label='Normal Teorica')
        axes[0].legend()
        
        # Q-Q Plot
        stats.probplot(residuos, dist="norm", plot=axes[1])
        axes[1].set_title('Q-Q Plot (Normalidade)')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('c:/Users/Administrador/Faculdade-Impacta/Iniciação-cientifica/project/modelagem/teste_normalidade.png', dpi=300)
        print("\nGrafico salvo: teste_normalidade.png")
        plt.show()
        
        self.resultados['normalidade'] = {'statistic': statistic, 'p_value': p_value}
        return statistic, p_value
    
    def teste_homocedasticidade(self, X_test, y_test):
        """
        Teste de Homocedasticidade (Breusch-Pagan)
        H0: Variância dos erros é constante (homocedasticidade)
        """
        print("\n" + "="*70)
        print("TESTE 2: HOMOCEDASTICIDADE (Breusch-Pagan)")
        print("="*70)
        
        X_scaled = self.scaler.fit_transform(X_test)
        self.modelo.fit(X_scaled, y_test)
        
        y_pred = self.modelo.predict(X_scaled)
        residuos = y_test - y_pred
        residuos_quadrados = residuos ** 2
        
        # Modelo auxiliar: residuos² ~ X
        modelo_aux = LinearRegression()
        modelo_aux.fit(X_scaled, residuos_quadrados)
        r2_aux = r2_score(residuos_quadrados, modelo_aux.predict(X_scaled))
        
        # Estatística LM
        n = len(X_test)
        lm_statistic = n * r2_aux
        p_value = 1 - stats.chi2.cdf(lm_statistic, X_test.shape[1])
        
        print(f"\nEstatistica LM: {lm_statistic:.6f}")
        print(f"P-valor: {p_value:.6f}")
        print(f"\nInterpretacao (alpha=0.05):")
        if p_value > 0.05:
            print("  Nao rejeitamos H0: Variancia dos erros e constante")
            print("  O modelo atende a suposicao de homocedasticidade")
        else:
            print("  Rejeitamos H0: Heterocedasticidade detectada")
            print("  A variancia dos erros muda com os valores preditos")
        
        # Visualização
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Resíduos vs Valores Preditos
        axes[0].scatter(y_pred, residuos, alpha=0.5, s=10, color='steelblue')
        axes[0].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[0].set_xlabel('Valores Preditos')
        axes[0].set_ylabel('Residuos')
        axes[0].set_title('Residuos vs Valores Preditos')
        axes[0].grid(True, alpha=0.3)
        
        # Resíduos Absolutos vs Valores Preditos
        axes[1].scatter(y_pred, np.abs(residuos), alpha=0.5, s=10, color='coral')
        axes[1].set_xlabel('Valores Preditos')
        axes[1].set_ylabel('|Residuos|')
        axes[1].set_title('Residuos Absolutos vs Valores Preditos')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('c:/Users/Administrador/Faculdade-Impacta/Iniciação-cientifica/project/modelagem/teste_homocedasticidade.png', dpi=300)
        print("\nGrafico salvo: teste_homocedasticidade.png")
        plt.show()
        
        self.resultados['homocedasticidade'] = {'lm_statistic': lm_statistic, 'p_value': p_value}
        return lm_statistic, p_value
    
    def teste_multicolinearidade(self, X_test):
        """
        Teste de Multicolinearidade (VIF - Variance Inflation Factor)
        VIF < 5: Baixa multicolinearidade
        VIF > 10: Alta multicolinearidade (problemático)
        """
        print("\n" + "="*70)
        print("TESTE 3: MULTICOLINEARIDADE (VIF)")
        print("="*70)
        
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        
        X_scaled = self.scaler.fit_transform(X_test)
        
        vif_data = pd.DataFrame()
        vif_data["Feature"] = X_test.columns
        vif_data["VIF"] = [variance_inflation_factor(X_scaled, i) for i in range(X_scaled.shape[1])]
        
        print("\nVariance Inflation Factor (VIF) por feature:")
        print(vif_data.to_string(index=False))
        
        print(f"\nInterpretacao:")
        problemas = vif_data[vif_data['VIF'] > 10]
        if len(problemas) > 0:
            print(f"  ATENCAO: {len(problemas)} feature(s) com VIF > 10")
            print("  Alta multicolinearidade detectada:")
            for _, row in problemas.iterrows():
                print(f"    - {row['Feature']}: VIF = {row['VIF']:.2f}")
        else:
            print("  Multicolinearidade aceitavel (VIF < 10 para todas as features)")
        
        # Visualização
        plt.figure(figsize=(10, 6))
        colors = ['red' if v > 10 else 'orange' if v > 5 else 'green' for v in vif_data['VIF']]
        plt.barh(vif_data['Feature'], vif_data['VIF'], color=colors, edgecolor='black')
        plt.axvline(x=5, color='orange', linestyle='--', label='VIF = 5 (Moderado)')
        plt.axvline(x=10, color='red', linestyle='--', label='VIF = 10 (Problematico)')
        plt.xlabel('VIF')
        plt.title('Variance Inflation Factor por Feature')
        plt.legend()
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig('c:/Users/Administrador/Faculdade-Impacta/Iniciação-cientifica/project/modelagem/teste_multicolinearidade.png', dpi=300)
        print("\nGrafico salvo: teste_multicolinearidade.png")
        plt.show()
        
        self.resultados['multicolinearidade'] = vif_data.to_dict()
        return vif_data
    
    def teste_significancia_coeficientes(self, X_test, y_test):
        """
        Teste t de Significância dos Coeficientes
        H0: Coeficiente = 0 (não significativo)
        """
        print("\n" + "="*70)
        print("TESTE 4: SIGNIFICANCIA DOS COEFICIENTES (Teste t)")
        print("="*70)
        
        from statsmodels.api import OLS, add_constant
        
        X_scaled = self.scaler.fit_transform(X_test)
        X_with_const = add_constant(X_scaled)
        
        modelo_ols = OLS(y_test, X_with_const).fit()
        
        print("\nResumo dos Coeficientes:")
        print(modelo_ols.summary().tables[1])
        
        print(f"\nInterpretacao (alpha=0.05):")
        print("  P-valor < 0.05: Coeficiente significativo")
        print("  P-valor >= 0.05: Coeficiente nao significativo")
        
        # Extrair informações
        coef_summary = pd.DataFrame({
            'Feature': ['Intercept'] + list(X_test.columns),
            'Coeficiente': modelo_ols.params,
            'Std Error': modelo_ols.bse,
            't-statistic': modelo_ols.tvalues,
            'P-valor': modelo_ols.pvalues,
            'Significativo': modelo_ols.pvalues < 0.05
        })
        
        print(f"\nFeatures significativas: {coef_summary['Significativo'].sum() - 1}/{len(X_test.columns)}")
        
        self.resultados['significancia'] = coef_summary.to_dict()
        return modelo_ols
    
    def analise_residuos_completa(self, X_test, y_test):
        """Análise completa dos resíduos"""
        print("\n" + "="*70)
        print("ANALISE COMPLETA DOS RESIDUOS")
        print("="*70)
        
        X_scaled = self.scaler.fit_transform(X_test)
        self.modelo.fit(X_scaled, y_test)
        
        y_pred = self.modelo.predict(X_scaled)
        residuos = y_test - y_pred
        
        # Estatísticas descritivas
        print("\nEstatisticas Descritivas dos Residuos:")
        print(f"  Media: {residuos.mean():.6f}")
        print(f"  Mediana: {np.median(residuos):.6f}")
        print(f"  Desvio Padrao: {residuos.std():.6f}")
        print(f"  Minimo: {residuos.min():.4f}")
        print(f"  Maximo: {residuos.max():.4f}")
        print(f"  Assimetria (Skewness): {stats.skew(residuos):.4f}")
        print(f"  Curtose: {stats.kurtosis(residuos):.4f}")
        
        # Visualização 4 painéis
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Resíduos vs Índice
        axes[0, 0].scatter(range(len(residuos)), residuos, alpha=0.5, s=10)
        axes[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[0, 0].set_xlabel('Indice')
        axes[0, 0].set_ylabel('Residuos')
        axes[0, 0].set_title('Residuos vs Indice')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Histograma com densidade
        axes[0, 1].hist(residuos, bins=50, density=True, alpha=0.7, edgecolor='black')
        mu, std = residuos.mean(), residuos.std()
        x = np.linspace(residuos.min(), residuos.max(), 100)
        axes[0, 1].plot(x, stats.norm.pdf(x, mu, std), 'r-', linewidth=2)
        axes[0, 1].set_xlabel('Residuos')
        axes[0, 1].set_ylabel('Densidade')
        axes[0, 1].set_title('Distribuicao dos Residuos')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Resíduos vs Valores Preditos
        axes[1, 0].scatter(y_pred, residuos, alpha=0.5, s=10)
        axes[1, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[1, 0].set_xlabel('Valores Preditos')
        axes[1, 0].set_ylabel('Residuos')
        axes[1, 0].set_title('Residuos vs Valores Preditos')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Resíduos Padronizados
        residuos_padronizados = (residuos - residuos.mean()) / residuos.std()
        axes[1, 1].scatter(y_pred, residuos_padronizados, alpha=0.5, s=10)
        axes[1, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[1, 1].axhline(y=3, color='orange', linestyle='--', linewidth=1, label='+3σ')
        axes[1, 1].axhline(y=-3, color='orange', linestyle='--', linewidth=1, label='-3σ')
        axes[1, 1].set_xlabel('Valores Preditos')
        axes[1, 1].set_ylabel('Residuos Padronizados')
        axes[1, 1].set_title('Residuos Padronizados vs Valores Preditos')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('c:/Users/Administrador/Faculdade-Impacta/Iniciação-cientifica/project/modelagem/analise_residuos.png', dpi=300)
        print("\nGrafico salvo: analise_residuos.png")
        plt.show()
    
    def executar_todos_testes(self, X_test, y_test):
        """Executa todos os testes estatísticos"""
        print("\n" + "="*70)
        print("EXECUTANDO BATERIA COMPLETA DE TESTES ESTATISTICOS")
        print("="*70)
        
        # Teste 1: Normalidade
        self.teste_normalidade_residuos(X_test, y_test)
        
        # Teste 2: Homocedasticidade
        self.teste_homocedasticidade(X_test, y_test)
        
        # Teste 3: Multicolinearidade
        self.teste_multicolinearidade(X_test)
        
        # Teste 4: Significância
        self.teste_significancia_coeficientes(X_test, y_test)
        
        # Análise de Resíduos
        self.analise_residuos_completa(X_test, y_test)
        
        # Relatório Final
        print("\n" + "="*70)
        print("RELATORIO FINAL DOS TESTES ESTATISTICOS")
        print("="*70)
        print("\n1. Normalidade dos Residuos:")
        print(f"   P-valor: {self.resultados['normalidade']['p_value']:.6f}")
        print(f"   Status: {'APROVADO' if self.resultados['normalidade']['p_value'] > 0.05 else 'REPROVADO'}")
        
        print("\n2. Homocedasticidade:")
        print(f"   P-valor: {self.resultados['homocedasticidade']['p_value']:.6f}")
        print(f"   Status: {'APROVADO' if self.resultados['homocedasticidade']['p_value'] > 0.05 else 'REPROVADO'}")
        
        print("\n3. Multicolinearidade:")
        print("   Verificar graficos para detalhes")
        
        print("\n4. Significancia dos Coeficientes:")
        print("   Verificar tabela de coeficientes para detalhes")
        
        print("\n" + "="*70)


if __name__ == "__main__":
    print("="*70)
    print("TESTES ESTATISTICOS - APRENDIZADO FEDERADO")
    print("Regressao Linear com California Housing Dataset")
    print("="*70)
    
    # Carregar dados
    print("\nCarregando California Housing Dataset...")
    california = fetch_california_housing()
    X = pd.DataFrame(california.data, columns=california.feature_names)
    y = pd.Series(california.target, name='MedHouseVal')
    
    # Dividir dados
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    print(f"  Treino: {len(X_train)} amostras")
    print(f"  Teste: {len(X_test)} amostras")
    
    # Criar testador
    testador = TestadorEstatistico(X_train, y_train)
    
    # Executar todos os testes
    testador.executar_todos_testes(X_test, y_test)
    
    print("\n" + "="*70)
    print("TESTES CONCLUIDOS")
    print("Verifique os graficos salvos na pasta modelagem/")
    print("="*70)
