# 06 - Guias de Uso

## Visão Geral

Este documento fornece guias práticos para instalar, configurar, executar e estender o sistema de Aprendizado Federado com detecção de outliers.

## Guia de Instalação

### Requisitos do Sistema

**Hardware Mínimo**:
- CPU: 2 cores, 2.0 GHz
- RAM: 4 GB
- Disco: 500 MB disponível

**Hardware Recomendado**:
- CPU: 4+ cores, 2.5+ GHz
- RAM: 8+ GB
- Disco: 2 GB disponível (para datasets maiores)

**Software**:
- Python: 3.10 ou superior
- pip: 21.0+
- Git: 2.30+ (opcional, para clonar repositório)

**Sistemas Operacionais Testados**:
- Windows 10/11
- Linux (Ubuntu 20.04+, Debian 11+)
- macOS 12+ (Monterey ou superior)

### Passo 1: Clonar o Repositório

```bash
# Via HTTPS
git clone https://github.com/seu-usuario/federated-learning-outlier-detection.git
cd federated-learning-outlier-detection

# Via SSH
git clone git@github.com:seu-usuario/federated-learning-outlier-detection.git
cd federated-learning-outlier-detection
```

**Sem Git**:
1. Baixar ZIP do repositório
2. Extrair em `C:\Users\Administrador\Faculdade-Impacta\Iniciação-cientifica`
3. Abrir terminal na pasta `project/`

### Passo 2: Criar Ambiente Virtual

**Windows (PowerShell)**:
```powershell
# Criar ambiente virtual
python -m venv venv

# Ativar ambiente
.\venv\Scripts\Activate.ps1

# Se houver erro de política de execução:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Linux/macOS**:
```bash
# Criar ambiente virtual
python3 -m venv venv

# Ativar ambiente
source venv/bin/activate
```

**Verificar Ativação**:
```bash
# Deve mostrar (venv) no início do prompt
(venv) PS C:\...\project>
```

### Passo 3: Instalar Dependências

```bash
# Atualizar pip
python -m pip install --upgrade pip

# Instalar todas as dependências
pip install -r dependencies/requirements.txt

# Verificar instalação
pip list
```

**Saída Esperada**:
```
Package         Version
--------------- --------
numpy           1.24.3
pandas          2.0.2
scikit-learn    1.3.0
matplotlib      3.7.1
seaborn         0.12.2
```

### Passo 4: Verificar Instalação

```bash
# Testar importações
python -c "import sklearn, pandas, numpy, matplotlib; print('OK')"

# Deve imprimir: OK
```

**Se houver erros**:
```bash
# Reinstalar dependências
pip install --force-reinstall -r dependencies/requirements.txt
```

## Guia de Quickstart

### Executar Exemplo Básico (30 segundos)

```bash
# Navegar para pasta de modelagem
cd project/modelagem

# Executar teste simples (sem ataques)
python teste_iris.py
```

**Saída Esperada**:
```
[Servidor] Iniciando Aprendizado Federado com 3 clientes, 5 rodadas...
[Rodada 1/5] Treinamento local...
[Rodada 1/5] Cliente_1: R²=0.87, MSE=0.08, MAE=0.21
[Rodada 1/5] Cliente_2: R²=0.86, MSE=0.09, MAE=0.22
[Rodada 1/5] Cliente_3: R²=0.88, MSE=0.07, MAE=0.20
[Rodada 1/5] Agregação: 3 aceitos, 0 outliers
...
[Rodada 5/5] Modelo Global: R²=0.91, MSE=0.05, MAE=0.17
[Servidor] Treinamento concluído!
Gráficos salvos em: resultados_fl.png
```

**Visualizar Gráficos**:
- Abrir `resultados_fl.png` (gerado na pasta `modelagem/`)
- Ver 4 subplots: R², MSE, MAE, clientes aceitos

### Executar com Ataques (1 minuto)

```bash
# Executar teste com 1 atacante
python modelagem.py
```

**Diferença na Saída**:
```
[Rodada 1/5] Cliente_2_MALICIOSO: R²=-1.23, MSE=15.67, MAE=3.45
[Rodada 1/5] Agregação: 3 aceitos, 1 outlier
[Rodada 1/5] Outliers detectados: ['Cliente_2_MALICIOSO']
```

## Guia de Configuração

### Estrutura de Configuração

Não há arquivo de config separado. Configurações estão no início de `modelagem.py`:

```python
# Linha 547-598 (função main)
def main():
    # === CONFIGURAÇÕES ===
    dataset_path = "../data/iris/iris.csv"
    target_column = "petal_width"
    max_rodadas = 5
    num_clientes = 4
    
    # Tipos de ataque: "nenhum", "dados", "modelo_invertidos", "modelo_randomizados"
    configuracao_clientes = [
        {"id": "Cliente_1", "ataque": "nenhum"},
        {"id": "Cliente_2_MALICIOSO", "ataque": "dados"},  # ATACANTE
        {"id": "Cliente_3", "ataque": "nenhum"},
        {"id": "Cliente_4", "ataque": "nenhum"},
    ]
```

### Parâmetros Configuráveis

#### 1. Dataset

**Mudar Dataset**:
```python
# Opção 1: Iris (padrão)
dataset_path = "../data/iris/iris.csv"
target_column = "petal_width"

# Opção 2: Penguins
dataset_path = "../data/penguin/penguins.csv"
target_column = "body_mass_g"

# Opção 3: Weight-Height
dataset_path = "../data/weight-height/weight_height.csv"
target_column = "Weight"
```

#### 2. Número de Rodadas

```python
# Convergência rápida (testes)
max_rodadas = 5

# Convergência completa (produção)
max_rodadas = 10

# Análise de longo prazo
max_rodadas = 20
```

**Recomendação**: 5 rodadas para Iris, 10+ para datasets maiores

#### 3. Número de Clientes

```python
# Mínimo funcional
num_clientes = 3

# Padrão recomendado
num_clientes = 4

# Escalabilidade
num_clientes = 10  # Requer ajuste em configuracao_clientes
```

**Limitações**:
- Mínimo: 2 clientes (federado não faz sentido com 1)
- Máximo testado: 100 clientes (tempo ~25s/rodada)

#### 4. Configuração de Ataques

**Cenário 1: Baseline (Sem Ataques)**
```python
configuracao_clientes = [
    {"id": "Cliente_1", "ataque": "nenhum"},
    {"id": "Cliente_2", "ataque": "nenhum"},
    {"id": "Cliente_3", "ataque": "nenhum"},
]
```

**Cenário 2: Data Poisoning (30% ruído)**
```python
configuracao_clientes = [
    {"id": "Cliente_1", "ataque": "nenhum"},
    {"id": "Cliente_2_MAL", "ataque": "dados"},  # ATACANTE
    {"id": "Cliente_3", "ataque": "nenhum"},
]
```

**Cenário 3: Model Poisoning - Invertidos**
```python
configuracao_clientes = [
    {"id": "Cliente_1", "ataque": "nenhum"},
    {"id": "Cliente_2_MAL", "ataque": "modelo_invertidos"},  # ATACANTE
    {"id": "Cliente_3", "ataque": "nenhum"},
]
```

**Cenário 4: Model Poisoning - Randomizados**
```python
configuracao_clientes = [
    {"id": "Cliente_1", "ataque": "nenhum"},
    {"id": "Cliente_2_MAL", "ataque": "modelo_randomizados"},  # ATACANTE
    {"id": "Cliente_3", "ataque": "nenhum"},
]
```

**Cenário 5: Múltiplos Atacantes**
```python
configuracao_clientes = [
    {"id": "Cliente_1", "ataque": "nenhum"},
    {"id": "Cliente_2_MAL", "ataque": "dados"},           # ATACANTE 1
    {"id": "Cliente_3", "ataque": "nenhum"},
    {"id": "Cliente_4_MAL", "ataque": "modelo_invertidos"},  # ATACANTE 2
    {"id": "Cliente_5", "ataque": "nenhum"},
]
```

#### 5. Threshold MAD

**Localização**: Linha 154 em `modelagem.py`

```python
# Padrão: 3σ (99.7% confiança)
threshold = np.median(desvios) + 3 * mad

# Mais conservador (menos falsos positivos, pode perder ataques)
threshold = np.median(desvios) + 4 * mad

# Mais agressivo (detecta mais, risco de falsos positivos)
threshold = np.median(desvios) + 2 * mad
```

**Recomendação**: Manter 3σ (balanceado)

### Configuração Avançada

#### Divisão de Dados (IID vs Non-IID)

**Atual**: IID (shuffle + split uniforme)

```python
# Linha 568
dados = dados.sample(frac=1, random_state=42).reset_index(drop=True)
```

**Non-IID (heterogêneo)**:
```python
# Exemplo: Cliente 1 recebe apenas setosa, Cliente 2 versicolor, etc.
def dividir_noniid(dados, num_clientes):
    especies = dados['species'].unique()
    chunks = []
    for i in range(num_clientes):
        chunk = dados[dados['species'] == especies[i % len(especies)]]
        chunks.append(chunk)
    return chunks
```

#### Customizar Ataques

**Aumentar Intensidade de Data Poisoning**:
```python
# Linha 487 em ClienteMalicioso
ruido = np.random.normal(0, self.dados[col].std() * 5, n_envenenadas)  # 3→5
```

**Novo Tipo de Ataque**:
```python
def envenenar_modelo(self):
    pesos = self.modelo_local.obter_pesos()
    
    if "escalonados" in self.tipo_ataque:
        # Novo ataque: amplifica coeficientes por 10
        pesos['coef'] = pesos['coef'] * 10
    
    self.modelo_local.atualizar_pesos(pesos)
```

## Guia de Execução

### Executar Testes Automatizados

```bash
# Teste 1: Sem ataques
cd project/modelagem
python teste_iris.py

# Verificar sucesso
echo $?  # Linux/macOS
echo $LASTEXITCODE  # PowerShell
# 0 = sucesso, 1 = falha
```

### Executar Sistema Completo

```bash
# Com saída padrão
python modelagem.py

# Redirecionar saída para arquivo (log)
python modelagem.py > experimento_$(date +%Y%m%d_%H%M%S).log

# PowerShell
python modelagem.py > "experimento_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"
```

### Executar Múltiplos Experimentos (Batch)

**Criar script batch** (`run_experiments.sh` - Linux/macOS):
```bash
#!/bin/bash
for i in {1..10}; do
    echo "Executando experimento $i..."
    python modelagem.py > logs/exp_$i.log
    mv resultados_fl.png resultados/exp_$i.png
done
echo "Todos os experimentos concluídos!"
```

**PowerShell** (`run_experiments.ps1`):
```powershell
for ($i=1; $i -le 10; $i++) {
    Write-Host "Executando experimento $i..."
    python modelagem.py > "logs\exp_$i.log"
    Move-Item resultados_fl.png "resultados\exp_$i.png"
}
Write-Host "Todos os experimentos concluídos!"
```

**Executar**:
```bash
# Linux/macOS
chmod +x run_experiments.sh
./run_experiments.sh

# PowerShell
.\run_experiments.ps1
```

### Monitorar Progresso

**Tempo Real**:
```bash
# Linux/macOS
python modelagem.py | tee experimento.log

# PowerShell
python modelagem.py | Tee-Object experimento.log
```

**Ver últimas linhas do log**:
```bash
# Linux/macOS
tail -f experimento.log

# PowerShell
Get-Content experimento.log -Wait
```

## Guia de Interpretação de Resultados

### Entender Métricas

#### R² Score (Coeficiente de Determinação)

**Interpretação**:
- **0.90 - 1.00**: Excelente (modelo explica 90%+ da variância)
- **0.80 - 0.90**: Bom (modelo útil)
- **0.70 - 0.80**: Razoável (pode melhorar)
- **< 0.70**: Ruim (modelo inadequado)

**No Iris**:
- R² = 0.91 → Modelo excelente
- Consegue prever petal width com alta precisão

#### MSE (Mean Squared Error)

**Interpretação**:
- Unidade: (cm)² (quadrado da unidade do target)
- Penaliza erros grandes (outliers pesam mais)
- Quanto menor, melhor

**No Iris**:
- MSE = 0.0558 → Erro baixo
- √MSE = 0.236 cm → Erro típico de 0.24 cm

#### MAE (Mean Absolute Error)

**Interpretação**:
- Unidade: cm (mesma unidade do target)
- Erro médio absoluto (interpretável)
- Menos sensível a outliers que MSE

**No Iris**:
- MAE = 0.17 cm → Erro médio de 0.17 cm
- Petal width varia de 0.1 a 2.5 cm → erro relativo ~7%

### Analisar Gráficos

**Convergência Normal**:
- R² subindo suavemente
- MSE/MAE descendo suavemente
- Estabiliza em 3-5 rodadas

**Problema: Não Converge**:
- R² oscila ou cai
- MSE/MAE sobem
- Possíveis causas:
  - Muitos atacantes (> 50%)
  - Threshold MAD muito agressivo
  - Dados insuficientes

**Problema: Falsos Positivos**:
- Gráfico 4 mostra clientes honestos rejeitados
- Possíveis causas:
  - Threshold MAD muito conservador
  - Dados Non-IID extremos
  - Bug no código

### Interpretar Outliers Detectados

**Saída Típica**:
```
[Rodada 1/5] Outliers detectados: ['Cliente_2_MALICIOSO']
[Rodada 2/5] Outliers detectados: ['Cliente_2_MALICIOSO']
[Rodada 3/5] Outliers detectados: ['Cliente_2_MALICIOSO']
```

**Análise**:
- ✅ Atacante detectado consistentemente (100% taxa)
- ✅ Mesmo cliente em todas as rodadas (comportamento esperado)
- ✅ Sistema funcionando corretamente

**Saída Problemática**:
```
[Rodada 1/5] Outliers detectados: ['Cliente_2_MALICIOSO', 'Cliente_3']
[Rodada 2/5] Outliers detectados: ['Cliente_2_MALICIOSO']
[Rodada 3/5] Outliers detectados: []
```

**Análise**:
- ⚠️ Cliente_3 honesto foi rejeitado (falso positivo)
- ⚠️ Rodada 3 não detectou atacante (falso negativo)
- 🔍 Investigar: Threshold, dados, implementação

## Guia de Extensão

### Adicionar Novo Dataset

**Passo 1**: Preparar CSV
```csv
feature1,feature2,feature3,target
1.2,3.4,5.6,7.8
...
```

**Passo 2**: Colocar em `project/data/meu-dataset/`

**Passo 3**: Ajustar `main()` em `modelagem.py`
```python
dataset_path = "../data/meu-dataset/dados.csv"
target_column = "target"  # Nome da coluna alvo
```

**Passo 4**: Executar
```bash
python modelagem.py
```

### Adicionar Novo Modelo

**Passo 1**: Criar novo wrapper em `modelagem.py`
```python
from sklearn.ensemble import RandomForestRegressor

class ModeloRandomForest(Modelo):
    def __init__(self):
        self._modelo_interno = RandomForestRegressor(n_estimators=100)
    
    def obter_pesos(self):
        # Serializar árvores (complexo!)
        # Para simplificar, pode não suportar FL diretamente
        return {'trees': self._modelo_interno.estimators_}
    
    def atualizar_pesos(self, pesos):
        # Desserializar árvores
        self._modelo_interno.estimators_ = pesos['trees']
```

**Passo 2**: Usar no servidor
```python
servidor = ServidorFederado(max_rodadas=5)
servidor.modelo_global = ModeloRandomForest()
```

**Desafio**: Modelos complexos (Random Forest, XGBoost) não agregam bem com média simples

### Adicionar Novo Algoritmo de Agregação

**Passo 1**: Implementar em `ServidorFederado`
```python
def _agregar_krum(self, k=2):
    """Krum: Seleciona k clientes com menores distâncias agregadas"""
    todos_coefs = []
    for cliente in self.clientes:
        coef = cliente.modelo_local.get_coef()
        if coef is not None:
            todos_coefs.append(coef)
    
    # Calcular distâncias pairwise
    n = len(todos_coefs)
    scores = []
    for i in range(n):
        distancias = []
        for j in range(n):
            if i != j:
                dist = np.linalg.norm(todos_coefs[i] - todos_coefs[j])
                distancias.append(dist)
        # Soma das k menores distâncias
        scores.append(sum(sorted(distancias)[:k]))
    
    # Selecionar cliente com menor score
    melhor_idx = np.argmin(scores)
    pesos_agregados = {
        'coef': todos_coefs[melhor_idx],
        'intercept': self.clientes[melhor_idx].modelo_local.get_intercept()
    }
    self.modelo_global.atualizar_pesos(pesos_agregados)
```

**Passo 2**: Usar na agregação
```python
# Substituir _agregar_modelos() por _agregar_krum()
self._agregar_krum(k=2)
```

### Adicionar Logs Estruturados

**Passo 1**: Importar logging
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('federated_learning.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
```

**Passo 2**: Substituir `print()` por `logger`
```python
# Antes
print(f"[Servidor] Iniciando Aprendizado Federado...")

# Depois
logger.info("Servidor iniciando Aprendizado Federado com %d clientes", len(self.clientes))
```

### Adicionar Dashboard Interativo

**Passo 1**: Instalar Streamlit
```bash
pip install streamlit
```

**Passo 2**: Criar `dashboard.py`
```python
import streamlit as st
import pandas as pd

st.title("Dashboard Aprendizado Federado")

# Upload de arquivo de resultados
uploaded_file = st.file_uploader("Upload experimento.log", type="log")

if uploaded_file:
    # Parsear log e exibir métricas
    st.metric("R² Final", 0.91)
    st.metric("Taxa de Detecção", "100%")
    
    # Gráfico interativo
    st.line_chart(dados_historico)
```

**Passo 3**: Executar
```bash
streamlit run dashboard.py
```

## Troubleshooting

### Problema: ModuleNotFoundError

**Erro**:
```
ModuleNotFoundError: No module named 'sklearn'
```

**Solução**:
```bash
# Verificar ambiente ativado
which python  # Linux/macOS
Get-Command python  # PowerShell

# Deve mostrar path do venv
# Se não, ativar:
source venv/bin/activate  # Linux/macOS
.\venv\Scripts\Activate.ps1  # PowerShell

# Reinstalar dependências
pip install -r dependencies/requirements.txt
```

### Problema: FileNotFoundError

**Erro**:
```
FileNotFoundError: [Errno 2] No such file or directory: '../data/iris/iris.csv'
```

**Solução**:
```bash
# Verificar diretório atual
pwd  # Linux/macOS
Get-Location  # PowerShell

# Deve estar em: .../project/modelagem/
# Se não, navegar:
cd project/modelagem

# Verificar dataset existe
ls ../data/iris/iris.csv
```

### Problema: Gráficos Não Aparecem

**Sintoma**: `resultados_fl.png` não é criado

**Solução**:
```bash
# Verificar backend matplotlib
python -c "import matplotlib; print(matplotlib.get_backend())"

# Se for 'Agg', está correto (salva arquivo)
# Se for 'TkAgg', pode ter problema

# Forçar backend Agg
export MPLBACKEND=Agg  # Linux/macOS
$env:MPLBACKEND = "Agg"  # PowerShell

# Executar novamente
python modelagem.py
```

### Problema: Todos Clientes Detectados como Outliers

**Sintoma**:
```
[Rodada 1/5] Agregação: 0 aceitos, 4 outliers
[ALERTA] Todos os clientes foram detectados como outliers!
```

**Causas**:
1. Threshold MAD muito conservador
2. Dados muito heterogêneos (Non-IID extremo)
3. Bug na implementação

**Solução**:
```python
# Ajustar threshold (linha 154)
threshold = np.median(desvios) + 2 * mad  # 3→2

# Verificar divisão de dados (deve ser IID)
dados = dados.sample(frac=1, random_state=42)  # Shuffle
```

### Problema: Performance Lenta

**Sintoma**: Rodada demora > 10 segundos

**Soluções**:

1. **Reduzir clientes**:
```python
num_clientes = 4  # Em vez de 10+
```

2. **Reduzir rodadas**:
```python
max_rodadas = 5  # Em vez de 20
```

3. **Desabilitar gráficos temporariamente**:
```python
# Comentar linha 367
# self.gerar_graficos()
```

4. **Usar dataset menor**:
```python
# Iris: 150 amostras (rápido)
# NSL-KDD: 125k amostras (lento)
```

## Referências Rápidas

### Comandos Essenciais

```bash
# Ativar ambiente
source venv/bin/activate  # Linux/macOS
.\venv\Scripts\Activate.ps1  # PowerShell

# Instalar dependências
pip install -r dependencies/requirements.txt

# Executar sistema
python modelagem.py

# Executar testes
python teste_iris.py

# Ver ajuda
python modelagem.py --help  # Não implementado ainda
```

### Estrutura de Arquivos

```
project/
├── modelagem/
│   ├── modelagem.py          → Sistema principal
│   ├── teste_iris.py         → Testes automatizados
│   ├── resultados_fl.png     → Gráficos gerados
│   └── README.md             → Documentação técnica
├── data/
│   └── iris/
│       └── iris.csv          → Dataset Iris
└── dependencies/
    └── requirements.txt      → Dependências Python
```

### Atalhos Úteis

**Limpar cache Python**:
```bash
find . -type d -name __pycache__ -exec rm -rf {} +  # Linux/macOS
Get-ChildItem -Recurse -Directory __pycache__ | Remove-Item -Recurse  # PowerShell
```

**Gerar relatório de experimento**:
```bash
python modelagem.py > relatorio.txt 2>&1
```

**Contar linhas de código**:
```bash
cloc modelagem.py  # Requer cloc instalado
```

---

**Última atualização**: Fevereiro 2026
**Dúvidas**: Abrir issue no repositório ou consultar docs/
