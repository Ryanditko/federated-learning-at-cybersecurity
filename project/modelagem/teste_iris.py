"""
Teste Simples do Sistema de Aprendizado Federado com Iris Dataset
Valida o funcionamento básico do sistema sem ataques
"""
import sys
import os
sys.path.append(os.path.dirname(__file__))

from modelagem import (
    ServidorFederado,
    ClienteMalicioso,
    carregar_dataset_iris,
    dividir_dados_clientes
)

def teste_simples_sem_ataques():
    """Testa o sistema com todos os clientes honestos"""
    print("="*70)
    print("TESTE 1: Sistema FL com TODOS os clientes honestos")
    print("="*70)
    
    # Carrega dataset
    X, y = carregar_dataset_iris()
    
    # Divide dados entre 3 clientes
    dados_clientes, dados_validacao = dividir_dados_clientes(X, y, n_clientes=3, validacao_size=0.2)
    
    # Cria servidor
    servidor = ServidorFederado(max_rodadas=5, dados_validacao=dados_validacao)
    
    # Adiciona clientes honestos
    print("\nConfigurando 3 clientes honestos:")
    for i in range(3):
        servidor.adicionar_cliente(
            ClienteMalicioso(f"Cliente_{i+1}_Honesto", dados_clientes[i], "target", "nenhum")
        )
    
    # Executa
    servidor.executar_aprendizado_federado()
    
    # Verifica resultados
    print("\n" + "="*70)
    print("RESULTADOS DO TESTE:")
    print("="*70)
    
    if servidor.historico_r2_global:
        r2_final = servidor.historico_r2_global[-1]
        r2_inicial = servidor.historico_r2_global[0]
        
        print(f"R² inicial: {r2_inicial:.4f}")
        print(f"R² final: {r2_final:.4f}")
        print(f"Melhoria: {(r2_final - r2_inicial):.4f}")
        
        if r2_final > 0.7:
            print("\nRESULTADO: SUCESSO - Modelo convergiu com boa qualidade")
        else:
            print("\nRESULTADO: ALERTA - Modelo pode precisar de mais rodadas")
    
    print(f"\nOutliers detectados: {len(servidor.outliers_detectados)}")
    print("="*70)


def teste_com_um_ataque():
    """Testa o sistema com 1 cliente malicioso"""
    print("\n\n" + "="*70)
    print("TESTE 2: Sistema FL com 1 cliente malicioso (envenenamento de dados)")
    print("="*70)
    
    # Carrega dataset
    X, y = carregar_dataset_iris()
    
    # Divide dados entre 4 clientes
    dados_clientes, dados_validacao = dividir_dados_clientes(X, y, n_clientes=4, validacao_size=0.2)
    
    # Cria servidor
    servidor = ServidorFederado(max_rodadas=5, dados_validacao=dados_validacao)
    
    # Adiciona clientes
    print("\nConfigurando clientes:")
    print("  - 3 clientes honestos")
    print("  - 1 cliente malicioso (envenenamento de dados)")
    
    servidor.adicionar_cliente(
        ClienteMalicioso("Cliente_1_Honesto", dados_clientes[0], "target", "nenhum")
    )
    servidor.adicionar_cliente(
        ClienteMalicioso("Cliente_2_MALICIOSO", dados_clientes[1], "target", "dados")
    )
    servidor.adicionar_cliente(
        ClienteMalicioso("Cliente_3_Honesto", dados_clientes[2], "target", "nenhum")
    )
    servidor.adicionar_cliente(
        ClienteMalicioso("Cliente_4_Honesto", dados_clientes[3], "target", "nenhum")
    )
    
    # Executa
    servidor.executar_aprendizado_federado()
    
    # Verifica resultados
    print("\n" + "="*70)
    print("RESULTADOS DO TESTE:")
    print("="*70)
    
    if servidor.historico_r2_global:
        r2_final = servidor.historico_r2_global[-1]
        print(f"R² final: {r2_final:.4f}")
    
    print(f"\nOutliers detectados: {len(servidor.outliers_detectados)}")
    
    if servidor.outliers_detectados:
        print("\nDetecções realizadas:")
        for deteccao in servidor.outliers_detectados:
            print(f"  Rodada {deteccao['rodada']}: {deteccao['clientes']}")
        
        if "Cliente_2_MALICIOSO" in str(servidor.outliers_detectados):
            print("\nRESULTADO: SUCESSO - Cliente malicioso foi detectado!")
        else:
            print("\nRESULTADO: PARCIAL - Cliente malicioso não foi detectado")
    else:
        print("\nRESULTADO: FALHA - Nenhum outlier detectado")
    
    print("="*70)


if __name__ == "__main__":
    # Executa testes
    teste_simples_sem_ataques()
    teste_com_um_ataque()
    
    print("\n\n" + "="*70)
    print("TESTES CONCLUÍDOS")
    print("="*70)
