import os
import shutil

# Diretório base
BASE_DIR = r"C:\Users\Administrador\Faculdade-Impacta\Iniciação-cientifica\project\notebooks\nsl-kdd\teste-images"

# Tipos de ataque
TIPOS_ATAQUE = [
    "neptune",
    "satan", 
    "ipsweep",
    "portsweep",
    "smurf",
    "nmap",
    "back",
    "teardrop",
    "warezclient",
    "pod",
    "guess_passwd"
]

print("=" * 80)
print("🗂️  ORGANIZANDO IMAGENS POR TIPO DE ATAQUE")
print("=" * 80)

# Criar pastas para cada tipo de ataque
for ataque in TIPOS_ATAQUE:
    pasta_ataque = os.path.join(BASE_DIR, ataque.upper())
    os.makedirs(pasta_ataque, exist_ok=True)
    print(f"✓ Pasta criada: {ataque.upper()}")

print("\n" + "=" * 80)
print("📁 MOVENDO IMAGENS PARA SUAS RESPECTIVAS PASTAS")
print("=" * 80)

# Listar todos os arquivos PNG no diretório base
arquivos = [f for f in os.listdir(BASE_DIR) if f.endswith('.png')]

contador = {ataque: 0 for ataque in TIPOS_ATAQUE}

for arquivo in arquivos:
    # Verificar qual tipo de ataque está no nome do arquivo
    movido = False
    for ataque in TIPOS_ATAQUE:
        if ataque in arquivo.lower():
            origem = os.path.join(BASE_DIR, arquivo)
            destino = os.path.join(BASE_DIR, ataque.upper(), arquivo)
            
            # Mover o arquivo
            shutil.move(origem, destino)
            contador[ataque] += 1
            movido = True
            break
    
    if not movido:
        # Arquivos que não são de nenhum ataque específico (resumos gerais)
        print(f"  ℹ️  Mantido na raiz: {arquivo}")

print("\n" + "=" * 80)
print("✅ ORGANIZAÇÃO CONCLUÍDA!")
print("=" * 80)

for ataque in TIPOS_ATAQUE:
    if contador[ataque] > 0:
        print(f"  {ataque.upper():20s} → {contador[ataque]:3d} imagens")

print("\n" + "=" * 80)
print(f"📂 Estrutura organizada em: {BASE_DIR}")
print("=" * 80)
