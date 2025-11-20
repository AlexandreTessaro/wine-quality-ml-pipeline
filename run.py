"""
Script auxiliar para executar o pipeline
Resolve problemas de caminho automaticamente
"""
import os
import sys
from pathlib import Path

# Encontrar o diretório do script
script_dir = Path(__file__).parent.absolute()
os.chdir(script_dir)
sys.path.insert(0, str(script_dir))
sys.path.insert(0, str(script_dir / 'src'))

# Executar main
if __name__ == "__main__":
    from main import main
    main()

