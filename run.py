#!/usr/bin/env python3
"""
Lanceur de la Forge TTU-MC³
"""

import subprocess
import sys
import os

def check_dependencies():
    """Vérifie et installe les dépendances si nécessaire"""
    required = ['streamlit', 'plotly', 'pandas', 'numpy']
    missing = []
    
    for package in required:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing.append(package)
    
    if missing:
        print("📦 Installation des dépendances manquantes...")
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing)
        print("✅ Dépendances installées avec succès!")
    
    return True

def main():
    """Fonction principale"""
    print("""
    ⚒️╔══════════════════════════════════════════╗⚒️
    ⚒️║      LANCEUR DE LA FORGE TTU-MC³         ⚒️
    ⚒️║   Théorie Triadique Unifiée - MC³        ⚒️
    ⚒️╚══════════════════════════════════════════╝⚒️
    """)
    
    # Vérification des dépendances
    if not check_dependencies():
        print("❌ Échec de la vérification des dépendances")
        return
    
    # Démarrage de Streamlit
    print("🚀 Démarrage de la Forge...")
    print("🌐 L'application sera disponible sur http://localhost:8501")
    print("🛑 Appuyez sur Ctrl+C pour arrêter")
    print("\n" + "="*50 + "\n")
    
    # Lancement de l'application
    os.system("streamlit run app.py")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚒️ Forge arrêtée. À bientôt!")
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
