# Configurer Python dans VSCode
- utiliser le développement dans les conteneurs: voir .devcontainer
- extensions: installer les extensions recommandées (voir dossier .vscode/extensions.json)

# Démarrer le conteneur de développement
- Menu F1/Dev Containers: Rebuild dev Containers

# Déboguer
- pas de config supplémentaire
- ouvrir un fichier .py
- dans le menu latéral de débug, cliquer sur la flèche verte

# Exécuter
- dans l'onglet "Terminal" sous le source, il y a un menu latéral à droite pour choisir:
  - bash: commandes bash comme par exemple `pip install monmodule`
  - Python: console python
- sélectionner des bouts de codes et faire Maj+Entrée pour les envoyer dans un terminal

# Installer des modules
- dans la console: 
```bash
pip install monmodule
pip freeze > requirements.txt
```
- Menu F1/Dev Containers: Rebuild dev Containers

# Autocomplétion
- Ctrl+Espace

# Vérifier syntaxe/bonnes pratiques
- F1/ MyPy: Recheck Workspace (ou Restart Daemon)
