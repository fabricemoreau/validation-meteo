données depuis 2010 inclus
7 paramètres quotidiens
- TN : température minimale (°C)
- TX : température maximale
- TM : température moyenne
- RR : précipitations (pluies)
- GLOT : rayonnement global (J/cm²): rayons lumineux se diffusant sur la surface de la terre
- ETP : évapotranspiration potentielle: somme de la transpiration du couvert végétal et de l'évaporation du sol qui pourrait se produire en cas d'approvisionement en eau suffisant. 
D'autres paramètres à l'échelle horaire

Cas 1 :
En utilisant un export de la base de données météo contenant les données brutes et données validées:
- priorité 1: détection d'erreurs: classification. Il y a déjà des tests simples, mais traitant les paramètres de façon indépendnante. L'idée est d'expérimenter une méthode CNN prenant en compte l'ensemble des paramètres. La sortie est d'indiquer, pour chaque donnée de chaque paramètre, s'il y a détection d'une anomalie.
Les données étant d'usage stratégiques, on cherche en théorie à maximiser la sensibilité (peu de faux négatifs). Néanmoins, un équilibre avec la spécificité est à trouver pour limiter le nombre d'anomalies à analyser. 
Le premier objectif est que l'algorithme s'approche le plus possible des anomalies détectées dans le jeu de donnée. 
On considérera que les données corrigées dans le jeu de donnée qui ont un écart faible avec la donnée d'origine ne sont pas des anomalies (seuils à préciser)
Le réentrainement du modèle ne doit pas être trop fréquent pour avoir une qualité constante et obtenir suffisamment de données supplémentaires: tous les 2 ans est suffisant
- priorité 2: remplacement des données manquantes: régression. Il y a déjà une méthode basées sur l'interpolation spatiale et temporelle traitant tous les paramètres de façon indépendante. L'idée est d'expérimenter une méthode CNN prenant en compte l'ensemble des paramètres.
La correction de la pluie sera écartée: actuellement, les données corrigées de la base ne sont pas toujours expertisées
L'objectif est d'obtenir les valeurs estimées les plus proches possibles des valeurs corrigées. On pourra simuler des données manquantes dans le jeu de données pour entrainer le modèle.

intégration au processus métier: pour intégration dans le processus actuel, livrer les modèles sous forme d'application CLI qui prennent en paramètres les données nécessaires (ou sous forme de fichiers)
déploiement: Déploiement sur une machine de test reliée à une base de test: on compare la détection d'anomalie entre la chaîne en production (qui reste la référence) et le nouvel algorithme: production d'un rapport qui identifie les différences de détection d'anomalies et les différences de corrections proposées. Olivier annote les différences pertinentes pendant un mois.
risques: évaluation pendant une période climatique qui ne représente pas une année complète
Au bout d'un mois, décision si mise en production ou amélioration algo
KPI: calculer régulièrement :
- corrections retenues en production, anomalies détectées par algo
- ...
lancement
- documentation algo, démarche, résultats
- 
- communication aux ingénieurs recherche et offres de service: 

- *non priorisé ici: la validation par le responsable pourrait être améliorere par de l'ergonomie, notamment en présentant les données de façon groupées et mise en forme par groupe.* 


Cas 2 :
- on peut s'appuyer sur les données d'origine radar pour fusionner les données dans le cas des pluies
