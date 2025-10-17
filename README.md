#  Prédiction du Temps de Livraison - Projet Machine Learning

## 📋 Table des Matières
- [Contexte du Projet](#contexte-du-projet)
- [Objectifs](#objectifs)
- [Données](#données)
- [Architecture du Projet](#architecture-du-projet)
- [Installation](#installation)
- [Méthodologie](#méthodologie)
- [Résultats](#résultats)
- [Tests Automatisés](#tests-automatisés)
- [Utilisation](#utilisation)
- [Technologies Utilisées](#technologies-utilisées)
- [Auteur](#auteur)

---

## Contexte du Projet

Ce projet a été développé dans le cadre d'une mission de data science pour une entreprise de logistique et de livraison. L'objectif principal est de créer un modèle de Machine Learning capable de prédire avec précision le temps total d'une livraison, depuis la commande jusqu'à la réception par le client.

### Problématique Actuelle
- Estimations manuelles du temps de livraison
- Absence de modèle de prédiction fiable
- Retards fréquents générant une insatisfaction client
- Besoin d'automatisation et d'optimisation des tournées

### Bénéfices Attendus
- ✅ Anticiper les retards potentiels
- ✅ Informer les clients en temps réel
- ✅ Optimiser l'organisation des tournées
- ✅ Améliorer la satisfaction client

---

## Objectifs

Développer un modèle de régression performant pour prédire la variable cible **DeliveryTime** (temps total de livraison en minutes) en exploitant les facteurs suivants :

| Variable | Description |
|----------|-------------|
| `Distance_km` | Distance entre le restaurant et l'adresse de livraison |
| `Traffic_Level` | Niveau de trafic (Low, Medium, High) |
| `Vehicle_Type` | Type de véhicule utilisé (Bike, Scooter, Car) |
| `Time_of_Day` | Heure de la journée (Morning, Afternoon, Evening, Night) |
| `Courier_Experience` | Expérience du livreur (en années) |
| `Weather` | Conditions météorologiques (Sunny, Rainy, Cloudy, Stormy) |
| `Preparation_Time` | Temps de préparation de la commande (en minutes) |

---

## Données

Les données utilisées comprennent 7 variables explicatives (features) pour prédire le temps de livraison. Le dataset contient des variables numériques et catégorielles nécessitant un prétraitement adapté.

---

## Architecture du Projet

```
prediction-du-temps-de-livraison/
│
├── .github/
│ └── workflows/
│ └── python_tests.yml
├── dataset.csv
├── main.ipynb
├── script.py
├── test_pipeline.py
├── requirements.txt
└── README.md
```

---

## Installation

### Étapes d'Installation

```bash
# Cloner le repository
git clone https://github.com/Khaoula1025/Prediction-du-Temps-de-Livraison.git

cd Prediction-du-Temps-de-Livraison

# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt
```

### Dépendances Principales
```
pandas
numpy
scikit-learn
matplotlib
seaborn
pytest
```

---

## Méthodologie

### 1. Analyse Exploratoire des Données (EDA)

#### Visualisations Réalisées
- **Heatmap de Corrélation** : Identification des relations entre variables numériques
- **Countplots** : Distribution des variables catégorielles
- **Boxplots** : Relation entre variables catégorielles et temps de livraison
- **Distribution de la Cible** : Analyse de la normalité et des outliers

#### Insights Clés
- Corrélation forte entre Distance_km et DeliveryTime
- Impact significatif du Traffic_Level sur les temps de livraison
- Variations selon le Time_of_Day

### 2. Prétraitement des Données

```python
# Pipeline de prétraitement
- StandardScaler pour variables numériques
- OneHotEncoder pour variables catégorielles
- SelectKBest avec f_regression pour sélection de features
```

### 3. Modélisation

#### Modèles Testés
1. **RandomForestRegressor**
   - Grid d'hyperparamètres testés
   - Avantages : robuste, gère bien les non-linéarités
   
2. **Support Vector Regression (SVR)**
   - Grid d'hyperparamètres testés
   - Avantages : efficace en haute dimension

#### GridSearchCV
```python
# Configuration
- Validation croisée : 5-fold CV
- Métrique d'optimisation : MAE (Mean Absolute Error)
- Scoring secondaire : R²
```

### 4. Pipeline Scikit-Learn (Bonus)

Pipeline intégré comprenant :
1. Prétraitement automatisé
2. Sélection de features
3. Modélisation
4. Prévention des fuites de données

---

## Résultats

### Performance des Modèles

| Modèle | MAE (Test) | R² (Test) |
|--------|------------|-----------|
| RandomForest | 6.86 | 0.78 | 
| SVR | 6.01  | 0.81|

### Modèle Retenu
**SVR** avec les hyperparamètres suivants :
```python
# Meilleurs hyperparamètres
{
    'model__C': 10, 
    'model__gamma': 'auto', 
    'model__kernel': 'rbf'
}
```

### Justification du Choix
Le modèle **SVR (Support Vector Regression)** a été retenu car il offre le meilleur compromis entre performance, temps de calcul et interprétabilité.
Il a obtenu **les meilleurs résultats (MAE le plus faible, R² le plus élevé)** tout en restant rapide à entraîner sur notre jeu de données.
Son ajustement flexible via les hyperparamètres (C, kernel, gamma) le rend particulièrement adapté à la prédiction du temps de livraison.

### Analyse des Erreurs
- MAE moyenne : 6 minutes
- Interprétation : En moyenne, le modèle se trompe de 6 minutes
- Performance acceptable pour les objectifs métier

---

## Tests Automatisés

### Tests Unitaires Implémentés

```python
# test_model.py
1. test_cleanData() : Vérification format et dimensions
2. test_mae_threshold() : MAE < seuil acceptable
```

### Exécution des Tests
```bash
# Lancer tous les tests
pytest -v

# Lancer un test spécifique
pytest tests/test_pipeline.py::test_mae_threshold -v
```

### Intégration Continue (CI/CD)
GitHub Actions configuré pour :
- Exécution automatique des tests à chaque push
- Validation de l'environnement Python
- Installation des dépendances
- Rapport de couverture des tests

---

## Utilisation

Pour utiliser le modèle entraîné :

```python
# Charger le modèle
from script import load_model

# Faire une prédiction
model = load_model()
prediction = model.predict(new_data)
```

---

## Technologies Utilisées

- **Langage** : Python 3.11
- **ML/Data Science** : scikit-learn, pandas, numpy
- **Visualisation** : matplotlib, seaborn
- **Tests** : pytest
- **CI/CD** : GitHub Actions
- **Gestion de Projet** : Jira
- **Versioning** : Git/GitHub

---

## Gestion de Projet

### Organisation avec Jira
| **Key**   | **Task**                     | **Description / Summary**                                                                  | **Status** | **Due Date** |
| --------- | ---------------------------- | ------------------------------------------------------------------------------------------ | ---------- | ------------ |
| PDTDLB-1  | **Project Setup**            | Initial setup of the project environment, structure, and dependencies.                     | 🟩 DONE    | 13 Oct 2025  |
| PDTDLB-4  | **Data Analysis & Cleaning** | Performed data exploration, handled missing values, and prepared the dataset for modeling. | 🟩 DONE    | 13 Oct 2025  |
| PDTDLB-7  | **Preprocessing**            | Applied data preprocessing techniques such as encoding, scaling, and feature engineering.  | 🟩 DONE    | 14 Oct 2025  |
| PDTDLB-12 | **Model Training**           | Trained machine learning models and optimized performance.                                 | 🟩 DONE    | 16 Oct 2025  |
| PDTDLB-18 | **Pipeline (Bonus)**         | Implemented a pipeline to automate data transformation and model execution.                | 🟩 DONE    | 16 Oct 2025  |
| PDTDLB-21 | **Testing & Validation**     | Evaluated model accuracy using validation and testing metrics.                             | 🟩 DONE    | 16 Oct 2025  |
| PDTDLB-24 | **GitHub Actions (Bonus)**   | Set up CI/CD automation for testing and deployment using GitHub Actions.                   | 🟩 DONE    | 16 Oct 2025  |
| PDTDLB-25 | **Report**                   | Compiled project results, findings, and insights into a final report.                      | 🟩 DONE    | 16 Oct 2025  |

### Workflow Git
```
main (production)
  └── develop (développement)
       └── feature/* (fonctionnalités)
```

---

## Rapport de Synthèse

### Démarche Suivie
1. Exploration approfondie des données
2. Identification des patterns et corrélations
3. Prétraitement adapté aux types de variables
4. Sélection rigoureuse des features
5. Comparaison de modèles avec validation croisée
6. Optimisation des hyperparamètres
7. Validation sur jeu de test
8. Mise en place de tests automatisés

### Résultats Obtenus
Le modèle développé permet de prédire le temps de livraison avec une erreur moyenne de 6 minutes, ce qui répond aux objectifs métier fixés. L'automatisation via pipeline et l'intégration continue garantissent la reproductibilité et la maintenabilité du code.

---

## Auteur

**Mohamed boukhali**
- 
- Email : elboukhalimohammed@gmail.com

---



---

**Date de dernière mise à jour** : 17/10/2025  
