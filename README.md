### Objectif global  
Développer un système intelligent de reconnaissance faciale enrichie en temps réel, intégrant l’analyse de l’âge, du genre, de l’émotion et de la direction du regard. Ce système est destiné à des applications telles que la sécurité, le contrôle d’accès ou la surveillance.

---

### Modules principaux

#### 1. Reconnaissance faciale avec métadonnées  
- Détection des visages via OpenCV  
- Encodage des visages avec `face_recognition`  
- Récupération des métadonnées (nom, âge, genre) depuis un fichier Excel  
- Affichage des informations si une correspondance est trouvée, sinon refus d’accès  
- Bibliothèques : `face_recognition`, `OpenCV`, `pandas`, `os`

#### 2. Détection de l’âge (via CNN)  
- Images classées par tranches d’âge : 6–20, 25–30, 42–48, 60–98  
- Modèle CNN entraîné sur des images 48x48 en niveaux de gris  
- Prédiction de la tranche d’âge en direct via webcam  
- Bibliothèques : `OpenCV`, `Keras`, `NumPy`, `scikit-learn`

#### 3. Détection d’émotions (via CNN)  
- Sept émotions détectées : angry, disgust, fear, happy, sad, surprise, neutral  
- Images organisées par classe émotionnelle  
- Modèle CNN entraîné et utilisé en temps réel sur visage détecté  
- Bibliothèques : `OpenCV`, `Keras`, `NumPy`, `scikit-learn`

#### 4. Détection de la direction du regard (eye-tracking)  
- Images classées en deux catégories : `left_look` (0) et `right_look` (1)  
- Modèle CNN entraîné sur des images 48x48 en niveaux de gris  
- En temps réel, détection du regard vers la droite interprétée comme comportement suspect  
- Bibliothèques : `OpenCV`, `Keras`, `NumPy`, `scikit-learn`

---

### Fonctionnement en temps réel  
- Activation de la webcam avec OpenCV  
- À chaque frame : détection, prétraitement, prédiction  
- Affichage dynamique des rectangles autour des visages, des informations ou alertes  
- Arrêt du système avec la touche `q`

---

### Fichiers des modèles entraînés

| Tâche                  | Fichier modèle                |
|------------------------|-------------------------------|
| Détection d’âge        | `age_detection_model.h5`      |
| Détection d’émotion    | `emotion_detection_model.h5`  |
| Détection du regard    | `eye_tracking_model.h5`       |

---
