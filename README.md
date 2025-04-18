🎯 Objectif global
Créer un système intelligent de reconnaissance faciale enrichie en temps réel (âge, genre, émotion, regard) pour des applications de sécurité, contrôle d'accès ou surveillance.

🧠 Modules principaux :
1. 🔍 Reconnaissance faciale avec métadonnées
📷 Détection de visages avec OpenCV


🧠 Encodage facial avec face_recognition


📊 Métadonnées (nom, âge, genre) récupérées depuis un fichier Excel


✅ Affichage des infos si correspondance trouvée, ❌ sinon refus d’accès


📚 Libs : face_recognition, OpenCV, pandas, os

2. 📅 Détection d’âge (via CNN)
📂 Images triées par tranche d’âge (6–20, 25–30, 42–48, 60–98)


🏗️ Modèle CNN entraîné sur images 48x48 (niveaux de gris)


🧪 Prédiction de tranche d’âge en live via webcam


📦 Libs : OpenCV, Keras, NumPy, scikit-learn

3. 🙂 Détection d’émotions (via CNN)
7 émotions : angry, disgust, fear, happy, sad, surprise, neutral


📁 Images organisées par émotion


🧠 Modèle CNN entraîné, puis utilisé en temps réel sur visage détecté


📦 Libs : OpenCV, Keras, NumPy, scikit-learn

4. 👁️ Détection de la direction du regard (eye-tracking)
📂 Images d’yeux classées en left_look (0) et right_look (1)


🧠 Modèle CNN entraîné (48x48, grayscale)


📹 En temps réel : si regard détecté à droite → comportement suspect


🔒 Utilisation possible en sécurité/surveillance


📦 Libs : OpenCV, Keras, NumPy, scikit-learn

🎥 Fonctionnement en temps réel (pour tous les modules)
📸 Webcam activée avec OpenCV


🧠 Pour chaque frame : détection → prétraitement → prédiction


🖼️ Affichage dynamique : rectangles, texte d’info ou alerte


🛑 Arrêt avec touche q



💾 Modèles enregistrés :
Tâche
Fichier modèle
Âge
age_detection_model.h5
Émotion
emotion_detection_model.h5
Direction du regard
eye_tracking_model.h5

