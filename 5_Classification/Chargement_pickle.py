import numpy as np
import pickle
import inspect

# Charger le modèle sauvegardé
model_filename = 'models/best_model_f_classif_LinearSVC_mean_k41.pkl'
# Charger le modèle avec pickle
with open(model_filename, 'rb') as f:
    pipeline = pickle.load(f)

# Inspect each step in the pipeline
for step_name, step_model in pipeline.steps:
    print(f"Step: {step_name}")
    print(f"Type: {type(step_model)}")

    # If it's a function or model, you can inspect its details
    if hasattr(step_model, 'coef_'):  # For classifiers like LinearSVC
        print(f"Model coefficients: {step_model.coef_}")
        print(f"coef nb : {np.shape(step_model.coef_)}")
    if hasattr(step_model, 'get_params'):  # For models or selectors
        print(f"Parameters: {step_model.get_params()}")
    
    # If it's a function, inspect its signature and source code
    if callable(step_model):
        print(f"Signature: {inspect.signature(step_model)}")
        print(f"Source code: {inspect.getsource(step_model)}")
# Charger les nouvelles données que vous souhaitez prédire
# Assurez-vous que ces données sont formatées de la même manière que celles utilisées pour l'entraînement.
X_new = np.load('Ind1_F2/stacked_mean.npy')  # Exemple d'une nouvelle matrice de données
print(np.shape(X_new))

# Utilisation du modèle pour faire des prédictions
y_pred = pipeline.predict(X_new)
