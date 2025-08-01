import numpy as np
import pandas as pd 
from prepare_dataset import prepare_dataset
from scipy.signal import savgol_filter
from fastai.tabular.all import *
from fastai.callback.tracker import EarlyStoppingCallback
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import os
from pathlib import Path

# --- Configuración inicial ---
spectra_folder = r"/home/cab/Documents/CLEANED_SPECTRA_Guapos_CH3CN/ALL"
logn_model_file = "logn_regressor_CH3CN_improved.pkl"
tex_model_file = "tex_regressor_CH3CN.pkl"
force_retrain = False
sample_every = 10  # Submuestreo de frecuencias para reducir dimensionalidad

# Directorios de resultados
current_dir = Path(os.getcwd())
plots_dir = current_dir / "training_plots"
models_dir = current_dir / "saved_models"
plots_dir.mkdir(exist_ok=True)
models_dir.mkdir(exist_ok=True)

print(f"Directorio actual: {current_dir}")

# --- Carga y preprocesamiento de datos ---
print("\nCargando y preparando dataset...")
X, y, texs = prepare_dataset(spectra_folder, n_points=30000)

# Verificación de datos
print(f"\nResumen de datos cargados:")
print(f"- Forma de X (espectros): {X.shape}")
print(f"- Mín/Media/Máx de intensidades: {np.min(X):.2e}/{np.mean(X):.2e}/{np.max(X):.2e}")
print(f"- Ejemplo de logN: {y[:5]}...")
print(f"- Ejemplo de Tex: {texs[:5]}...")

# Ajuste de Tex si es necesario
if len(texs) != len(y):
    print("\n¡Advertencia! Número de Tex no coincide con espectros. Usando valor medio.")
    texs = np.full(len(y), np.mean(texs))

# Procesamiento de espectros
print("\nProcesando espectros...")
X = savgol_filter(X, window_length=11, polyorder=3)  # Suavizado
X_norm = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)

# Visualización de espectros de ejemplo
plt.figure(figsize=(12, 6))
for i in np.random.choice(len(X), 5, replace=False):
    plt.plot(X_norm[i], alpha=0.6, label=f'logN={y[i]:.1f}, Tex={texs[i]:.1f}')
plt.title("Espectros normalizados (ejemplos aleatorios)")
plt.xlabel("Puntos de frecuencia")
plt.ylabel("Intensidad normalizada")
plt.legend()
plt.savefig(plots_dir / "ejemplo_espectros.png", dpi=150)
plt.close()

# --- Preparación del DataFrame ---
print("\nCreando DataFrame...")
data_dict = {'logN': y, 'Tex': texs}
for i in range(0, X_norm.shape[1], sample_every):
    data_dict[f'spec_{i}'] = X_norm[:, i]
spectra_df = pd.DataFrame(data_dict)

# --- División de datos ---
splits = RandomSplitter(valid_pct=0.15, seed=42)(range_of(spectra_df))

# --- Configuración del modelo ---
def get_learner(df, y_col):
    dls = TabularDataLoaders.from_df(
        df, 
        y_names=y_col, 
        splits=splits, 
        bs=64, 
        procs=[Normalize],
        cont_names = [col for col in df.columns if col.startswith('spec_')]
    )
    
    return tabular_learner(
        dls,
        layers=[512, 256, 128],
        metrics=[mae, rmse],
        loss_func=MSELossFlat(),
        y_range=(float(df[y_col].min()*0.9), float(df[y_col].max()*1.1))
    )

# --- Entrenamiento ---
def train_model(learner, model_path, model_name):
    print(f"\nEntrenando modelo {model_name}...")
    
    # Busqueda de learning rate
    print("Buscando mejor learning rate...")
    lr_min, lr_steep = learner.lr_find(suggest_funcs=(minimum, steep))
    print(f"LR sugerido: min={lr_min:.2e}, steep={lr_steep:.2e}")
    
    # Entrenamiento con early stopping
    learner.fit_one_cycle(
        50,
        lr_max=lr_steep,
        cbs=[
            EarlyStoppingCallback(monitor='valid_loss', patience=5, min_delta=0.1),
            SaveModelCallback(monitor='valid_loss', fname=model_path.stem)
        ]
    )
    
    return learner

# --- Entrenamiento principal ---
try:
    # Modelo para logN
    print("\n=== MODELO logN ===")
    logn_path = models_dir / logn_model_file
    learn_logn = get_learner(spectra_df, 'logN')
    
    if force_retrain or not logn_path.exists():
        learn_logn = train_model(learn_logn, logn_path, 'logN')
    else:
        learn_logn.load(logn_path.stem)
        print(f"Modelo cargado: {logn_path}")

    # Modelo para Tex
    print("\n=== MODELO Tex ===")
    tex_path = models_dir / tex_model_file
    learn_tex = get_learner(spectra_df, 'Tex')
    
    if force_retrain or not tex_path.exists():
        learn_tex = train_model(learn_tex, tex_path, 'Tex')
    else:
        learn_tex.load(tex_path.stem)
        print(f"Modelo cargado: {tex_path}")

    # --- Evaluación ---
    def evaluate_and_plot(learner, target_name):
        try:
            preds, targs = learner.get_preds()
            targs = targs.numpy().flatten()
            preds = preds.numpy().flatten()
            
            # Métricas
            mae_val = mean_absolute_error(targs, preds)
            r2_val = r2_score(targs, preds)
            
            print(f"\nEvaluación {target_name}:")
            print(f"- MAE: {mae_val:.4f}")
            print(f"- R2: {r2_val:.4f}")
            
            # Gráfico de predicciones
            plt.figure(figsize=(8, 6))
            plt.scatter(targs, preds, alpha=0.6)
            plt.plot([min(targs), max(targs)], [min(targs), max(targs)], 'r--')
            plt.xlabel(f'True {target_name}')
            plt.ylabel(f'Predicted {target_name}')
            plt.title(f'{target_name} Predictions\nMAE: {mae_val:.3f}, R2: {r2_val:.3f}')
            plt.grid(True)
            plt.savefig(plots_dir / f'{target_name}_predictions.png', dpi=300)
            plt.close()
            
            # Histograma de errores
            errors = preds - targs
            plt.figure(figsize=(8, 6))
            plt.hist(errors, bins=30, edgecolor='k')
            plt.xlabel('Error (Predicho - Real)')
            plt.ylabel('Frecuencia')
            plt.title(f'Distribución de errores - {target_name}')
            plt.grid(True)
            plt.savefig(plots_dir / f'{target_name}_errors.png', dpi=300)
            plt.close()
            
            return targs, preds
        except Exception as e:
            print(f"Error en evaluación de {target_name}: {str(e)}")
            return None, None

    logn_true, logn_pred = evaluate_and_plot(learn_logn, 'logN')
    tex_true, tex_pred = evaluate_and_plot(learn_tex, 'Tex')

except Exception as e:
    print(f"\nERROR durante el entrenamiento: {str(e)}")
    raise

print("\nProceso completado. Resultados guardados en:")
print(f"- Gráficos: {plots_dir}")
print(f"- Modelos: {models_dir}")