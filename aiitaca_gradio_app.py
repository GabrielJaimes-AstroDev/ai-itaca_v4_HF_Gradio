import gradio as gr
import os
import numpy as np
import matplotlib.pyplot as plt
from core_functions_3 import *
from cube_functions import *
import tempfile
import plotly.graph_objects as go
import plotly.express as px
import tensorflow as tf
import gdown
import shutil
import time
from astropy.io import fits
import warnings
from paths import GDRIVE_FOLDER_URL, TEMP_MODEL_DIR
# Añade esto al inicio del archivo para manejar la caché de modelos
import os
from pathlib import Path

# Configura rutas para Hugging Face
CACHE_DIR = Path("model_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# Modifica la función de descarga para usar la caché
def download_models_from_drive(folder_url, output_dir):
    output_dir = CACHE_DIR / output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Resto de la función igual que antes...

warnings.filterwarnings('ignore')

# Configuración inicial
if not os.path.exists(TEMP_MODEL_DIR):
    os.makedirs(TEMP_MODEL_DIR)

# Función para descargar modelos
def download_models_from_drive(folder_url, output_dir):
    model_files = [f for f in os.listdir(output_dir) if f.endswith('.keras')]
    data_files = [f for f in os.listdir(output_dir) if f.endswith('.npz')]

    if model_files and data_files:
        return model_files, data_files, True

    try:
        gdown.download_folder(
            folder_url, 
            output=output_dir, 
            quiet=True,
            use_cookies=False
        )
        
        model_files = [f for f in os.listdir(output_dir) if f.endswith('.keras')]
        data_files = [f for f in os.listdir(output_dir) if f.endswith('.npz')]
        
        return model_files, data_files, True
    except Exception as e:
        print(f"Error downloading models: {str(e)}")
        return [], [], False

# Descargar modelos al iniciar
model_files, data_files, models_downloaded = download_models_from_drive(GDRIVE_FOLDER_URL, TEMP_MODEL_DIR)

# Funciones de análisis
def analyze_spectrum_wrapper(file_path, selected_model, freq_unit, intensity_unit, 
                           sigma_emission, window_size, sigma_threshold, 
                           fwhm_ghz, tolerance_ghz, min_peak_height_ratio,
                           top_n_lines, top_n_similar):
    try:
        # Configuración
        config = {
            'trained_models_dir': TEMP_MODEL_DIR,
            'peak_matching': {
                'sigma_emission': sigma_emission,
                'window_size': window_size,
                'sigma_threshold': sigma_threshold,
                'fwhm_ghz': fwhm_ghz,
                'tolerance_ghz': tolerance_ghz,
                'min_peak_height_ratio': min_peak_height_ratio,
                'top_n_lines': top_n_lines,
                'debug': True,
                'top_n_similar': top_n_similar
            },
            'units': {
                'frequency': freq_unit,
                'intensity': intensity_unit
            }
        }
        
        # Cargar modelo
        mol_name = selected_model.replace('_model.keras', '')
        model_path = os.path.join(TEMP_MODEL_DIR, selected_model)
        model = tf.keras.models.load_model(model_path)
        
        # Cargar datos de entrenamiento
        data_file = os.path.join(TEMP_MODEL_DIR, f'{mol_name}_train_data.npz')
        with np.load(data_file) as data:
            train_freq = data['train_freq']
            train_data = data['train_data']
            train_logn = data['train_logn']
            train_tex = data['train_tex']
            headers = data['headers']
            filenames = data['filenames']
        
        # Analizar espectro
        results = analyze_spectrum(
            file_path, model, train_data, train_freq,
            filenames, headers, train_logn, train_tex,
            config, mol_name
        )
        
        # Ajustar unidades
        freq_conversion = {
            "GHz": 1e9,
            "MHz": 1e6,
            "kHz": 1e3,
            "Hz": 1.0
        }
        
        intensity_conversion = {
            "K": 1.0,
            "Jy": 1.0
        }
        
        if freq_unit != 'GHz':
            results['input_freq'] = results['input_freq'] * 1e9 / freq_conversion[freq_unit]
            results['best_match']['x_synth'] = results['best_match']['x_synth'] * 1e9 / freq_conversion[freq_unit]
        
        if intensity_unit != 'K':
            results['input_spec'] = results['input_spec'] * intensity_conversion[intensity_unit]
            results['best_match']['y_synth'] = results['best_match']['y_synth'] * intensity_conversion[intensity_unit]
        
        # Crear figura interactiva
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=results['input_freq'],
            y=results['input_spec'],
            mode='lines',
            name='Input Spectrum',
            line=dict(color='white', width=2)))
        
        fig.add_trace(go.Scatter(
            x=results['best_match']['x_synth'],
            y=results['best_match']['y_synth'],
            mode='lines',
            name='Best Match',
            line=dict(color='red', width=2)))
        
        fig.update_layout(
            plot_bgcolor='#0D0F14',
            paper_bgcolor='#0D0F14',
            margin=dict(l=50, r=50, t=60, b=50),
            xaxis_title=f'Frequency ({freq_unit})',
            yaxis_title=f'Intensity ({intensity_unit})',
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            height=600,
            font=dict(color='white'),
            xaxis=dict(gridcolor='#3A3A3A'),
            yaxis=dict(gridcolor='#3A3A3A')
        )
        
        # Crear resumen de resultados
        summary = f"""
        <div style="background-color: #0D0F14; padding: 15px; border-radius: 5px; color: white;">
            <h3 style="color: #1E88E5; margin-top: 0;">Detection of Physical Parameters</h3>
            <p><strong>LogN:</strong> {results['best_match']['logn']:.2f} cm⁻²</p>
            <p><strong>Tex:</strong> {results['best_match']['tex']:.2f} K</p>
            <p><strong>File (Top CNN Train):</strong> {results['best_match']['filename']}</p>
        </div>
        """
        
        return summary, fig, results
    
    except Exception as e:
        return f"Error during analysis: {e}", None, None

# Función para visualizar cubos FITS
def visualize_cube(cube_file, channel, scale, show_rms):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".fits") as tmp_cube:
            tmp_cube.write(cube_file)
            tmp_cube_path = tmp_cube.name
        
        cube_info = load_alma_cube(tmp_cube_path)
        
        if len(cube_info['data'].shape) == 3:
            img_data = cube_info['data'][channel, :, :]
        else:
            img_data = cube_info['data']
        
        if scale == "Log":
            img_data = np.log10(img_data - np.nanmin(img_data) + 1)
        elif scale == "Sqrt":
            img_data = np.sqrt(img_data - np.nanmin(img_data))
        
        fig = px.imshow(
            img_data,
            origin='lower',
            color_continuous_scale='viridis',
            labels={'color': 'Intensity (K)'},
            title=f"Channel {channel}" + (f" ({cube_info['freq_axis'][channel]/1e9:.4f} GHz)" if cube_info['freq_axis'] is not None else "")
        )
        
        fig.update_layout(
            plot_bgcolor='#0D0F14',
            paper_bgcolor='#0D0F14',
            margin=dict(l=50, r=50, t=80, b=50),
            xaxis_title="RA (pixels)",
            yaxis_title="Dec (pixels)",
            font=dict(color='white'),
            xaxis=dict(gridcolor='#3A3A3A'),
            yaxis=dict(gridcolor='#3A3A3A')
        )
        
        os.unlink(tmp_cube_path)
        return fig, cube_info
    
    except Exception as e:
        return f"Error processing ALMA cube: {str(e)}", None

# Interfaz de Gradio
with gr.Blocks(theme=gr.themes.Default(primary_hue="blue", secondary_hue="gray")) as demo:
    gr.Markdown("# AI-ITACA | Spectrum Analyzer")
    gr.Markdown("### Molecular Spectrum Analysis Tool")
    
    with gr.Tabs():
        with gr.TabItem("Molecular Analyzer"):
            with gr.Row():
                with gr.Column():
                    file_input = gr.File(label="Input Spectrum File", file_types=[".txt", ".dat", ".fits", ".spec"])
                    model_dropdown = gr.Dropdown(model_files, label="Select Molecule Model")
                    
                    with gr.Accordion("Units Configuration", open=False):
                        freq_unit = gr.Dropdown(["GHz", "MHz", "kHz", "Hz"], value="GHz", label="Frequency Units")
                        intensity_unit = gr.Dropdown(["K", "Jy"], value="K", label="Intensity Units")
                    
                    with gr.Accordion("Peak Matching Parameters", open=False):
                        sigma_emission = gr.Slider(0.1, 5.0, value=1.5, step=0.1, label="Sigma Emission")
                        window_size = gr.Slider(1, 20, value=3, step=1, label="Window Size")
                        sigma_threshold = gr.Slider(0.1, 5.0, value=2.0, step=0.1, label="Sigma Threshold")
                        fwhm_ghz = gr.Slider(0.01, 0.5, value=0.05, step=0.01, label="FWHM (GHz)")
                        tolerance_ghz = gr.Slider(0.01, 1.0, value=0.1, step=0.01, label="Tolerance (GHz)")
                        min_peak_height_ratio = gr.Slider(0.1, 1.0, value=0.3, step=0.05, label="Min Peak Height Ratio")
                        top_n_lines = gr.Slider(5, 100, value=30, step=5, label="Top N Lines")
                        top_n_similar = gr.Slider(50, 5000, value=50, step=50, label="Top N Similar")
                    
                    analyze_btn = gr.Button("Analyze Spectrum", variant="primary")
                
                with gr.Column():
                    results_summary = gr.HTML()
                    plot_output = gr.Plot()
        
            analyze_btn.click(
                fn=analyze_spectrum_wrapper,
                inputs=[file_input, model_dropdown, freq_unit, intensity_unit,
                       sigma_emission, window_size, sigma_threshold,
                       fwhm_ghz, tolerance_ghz, min_peak_height_ratio,
                       top_n_lines, top_n_similar],
                outputs=[results_summary, plot_output]
            )
        
        with gr.TabItem("Cube Visualizer"):
            with gr.Row():
                with gr.Column():
                    cube_input = gr.File(label="Upload ALMA Cube (FITS format)", file_types=[".fits"])
                    channel_slider = gr.Slider(0, 100, value=50, step=1, label="Channel")
                    scale_radio = gr.Radio(["Linear", "Log", "Sqrt"], value="Linear", label="Image Scale")
                    show_rms_check = gr.Checkbox(value=True, label="Show RMS noise level")
                    visualize_btn = gr.Button("Visualize Cube", variant="primary")
                
                with gr.Column():
                    cube_plot = gr.Plot()
            
            def update_channel_slider(cube_file):
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".fits") as tmp_cube:
                        tmp_cube.write(cube_file)
                        tmp_cube_path = tmp_cube.name
                    
                    cube_info = load_alma_cube(tmp_cube_path)
                    os.unlink(tmp_cube_path)
                    
                    if len(cube_info['data'].shape) == 3:
                        max_chan = cube_info['n_chan'] - 1
                        return gr.Slider(maximum=max_chan, value=max_chan//2)
                    return gr.Slider(maximum=100, value=50)
                except:
                    return gr.Slider(maximum=100, value=50)
            
            cube_input.change(
                fn=update_channel_slider,
                inputs=cube_input,
                outputs=channel_slider
            )
            
            visualize_btn.click(
                fn=visualize_cube,
                inputs=[cube_input, channel_slider, scale_radio, show_rms_check],
                outputs=[cube_plot]
            )

# Iniciar la aplicación
demo.launch()
