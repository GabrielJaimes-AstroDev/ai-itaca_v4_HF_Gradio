import os
import numpy as np
import tempfile
import plotly.graph_objects as go
from scipy.interpolate import interp1d
from astropy.io import fits
import joblib
import pandas as pd
import warnings
from io import StringIO
import matplotlib.pyplot as plt
import gradio as gr
from glob import glob
import shutil

# =============================================
# CONFIGURACIÓN DE PATHS LOCALES
# =============================================
LOCAL_MODEL_DIR = "RF_Models"
LOCAL_FILTER_DIR = "RF_Filters"

# =============================================
# HELPER FUNCTIONS (Mismos que en la versión Streamlit)
# =============================================
def list_local_files(directory):
    """Recursively list all local files with detailed information"""
    file_list = []
    try:
        for root, dirs, files in os.walk(directory):
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            
            for file in files:
                if not file.startswith('.'):
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, directory)
                    try:
                        size = os.path.getsize(full_path)
                        size_str = f"{size/1024:.1f} KB" if size > 1024 else f"{size} B"
                        file_list.append({
                            'path': rel_path,
                            'size': size_str,
                            'full_path': full_path,
                            'is_file': True,
                            'parent_dir': os.path.basename(root)
                        })
                    except Exception as e:
                        file_list.append({
                            'path': rel_path,
                            'size': 'Error',
                            'full_path': full_path,
                            'is_file': True,
                            'parent_dir': os.path.basename(root)
                        })
    except Exception as e:
        print(f"Error listing files in {directory}: {str(e)}")
    return file_list

def robust_read_file(file_path):
    """Read spectrum or filter files with robust format handling"""
    try:
        if file_path.endswith('.fits'):
            with fits.open(file_path) as hdul:
                return hdul[1].data['freq'], hdul[1].data['intensity']
        
        with open(file_path, 'rb') as f:
            content = f.read()
        
        for encoding in ['utf-8', 'latin-1', 'ascii']:
            try:
                decoded = content.decode(encoding)
                lines = decoded.splitlines()
                
                data_lines = []
                for line in lines:
                    stripped = line.strip()
                    if stripped and not stripped.startswith(('!', '//', '#')):
                        cleaned = stripped.replace(',', '.')
                        data_lines.append(cleaned)
                
                if not data_lines:
                    continue
                
                data = np.genfromtxt(data_lines)
                if data.ndim == 2 and data.shape[1] >= 2:
                    return data[:, 0], data[:, 1]
                
            except (UnicodeDecodeError, ValueError):
                continue
        
        raise ValueError("Could not read the file with any standard encoding")
    
    except Exception as e:
        print(f"Error reading file {os.path.basename(file_path)}: {str(e)}")
        return None, None

def apply_spectral_filter(spectrum_freq, spectrum_intensity, filter_path):
    """Apply spectral filter with robust handling"""
    try:
        filter_freq, filter_intensity = robust_read_file(filter_path)
        if filter_freq is None:
            return None
        
        if np.mean(filter_freq) > 1e6:
            filter_freq = filter_freq / 1e9
        
        max_intensity = np.max(filter_intensity)
        if max_intensity > 0:
            filter_intensity = filter_intensity / max_intensity
        
        mask = filter_intensity > 0.01
        
        valid_points = (~np.isnan(spectrum_intensity)) & (~np.isinf(spectrum_intensity))
        if np.sum(valid_points) < 2:
            raise ValueError("Spectrum doesn't have enough valid points")
        
        interp_func = interp1d(
            spectrum_freq[valid_points],
            spectrum_intensity[valid_points],
            kind='linear',
            bounds_error=False,
            fill_value=0.0
        )
        
        filtered_data = interp_func(filter_freq) * filter_intensity
        filtered_data = np.clip(filtered_data, 0, None)
        
        full_filtered = np.zeros_like(filter_freq)
        full_filtered[mask] = filtered_data[mask]
        
        return {
            'freq': filter_freq,
            'intensity': full_filtered,
            'filter_profile': filter_intensity,
            'mask': mask,
            'filter_name': os.path.splitext(os.path.basename(filter_path))[0],
            'parent_dir': os.path.basename(os.path.dirname(filter_path))
        }
    
    except Exception as e:
        print(f"Error applying filter {os.path.basename(filter_path)}: {str(e)}")
        return None

def find_available_models(model_dir):
    """Find all available model directories that contain the required files"""
    required_files = {
        'rf_tex': 'random_forest_tex.pkl',
        'rf_logn': 'random_forest_logn.pkl',
        'x_scaler': 'x_scaler.pkl',
        'tex_scaler': 'tex_scaler.pkl',
        'logn_scaler': 'logn_scaler.pkl'
    }
    
    available_models = []
    
    try:
        for root, dirs, files in os.walk(model_dir):
            has_all_files = True
            for req_file in required_files.values():
                if req_file not in files:
                    has_all_files = False
                    break
            
            if has_all_files:
                model_name = os.path.basename(root)
                available_models.append({
                    'name': model_name,
                    'path': root
                })
    except Exception as e:
        print(f"Error searching for models: {str(e)}")
    
    return available_models

def load_prediction_models(model_dir):
    """Load models without any output"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            rf_tex = joblib.load(os.path.join(model_dir, 'random_forest_tex.pkl'))
            rf_logn = joblib.load(os.path.join(model_dir, 'random_forest_logn.pkl'))
            x_scaler = joblib.load(os.path.join(model_dir, 'x_scaler.pkl'))
            tex_scaler = joblib.load(os.path.join(model_dir, 'tex_scaler.pkl'))
            logn_scaler = joblib.load(os.path.join(model_dir, 'logn_scaler.pkl'))
            return rf_tex, rf_logn, x_scaler, tex_scaler, logn_scaler
        except:
            return None, None, None, None, None

def process_spectrum_for_prediction(file_path):
    """Completely silent processing"""
    try:
        with open(file_path, 'r') as f:
            lines = [line.strip() for line in f if line.strip() and not line.strip().startswith(('!', '//', '#'))]
        
        data = pd.read_csv(StringIO('\n'.join(lines)), 
                     sep='\s+', header=None, names=['freq', 'intensity'],
                     dtype=np.float32).dropna()
        
        if len(data) < 1000:
            return None
            
        freq = data['freq'].values
        intensity = data['intensity'].values
        
        normalized_freq = (freq - freq.min()) / (freq.max() - freq.min())
        interp_func = interp1d(normalized_freq, intensity, kind='linear', 
                              bounds_error=False, fill_value="extrapolate")
        interpolated = interp_func(np.linspace(0, 1, 64610))
        
        if np.any(np.isnan(interpolated)):
            interpolated = np.nan_to_num(interpolated)
        
        if np.max(interpolated) != np.min(interpolated):
            return (interpolated - np.min(interpolated)) / (np.max(interpolated) - np.min(interpolated))
        return np.zeros_like(interpolated)
    except:
        return None

def run_prediction(filtered_file_path, model_dir):
    """Completely clean prediction function"""
    models = load_prediction_models(model_dir)
    if None in models:
        return None, None
        
    scaled_spectrum = process_spectrum_for_prediction(filtered_file_path)
    if scaled_spectrum is None:
        return None, None
        
    rf_tex, rf_logn, x_scaler, tex_scaler, logn_scaler = models
    scaled_spectrum = x_scaler.transform([scaled_spectrum])
    
    tex_pred = tex_scaler.inverse_transform(rf_tex.predict(scaled_spectrum).reshape(-1, 1))[0,0]
    logn_pred = logn_scaler.inverse_transform(rf_logn.predict(scaled_spectrum).reshape(-1, 1))[0,0]
    
    return tex_pred, logn_pred

def plot_prediction_results(tex_pred, logn_pred):
    """Plot the prediction results cleanly"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.scatter(18.1857, logn_pred, c='red', s=200, edgecolors='black')
    ax1.annotate(f"Pred: {logn_pred:.2f}", 
                (18.1857, logn_pred),
                textcoords="offset points",
                xytext=(15,15), ha='center', fontsize=12, color='red')
    ax1.set_xlabel('LogN de referencia')
    ax1.set_ylabel('LogN predicho')
    ax1.set_title('Predicción de LogN')
    
    ax2.scatter(203.492, tex_pred, c='red', s=200, edgecolors='black')
    ax2.annotate(f"Pred: {tex_pred:.1f}", 
                (203.492, tex_pred),
                textcoords="offset points",
                xytext=(15,15), ha='center', fontsize=12, color='red')
    ax2.set_xlabel('Tex de referencia (K)')
    ax2.set_ylabel('Tex predicho (K)')
    ax2.set_title('Predicción de Tex')
    
    plt.tight_layout()
    return fig

# =============================================
# GRADIO INTERFACE
# =============================================
def process_spectrum(input_file, selected_model):
    # Create temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(input_file.name)[1]) as tmp_file:
        tmp_file.write(input_file.read())
        tmp_path = tmp_file.name
    
    try:
        input_freq, input_spec = robust_read_file(tmp_path)
        if input_freq is None:
            return None, None, None, "Error: Could not read the spectrum file"
        
        original_spectrum = {
            'freq': input_freq,
            'intensity': input_spec
        }
        
        filter_files = []
        for root, _, files in os.walk(LOCAL_FILTER_DIR):
            for file in files:
                if file.endswith('.txt'):
                    filter_files.append(os.path.join(root, file))
        
        if not filter_files:
            return None, None, None, "Error: No filter files found in the filters directory"
        
        filtered_spectra = []
        failed_filters = []
        
        for i, filter_file in enumerate(filter_files):
            filter_name = os.path.splitext(os.path.basename(filter_file))[0]
            
            result = apply_spectral_filter(input_freq, input_spec, filter_file)
            if result is not None:
                output_filename = f"filtered_{result['filter_name']}.txt"
                output_path = os.path.join(tempfile.gettempdir(), output_filename)
                
                header = f"!xValues(GHz)\tyValues(K)\n!Filter applied: {result['filter_name']}"
                np.savetxt(
                    output_path,
                    np.column_stack((result['freq'], result['intensity'])),
                    header=header,
                    delimiter='\t',
                    fmt=['%.10f', '%.6e'],
                    comments=''
                )
                
                filtered_spectra.append({
                    'name': result['filter_name'],
                    'filtered_data': result,
                    'output_path': output_path,
                    'parent_dir': result['parent_dir']
                })
            else:
                failed_filters.append(os.path.basename(filter_file))
        
        if not filtered_spectra:
            return None, None, None, f"Error: No filters were successfully applied. {len(failed_filters)} filters failed."
        
        # Create main plot
        fig_main = go.Figure()
        fig_main.add_trace(go.Scatter(
            x=original_spectrum['freq'],
            y=original_spectrum['intensity'],
            mode='lines',
            name='Original Spectrum',
            line=dict(color='blue', width=2))
        )
        
        for result in filtered_spectra:
            fig_main.add_trace(go.Scatter(
                x=result['filtered_data']['freq'],
                y=result['filtered_data']['intensity'],
                mode='lines',
                name=f"Filtered: {result['name']}",
                line=dict(width=1.5))
            )
        
        fig_main.update_layout(
            title="Spectrum Filtering Results",
            xaxis_title="Frequency (GHz)",
            yaxis_title="Intensity (K)",
            hovermode="x unified",
            height=600
        )
        
        # Find CH3OCHO spectrum for prediction
        ch3ocho_result = next((r for r in filtered_spectra if "CH3OCHO" in r['name'].upper()), None)
        prediction_result = None
        prediction_fig = None
        
        if ch3ocho_result and selected_model:
            model_path = next((m['path'] for m in find_available_models(LOCAL_MODEL_DIR) if m['name'] == selected_model), None)
            if model_path:
                tex_pred, logn_pred = run_prediction(ch3ocho_result['output_path'], model_path)
                if tex_pred and logn_pred:
                    prediction_result = f"Tex: {tex_pred:.2f} K\nLogN: {logn_pred:.2f}"
                    prediction_fig = plot_prediction_results(tex_pred, logn_pred)
        
        return fig_main, filtered_spectra, prediction_fig, prediction_result if prediction_result else "No CH3OCHO prediction available"
    
    except Exception as e:
        return None, None, None, f"Processing error: {str(e)}"
    finally:
        os.unlink(tmp_path)

def create_filter_details(filtered_spectra):
    details = []
    for result in filtered_spectra:
        with gr.Box():
            with gr.Row():
                # Filter profile plot
                fig_filter = go.Figure()
                fig_filter.add_trace(go.Scatter(
                    x=result['filtered_data']['freq'],
                    y=result['filtered_data']['filter_profile'],
                    mode='lines',
                    name='Filter Profile',
                    line=dict(color='green'))
                fig_filter.update_layout(
                    title=f"Filter Profile: {result['name']}",
                    height=300
                )
                
                # Comparison plot
                fig_compare = go.Figure()
                fig_compare.add_trace(go.Scatter(
                    x=result['filtered_data']['freq'],
                    y=result['filtered_data']['intensity'],
                    mode='lines',
                    name='Filtered',
                    line=dict(color='red', width=1))
                )
                fig_compare.update_layout(
                    title=f"Filtered Spectrum: {result['name']}",
                    height=300
                )
                
                details.append((fig_filter, fig_compare, result['name']))
    
    return details

# Initialize available models
available_models = [m['name'] for m in find_available_models(LOCAL_MODEL_DIR)] if os.path.exists(LOCAL_MODEL_DIR) else []

# Create Gradio interface
with gr.Blocks(title="AI-ITACA | Spectrum Analyzer", theme="default") as demo:
    gr.Markdown("""
    # AI-ITACA | Artificial Intelligence Integral Tool for AstroChemical Analysis
    ## Molecular Spectrum Analyzer
    """)
    
    with gr.Row():
        with gr.Column():
            input_file = gr.File(label="Upload Spectrum File", file_types=[".txt", ".dat", ".fits", ".spec"])
            model_dropdown = gr.Dropdown(choices=available_models, label="Select Prediction Model", interactive=bool(available_models))
            submit_btn = gr.Button("Analyze Spectrum", variant="primary")
        
        with gr.Column():
            status_output = gr.Textbox(label="Status", interactive=False)
    
    with gr.Tabs():
        with gr.TabItem("Interactive Spectrum"):
            spectrum_plot = gr.Plot(label="Spectrum Visualization")
        
        with gr.TabItem("Filter Details"):
            filter_details = gr.Gallery(label="Filter Details", columns=2)
        
        with gr.TabItem("CH3OCHO Prediction"):
            with gr.Row():
                prediction_plot = gr.Plot(label="Prediction Results")
                prediction_output = gr.Textbox(label="Prediction Values", interactive=False)
    
    submit_btn.click(
        fn=process_spectrum,
        inputs=[input_file, model_dropdown],
        outputs=[spectrum_plot, filter_details, prediction_plot, status_output]
    )

# For Hugging Face Spaces
demo.launch()
