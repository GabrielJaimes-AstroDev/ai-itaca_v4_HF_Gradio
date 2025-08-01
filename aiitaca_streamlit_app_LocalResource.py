import streamlit as st
import os
import numpy as np
import tempfile
import plotly.graph_objects as go
import time
from scipy.interpolate import interp1d
from astropy.io import fits
import shutil
from glob import glob
import pandas as pd
import joblib
from io import StringIO
import warnings
import sys
import subprocess

# Configuración inicial
warnings.filterwarnings('ignore')

# =============================================
# VERIFICACIÓN DE DEPENDENCIAS
# =============================================
def verify_dependencies():
    """Verifica todas las dependencias requeridas"""
    required_packages = {
        'numpy': 'numpy',
        'scikit-learn': 'sklearn',
        'joblib': 'joblib',
        'astropy': 'astropy',
        'plotly': 'plotly',
        'scipy': 'scipy'
    }
    
    missing = []
    for pkg_name, pkg_import in required_packages.items():
        try:
            __import__(pkg_import)
        except ImportError:
            missing.append(pkg_name)
    
    if missing:
        st.error(f"Faltan dependencias: {', '.join(missing)}")
        if st.button("Instalar dependencias automáticamente"):
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing)
                st.success("Dependencias instaladas correctamente. Por favor reinicie la aplicación.")
                st.experimental_rerun()
            except subprocess.CalledProcessError:
                st.error("Error al instalar dependencias. Por favor instálelas manualmente.")
        return False
    return True

# Verificar dependencias al inicio
if not verify_dependencies():
    st.stop()

# =============================================
# INITIALIZE SESSION STATE
# =============================================
def initialize_session_state():
    """Inicializa todas las variables de estado necesarias"""
    if 'resources_downloaded' not in st.session_state:
        st.session_state.resources_downloaded = False
        st.session_state.MODEL_DIR = "RF_Models/1.ML_Performance_RDF_CH3OCHO_Noisy_Weight3_Sigma0_001_T1"  # Ruta relativa ajustada
        st.session_state.FILTER_DIR = "RF_Filters"  # Ruta ajustada para los filtros
        st.session_state.downloaded_files = {'models': [], 'filters': []}
        st.session_state.prediction_models_loaded = False
        st.session_state.prediction_models = None
        st.session_state.models_downloaded = False
    
    # Nuevos estados para manejar el procesamiento de archivos
    if 'file_processed' not in st.session_state:
        st.session_state.file_processed = False
        st.session_state.current_file = None
        st.session_state.filtered_results = []
        st.session_state.failed_filters = []
        st.session_state.spectrum_data = None
        st.session_state.tmp_file_path = None

initialize_session_state()

# =============================================
# PAGE CONFIGURATION
# =============================================
st.set_page_config(
    layout="wide", 
    page_title="AI-ITACA | Spectrum Analyzer",
    page_icon="🔭" 
)

# =============================================
# CSS STYLES (from external file)
# =============================================
def load_css_styles():
    """Carga los estilos CSS desde el archivo externo"""
    try:
        with open('styles.txt', 'r') as f:
            css_styles = f.read()
        st.markdown(f"<style>{css_styles}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("No se encontró el archivo de estilos CSS. La aplicación funcionará pero con estilos básicos.")

load_css_styles()

# =============================================
# HELPER FUNCTIONS
# =============================================
def list_downloaded_files(directory):
    """Lista recursivamente todos los archivos descargados"""
    file_list = []
    try:
        for root, dirs, files in os.walk(directory):
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            
            for file in files:
                if not file.startswith('.'):
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, directory)
                    try:
                        size = get_file_size(full_path)
                        file_list.append({
                            'path': rel_path,
                            'size': size,
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
        st.error(f"Error listing files in {directory}: {str(e)}")
    return file_list

def get_file_size(path):
    """Obtiene el tamaño de archivo en formato legible"""
    size = os.path.getsize(path)
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"

def robust_read_file(file_path):
    """Lee archivos de espectro o filtros con manejo robusto"""
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
        st.error(f"Error reading file {os.path.basename(file_path)}: {str(e)}")
        return None, None

def apply_spectral_filter(spectrum_freq, spectrum_intensity, filter_path):
    """Aplica filtro espectral con manejo robusto"""
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
        st.error(f"Error applying filter {os.path.basename(filter_path)}: {str(e)}")
        return None

def load_prediction_models(models_dir):
    """Carga los modelos de predicción con manejo mejorado de errores"""
    try:
        # Verificar estructura de directorios
        if not os.path.exists(models_dir):
            st.error(f"Models directory not found: {os.path.abspath(models_dir)}")
            st.error(f"Current working directory: {os.getcwd()}")
            st.error(f"Directory contents: {os.listdir()}")
            return None
        
        # Modelos requeridos con sus rutas exactas
        model_files = {
            'random_forest_tex': os.path.join(models_dir, 'random_forest_tex.pkl'),
            'random_forest_logn': os.path.join(models_dir, 'random_forest_logn.pkl'),
            'x_scaler': os.path.join(models_dir, 'x_scaler.pkl'),
            'tex_scaler': os.path.join(models_dir, 'tex_scaler.pkl'),
            'logn_scaler': os.path.join(models_dir, 'logn_scaler.pkl')
        }
        
        # Verificar que todos los archivos existan
        missing_models = [name for name, path in model_files.items() if not os.path.exists(path)]
        if missing_models:
            st.error(f"Missing model files: {', '.join(missing_models)}")
            return None
        
        # Cargar modelos
        models = {}
        try:
            models['rf_tex'] = joblib.load(model_files['random_forest_tex'])
            models['rf_logn'] = joblib.load(model_files['random_forest_logn'])
            models['x_scaler'] = joblib.load(model_files['x_scaler'])
            models['tex_scaler'] = joblib.load(model_files['tex_scaler'])
            models['logn_scaler'] = joblib.load(model_files['logn_scaler'])
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return None
        
        return models
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None

def process_spectrum_for_prediction(freq, intensity, interpolation_length=64610, min_required_points=1000):
    """Procesa el espectro para predicción"""
    try:
        data = pd.DataFrame({
            'freq': freq,
            'intensity': intensity
        })
        
        data = data.replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(data) < min_required_points:
            st.error(f"Spectrum has too few valid points ({len(data)})")
            return None
        
        min_freq = data['freq'].min()
        max_freq = data['freq'].max()
        freq_range = max_freq - min_freq
        if freq_range == 0:
            st.error("Zero frequency range in spectrum")
            return None
        
        normalized_freq = (data['freq'] - min_freq) / freq_range
        
        interp_func = interp1d(
            normalized_freq, 
            data['intensity'], 
            kind='linear', 
            bounds_error=False, 
            fill_value=(data['intensity'].iloc[0], data['intensity'].iloc[-1])
        )
        
        new_freq = np.linspace(0, 1, interpolation_length)
        interpolated_intensity = interp_func(new_freq).astype(np.float32)
        
        if np.any(np.isnan(interpolated_intensity)):
            nan_indices = np.where(np.isnan(interpolated_intensity))[0]
            for idx in nan_indices:
                left_idx = max(0, idx-1)
                right_idx = min(interpolation_length-1, idx+1)
                interpolated_intensity[idx] = np.mean(interpolated_intensity[[left_idx, right_idx]])
        
        min_intensity = np.min(interpolated_intensity)
        max_intensity = np.max(interpolated_intensity)
        if max_intensity != min_intensity:
            scaled_intensity = (interpolated_intensity - min_intensity) / (max_intensity - min_intensity)
        else:
            scaled_intensity = np.zeros_like(interpolated_intensity)
        
        return scaled_intensity
    except Exception as e:
        st.error(f"Error processing spectrum for prediction: {str(e)}")
        return None

def make_predictions(spectrum_data, models):
    """Realiza predicciones usando los modelos cargados"""
    try:
        scaled_spectrum = models['x_scaler'].transform([spectrum_data])
        
        tex_pred_scaled = models['rf_tex'].predict(scaled_spectrum)
        tex_pred = models['tex_scaler'].inverse_transform(tex_pred_scaled.reshape(-1, 1))[0,0]
        
        logn_pred_scaled = models['rf_logn'].predict(scaled_spectrum)
        logn_pred = models['logn_scaler'].inverse_transform(logn_pred_scaled.reshape(-1, 1))[0,0]
        
        return tex_pred, logn_pred
    except Exception as e:
        st.error(f"Error making predictions: {str(e)}")
        return None, None

def plot_prediction_results(tex_pred, logn_pred):
    """Crea figura interactiva para resultados de predicción"""
    fig = go.Figure()
    
    # LogN plot
    fig.add_trace(go.Scatter(
        x=[18.1857],
        y=[logn_pred],
        mode='markers',
        marker=dict(color='red', size=15, line=dict(width=2, color='black')),
        name='Predicted LogN',
        text=[f"Pred: {logn_pred:.2f}"],
        hoverinfo='text',
        xaxis='x1',
        yaxis='y1'
    ))
    
    # Tex plot
    fig.add_trace(go.Scatter(
        x=[203.492],
        y=[tex_pred],
        mode='markers',
        marker=dict(color='red', size=15, line=dict(width=2, color='black')),
        name='Predicted Tex',
        text=[f"Pred: {tex_pred:.1f} K"],
        hoverinfo='text',
        xaxis='x2',
        yaxis='y2'
    ))
    
    fig.update_layout(
        title='Prediction Results',
        grid=dict(rows=1, columns=2, pattern='independent'),
        plot_bgcolor='#0D0F14',
        paper_bgcolor='#0D0F14',
        font=dict(color='white'),
        height=400
    )
    
    fig.update_xaxes(title_text='Reference LogN', row=1, col=1)
    fig.update_yaxes(title_text='Predicted LogN', row=1, col=1)
    fig.update_xaxes(title_text='Reference Tex (K)', row=1, col=2)
    fig.update_yaxes(title_text='Predicted Tex (K)', row=1, col=2)
    
    return fig

def ensure_resources_downloaded():
    """Asegura que los recursos se descarguen solo una vez"""
    if not st.session_state.models_downloaded:
        with st.spinner("🔄 Loading local models..."):
            # Verificar si el directorio existe
            if not os.path.exists(st.session_state.MODEL_DIR):
                st.error(f"Model directory not found at: {os.path.abspath(st.session_state.MODEL_DIR)}")
                st.error(f"Current working directory: {os.getcwd()}")
                st.error(f"Directory contents: {os.listdir()}")
                return
            
            st.session_state.downloaded_files['models'] = list_downloaded_files(st.session_state.MODEL_DIR)
            
            if st.session_state.downloaded_files['models']:
                st.session_state.resources_downloaded = True
                st.session_state.models_downloaded = True
                st.success("Local models loaded successfully!")
            else:
                st.error("No model files found in the specified directory.")

def process_uploaded_file(input_file):
    """Procesa el archivo subido y almacena los resultados en session_state"""
    try:
        # Limpiar resultados anteriores
        st.session_state.filtered_results = []
        st.session_state.failed_filters = []
        st.session_state.file_processed = False
        
        # Crear archivo temporal
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(input_file.getvalue())
            st.session_state.tmp_file_path = tmp_file.name
        
        # Leer el archivo
        input_freq, input_spec = robust_read_file(st.session_state.tmp_file_path)
        if input_freq is None:
            raise ValueError("Could not read the spectrum file")
        
        st.session_state.spectrum_data = (input_freq, input_spec)
        
        # Obtener lista de filtros (si existen)
        filter_files = []
        if st.session_state.FILTER_DIR and os.path.exists(st.session_state.FILTER_DIR):
            for root, _, files in os.walk(st.session_state.FILTER_DIR):
                for file in files:
                    if file.endswith('.txt'):
                        filter_files.append(os.path.join(root, file))
        
        # Aplicar filtros si existen
        if filter_files:
            with st.spinner("🔍 Applying filters..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, filter_file in enumerate(filter_files):
                    filter_name = os.path.splitext(os.path.basename(filter_file))[0]
                    status_text.text(f"Processing filter {i+1}/{len(filter_files)}: {filter_name}")
                    progress_bar.progress((i + 1) / len(filter_files))
                    
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
                        
                        st.session_state.filtered_results.append({
                            'name': result['filter_name'],
                            'original_freq': input_freq,
                            'original_intensity': input_spec,
                            'filtered_data': result,
                            'output_path': output_path
                        })
                    else:
                        st.session_state.failed_filters.append(os.path.basename(filter_file))
                
                progress_bar.empty()
                status_text.empty()
        
        st.session_state.file_processed = True
        
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        if st.session_state.tmp_file_path and os.path.exists(st.session_state.tmp_file_path):
            os.unlink(st.session_state.tmp_file_path)
        st.session_state.file_processed = False
        raise

def display_results():
    """Muestra los resultados del procesamiento"""
    if not st.session_state.file_processed:
        st.warning("No hay resultados para mostrar")
        return
    
    st.markdown(f'<div class="success-box">✅ Spectrum processed successfully</div>', unsafe_allow_html=True)
    
    if st.session_state.filtered_results:
        st.markdown(f'<div class="success-box">✅ Successfully applied {len(st.session_state.filtered_results)} filters</div>', unsafe_allow_html=True)
    
    if st.session_state.failed_filters:
        st.markdown(f'<div class="warning-box">⚠ Failed to apply {len(st.session_state.failed_filters)} filters: {", ".join(st.session_state.failed_filters)}</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["Interactive Spectrum", "Filter Details", "Molecular Parameters Prediction"])
    
    with tab1:
        fig_main = go.Figure()
        
        fig_main.add_trace(go.Scatter(
            x=st.session_state.spectrum_data[0],
            y=st.session_state.spectrum_data[1],
            mode='lines',
            name='Original Spectrum',
            line=dict(color='white', width=2))
        )
        
        for result in st.session_state.filtered_results:
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
            height=600,
            plot_bgcolor='#0D0F14',
            paper_bgcolor='#0D0F14',
            font=dict(color='white'),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        st.plotly_chart(fig_main, use_column_width=True)
    
    with tab2:
        if st.session_state.filtered_results:
            for result in st.session_state.filtered_results:
                with st.expander(f"Filter: {result['name']} (from {result['filtered_data']['parent_dir']})", expanded=True):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig_filter = go.Figure()
                        fig_filter.add_trace(go.Scatter(
                            x=result['filtered_data']['freq'],
                            y=result['filtered_data']['filter_profile'],
                            mode='lines',
                            name='Filter Profile',
                            line=dict(color='#1E88E5'))
                        )
                        fig_filter.update_layout(
                            title="Filter Profile",
                            height=300,
                            plot_bgcolor='#0D0F14',
                            paper_bgcolor='#0D0F14',
                            showlegend=False
                        )
                        st.plotly_chart(fig_filter, use_column_width=True)
                    
                    with col2:
                        fig_compare = go.Figure()
                        fig_compare.add_trace(go.Scatter(
                            x=result['original_freq'],
                            y=result['original_intensity'],
                            mode='lines',
                            name='Original',
                            line=dict(color='white', width=1))
                        )
                        fig_compare.add_trace(go.Scatter(
                            x=result['filtered_data']['freq'],
                            y=result['filtered_data']['intensity'],
                            mode='lines',
                            name='Filtered',
                            line=dict(color='#FF5722', width=1))
                        )
                        fig_compare.update_layout(
                            title="Original vs Filtered",
                            height=300,
                            plot_bgcolor='#0D0F14',
                            paper_bgcolor='#0D0F14',
                            showlegend=False
                        )
                        st.plotly_chart(fig_compare, use_column_width=True)
                    
                    with open(result['output_path'], 'rb') as f:
                        st.download_button(
                            label=f"Download {result['name']} filtered spectrum",
                            data=f,
                            file_name=os.path.basename(result['output_path']),
                            mime='text/plain',
                            key=f"download_{result['name']}",
                            use_column_width=True
                        )
        else:
            st.info("No filters were applied to this spectrum")
    
    with tab3:
        st.markdown("## Molecular Parameters Prediction")
        st.markdown("Predict LogN (column density) and Tex (excitation temperature) using machine learning models.")
        
        if st.session_state.prediction_models is None:
            with st.spinner("🔄 Loading prediction models..."):
                st.session_state.prediction_models = load_prediction_models(st.session_state.MODEL_DIR)
        
        if st.session_state.prediction_models:
            # Usamos el espectro original si no hay filtros aplicados
            if st.session_state.filtered_results:
                filter_options = [f"{res['name']} (from {res['filtered_data']['parent_dir']})" for res in st.session_state.filtered_results]
                selected_filter = st.selectbox(
                    "Select filtered spectrum for prediction:",
                    options=filter_options,
                    index=0
                )
                
                selected_index = filter_options.index(selected_filter)
                selected_result = st.session_state.filtered_results[selected_index]
                freq_data = selected_result['filtered_data']['freq']
                intensity_data = selected_result['filtered_data']['intensity']
            else:
                st.info("Using original spectrum for prediction (no filters applied)")
                freq_data = st.session_state.spectrum_data[0]
                intensity_data = st.session_state.spectrum_data[1]
            
            with st.spinner("Processing spectrum for prediction..."):
                processed_spectrum = process_spectrum_for_prediction(freq_data, intensity_data)
            
            if processed_spectrum is not None:
                with st.spinner("Making predictions..."):
                    tex_pred, logn_pred = make_predictions(processed_spectrum, st.session_state.prediction_models)
                
                if tex_pred is not None and logn_pred is not None:
                    st.success("Prediction completed successfully!")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(label="Predicted Excitation Temperature (Tex)", value=f"{tex_pred:.2f} K")
                    with col2:
                        st.metric(label="Predicted Column Density (LogN)", value=f"{logn_pred:.2f}")
                    
                    st.plotly_chart(
                        plot_prediction_results(tex_pred, logn_pred),
                        use_column_width=True
                    )
                    
                    with st.expander("Prediction Details"):
                        st.markdown(f"""
                        **Source used:** {'Filtered: ' + selected_result['name'] if st.session_state.filtered_results else 'Original spectrum'}  
                        **Number of points in spectrum:** {len(freq_data)}  
                        **Intensity range:** {np.min(intensity_data):.2e} to {np.max(intensity_data):.2e} K
                        """)
                else:
                    st.error("Failed to make predictions")
            else:
                st.error("Failed to process spectrum for prediction")
        else:
            st.error("Prediction models could not be loaded. Please check if models are properly placed in the model directory.")

# =============================================
# HEADER
# =============================================
st.image("NGC6523_BVO_2.jpg", use_column_width=True)

col1, col2 = st.columns([1, 3])
with col1:
    st.empty()
    
with col2:
    st.markdown('<p class="main-title">AI-ITACA | Artificial Intelligence Integral Tool for AstroChemical Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Molecular Spectrum Analyzer</p>', unsafe_allow_html=True)

with open('description.txt', 'r') as f:
    description_content = f.read()
st.markdown(f"<div class='description-panel'>{description_content}</div>", unsafe_allow_html=True)

# =============================================
# MAIN INTERFACE
# =============================================
ensure_resources_downloaded()

# =============================================
# SIDEBAR - CONFIGURATION
# =============================================
st.sidebar.title("Configuration")

with st.sidebar:
    st.header("📁 Local Resources")
    
    if st.session_state.MODEL_DIR and os.path.exists(st.session_state.MODEL_DIR):
        st.subheader("Models Directory")
        st.code(f"{os.path.abspath(st.session_state.MODEL_DIR)}", language="bash")
        
        if st.session_state.downloaded_files['models']:
            st.markdown("**Model files:**")
            model_files_text = "\n".join(
                f"- {file['path']} ({file['size']})" 
                for file in st.session_state.downloaded_files['models']
            )
            st.text_area("Model files list", value=model_files_text, height=150, label_visibility="collapsed")
        else:
            st.warning("No model files found")
    
    if st.session_state.FILTER_DIR and os.path.exists(st.session_state.FILTER_DIR):
        st.subheader("Filters Directory")
        st.code(f"{os.path.abspath(st.session_state.FILTER_DIR)}", language="bash")
        
        if st.session_state.downloaded_files['filters']:
            st.markdown("**Filter files:**")
            filter_files_text = "\n".join(
                f"- {file['path']} ({file['size']})" 
                for file in st.session_state.downloaded_files['filters']
            )
            st.text_area("Filter files list", value=filter_files_text, height=150, label_visibility="collapsed")
        else:
            st.warning("No filter files found")

# Input file uploader - ahora manejado con estado
input_file = st.sidebar.file_uploader(
    "Input Spectrum File",
    type=['.txt', '.dat', '.fits', '.spec'],
    help="Upload your spectrum file (TXT, DAT, FITS, SPEC)",
    key="file_uploader"
)

# Procesar archivo si se subió uno nuevo
if input_file is not None and (not st.session_state.file_processed or st.session_state.current_file != input_file.name):
    try:
        st.session_state.current_file = input_file.name
        process_uploaded_file(input_file)
        st.experimental_rerun()  # Corregido: reemplazado st.rerun() por st.experimental_rerun()
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")

# Mostrar resultados si ya se procesó un archivo
if st.session_state.file_processed:
    display_results()
elif input_file is None:
    st.info("ℹ️ Please upload a spectrum file to begin analysis")

# Limpieza final al cerrar
if st.session_state.tmp_file_path and os.path.exists(st.session_state.tmp_file_path):
    os.unlink(st.session_state.tmp_file_path)

# =============================================
# INSTRUCTIONS
# =============================================
st.sidebar.markdown("""
**Instructions:**
1. Upload your spectrum file
2. The system will process it using local models
3. View results in the interactive tabs
4. Predict molecular parameters in the Prediction tab
5. Download filtered spectra as needed

**Supported formats:**
- Text files (.txt, .dat)
- FITS files (.fits)
- Spectrum files (.spec)

**Note:** The application uses pre-installed models from your local directory.
""")
