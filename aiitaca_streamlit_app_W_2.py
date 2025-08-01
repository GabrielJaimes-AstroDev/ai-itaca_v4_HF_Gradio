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
import joblib
import pandas as pd
import warnings
from io import StringIO
import matplotlib.pyplot as plt

# =============================================
# CONFIGURACIÓN DE PATHS LOCALES
# =============================================
LOCAL_MODEL_DIR = "RF_Models"
LOCAL_FILTER_DIR = "RF_Filters"

# =============================================
# INITIALIZE SESSION STATE
# =============================================
if not hasattr(st.session_state, 'resources_loaded'):
    st.session_state.resources_loaded = False
    st.session_state.MODEL_DIR = None
    st.session_state.FILTER_DIR = None
    st.session_state.local_files = {'models': [], 'filters': []}
    st.session_state.prediction_models_loaded = False
    st.session_state.prediction_results = None
    st.session_state.original_spectrum = None
    st.session_state.filtered_spectra = []
    st.session_state.available_models = []

# =============================================
# PAGE CONFIGURATION
# =============================================
st.set_page_config(
    layout="wide", 
    page_title="AI-ITACA | Spectrum Analyzer",
    page_icon="🔭" 
)

# =============================================
# CSS STYLES
# =============================================
css_styles = """
<style>
.main-title {
    font-size: 28px !important;
    font-weight: bold !important;
    color: #FFFFFF !important;
    margin-bottom: 0.2rem !important;
}
.subtitle {
    font-size: 18px !important;
    color: #BBBBBB !important;
    margin-bottom: 1.5rem !important;
}
.description-panel {
    background-color: #1E1E1E;
    padding: 15px;
    border-radius: 10px;
    margin-bottom: 20px;
    border-left: 5px solid #4CAF50;
}
.success-box {
    background-color: #2E7D32;
    color: white;
    padding: 10px;
    border-radius: 5px;
    margin: 10px 0;
}
.warning-box {
    background-color: #FF8F00;
    color: white;
    padding: 10px;
    border-radius: 5px;
    margin: 10px 0;
}
.error-box {
    background-color: #C62828;
    color: white;
    padding: 10px;
    border-radius: 5px;
    margin: 10px 0;
}
.tree-view {
    font-family: monospace;
    margin-left: 15px;
}
.directory {
    color: #4FC3F7;
    margin: 5px 0;
    font-weight: bold;
}
.file {
    color: #E0E0E0;
    margin-left: 20px;
}
.size {
    color: #FF9800;
    font-style: italic;
}
.file-explorer-header {
    font-size: 18px;
    font-weight: bold;
    color: #4FC3F7;
    margin-bottom: 10px;
    border-bottom: 1px solid #444;
    padding-bottom: 5px;
}
.file-explorer-item {
    padding: 5px;
    border-bottom: 1px dotted #444;
}
.progress-text {
    font-size: 14px;
    color: #BBBBBB;
    margin-bottom: 5px;
}
.progress-container {
    width: 100%;
    background-color: #1E1E1E;
    border-radius: 5px;
    margin: 10px 0;
}
.progress-bar {
    height: 20px;
    background-color: #4CAF50;
    border-radius: 5px;
    width: 0%;
    transition: width 0.5s;
    text-align: center;
    color: white;
    line-height: 20px;
}
</style>
"""
st.markdown(css_styles, unsafe_allow_html=True)

# =============================================
# HELPER FUNCTIONS
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
    """Get file size in human-readable format"""
    size = os.path.getsize(path)
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"

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
        st.error(f"Error reading file {os.path.basename(file_path)}: {str(e)}")
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
        st.error(f"Error applying filter {os.path.basename(filter_path)}: {str(e)}")
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
        st.error(f"Error searching for models: {str(e)}")
    
    return available_models

# =============================================
# PREDICTION FUNCTIONS (OPTIMIZED)
# =============================================

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
    with st.spinner("Calculando predicción..."):
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
# HEADER
# =============================================
st.image("NGC6523_BVO_2.jpg", use_column_width=True)

col1, col2 = st.columns([1, 3])
with col1:
    st.empty()
    
with col2:
    st.markdown('<p class="main-title">AI-ITACA | Artificial Intelligence Integral Tool for AstroChemical Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Molecular Spectrum Analyzer</p>', unsafe_allow_html=True)

description_content = """
<h3>About AI-ITACA</h3>
<p>AI-ITACA is a powerful tool for analyzing molecular spectra using machine learning models.</p>

<h3>How to Use</h3>
<ol>
    <li>Upload your spectrum file (TXT, DAT, FITS, SPEC format)</li>
    <li>The system will automatically apply all available filters</li>
    <li>View results in the interactive tabs</li>
    <li>Make predictions for CH3OCHO parameters</li>
</ol>
"""
st.markdown(f"<div class='description-panel'>{description_content}</div>", unsafe_allow_html=True)

# =============================================
# MAIN INTERFACE
# =============================================
if not st.session_state.resources_loaded:
    st.session_state.MODEL_DIR = LOCAL_MODEL_DIR
    st.session_state.FILTER_DIR = LOCAL_FILTER_DIR
    
    if not os.path.exists(st.session_state.MODEL_DIR):
        st.error(f"Model directory not found: {st.session_state.MODEL_DIR}")
    if not os.path.exists(st.session_state.FILTER_DIR):
        st.warning(f"Filter directory not found: {st.session_state.FILTER_DIR}")
    
    try:
        st.session_state.local_files['models'] = list_local_files(st.session_state.MODEL_DIR)
        st.session_state.local_files['filters'] = list_local_files(st.session_state.FILTER_DIR)
        st.session_state.available_models = find_available_models(st.session_state.MODEL_DIR)
        st.session_state.resources_loaded = True
    except Exception as e:
        st.error(f"Error processing local files: {str(e)}")

# =============================================
# SIDEBAR - CONFIGURATION
# =============================================
st.sidebar.title("Configuration")

with st.sidebar:
    st.header("📁 Local Resources")
    
    if st.session_state.MODEL_DIR and os.path.exists(st.session_state.MODEL_DIR):
        st.subheader("Models Directory")
        st.code(f"./{st.session_state.MODEL_DIR}", language="bash")
        
        if st.session_state.available_models:
            st.markdown("**Available models:**")
            for model in st.session_state.available_models:
                st.markdown(f"- {model['name']}")
        else:
            st.warning("No valid model configurations found")
    
    if st.session_state.FILTER_DIR and os.path.exists(st.session_state.FILTER_DIR):
        st.subheader("Filters Directory")
        st.code(f"./{st.session_state.FILTER_DIR}", language="bash")
        
        if st.session_state.local_files['filters']:
            st.markdown("**Filter files:**")
            filter_files_text = "\n".join(
                f"- {file['path']} ({file['size']})" 
                for file in st.session_state.local_files['filters']
            )
            st.text_area("Filter files list", value=filter_files_text, height=150, label_visibility="collapsed")
        else:
            st.warning("No filter files found")

    if st.button("🔄 Reload Resources"):
        st.session_state.local_files['models'] = list_local_files(st.session_state.MODEL_DIR)
        st.session_state.local_files['filters'] = list_local_files(st.session_state.FILTER_DIR)
        st.session_state.available_models = find_available_models(st.session_state.MODEL_DIR)
        st.experimental_rerun()

input_file = st.sidebar.file_uploader(
    "Input Spectrum File",
    type=['.txt', '.dat', '.fits', '.spec'],
    help="Upload your spectrum file (TXT, DAT, FITS, SPEC)"
)

# =============================================
# MAIN PROCESSING
# =============================================
if input_file is not None and st.session_state.MODEL_DIR and st.session_state.FILTER_DIR:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(input_file.getvalue())
        tmp_path = tmp_file.name
    
    try:
        input_freq, input_spec = robust_read_file(tmp_path)
        if input_freq is None:
            raise ValueError("Could not read the spectrum file")
        
        st.session_state.original_spectrum = {
            'freq': input_freq,
            'intensity': input_spec
        }
        
        filter_files = []
        for root, _, files in os.walk(st.session_state.FILTER_DIR):
            for file in files:
                if file.endswith('.txt'):
                    filter_files.append(os.path.join(root, file))
        
        if not filter_files:
            raise ValueError("No filter files found in the filters directory")
        
        st.session_state.filtered_spectra = []
        failed_filters = []
        
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
                    
                    st.session_state.filtered_spectra.append({
                        'name': result['filter_name'],
                        'filtered_data': result,
                        'output_path': output_path,
                        'parent_dir': result['parent_dir']
                    })
                else:
                    failed_filters.append(os.path.basename(filter_file))
            
            progress_bar.empty()
            status_text.empty()
        
        if not st.session_state.filtered_spectra:
            raise ValueError(f"No filters were successfully applied. {len(failed_filters)} filters failed.")
        
        st.markdown(f'<div class="success-box">✅ Successfully applied {len(st.session_state.filtered_spectra)} filters</div>', unsafe_allow_html=True)
        
        if failed_filters:
            st.markdown(f'<div class="warning-box">⚠ Failed to apply {len(failed_filters)} filters: {", ".join(failed_filters)}</div>', unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs(["Interactive Spectrum", "Filter Details", "CH3OCHO Prediction"])
        
        with tab1:
            fig_main = go.Figure()
            
            fig_main.add_trace(go.Scatter(
                x=st.session_state.original_spectrum['freq'],
                y=st.session_state.original_spectrum['intensity'],
                mode='lines',
                name='Original Spectrum',
                line=dict(color='white', width=2))
            )
            
            for result in st.session_state.filtered_spectra:
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
            for result in st.session_state.filtered_spectra:
                with st.expander(f"Filter: {result['name']} (from {result['parent_dir']})", expanded=True):
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
                            x=st.session_state.original_spectrum['freq'],
                            y=st.session_state.original_spectrum['intensity'],
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
                            use_container_width=True
                        )
        
        with tab3:
            st.markdown("<h2 style='text-align: center;'>CH3OCHO Spectral Prediction</h2>", unsafe_allow_html=True)
            
            ch3ocho_result = next((r for r in st.session_state.filtered_spectra if "CH3OCHO" in r['name'].upper()), None)
            
            if ch3ocho_result:
                st.success(f"CH3OCHO spectrum ready: {ch3ocho_result['name']}")
                
                if st.session_state.available_models:
                    selected_model = st.selectbox(
                        "Select prediction model:",
                        [m['name'] for m in st.session_state.available_models],
                        key="model_selector"
                    )
                    
                    model_path = next(m['path'] for m in st.session_state.available_models if m['name'] == selected_model)
                    
                    if st.button("Run Prediction", key="run_prediction"):
                        with st.spinner("Processing..."):
                            tex_pred, logn_pred = run_prediction(ch3ocho_result['output_path'], model_path)
                        
                        if tex_pred and logn_pred:
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Predicted Tex", f"{tex_pred:.2f} K")
                            with col2:
                                st.metric("Predicted LogN", f"{logn_pred:.2f}")
                            
                            st.pyplot(plot_prediction_results(tex_pred, logn_pred))
                            
                            st.download_button(
                                "Download Results",
                                f"Tex: {tex_pred:.2f} K\nLogN: {logn_pred:.2f}",
                                "prediction_results.txt"
                            )
                        else:
                            st.error("Prediction failed - please check your input and models")
            else:
                st.warning("No CH3OCHO spectrum found - check your filters")
    
    except Exception as e:
        st.markdown(f'<div class="error-box">❌ Processing error: {str(e)}</div>', unsafe_allow_html=True)
    finally:
        os.unlink(tmp_path)

elif not st.session_state.MODEL_DIR or not st.session_state.FILTER_DIR:
    st.markdown("""
    <div class="error-box">
    ❌ Required local resources not found.<br><br>
    Please ensure you have:
    <ol>
        <li>Created a 'RF_Models' directory with the model files</li>
        <li>Created a 'RF_Filters' directory with filter files</li>
        <li>Placed them in the correct location</li>
    </ol>
    Click the 'Reload Resources' button in the sidebar after setting up the directories.
    </div>
    """, unsafe_allow_html=True)
else:
    st.info("ℹ️ Please upload a spectrum file to begin analysis")

# =============================================
# INSTRUCTIONS
# =============================================
st.sidebar.markdown("""
**Instructions:**
1. Upload your spectrum file
2. The system will automatically apply all filters
3. View results in the interactive tabs
4. Download filtered spectra as needed

**Supported formats:**
- Text files (.txt, .dat)
- FITS files (.fits)
- Spectrum files (.spec)
""")
