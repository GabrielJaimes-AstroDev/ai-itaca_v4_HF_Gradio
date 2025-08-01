PROJECT_DESCRIPTION = """
<div class="description-panel" style="text-align: justify;">
A remarkable upsurge in the complexity of molecules identified in the interstellar medium (ISM) is currently occurring, with over 80 new species discovered in the last three years. A number of them have been emphasized by prebiotic experiments as vital molecular building blocks of life. Since our Solar System was formed from a molecular cloud in the ISM, it prompts the query as to whether the rich interstellar chemical reservoir could have played a role in the emergence of life. The improved sensitivities of state-of-the-art astronomical facilities, such as the Atacama Large Millimeter/submillimeter Array (ALMA) and the James Webb Space Telescope (JWST), are revolutionizing the discovery of new molecules in space. However, we are still just scraping the tip of the iceberg. We are far from knowing the complete catalogue of molecules that astrochemistry can offer, as well as the complexity they can reach.<br><br>
<strong>Artificial Intelligence Integral Tool for AstroChemical Analysis (AI-ITACA)</strong>, proposes to combine complementary machine learning (ML) techniques to address all the challenges that astrochemistry is currently facing. AI-ITACA will significantly contribute to the development of new AI-based cutting-edge analysis software that will allow us to make a crucial leap in the characterization of the level of chemical complexity in the ISM, and in our understanding of the contribution that interstellar chemistry might have in the origin of life.
</div>
"""

PARAMS_EXPLANATION = """
<div class="description-panel">
<h3 style="text-align: center; margin-top: 0; color: black; border-bottom: 2px solid #1E88E5; padding-bottom: 10px;">Technical Parameters Guide</h3>
                
<div style="margin-bottom: 25px;">
<h4 style="color: #1E88E5; border-bottom: 1px solid #ddd; padding-bottom: 5px; margin-top: 15px;">🔬 Peak Detection</h4>
<p><strong>Sigma Emission (1.5):</strong> Threshold for peak detection in standard deviations (σ) of the noise. 
<span style="display: block; margin-left: 20px; font-size: 0.92em; color: #555;">Higher values reduce false positives but may miss weak peaks. Typical range: 1.0-3.0</span></p>
                
<p><strong>Window Size (3):</strong> Points in Savitzky-Golay smoothing kernel. 
<span style="display: block; margin-left: 20px; font-size: 0.92em; color: #555;">Odd integers only. Larger values smooth noise but blur close peaks.</span></p>
                
<p><strong>Sigma Threshold (2.0):</strong> Minimum peak prominence (σ). 
<span style="display: block; margin-left: 20px; font-size: 0.92em; color: #555;">Filters low-significance features. Critical for crowded spectra.</span></p>
                
<p><strong>FWHM (0.05 GHz):</strong> Expected line width at half maximum. 
<span style="display: block; margin-left: 20px; font-size: 0.92em; color: #555;">Should match your instrument's resolution. Affects line fitting.</span></p>
</div>
                    
<div style="margin-bottom: 25px;">
<h4 style="color: #1E88E5; border-bottom: 1px solid #ddd; padding-bottom: 5px; margin-top: 15px;">🔄 Matching Parameters</h4>
<p><strong>Tolerance (0.1 GHz):</strong> Frequency matching window. 
<span style="display: block; margin-left: 20px; font-size: 0.92em; color: #555;">Accounts for Doppler shifts (±20 km/s at 100 GHz). Increase for broad lines.</span></p>
                
<p><strong>Min Peak Ratio (0.3):</strong> Relative intensity cutoff. 
<span style="display: block; margin-left: 20px; font-size: 0.92em; color: #555;">Peaks below this fraction of strongest line are excluded. Range: 0.1-0.5.</span></p>
</div>
                
 <div>
 <h4 style="color: #1E88E5; border-bottom: 1px solid #ddd; padding-bottom: 5px; margin-top: 15px;">📊 Output Settings</h4>
<p><strong>Top N Lines (30):</strong> Lines displayed in results. 
<span style="display: block; margin-left: 20px; font-size: 0.92em; color: #555;">Doesn't affect analysis quality, only visualization density.</span></p>
                
<p><strong>Top N Similar (800):</strong> Synthetic spectra retained. 
<span style="display: block; margin-left: 20px; font-size: 0.92em; color: #555;">Higher values improve accuracy but increase runtime. Max: 2000.</span></p>
</div>
                
<div style="margin-top: 20px; padding: 12px; background-color: #f8f9fa; border-radius: 5px; border-left: 4px solid #1E88E5;">
<p style="margin: 0; font-size: 0.9em;"><strong>Pro Tip:</strong> For ALMA data (high resolution), start with FWHM=0.05 GHz and Tolerance=0.05 GHz. For single-dish telescopes, try FWHM=0.2 GHz.</p>
</div>
</div>
"""


TRAINING_DATASET = {
    "content": """
    **Training Dataset Parameters**

    | Molecule        | Tex Range (K) | Tex Step | LogN Range (cm⁻²) | LogN Step | Frequency Range (GHz) |
    |-----------------|---------------|----------|-------------------|-----------|-----------------------|
    | CO              | 20 - 380      | 5        | 12 - 19.2         | 0.1       | 80 - 300              |
    | SiO             | 20 - 380      | 5        | 12 - 19.2         | 0.1       | 80 - 300              |
    | HCO⁺ v=0,1,2    | 20 - 380      | 5        | 12 - 19.2         | 0.1       | 80 - 300              |
    | CH3CN           | 20 - 380      | 5        | 12 - 19.2         | 0.1       | 80 - 300              |
    | HNC             | 20 - 380      | 5        | 12 - 19.2         | 0.1       | 80 - 300              |
    | SO              | 20 - 380      | 5        | 12 - 19.2         | 0.1       | 80 - 300              |
    | CH3OCHO_Yebes   | 20 - 350      | 5        | 12 - 19.2         | 0.1       | 20 - 50               |
    | CH3OCHO         | 120 - 380     | 5        | 12 - 19.2         | 0.1       | 80 - 300              |

    *Note: Generated using LTE radiative transfer models under typical ISM conditions.*
    """
}

# Main titles
MAIN_TITLE = "AI-ITACA | Artificial Intelligence Integral Tool for AstroChemical Analysis"
SUBTITLE = "Molecular Spectrum Analyzer"

# texts.py - Contenido con estilos originales

FLOW_OF_WORK = """
<div style="margin-top: 20px;">
<h4 style="color: #1E88E5; margin-bottom: 10px;">Analysis Pipeline Steps:</h4>
<ol style="color: white; padding-left: 20px;">
<li><strong>Spectrum Input:</strong> Upload your observational spectrum</li>
<li><strong>Pre-processing:</strong> Noise reduction and baseline correction</li>
<li><strong>Peak Detection:</strong> Identify significant spectral features</li>
<li><strong>Model Matching:</strong> Compare with synthetic spectra database</li>
<li><strong>Parameter Estimation:</strong> Determine physical conditions (T<sub>ex</sub>, logN)</li>
<li><strong>Visualization:</strong> Interactive comparison of observed vs synthetic spectra</li>
</ol>
</div>

<div class="pro-tip" style="margin-top: 20px; padding: 12px; background-color: #0D0F14; border-left: 4px solid #1E88E5; border-radius: 4px;">
<p style="margin: 0; color: white; font-size: 0.9em;"><strong>Note:</strong> The complete analysis typically takes 30-90 seconds depending on spectrum complexity and selected parameters.</p>
</div>
"""

ACKNOWLEDGMENTS = """
<div class="description-panel" style="text-align: justify; color: white; padding: 10px;">
"The funding for these actions/grants and contracts comes from the European Union's Recovery and Resilience Facility-Next Generation, in the framework of the General Invitation of the Spanish Government's public business entity Red.es to participate in talent attraction and retention programmes within Investment 4 of Component 19 of the Recovery, Transformation and Resilience Plan."
</div>
"""

# Cube Visualizer description
CUBE_VISUALIZER_DESCRIPTION = """
<div class="description-panel">
<h3 style="text-align: center; margin-top: 0; color: black; border-bottom: 2px solid #1E88E5; padding-bottom: 10px;">3D Spectral Cube Visualization</h3>
<p>Upload and visualize ALMA spectral cubes (FITS format) up to 2GB in size. Explore different channels and create integrated intensity maps with these features:</p>
    
<ul style="color: black;">
<li><strong>Interactive Channel Navigation:</strong> Scroll through spectral dimensions</li>
<li><strong>Region Selection:</strong> Extract spectra from specific spatial regions</li>
<li><strong>Dynamic Scaling:</strong> Linear, logarithmic, or square root intensity scaling</li>
<li><strong>Frequency Information:</strong> Automatic detection of spectral axis</li>
</ul>
    
<div class="pro-tip">
<p><strong>Pro Tip:</strong> For best performance with large cubes, select smaller regions when extracting spectra.</p>
</div>
</div>
"""


# Training Dataset (example - include your full content)
TRAINING_DATASET = """
<table class="training-table">
<your full training dataset table here>
</table>
"""
