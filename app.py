import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings('ignore')

# Konfigurasi Halaman
st.set_page_config(
    page_title="Quantum Tunnelling Simulator",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CSS & HTML STYLING
# =============================================================================
custom_css = """
<style>
    .main-header {
        background: linear-gradient(135deg, #004e80, #0077b6);
        color: #ffffff;
        padding: 2.5rem 2rem;
        text-align: center;
        border-radius: 0 0 16px 16px;
        margin-bottom: 24px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.25);
    }
    .main-header h1 {
        margin: 0;
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    .main-header p {
        margin: 10px 0 0;
        font-size: 1.05rem;
        font-weight: 400;
        opacity: 0.95;
    }
    .card-container {
        background: #ffffff;
        padding: 24px;
        border-radius: 12px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.08);
        margin: 16px 0;
        border: 1px solid #eaeaea;
    }
    .card-container h3 {
        margin-top: 0;
        margin-bottom: 8px;
        color: #1f2937;
    }
    .card-container .caption {
        color: #6b7280;
        font-size: 0.95rem;
        margin-bottom: 20px;
        line-height: 1.6;
    }
    .footer {
        text-align: center;
        padding: 24px 0;
        color: #555555;
        font-size: 0.92rem;
        border-top: 2px solid #eaeaea;
        margin-top: 40px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        padding: 0 20px;
        font-weight: 600;
    }
    /* Perbaikan tampilan math */
    .katex {
        font-size: 1.05em !important;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# =============================================================================
# HEADER
# =============================================================================
st.markdown("""
<div class="main-header">
    <h1>Quantum Tunnelling Simulator</h1>
    <p>Dikembangkan oleh Felix Marcellino Henrikus, S.Si.<br>
    Program Studi Magister Sains Data, UKSW Salatiga<br>
    Untuk digunakan dalam pembelajaran Fisika Kuantum di S1 Fisika, UKSW Salatiga</p>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# PHYSICS UTILITIES
# =============================================================================
HBAR = 1.0545718e-34   # J·s
M_E  = 9.1093837e-31   # kg
EV_J = 1.602176634e-19 # J/eV

def calculate_transmission(E_eV, V0_eV, L_nm, m_factor=1.0):
    """Menghitung koefisien transmisi T untuk penghalang persegi."""
    E = np.asarray(E_eV, dtype=float) * EV_J
    V0 = np.asarray(V0_eV, dtype=float) * EV_J
    L = np.asarray(L_nm, dtype=float) * 1e-9
    m = m_factor * M_E

    E, V0, L = np.broadcast_arrays(E, V0, L)
    out_shape = E.shape
    T = np.zeros(out_shape, dtype=float)

    mask_t = E < V0
    if np.any(mask_t):
        kappa = np.sqrt(2 * m * (V0[mask_t] - E[mask_t])) / HBAR
        denom = 4 * E[mask_t] * (V0[mask_t] - E[mask_t])
        T[mask_t] = 1.0 / (1.0 + (V0[mask_t]**2 * np.sinh(kappa * L[mask_t])**2) / denom)

    mask_a = E > V0
    if np.any(mask_a):
        k2 = np.sqrt(2 * m * (E[mask_a] - V0[mask_a])) / HBAR
        denom = 4 * E[mask_a] * (E[mask_a] - V0[mask_a])
        T[mask_a] = 1.0 / (1.0 + (V0[mask_a]**2 * np.sin(k2 * L[mask_a])**2) / denom)

    mask_e = np.isclose(E, V0)
    if np.any(mask_e):
        T[mask_e] = 1.0 / (1.0 + (m * V0[mask_e] * L[mask_e]**2) / (2 * HBAR**2))

    T = np.clip(T, 0, 1)
    return T.item() if T.ndim == 0 else T

def get_wavefunction_profile(E_eV, V0_eV, L_nm, m_factor=1.0):
    """Mengembalikan posisi (nm) dan kerapatan probabilitas |ψ|²."""
    E = E_eV * EV_J
    V0 = V0_eV * EV_J
    L = L_nm * 1e-9
    m = m_factor * M_E

    k1 = np.sqrt(2 * m * E) / HBAR
    kappa = np.sqrt(2 * m * abs(V0 - E)) / HBAR if E != V0 else 1e-9

    if E < V0:
        D = (k1**2 + kappa**2) * np.sinh(kappa * L) + 2j * k1 * kappa * np.cosh(kappa * L)
        R = (k1**2 + kappa**2) * np.sinh(kappa * L) / D
        T_amp = 2j * k1 * kappa * np.exp(-1j * k1 * L) / D
        A = ((1 + R) * kappa + 1j * k1 * (1 - R)) / (2 * kappa)
        B = (1 + R) - A
        k2_eff = None
    else:
        k2 = np.sqrt(2 * m * (E - V0)) / HBAR
        D = (k1**2 - k2**2) * np.sin(k2 * L) + 2j * k1 * k2 * np.cos(k2 * L)
        R = (k1**2 - k2**2) * np.sin(k2 * L) / D
        T_amp = 2j * k1 * k2 * np.exp(-1j * k1 * L) / D
        A = ((1 + R) * k2 + k1 * (1 - R)) / (2 * k2)
        B = (1 + R) - A
        k2_eff = k2

    x_nm = np.linspace(-2.0, L_nm + 3.0, 1200)
    x_si = x_nm * 1e-9
    psi_sq = np.zeros_like(x_nm)

    m1 = x_nm < 0
    psi_sq[m1] = np.abs(np.exp(1j * k1 * x_si[m1]) + R * np.exp(-1j * k1 * x_si[m1]))**2

    m2 = (x_nm >= 0) & (x_nm <= L_nm)
    if E < V0:
        psi_sq[m2] = np.abs(A * np.exp(kappa * x_si[m2]) + B * np.exp(-kappa * x_si[m2]))**2
    else:
        psi_sq[m2] = np.abs(A * np.exp(1j * k2_eff * x_si[m2]) + B * np.exp(-1j * k2_eff * x_si[m2]))**2

    m3 = x_nm > L_nm
    psi_sq[m3] = np.abs(T_amp * np.exp(1j * k1 * x_si[m3]))**2

    psi_sq /= np.max(psi_sq) if np.max(psi_sq) > 0 else 1.0
    return x_nm, psi_sq, L_nm

# =============================================================================
# SIDEBAR INPUTS
# =============================================================================
with st.sidebar:
    st.header("⚙️ Parameter Partikel & Penghalang")
    E_eV = st.slider("Energi Partikel (E) [eV]", 0.1, 12.0, 3.0, 0.1)
    V0_eV = st.slider("Tinggi Penghalang (V₀) [eV]", 1.0, 15.0, 5.0, 0.1)
    L_nm = st.slider("Lebar Penghalang (L) [nm]", 0.1, 2.5, 0.5, 0.05)
    m_factor = st.slider("Massa Partikel (dalam mₑ)", 0.1, 5.0, 1.0, 0.1)

# =============================================================================
# MAIN CONTENT TABS
# =============================================================================
tab1, tab2 = st.tabs([
    "📈 Potensial & Profil Gelombang",
    "📊 Analisis Probabilitas Tunneling"
])

# --- TAB 1: POTENSIAL & GELOMBANG ---
with tab1:
    st.markdown("""
    <div class="card-container">
        <h3>📊 Visualisasi Incoming, Reflected & Transmitted Wave</h3>
        <p class="caption">Grafik menunjukkan kerapatan probabilitas $|\psi(x)|^2$ pada tiga wilayah: sebelum, di dalam, dan setelah penghalang potensial.</p>
    </div>
    """, unsafe_allow_html=True)
    
    x_nm, psi_sq, barrier_L = get_wavefunction_profile(E_eV, V0_eV, L_nm, m_factor)
    T_val = calculate_transmission(E_eV, V0_eV, L_nm, m_factor)
    R_val = 1.0 - T_val

    fig_wf = go.Figure()
    
    # Penghalang Potensial
    fig_wf.add_vrect(
        x0=0, x1=barrier_L, 
        fillcolor="rgba(255,165,0,0.2)", 
        line_color="orange", 
        line_width=1,
        layer="below"
    )
    
    # Garis referensi y=0
    fig_wf.add_hline(y=0, line_dash="dot", line_color="gray")
    
    # Gelombang
    fig_wf.add_trace(go.Scatter(
        x=x_nm, y=psi_sq, 
        mode='lines', 
        line=dict(color='#1f77b4', width=3), 
        name="|ψ(x)|²"
    ))
    
    # Batas-batas penghalang (dipindahkan ke luar area)
    fig_wf.add_vline(x=0, line_dash="dash", line_color="#ff7f0e")
    fig_wf.add_annotation(
        x=-0.3, y=0.9, 
        text="Batas Masuk", 
        showarrow=False,
        font=dict(size=11, color="#ff7f0e"),
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="#ff7f0e",
        borderwidth=1,
        borderpad=4
    )
    
    fig_wf.add_vline(x=barrier_L, line_dash="dash", line_color="#ff7f0e")
    fig_wf.add_annotation(
        x=barrier_L + 0.3, y=0.9, 
        text="Batas Keluar", 
        showarrow=False,
        font=dict(size=11, color="#ff7f0e"),
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="#ff7f0e",
        borderwidth=1,
        borderpad=4
    )

    fig_wf.update_layout(
        height=400,
        xaxis_title="Posisi x [nm]",
        yaxis_title="Kerapatan Probabilitas (Ternormalisasi)",
        template="plotly_white",
        showlegend=True,
        legend=dict(y=1.1, x=0.01),
        margin=dict(l=60, r=60, t=40, b=60)
    )
    fig_wf.update_yaxes(range=[0, 1.1])
    st.plotly_chart(fig_wf, use_container_width=True)

    st.markdown(f"""
    <div class="card-container">
        <h4>📈 Hasil Perhitungan Koefisien:</h4>
        <ul>
            <li><strong>Probabilitas Transmisi (T):</strong> {T_val:.4f} ({T_val*100:.2f}%)</li>
            <li><strong>Probabilitas Refleksi (R):</strong> {R_val:.4f} ({R_val*100:.2f}%)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# --- TAB 2: PROBABILITAS TUNNELING ---
with tab2:
    st.markdown("""
    <div class="card-container">
        <h3> Hubungan Energi & Lebar Barrier terhadap Probabilitas Tunneling</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Kurva T vs E
    E_range = np.linspace(0.1, V0_eV + 5.0, 150)
    T_vs_E = calculate_transmission(E_range, V0_eV, L_nm, m_factor)
    
    # Kurva T vs L
    L_range = np.linspace(0.05, 2.5, 150)
    T_vs_L = calculate_transmission(E_eV, V0_eV, L_range, m_factor)

    fig_prob = make_subplots(rows=1, cols=2, subplot_titles=("Transmisi vs Energi (E)", "Transmisi vs Lebar Barrier (L)"))
    
    fig_prob.add_trace(go.Scatter(x=E_range, y=T_vs_E, mode='lines', line=dict(color='#2ca02c', width=3), name="T(E)"), row=1, col=1)
    fig_prob.add_vline(x=E_eV, line_dash="dash", line_color="red", row=1, col=1)
    fig_prob.add_annotation(x=E_eV, y=0.5, text=f"E = {E_eV} eV", showarrow=False, row=1, col=1)

    fig_prob.add_trace(go.Scatter(x=L_range, y=T_vs_L, mode='lines', line=dict(color='#9467bd', width=3), name="T(L)"), row=1, col=2)
    fig_prob.add_vline(x=L_nm, line_dash="dash", line_color="red", row=1, col=2)
    fig_prob.add_annotation(x=L_nm, y=0.5, text=f"L = {L_nm} nm", showarrow=False, row=1, col=2)

    fig_prob.update_layout(height=420, template="plotly_white")
    fig_prob.update_xaxes(title_text="Energi E [eV]", row=1, col=1)
    fig_prob.update_yaxes(title_text="Probabilitas Transmisi", row=1, col=1)
    fig_prob.update_xaxes(title_text="Lebar Barrier L [nm]", row=1, col=2)
    fig_prob.update_yaxes(title_text="Probabilitas Transmisi", row=1, col=2)
    st.plotly_chart(fig_prob, use_container_width=True)

# =============================================================================
# FOOTER
# =============================================================================
st.markdown('<div class="footer">© 2026 - Felix Marcellino Henrikus, S.Si. - UKSW Salatiga</div>', unsafe_allow_html=True)
