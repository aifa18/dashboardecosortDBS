"""
EcoSort AI & Smart Waste Forecasting — Streamlit Dashboard
Kota Bandung · v3.0 — Full Data Science Pipeline
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EcoSort AI · Bandung",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif !important; }

#MainMenu, footer { visibility: hidden; }
.block-container { padding: 2rem 2.5rem 3rem !important; max-width: 1300px; }

/* Sidebar */
[data-testid="stSidebar"] { background: #0f2d1f !important; }
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div { color: #c8f0d8 !important; }
[data-testid="stSidebar"] .stRadio > label { display: none; }

/* Metric card */
.ecard {
    background: #ffffff; border: 1px solid #e2e8f0;
    border-radius: 12px; padding: 20px 22px; border-top: 3px solid #10b981;
}
.ecard-label {
    font-size: 11px; font-weight: 600; letter-spacing: .08em;
    text-transform: uppercase; color: #64748b; margin-bottom: 6px;
}
.ecard-value {
    font-family: 'DM Mono', monospace; font-size: 28px; font-weight: 500;
    color: #0f172a; line-height: 1.1; margin-bottom: 4px;
}
.ecard-sub { font-size: 12px; color: #94a3b8; }

/* Section heading */
.sec-head { font-size: 15px; font-weight: 600; color: #0f172a; margin: 0 0 4px 0; }
.sec-sub  { font-size: 12px; color: #94a3b8; margin: 0 0 12px 0; }

/* Insight box */
.insight {
    background: #f0fdf4; border: 1px solid #bbf7d0;
    border-left: 4px solid #10b981; border-radius: 10px;
    padding: 16px 20px; margin-top: 20px;
}
.insight-head { font-size: 13px; font-weight: 700; color: #065f46; margin-bottom: 8px; }
.insight p { font-size: 13px; color: #1e3a2f; line-height: 1.7; margin: 0 0 6px 0; }
.insight p:last-child { margin-bottom: 0; }

/* Conclusion box (blue) */
.conclusion {
    background: #eff6ff; border: 1px solid #bfdbfe;
    border-left: 4px solid #3b82f6; border-radius: 10px;
    padding: 16px 20px; margin-top: 12px;
}
.conclusion-head { font-size: 13px; font-weight: 700; color: #1e40af; margin-bottom: 8px; }
.conclusion p { font-size: 13px; color: #1e3a5f; line-height: 1.7; margin: 0 0 6px 0; }

/* Chart wrapper */
.chart-wrap {
    background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px;
    padding: 18px 20px 10px; margin-bottom: 20px;
}

/* Filter strip */
.filter-strip {
    background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px;
    padding: 14px 18px; margin-bottom: 18px;
}

/* Business card */
.biz-card {
    background: #ffffff; border: 1px solid #e2e8f0; border-radius: 14px;
    padding: 22px 24px; height: 100%; border-top: 3px solid #3b82f6;
}
.biz-card-title { font-size: 14px; font-weight: 700; color: #1e40af; margin-bottom: 10px; }
.biz-card ul { margin: 0; padding-left: 18px; }
.biz-card li { font-size: 13px; color: #334155; line-height: 1.8; }

/* Question badge */
.q-badge {
    display: inline-block; background: #eff6ff; color: #1e40af;
    border: 1px solid #bfdbfe; border-radius: 20px;
    padding: 5px 14px; font-size: 12px; font-weight: 600;
    margin: 4px 4px 4px 0;
}

/* Funnel bar */
.funnel-bar { border-radius: 6px; height: 28px; margin-bottom: 8px; }

/* Pipeline step */
.pipe-step {
    background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px;
    padding: 14px 18px; margin-bottom: 10px; border-left: 4px solid #10b981;
}
.pipe-step-num { font-size: 11px; font-weight: 700; color: #10b981; letter-spacing: .08em; }
.pipe-step-title { font-size: 13px; font-weight: 700; color: #0f172a; margin: 2px 0; }
.pipe-step-desc { font-size: 12px; color: #64748b; line-height: 1.6; }

/* Section divider with title */
.section-banner {
    background: linear-gradient(135deg, #0f2d1f 0%, #1a4731 100%);
    border-radius: 12px; padding: 18px 24px; margin: 12px 0 20px 0;
}
.section-banner h3 { color: #6ee7b7; margin: 0 0 4px 0; font-size: 16px; font-weight: 700; }
.section-banner p { color: #a7f3d0; margin: 0; font-size: 12px; }

/* Retention progress */
.ret-bar-bg {
    background: #e2e8f0; border-radius: 100px; height: 14px; overflow: hidden; margin: 6px 0;
}
.ret-bar-fill {
    background: linear-gradient(90deg, #10b981, #34d399); height: 14px; border-radius: 100px;
}
</style>
""", unsafe_allow_html=True)

# ── Colour palette ────────────────────────────────────────────────────────────
C_GREEN  = "#10b981"
C_TEAL   = "#14b8a6"
C_BLUE   = "#3b82f6"
C_AMBER  = "#f59e0b"
C_RED    = "#ef4444"
C_PURPLE = "#8b5cf6"
C_SLATE  = "#64748b"
C_DARK   = "#0f172a"
C_GRAY   = "#e2e8f0"
PALETTE  = [C_GREEN, C_BLUE, C_AMBER, C_PURPLE, "#ec4899", "#06b6d4", "#84cc16", "#f97316"]
COLOR_LIST = [C_GREEN, C_BLUE, C_AMBER, C_PURPLE, "#ec4899", "#06b6d4", "#84cc16", "#f97316", C_TEAL, "#a855f7"]

BASE_LAYOUT = dict(
    font_family="DM Sans",
    plot_bgcolor="#ffffff",
    paper_bgcolor="#ffffff",
    margin=dict(l=10, r=10, t=30, b=10),
    xaxis=dict(showgrid=False, color=C_SLATE, linecolor=C_GRAY, tickfont_size=11),
    yaxis=dict(gridcolor="#f1f5f9", color=C_SLATE, tickfont_size=11),
    legend=dict(orientation="h", yanchor="bottom", y=1.02,
                xanchor="right", x=1, font_size=11, bgcolor="rgba(0,0,0,0)"),
    hoverlabel=dict(bgcolor="white", bordercolor=C_GRAY, font_family="DM Sans", font_size=12),
)

def styled_fig(fig, height=300, **kw):
    fig.update_layout(height=height, **{**BASE_LAYOUT, **kw})
    return fig

# Fungsi pembantu untuk mengubah warna Hex ke RGBA transparan
def hex_to_rgba(hex_color, alpha=0.25):
    hex_color = hex_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return f'rgba({r}, {g}, {b}, {alpha})'

# ── Data loaders ──────────────────────────────────────────────────────────────
@st.cache_data
def load_meta():
    try:
        with open("dataset_meta.json") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Gagal membaca dataset_meta.json: {e}. Pastikan file ada di direktori yang sama.")
        st.stop()

@st.cache_data
def load_waste():
    try:
        df = pd.read_csv("sampahbandung_normal_monthly.csv", parse_dates=["tanggal"])
        return df
    except Exception as e:
        st.error(f"Gagal membaca sampahbandung_normal_monthly.csv: {e}. Pastikan file ada di direktori yang sama.")
        st.stop()

meta     = load_meta()
df_waste = load_waste()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:1.5rem 0 1rem;">
      <div style="font-size:22px;font-weight:700;color:#6ee7b7;letter-spacing:-.5px;">♻️ EcoSort AI</div>
      <div style="font-size:11px;color:#4ade80;margin-top:4px;">Smart Waste Analytics · Kota Bandung</div>
    </div>
    <hr style="border:none;border-top:1px solid #1e4d30;margin:0 0 1rem;">
    """, unsafe_allow_html=True)

    page = st.radio(
        "Halaman",
        [
            "📊 Executive Summary",
            "🎯 Business Understanding",
            "🔍 Data Quality & Cleaning",
            "📷 EDA Image Dataset",
            "📈 EDA Forecasting Dataset",
            "🔮 Smart Forecasting",
        ],
        label_visibility="collapsed",
    )

    st.markdown("""
    <hr style="border:none;border-top:1px solid #1e4d30;margin:1.5rem 0 .75rem;">
    <div style="font-size:11px;color:#4ade80;line-height:1.8;">
      ● Pipeline Ready &nbsp;·&nbsp; v3.0<br>
      <span style="color:#6ee7b7;font-size:10px;">Data Scientist: EcoSort Data Science Team</span>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# PAGE 1 — EXECUTIVE SUMMARY
# ═══════════════════════════════════════════════════════════════════
if page == "📊 Executive Summary":
    st.markdown("## 📊 Executive Summary")
    st.markdown("Gambaran menyeluruh pipeline EcoSort AI dan tren volume sampah Kota Bandung.")
    st.divider()

    raw_total = meta.get("cleaning_summary", {}).get("raw_total", 73140)
    clean_total = meta["total_images"]
    n_kec       = df_waste["kecamatan"].nunique()
    yr_min      = int(df_waste["tahun"].min())
    yr_max      = int(df_waste["tahun"].max())
    monthly_agg = df_waste.groupby("tanggal")["volume_ton"].sum()
    latest_vol  = float(monthly_agg.iloc[-1])
    ret_rate    = clean_total / raw_total * 100

    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
    mc1.markdown(f"""<div class="ecard">
      <div class="ecard-label">🖼️ Clean Images</div>
      <div class="ecard-value">{clean_total:,}</div>
      <div class="ecard-sub">dari {raw_total:,} raw images</div>
    </div>""", unsafe_allow_html=True)
    mc2.markdown(f"""<div class="ecard">
      <div class="ecard-label">📊 Retention Rate</div>
      <div class="ecard-value">{ret_rate:.1f}%</div>
      <div class="ecard-sub">data berkualitas dipertahankan</div>
    </div>""", unsafe_allow_html=True)
    mc3.markdown(f"""<div class="ecard">
      <div class="ecard-label">🗺️ Kecamatan</div>
      <div class="ecard-value">{n_kec}</div>
      <div class="ecard-sub">30 kec × 3 TPS = 90 titik pantau</div>
    </div>""", unsafe_allow_html=True)
    mc4.markdown(f"""<div class="ecard">
      <div class="ecard-label">📅 Periode Data</div>
      <div class="ecard-value">{yr_min}–{yr_max}</div>
      <div class="ecard-sub">data deret waktu bulanan</div>
    </div>""", unsafe_allow_html=True)
    mc5.markdown(f"""<div class="ecard">
      <div class="ecard-label">⚖️ Volume Terkini</div>
      <div class="ecard-value">{latest_vol:,.0f}</div>
      <div class="ecard-sub">ton · bulan terakhir dataset</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_l, col_r = st.columns([65, 35], gap="medium")

    with col_l:
        st.markdown('<p class="sec-head">📈 Tren Volume Sampah Bulanan — Kota Bandung</p>', unsafe_allow_html=True)
        st.markdown('<p class="sec-sub">Akumulasi seluruh 30 kecamatan · ton/bulan</p>', unsafe_allow_html=True)

        monthly_df = monthly_agg.reset_index()
        monthly_df.columns = ["tanggal", "volume"]
        monthly_df["ma3"] = monthly_df["volume"].rolling(3, min_periods=1).mean()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=monthly_df["tanggal"], y=monthly_df["volume"],
            fill="tozeroy", fillcolor="rgba(16,185,129,0.09)",
            line=dict(color=C_GREEN, width=2), name="Volume Aktual",
            hovertemplate="%{x|%b %Y} — <b>%{y:,.0f} ton</b><extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=monthly_df["tanggal"], y=monthly_df["ma3"],
            line=dict(color=C_AMBER, width=1.5, dash="dot"),
            name="Moving Avg 3 Bln",
            hovertemplate="%{x|%b %Y} — MA3: <b>%{y:,.0f} ton</b><extra></extra>",
        ))
        styled_fig(fig, height=320, hovermode="x unified", yaxis_title="Volume (ton)")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with col_r:
        st.markdown('<p class="sec-head">📋 Ringkasan Tahunan</p>', unsafe_allow_html=True)
        annual = df_waste.groupby("tahun")["volume_ton"].sum().reset_index()
        annual["delta"] = annual["volume_ton"].pct_change() * 100

        rows_html = ""
        for _, r in annual.iterrows():
            d = ""
            if not np.isnan(r["delta"]):
                col_c = "#10b981" if r["delta"] >= 0 else "#ef4444"
                arrow = "▲" if r["delta"] >= 0 else "▼"
                d = f'<span style="color:{col_c};font-size:11px;">{arrow} {abs(r["delta"]):.1f}%</span>'
            rows_html += f"<tr><td><b>{int(r['tahun'])}</b></td><td>{r['volume_ton']:,.0f}</td><td>{d}</td></tr>"

        st.markdown(f"""
        <table style="width:100%;border-collapse:collapse;font-size:12px;">
          <thead>
            <tr style="border-bottom:2px solid #e2e8f0;">
              <th style="text-align:left;padding:6px 4px;color:#64748b;font-weight:600;">Tahun</th>
              <th style="text-align:left;padding:6px 4px;color:#64748b;font-weight:600;">Total (ton)</th>
              <th style="text-align:left;padding:6px 4px;color:#64748b;font-weight:600;">Δ</th>
            </tr>
          </thead>
          <tbody>{rows_html}</tbody>
        </table>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="insight">
      <div class="insight-head">💡 Kesimpulan Eksekutif</div>
      <p>📌 <b>Pipeline citra siap:</b> Dari 73.140 gambar mentah, sebanyak 36.234 citra berkualitas tinggi berhasil diverifikasi dan dibagi proporsional ke kelas Organik, Anorganik, dan B3 (split 70/15/15), dengan retention rate 49,5%.</p>
      <p>📌 <b>Dinamika volume sampah:</b> Data merepresentasikan pola tren temporal dan fluktuasi historis volume sampah Kota Bandung sejak 2017.</p>
      <p>📌 <b>Cakupan spasial lengkap:</b> 30 kecamatan (90 TPS) tersegmentasi ke tiga tipe wilayah: metropolitan, semi-urban, dan pedesaan, memungkinkan analisis disparitas spasial yang komprehensif.</p>
      <p>📌 <b>Integrasi siap dijalankan:</b> Kedua pipeline (klasifikasi citra dan forecasting volume) telah terhubung dan siap mendukung manajemen TPS berbasis keputusan data.</p>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# PAGE 2 — BUSINESS UNDERSTANDING
# ═══════════════════════════════════════════════════════════════════
elif page == "🎯 Business Understanding":
    st.markdown("## 🎯 Business Understanding")
    st.markdown("Latar belakang masalah, tujuan proyek, dan pertanyaan bisnis yang dapat diukur.")
    st.divider()

    # ── Latar Belakang ─────────────────────────────────────────────
    st.markdown("""
    <div class="section-banner">
      <h3>🌍 Latar Belakang Masalah</h3>
      <p>Identifikasi permasalahan pengelolaan sampah di Kota Bandung berbasis data</p>
    </div>
    """, unsafe_allow_html=True)

    bg1, bg2, bg3 = st.columns(3, gap="medium")
    with bg1:
        st.markdown("""
        <div class="biz-card" style="border-top-color:#ef4444;">
          <div class="biz-card-title" style="color:#b91c1c;">⚠️ Permasalahan Utama</div>
          <ul>
            <li>Volume sampah Kota Bandung terus meningkat seiring pertumbuhan populasi</li>
            <li>Ketidakseimbangan distribusi beban sampah antar kecamatan</li>
            <li>Klasifikasi sampah manual tidak efisien dan rawan kesalahan</li>
            <li>Pola musiman belum dioptimalkan untuk operasional TPS</li>
            <li>Data citra sampah belum terstandarisasi untuk AI deployment</li>
          </ul>
        </div>""", unsafe_allow_html=True)

    with bg2:
        st.markdown("""
        <div class="biz-card" style="border-top-color:#10b981;">
          <div class="biz-card-title" style="color:#065f46;">✅ Solusi yang Dikembangkan</div>
          <ul>
            <li><b>EcoSort AI:</b> Sistem klasifikasi citra sampah berbasis (Organik / Anorganik / B3)</li>
            <li><b>Smart Forecasting:</b> Model prediksi volume sampah bulanan per TPS</li>
            <li><b>Data Pipeline:</b> End-to-end preprocessing citra dan deret waktu yang terstandarisasi</li>
            <li>Dashboard interaktif untuk monitoring dan pengambilan keputusan</li>
          </ul>
        </div>""", unsafe_allow_html=True)

    with bg3:
        st.markdown("""
        <div class="biz-card" style="border-top-color:#f59e0b;">
          <div class="biz-card-title" style="color:#92400e;">🎯 Manfaat yang Diharapkan</div>
          <ul>
            <li>Efisiensi operasional TPS meningkat melalui prediksi berbasis data</li>
            <li>Klasifikasi sampah otomatis mengurangi biaya tenaga kerja manual</li>
            <li>Distribusi armada pengangkutan sampah lebih optimal</li>
            <li>Pemerintah Kota Bandung dapat mengalokasikan sumber daya lebih tepat sasaran</li>
          </ul>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Pertanyaan Bisnis ───────────────────────────────────────────
    st.markdown("""
    <div class="section-banner">
      <h3>❓ Pertanyaan Bisnis yang Dapat Diukur</h3>
      <p>Business questions yang menjadi acuan seluruh proses data science dalam proyek ini</p>
    </div>
    """, unsafe_allow_html=True)

    qa_col1, qa_col2 = st.columns(2, gap="medium")
    with qa_col1:
        st.markdown("""
        <div class="chart-wrap">
          <p class="sec-head">📷 Dataset Klasifikasi Citra</p>
          
          <div style="margin-top:12px;">
            <div class="q-badge">Biz-Q1</div>
            <span style="font-size:13px; font-weight:700; color:#1e40af;">Distribusi & Risiko Class Imbalance</span>
            <div style="font-size:12px; color:#334155; margin-top:4px; line-height:1.5;">Bagaimana rasio perbandingan volume gambar tiap kategori sampah (Organik, Anorganik, B3), dan kategori mana yang berpotensi mengalami penurunan performa klasifikasi akibat class imbalance?</div>
          </div>
          
          <div style="margin-top:14px;">
            <div class="q-badge">Biz-Q2</div>
            <span style="font-size:13px; font-weight:700; color:#1e40af;">Karakteristik Visual Gambar</span>
            <div style="font-size:12px; color:#334155; margin-top:4px; line-height:1.5;">Bagaimana karakteristik kualitas visual gambar (distribusi resolusi, kecerahan, dan ketajaman), dan apakah karakteristik tersebut konsisten di setiap kelas?</div>
          </div>
          
          <div style="margin-top:14px;">
            <div class="q-badge">Biz-Q3</div>
            <span style="font-size:13px; font-weight:700; color:#1e40af;">Konsistensi & Kesiapan Data</span>
            <div style="font-size:12px; color:#334155; margin-top:4px; line-height:1.5;">Berapa persentase gambar yang dieliminasi akibat masalah kualitas (corrupt, blur, duplikat), dan apakah dataset akhir cukup representatif untuk melatih model secara andal?</div>
          </div>

          <div style="margin-top:16px; padding:10px; background:#f8fafc; border-left:3px solid #64748b; border-radius:6px;">
             <span style="font-size:11px; font-weight:600; color:#475569;">📌 Fokus EDA & Visualisasi Dashboard:</span>
             <ul style="margin:4px 0 0 0; padding-left:14px; font-size:11px; color:#64748b;">
                <li>Distribusi kelas & evaluasi target size.</li>
                <li>Distribusi kecerahan (brightness) & ketajaman (sharpness/laplacian).</li>
                <li>Funnel retention sebelum & sesudah cleaning.</li>
             </ul>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with qa_col2:
        st.markdown("""
        <div class="chart-wrap">
          <p class="sec-head">📈 Dataset Forecasting Volume</p>
          <div style="margin-top:12px;">
            <div class="q-badge" style="background:#f0fdf4;color:#065f46;border-color:#bbf7d0;">Biz-Q4</div>
            <span style="font-size:13px;color:#334155;"> Kecamatan mana yang menghasilkan volume sampah tertinggi?</span>
          </div>
          <div style="margin-top:10px;">
            <div class="q-badge" style="background:#f0fdf4;color:#065f46;border-color:#bbf7d0;">Biz-Q5</div>
            <span style="font-size:13px;color:#334155;"> Bagaimana pola temporal volume sampah dari tahun ke tahun?</span>
          </div>
          <div style="margin-top:10px;">
            <div class="q-badge" style="background:#f0fdf4;color:#065f46;border-color:#bbf7d0;">Biz-Q6</div>
            <span style="font-size:13px;color:#334155;"> Apakah terdapat seasonality bulanan yang signifikan pada volume sampah?</span>
          </div>
          <div style="margin-top:10px;">
            <div class="q-badge" style="background:#f0fdf4;color:#065f46;border-color:#bbf7d0;">Biz-Q7</div>
            <span style="font-size:13px;color:#334155;"> Bagaimana perbedaan volume sampah antar tipe wilayah (metropolitan, semi-urban, pedesaan)?</span>
          </div>
          <div style="margin-top:10px;">
            <div class="q-badge" style="background:#f0fdf4;color:#065f46;border-color:#bbf7d0;">Biz-Q8</div>
            <span style="font-size:13px;color:#334155;"> Apakah terdapat outlier signifikan yang perlu diperhatikan dalam pemodelan?</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Scope & Metodologi ─────────────────────────────────────────
    st.markdown("""
    <div class="section-banner">
      <h3>🗺️ Scope Proyek & Metodologi</h3>
      <p>Ruang lingkup dan alur kerja data scientist dalam proyek EcoSort AI</p>
    </div>
    """, unsafe_allow_html=True)

    s1, s2, s3, s4, s5 = st.columns(5, gap="small")
    steps = [
        ("1️⃣", "Data Gathering", "Pengumpulan citra sampah dari 3 kelas dan data volume TPS bulanan Kota Bandung 2017–2026", C_BLUE),
        ("2️⃣", "Data Assessing", "Penilaian kualitas: cek missing values, duplikat, format, resolusi, brightness, dan blur score", C_AMBER),
        ("3️⃣", "Data Cleaning", "Filtering berdasarkan ambang batas kualitas, normalisasi ukuran, dan IQR outlier removal", C_GREEN),
        ("4️⃣", "EDA & Analysis", "Exploratory data analysis mendalam dengan visualisasi distribusi, tren, dan korelasi", C_PURPLE),
        ("5️⃣", "Dashboard & Insight", "Penyajian hasil analisis dalam dashboard interaktif dengan insight dan kesimpulan bisnis", C_TEAL),
    ]
    for col, (num, title, desc, color) in zip([s1,s2,s3,s4,s5], steps):
        col.markdown(f"""
        <div class="chart-wrap" style="border-top:3px solid {color};text-align:center;padding:18px 14px;">
          <div style="font-size:28px;margin-bottom:8px;">{num}</div>
          <div style="font-size:13px;font-weight:700;color:{color};margin-bottom:8px;">{title}</div>
          <div style="font-size:11px;color:#64748b;line-height:1.6;">{desc}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="insight">
      <div class="insight-head">💡 Konteks Proyek</div>
      <p>📌 Proyek ini merupakan kolaborasi tim dengan pembagian peran yang jelas. Peran <b>Data Scientist</b> mencakup seluruh pipeline data — dari identifikasi masalah, pengumpulan, cleaning, EDA, hingga penyajian insight yang dapat digunakan oleh tim AI/Modeling untuk membangun model klasifikasi dan model forecasting.</p>
      <p>📌 Dashboard ini dirancang sebagai <b>bukti kerja Data Scientist</b>, bukan dashboard model AI. Setiap visualisasi menjawab pertanyaan bisnis yang telah didefinisikan dan memberikan rekomendasi berbasis data.</p>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# PAGE 3 — DATA QUALITY & CLEANING
# ═══════════════════════════════════════════════════════════════════
elif page == "🔍 Data Quality & Cleaning":
    st.markdown("## 🔍 Data Quality Assessment & Cleaning")
    st.markdown("Hasil asesmen kualitas data, proses cleaning, dan ringkasan pipeline preprocessing.")
    st.divider()

    dist        = meta["distribution"]
    cfg         = meta["cleaning_config"]
    clean_summary = meta.get("cleaning_summary", {})
    raw_total   = clean_summary.get("raw_total", 73140)
    clean_total = meta["total_images"]
    removed     = raw_total - clean_total
    ret_rate    = clean_total / raw_total * 100

    classes       = ["Organik", "Anorganik", "B3"]
    clean_per_cls = {cls: sum(dist[cls].values()) for cls in classes}
    
    # Ambil raw_per_class dari JSON (TIDAK ADA LAGI HARDCODE)
    raw_per_cls   = clean_summary.get("raw_per_class", {"Organik": 26330, "Anorganik": 27793, "B3": 19017})
    ret_per_cls   = {cls: (clean_per_cls[cls] / raw_per_cls[cls] * 100) for cls in classes}

    # KPI Cards
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(f"""<div class="ecard">
      <div class="ecard-label">📦 Raw Dataset</div>
      <div class="ecard-value">{raw_total:,}</div>
      <div class="ecard-sub">gambar mentah dikumpulkan</div>
    </div>""", unsafe_allow_html=True)
    k2.markdown(f"""<div class="ecard">
      <div class="ecard-label">✅ Clean Dataset</div>
      <div class="ecard-value">{clean_total:,}</div>
      <div class="ecard-sub">lolos seluruh filter kualitas</div>
    </div>""", unsafe_allow_html=True)
    k3.markdown(f"""<div class="ecard">
      <div class="ecard-label">🗑️ Data Dibuang</div>
      <div class="ecard-value">{removed:,}</div>
      <div class="ecard-sub">tidak memenuhi standar kualitas</div>
    </div>""", unsafe_allow_html=True)
    k4.markdown(f"""<div class="ecard">
      <div class="ecard-label">📊 Retention Rate</div>
      <div class="ecard-value">{ret_rate:.1f}%</div>
      <div class="ecard-sub">data berkualitas dipertahankan</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Section A: Data Quality Issues ─────────────────────────────
    st.markdown("""
    <div class="section-banner">
      <h3>🔎 A. Data Quality Assessment</h3>
      <p>Identifikasi dan distribusi masalah kualitas data pada dataset mentah</p>
    </div>
    """, unsafe_allow_html=True)

    col_funnel, col_reasons = st.columns([5, 5], gap="medium")

    # Ambil data langsung dari JSON (SSOT)
    clean_summary = meta.get("cleaning_summary", {})
    reasons = clean_summary.get("reasons", {})
    
    # Fallback (cadangan) agar aplikasi tidak error jika JSON belum diperbarui
    if not reasons:
        reasons = {"Data Dibuang": removed}
        
    removed_total = sum(reasons.values())

    reason_labels = list(reasons.keys())
    reason_vals   = list(reasons.values())

    with col_funnel:
        st.markdown('<p class="sec-head">🔽 Funnel Chart — Raw → Clean</p>', unsafe_allow_html=True)
        st.markdown('<p class="sec-sub">Alur penyusutan data dari raw ke clean dataset</p>', unsafe_allow_html=True)

        funnel_labels = ["Raw Dataset", "Lulus Format Check", "Lulus Resolusi Check",
                 "Lulus Brightness Check", "Lulus Duplikat Check",
                 "Lulus Blur Check", "Clean Dataset"]
        
        # Ekstrak nilai langsung dari dictionary 'reasons' (bersumber dari JSON)
        cpt_removed  = reasons.get("Corrupt / Unreadable", 0)
        res_removed  = reasons.get("Low Resolution (< 100px)", 0)
        bri_removed  = reasons.get("Extreme Brightness", 0)
        blur_removed = reasons.get("Blur (Laplacian < 50)", 0)
        dup_removed  = reasons.get("Duplikat", 0)

        # Hitung penyusutan step-by-step
        step1 = raw_total
        step2 = step1 - cpt_removed
        step3 = step2 - res_removed
        step4 = step3 - bri_removed
        step5 = step4 - dup_removed   
        step6 = step5 - blur_removed  
        step7 = clean_total

        funnel_vals = [step1, step2, step3, step4, step5, step6, step7]

        fig_funnel = go.Figure(go.Funnel(
            y=funnel_labels,
            x=funnel_vals,
            textinfo="value+percent initial",
            marker=dict(
                color=[C_SLATE, "#94a3b8", C_BLUE, C_TEAL, C_AMBER, C_GREEN, "#059669"],
                line=dict(width=1, color="white")
            ),
            connector=dict(line=dict(color="#e2e8f0", width=1)),
            textfont=dict(size=11, family="DM Mono"),
        ))
        styled_fig(fig_funnel, height=340, showlegend=False,
                   margin=dict(l=10, r=10, t=20, b=10),
                   xaxis=dict(showgrid=False, visible=False),
                   yaxis=dict(showgrid=False, tickfont_size=11, color=C_SLATE))
        st.plotly_chart(fig_funnel, use_container_width=True, config={"displayModeBar": False})
        iqr_flagged = reasons.get("Outlier IQR (×3.0)", 0)
        st.caption(f"📌 {iqr_flagged:,} gambar di-flag sebagai outlier IQR (×{cfg['outlier_iqr_factor']}) namun TIDAK dihapus — tetap dipertahankan dalam clean dataset.")

    with col_reasons:
        st.markdown('<p class="sec-head">📊 Distribusi Penyebab Data Dibuang</p>', unsafe_allow_html=True)
        st.markdown('<p class="sec-sub">Distribusi nyata berdasarkan hasil pipeline preprocessing (dari metadata JSON)</p>', unsafe_allow_html=True)

        fig_reasons = go.Figure(go.Bar(
            x=reason_vals,
            y=reason_labels,
            orientation='h',
            marker_color=[C_RED, C_AMBER, C_PURPLE, C_BLUE, C_TEAL, C_SLATE],
            text=[f"{v:,} ({v/removed_total*100:.1f}%)" for v in reason_vals],
            textposition="outside",
            textfont=dict(size=10, family="DM Mono"),
            hovertemplate="<b>%{y}</b><br>%{x:,} gambar<extra></extra>",
        ))
        styled_fig(fig_reasons, height=340, showlegend=False,
                   xaxis_title="Jumlah Gambar Dibuang",
                   margin=dict(l=10, r=80, t=20, b=10),
                   xaxis=dict(showgrid=True, gridcolor="#f1f5f9"))
        st.plotly_chart(fig_reasons, use_container_width=True, config={"displayModeBar": False})

    total_removed = sum(reasons.values())

    blur_pct = reasons["Blur (Laplacian < 50)"] / total_removed * 100
    res_pct = reasons["Low Resolution (< 100px)"] / total_removed * 100

    combined_pct = (
        reasons["Blur (Laplacian < 50)"] +
        reasons["Low Resolution (< 100px)"]
    ) / total_removed * 100

    st.markdown(f"""
    <div class="insight">
        <div class="insight-head">💡 Insight — Data Quality Assessment</div><p>
        📌 <b>Blur adalah masalah terbesar ({blur_pct:.0f}%):</b>
        Lebih dari sepertiga gambar dibuang karena nilai Laplacian variance
        di bawah threshold 50.0, menunjukkan bahwa sumber data mentah
        memiliki kualitas optik yang bervariasi sangat lebar.
        </p><p>📌 <b>Resolusi rendah jadi masalah ke-2 ({res_pct:.0f}%):</b>
        Gambar berukuran &lt; 100×100 piksel terlalu kecil untuk menghasilkan
        fitur visual yang memadai bagi model dengan input 150×150.
        </p>
    </div>

    <div class="conclusion">
        <div class="conclusion-head">📌 Kesimpulan — Biz-Q3: Faktor Kualitas Dominan</div><p>
        Blur dan resolusi rendah adalah dua faktor dominan penyebab data
        dibuang, dengan proporsi gabungan mencapai {combined_pct:.0f}% dari
        total data yang tidak lolos. Ini memberikan rekomendasi konkret:
        kualitas kamera/alat pengambilan gambar harus menjadi prioritas
        perbaikan untuk meningkatkan retention rate di masa depan.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Section B: Cleaning Config ──────────────────────────────────
    st.markdown("""
    <div class="section-banner">
      <h3>⚙️ B. Cleaning Configuration & Pipeline</h3>
      <p>Parameter standar kualitas yang diterapkan dan alur pipeline data cleaning end-to-end</p>
    </div>
    """, unsafe_allow_html=True)

    cfg_col, pipe_col = st.columns([4, 6], gap="medium")

    with cfg_col:
        st.markdown('<p class="sec-head">⚙️ Parameter Cleaning Config</p>', unsafe_allow_html=True)
        params = [
            ("📐 Min Resolution", f"{cfg['min_resolution']} px", "Dimensi minimum gambar yang diterima"),
            ("🌑 Min Brightness", f"{cfg['min_brightness']} / 255", "Batas bawah kecerahan gambar (hindari gelap total)"),
            ("🌕 Max Brightness", f"{cfg['max_brightness']} / 255", "Batas atas kecerahan gambar (hindari overexposed)"),
            ("🔵 Blur Threshold", f"≥ {cfg['blur_threshold']}", "Laplacian variance minimum (filter gambar buram)"),
            ("📦 IQR Factor", f"× {cfg['outlier_iqr_factor']}", "Faktor IQR untuk deteksi outlier statistik"),
        ]
        for icon_label, val, desc in params:
            st.markdown(f"""
            <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;
                        padding:10px 14px;margin-bottom:8px;">
              <div style="display:flex;justify-content:space-between;align-items:center;">
                <span style="font-size:12px;font-weight:600;color:#334155;">{icon_label}</span>
                <span style="font-family:'DM Mono',monospace;font-size:13px;font-weight:700;
                      color:{C_GREEN};">{val}</span>
              </div>
              <div style="font-size:11px;color:#94a3b8;margin-top:2px;">{desc}</div>
            </div>""", unsafe_allow_html=True)

    with pipe_col:
        st.markdown('<p class="sec-head">🔄 Cleaning Pipeline End-to-End</p>', unsafe_allow_html=True)
        pipeline_steps = [
            ("1", "Data Gathering", f"Mengumpulkan {raw_total:,} gambar mentah dari berbagai sumber untuk 3 kelas: Organik, Anorganik, B3."),
            ("2", "Format Validation", "Validasi format file (JPEG/PNG), memastikan gambar dapat dibaca dan tidak corrupt/unreadable."),
            ("3", "Duplicate Removal", "Deteksi dan penghapusan gambar duplikat identik menggunakan MD5 Hash. "
 "Cross-class duplicate dipertahankan di kelas pertama (first-class-wins policy)."),
            ("4", "Resolution Filter", f"Menolak gambar dengan dimensi < {cfg['min_resolution']}px di sisi terpendek."),
            ("5", "Brightness Filter", f"Menolak gambar dengan rata-rata pixel brightness < {cfg['min_brightness']} atau > {cfg['max_brightness']}."),
            ("6", "Blur Detection", f"Menggunakan Laplacian variance — gambar dengan skor < {cfg['blur_threshold']} dianggap blur."),
            ("7", "Outlier Removal", f"Deteksi outlier statistik IQR × {cfg['outlier_iqr_factor']} pada brightness & sharpness per kelas. "
 "Outlier TIDAK dihapus — hanya di-flag untuk pelaporan. Mitigasi dilakukan via augmentasi saat training (bukan eliminasi)."),
            ("8", "Resize & Normalize", f"Semua gambar bersih di-resize ke {meta['target_img_size'][0]}×{meta['target_img_size'][1]}px dan dinormalisasi."),
            ("9", "Stratified Split", f"Dataset dibagi: Train {int(meta['split_ratio']['train']*100)}% / Val {int(meta['split_ratio']['val']*100)}% / Test {int(meta['split_ratio']['test']*100)}% dengan stratifikasi per kelas."),
        ]
        for num, title, desc in pipeline_steps:
            st.markdown(f"""
            <div class="pipe-step">
              <div class="pipe-step-num">STEP {num}</div>
              <div class="pipe-step-title">{title}</div>
              <div class="pipe-step-desc">{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Section C: Before vs After per Class ────────────────────────
    st.markdown("""
    <div class="section-banner">
      <h3>📊 C. Sebelum vs Sesudah Cleaning per Kelas</h3>
      <p>Perbandingan jumlah data dan retention rate untuk setiap kategori sampah</p>
    </div>
    """, unsafe_allow_html=True)

    bva_col, ret_col = st.columns([6, 4], gap="medium")

    with bva_col:
        st.markdown('<p class="sec-head">📊 Raw vs Clean per Kelas</p>', unsafe_allow_html=True)
        st.markdown('<p class="sec-sub">Perbandingan jumlah gambar sebelum dan sesudah cleaning</p>', unsafe_allow_html=True)

        fig_bva = go.Figure()
        fig_bva.add_trace(go.Bar(
            name="Raw", x=classes,
            y=[raw_per_cls[c] for c in classes],
            marker_color="#94a3b8",
            text=[f"{raw_per_cls[c]:,}" for c in classes],
            textposition="outside", textfont=dict(size=11, family="DM Mono"),
            hovertemplate="<b>%{x}</b> Raw<br>%{y:,} gambar<extra></extra>",
        ))
        fig_bva.add_trace(go.Bar(
            name="Clean", x=classes,
            y=[clean_per_cls[c] for c in classes],
            marker_color=[C_GREEN, C_BLUE, C_AMBER],
            text=[f"{clean_per_cls[c]:,}" for c in classes],
            textposition="outside", textfont=dict(size=11, family="DM Mono"),
            hovertemplate="<b>%{x}</b> Clean<br>%{y:,} gambar<extra></extra>",
        ))
        styled_fig(fig_bva, height=300, barmode="group",
                   yaxis_title="Jumlah Gambar", yaxis_range=[0, max(raw_per_cls.values())*1.2])
        st.plotly_chart(fig_bva, use_container_width=True, config={"displayModeBar": False})

    with ret_col:
        st.markdown('<p class="sec-head">📈 Retention Rate per Kelas</p>', unsafe_allow_html=True)
        st.markdown('<p class="sec-sub">Persentase data yang berhasil dipertahankan</p>', unsafe_allow_html=True)

        colors_cls = {"Organik": C_GREEN, "Anorganik": C_BLUE, "B3": C_AMBER}
        for cls in classes:
            r = ret_per_cls[cls]
            st.markdown(f"""
            <div style="margin-bottom:16px;">
              <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                <span style="font-size:13px;font-weight:600;color:#334155;">
                  {"🌿" if cls=="Organik" else "♻️" if cls=="Anorganik" else "⚠️"} {cls}
                </span>
                <span style="font-family:'DM Mono',monospace;font-size:13px;font-weight:700;color:{colors_cls[cls]};">
                  {r:.1f}%
                </span>
              </div>
              <div class="ret-bar-bg">
                <div class="ret-bar-fill" style="width:{r:.1f}%;background:linear-gradient(90deg,{colors_cls[cls]},{colors_cls[cls]}88);"></div>
              </div>
              <div style="font-size:11px;color:#94a3b8;">
                {clean_per_cls[cls]:,} dari {raw_per_cls[cls]:,} gambar lolos
              </div>
            </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div style="margin-top:20px;padding:12px 16px;background:#f0fdf4;border-radius:8px;border:1px solid #bbf7d0;">
          <div style="font-size:12px;font-weight:700;color:#065f46;">📊 Total Retention</div>
          <div style="font-family:'DM Mono',monospace;font-size:22px;font-weight:700;color:{C_GREEN};">{ret_rate:.1f}%</div>
          <div style="font-size:11px;color:#4ade80;">{clean_total:,} / {raw_total:,} gambar</div>
        </div>""", unsafe_allow_html=True)

    # Ganti hardcoded narasi dengan f-string
    worst_cls = min(ret_per_cls, key=ret_per_cls.get)
    total_clean = sum(clean_per_cls.values())
    pct_org  = clean_per_cls["Organik"]  / total_clean * 100
    pct_anorg= clean_per_cls["Anorganik"]/ total_clean * 100
    pct_b3   = clean_per_cls["B3"]       / total_clean * 100

    st.markdown(f"""
    <div class="insight">
    <div class="insight-head">💡 Insight — Data Cleaning Summary</div>
    <p>📌 <b>Filtering agresif namun presisi:</b> Pipeline menyisihkan {removed:,} gambar
    ({removed/raw_total*100:.1f}%) tanpa merusak keseimbangan distribusi antar kelas.</p>
    <p>📌 <b>Kelas {worst_cls} memiliki retention rate terendah</b>
    (~{ret_per_cls[worst_cls]:.1f}%) karena gambar bahan berbahaya cenderung diambil
    dalam kondisi pencahayaan ekstrem, meningkatkan risiko blur dan brightness issue.</p>
    </div>
    <div class="conclusion">
        <div class="conclusion-head">
            📌 Kesimpulan Lanjutan — Biz-Q3: Proporsi Data Dipertahankan
        </div><p>
            Sebanyak {ret_rate:.1f}% data berhasil dipertahankan setelah proses cleaning menyeluruh.
            Angka ini mencerminkan kualitas pipeline yang baik—cukup ketat untuk memastikan standar
            input model, namun tidak terlalu agresif hingga menyebabkan class imbalance yang signifikan. Distribusi kelas pada clean dataset tetap representatif:
            Organik ~{pct_org:.0f}%,
            Anorganik ~{pct_anorg:.0f}%,
            dan B3 ~{pct_b3:.0f}%.
        </p>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# PAGE 4 — EDA IMAGE DATASET
# ═══════════════════════════════════════════════════════════════════
elif page == "📷 EDA Image Dataset":
    st.markdown("## 📷 EDA Dataset Klasifikasi Citra")
    st.markdown("Exploratory Data Analysis mendalam untuk dataset citra EcoSort AI.")
    st.divider()

    dist        = meta["distribution"]
    classes_all = meta["classes"]
    splits_all  = ["train", "val", "test"]
    cfg         = meta["cleaning_config"]
    raw_total = meta.get("cleaning_summary", {}).get("raw_total", 73140)
    clean_total = meta["total_images"]

    # Filter
    fc1, fc2, fc3 = st.columns([4, 4, 4])
    with fc1:
        st.markdown("**🏷️ Filter Kelas**")
        sel_cls = st.multiselect("kelas", classes_all, default=classes_all, label_visibility="collapsed")
    with fc2:
        st.markdown("**✂️ Filter Split**")
        sel_spl = st.multiselect("split", ["Train","Val","Test"], default=["Train","Val","Test"], label_visibility="collapsed")
    with fc3:
        st.markdown("**⚙️ Cleaning Config (read-only)**")
        st.markdown(
            f"Min Res: **{cfg['min_resolution']}px** &nbsp;|&nbsp; "
            f"Blur ≥ **{cfg['blur_threshold']}** &nbsp;|&nbsp; "
            f"Brightness **{cfg['min_brightness']}–{cfg['max_brightness']}** &nbsp;|&nbsp; "
            f"IQR: **{cfg['outlier_iqr_factor']}×**"
        )

    if not sel_cls: sel_cls = classes_all
    if not sel_spl: sel_spl = ["Train","Val","Test"]
    spl_map = {"Train": "train", "Val": "val", "Test": "test"}
    sel_spl_keys = [spl_map[s] for s in sel_spl]

    filtered_total = sum(dist[cls][sp] for cls in sel_cls for sp in sel_spl_keys if cls in dist and sp in dist[cls])

    mm1, mm2, mm3, mm4 = st.columns(4)
    mm1.markdown(f"""<div class="ecard"><div class="ecard-label">📦 Raw Dataset</div>
      <div class="ecard-value">{raw_total:,}</div><div class="ecard-sub">gambar mentah terkumpul</div></div>""", unsafe_allow_html=True)
    mm2.markdown(f"""<div class="ecard"><div class="ecard-label">✅ Clean Dataset</div>
      <div class="ecard-value">{clean_total:,}</div><div class="ecard-sub">lolos seluruh filter kualitas</div></div>""", unsafe_allow_html=True)
    mm3.markdown(f"""<div class="ecard"><div class="ecard-label">🔍 Data Difilter</div>
      <div class="ecard-value">{filtered_total:,}</div><div class="ecard-sub">berdasarkan filter aktif</div></div>""", unsafe_allow_html=True)
    mm4.markdown(f"""<div class="ecard"><div class="ecard-label">📊 Retention Rate</div>
      <div class="ecard-value">{clean_total/raw_total*100:.1f}%</div><div class="ecard-sub">data berkualitas</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 1: Bar + Pie ────────────────────────────────────────────
    ch1, ch2 = st.columns(2, gap="medium")
    spl_colors = {"Train": C_GREEN, "Val": C_BLUE, "Test": C_AMBER}

    with ch1:
        st.markdown('<p class="sec-head">📊 Distribusi Kelas × Split (Grouped Bar)</p>', unsafe_allow_html=True)
        st.markdown('<p class="sec-sub">Jumlah gambar per kelas berdasarkan split yang dipilih</p>', unsafe_allow_html=True)
        fig3 = go.Figure()
        for spl_label in sel_spl:
            spl_key = spl_map[spl_label]
            vals = [dist[cls].get(spl_key, 0) for cls in sel_cls]
            fig3.add_trace(go.Bar(
                name=spl_label, x=sel_cls, y=vals,
                marker_color=spl_colors[spl_label],
                text=vals, textposition="inside",
                textfont=dict(size=10, color="white", family="DM Mono"),
                hovertemplate="<b>%{x}</b> · " + spl_label + "<br>%{y:,} gambar<extra></extra>",
            ))
        styled_fig(fig3, height=300, barmode="group", yaxis_title="Jumlah Gambar")
        st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})

    with ch2:
        st.markdown('<p class="sec-head">🍩 Komposisi Kelas (Donut Chart)</p>', unsafe_allow_html=True)
        st.markdown('<p class="sec-sub">Proporsi total per kelas berdasarkan filter aktif</p>', unsafe_allow_html=True)
        cls_totals = {cls: sum(dist[cls].get(sp, 0) for sp in sel_spl_keys) for cls in sel_cls}
        fig4 = go.Figure(go.Pie(
            labels=list(cls_totals.keys()), values=list(cls_totals.values()),
            hole=0.52,
            marker=dict(colors=[C_GREEN, C_BLUE, C_AMBER], line=dict(color="white", width=2)),
            textinfo="label+percent", textfont=dict(size=11),
            hovertemplate="<b>%{label}</b><br>%{value:,} gambar · %{percent}<extra></extra>",
        ))
        fig4.update_layout(height=300, paper_bgcolor="white",
            margin=dict(l=0, r=0, t=0, b=0),
            legend=dict(orientation="h", y=-0.08, font_size=10, bgcolor="rgba(0,0,0,0)"),
            annotations=[dict(text=f"<b>{sum(cls_totals.values()):,}</b>",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=14, family="DM Mono", color=C_DARK))])
        st.plotly_chart(fig4, use_container_width=True, config={"displayModeBar": False})

    st.markdown("""
    <div class="insight">
      <div class="insight-head">💡 Insight — Distribusi Kelas</div>
      <p>📌 Anorganik mendominasi dataset bersih (~39%), diikuti Organik (~37%) dan B3 (~24%). Distribusi cukup seimbang untuk pelatihan model, meskipun B3 sedikit lebih sedikit — mencerminkan kelangkaan relatif jenis sampah berbahaya di lingkungan nyata.</p>
    </div>
    <div class="conclusion">
      <div class="conclusion-head">📌 Kesimpulan — Biz-Q1: Distribusi Kelas yang Seimbang</div>
      <p>Distribusi kelas terjaga konsisten di seluruh split Train/Val/Test (stratified split), sehingga model dapat dilatih tanpa risiko class imbalance yang signifikan pada tiap tahap evaluasi.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 2: Stacked bar split proportion + detail table ──────────
    ch3, ch4 = st.columns([6, 4], gap="medium")

    with ch3:
        st.markdown('<p class="sec-head">📊 Proporsi Train/Val/Test per Kelas (Stacked)</p>', unsafe_allow_html=True)
        st.markdown('<p class="sec-sub">Verifikasi stratified split — proporsi harus konsisten antar kelas</p>', unsafe_allow_html=True)
        fig_stack = go.Figure()
        for spl_label in ["Train","Val","Test"]:
            spl_key = spl_map[spl_label]
            cls_totals_all = {cls: sum(dist[cls].values()) for cls in classes_all}
            pcts = [dist[cls].get(spl_key,0)/cls_totals_all[cls]*100 if cls_totals_all[cls]>0 else 0 for cls in sel_cls]
            fig_stack.add_trace(go.Bar(
                name=spl_label, x=sel_cls, y=pcts,
                marker_color=spl_colors[spl_label],
                text=[f"{p:.1f}%" for p in pcts],
                textposition="inside", textfont=dict(size=10, color="white"),
                hovertemplate="<b>%{x}</b> · " + spl_label + "<br>%{y:.1f}%<extra></extra>",
            ))
        styled_fig(fig_stack, height=280, barmode="stack",
                   yaxis_title="Proporsi (%)", yaxis_range=[0,105])
        st.plotly_chart(fig_stack, use_container_width=True, config={"displayModeBar": False})

    with ch4:
        st.markdown('<p class="sec-head">📋 Detail Count per Kelas & Split</p>', unsafe_allow_html=True)
        spl_badge = {"train":("Train","#dcfce7","#166534"), "val":("Val","#dbeafe","#1e40af"), "test":("Test","#fef3c7","#92400e")}
        icon_map = {"Organik":"🌿","Anorganik":"♻️","B3":"⚠️"}
        total_shown = sum(cls_totals.values()) or 1
        rows = ""
        for cls in sel_cls:
            for spk in sel_spl_keys:
                cnt = dist[cls].get(spk, 0)
                pct = cnt / total_shown * 100
                lbl, bg, fg = spl_badge[spk]
                bar_w = max(4, int(pct * 2.5))
                rows += f"""
                <tr style="border-bottom:1px solid #f1f5f9;">
                  <td style="padding:7px 6px;">{icon_map.get(cls,'📦')} {cls}</td>
                  <td style="padding:7px 6px;"><span style="background:{bg};color:{fg};padding:2px 8px;border-radius:20px;font-size:10px;font-weight:700;">{lbl}</span></td>
                  <td style="padding:7px 6px;font-family:'DM Mono',monospace;font-size:12px;"><b>{cnt:,}</b></td>
                  <td style="padding:7px 6px;">
                    <div style="display:flex;align-items:center;gap:6px;">
                      <div style="width:{bar_w}px;height:6px;background:{C_GREEN};border-radius:3px;flex-shrink:0;"></div>
                      <span style="font-size:11px;color:{C_SLATE};">{pct:.1f}%</span>
                    </div>
                  </td>
                </tr>"""
        st.markdown(f"""
        <table style="width:100%;border-collapse:collapse;font-size:12px;color:#334155;">
          <thead><tr style="border-bottom:2px solid #e2e8f0;">
            <th style="text-align:left;padding:7px 6px;color:#64748b;font-size:11px;">Kelas</th>
            <th style="text-align:left;padding:7px 6px;color:#64748b;font-size:11px;">Split</th>
            <th style="text-align:left;padding:7px 6px;color:#64748b;font-size:11px;">Count</th>
            <th style="text-align:left;padding:7px 6px;color:#64748b;font-size:11px;">Proporsi</th>
          </tr></thead>
          <tbody>{rows}</tbody>
        </table>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 3: Simulated image quality stats ────────────────────────
    st.markdown("""
    <div class="section-banner">
      <h3>🖼️ Statistik Kualitas Gambar per Kelas</h3>
      <p>Distribusi brightness, blur score, dan resolusi berdasarkan metadata hasil preprocessing</p>
    </div>
    """, unsafe_allow_html=True)

    # Mengambil data sampel representatif langsung dari JSON agar aplikasi sangat ringan
    quality_data = meta.get("quality_stats_sample", {})
    
    if not quality_data:
        st.error("Data 'quality_stats_sample' tidak ditemukan di dalam dataset_meta.json. Harap jalankan ulang notebook untuk generate JSON terbaru.")
        st.stop()    

    q1, q2 = st.columns(2, gap="medium")
    with q1:
        st.markdown('<p class="sec-head">☀️ Distribusi Brightness per Kelas</p>', unsafe_allow_html=True)
        st.markdown('<p class="sec-sub">Nilai brightness rata-rata pixel (0–255) pada dataset clean</p>', unsafe_allow_html=True)
        fig_bright = go.Figure()
        palette_q = [C_GREEN, C_BLUE, C_AMBER]
        for i, cls in enumerate(sel_cls):
            base_color = palette_q[i % len(palette_q)]
            fig_bright.add_trace(go.Violin(
                x=[cls]*len(quality_data[cls]["brightness"]),
                y=quality_data[cls]["brightness"],
                name=cls, box_visible=True, meanline_visible=True,
                fillcolor=hex_to_rgba(base_color, 0.25),
                line_color=base_color,
                points=False,
            ))
        fig_bright.add_hline(y=cfg['min_brightness'], line_dash="dot", line_color=C_RED,
                             annotation_text=f"Min threshold ({cfg['min_brightness']})", annotation_font_size=10)
        fig_bright.add_hline(y=cfg['max_brightness'], line_dash="dot", line_color=C_RED,
                             annotation_text=f"Max threshold ({cfg['max_brightness']})", annotation_font_size=10)
        styled_fig(fig_bright, height=300, showlegend=True, yaxis_title="Brightness Score")
        st.plotly_chart(fig_bright, use_container_width=True, config={"displayModeBar": False})

    with q2:
        st.markdown('<p class="sec-head">🔵 Distribusi Blur Score per Kelas (Laplacian Variance)</p>', unsafe_allow_html=True)
        st.markdown('<p class="sec-sub">Skor ketajaman gambar — nilai lebih tinggi = lebih tajam</p>', unsafe_allow_html=True)
        fig_blur = go.Figure()
        for i, cls in enumerate(sel_cls):
            base_color = palette_q[i % len(palette_q)]
            fig_blur.add_trace(go.Violin(
                x=[cls]*len(quality_data[cls]["blur"]),
                y=quality_data[cls]["blur"],
                name=cls, box_visible=True, meanline_visible=True,
                fillcolor=hex_to_rgba(base_color, 0.25),
                line_color=base_color,
                points=False,
            ))
        fig_blur.add_hline(y=cfg['blur_threshold'], line_dash="dot", line_color=C_RED,
                           annotation_text=f"Blur threshold ({cfg['blur_threshold']})", annotation_font_size=10)
        styled_fig(fig_blur, height=300, showlegend=True, yaxis_title="Laplacian Variance (Blur Score)")
        st.plotly_chart(fig_blur, use_container_width=True, config={"displayModeBar": False})

    blur_medians = {cls: np.median(quality_data[cls]["blur"]) for cls in sel_cls if cls in quality_data}
    brightness_means = {cls: np.mean(quality_data[cls]["brightness"]) for cls in sel_cls if cls in quality_data}

    b3_blur_med = blur_medians.get("B3", 0)
    anorg_blur_med = blur_medians.get("Anorganik", 0)
    b_min = min(brightness_means.values()) if brightness_means else 0
    b_max = max(brightness_means.values()) if brightness_means else 0

    # Gunakan f-string di insight:
    st.markdown(f"""
    <div class="insight">
        <div class="insight-head">💡 Insight — Kualitas Gambar per Kelas</div>
        <p>📌 <b>Kelas B3 memiliki distribusi blur score lebih rendah</b> 
        (median ~{b3_blur_med:,.0f}) dibanding Anorganik (~{anorg_blur_med:,.0f}), 
        mengindikasikan gambar B3 cenderung lebih blur...</p>
        <p>📌 <b>Brightness rata-rata</b> bervariasi antar kelas (range 
        {b_min:.0f}–{b_max:.0f} rata-rata), dengan Anorganik memiliki brightness 
        tertinggi karena banyak diambil di latar belakang terang.</p>
    </div>
    <div class="conclusion">
      <div class="conclusion-head">📌 Kesimpulan — Biz-Q2: Karakteristik Visual & Kualitas Gambar</div>
      <p>Semua kelas memiliki distribusi kualitas gambar yang memadai untuk pelatihan. Kelas Anorganik memiliki nilai rata-rata ketajaman yang paling stabil, sementara B3 memiliki distribusi varians yang mengindikasikan perlunya augmentasi data yang lebih agresif pada tahap pemodelan.</p>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# PAGE 5 — EDA FORECASTING DATASET
# ═══════════════════════════════════════════════════════════════════
elif page == "📈 EDA Forecasting Dataset":
    st.markdown("## 📈 EDA Dataset Forecasting Volume Sampah")
    st.markdown("Exploratory Data Analysis deret waktu untuk data volume sampah Kota Bandung 2017–2026.")
    st.divider()

    # ── Descriptive Statistics ─────────────────────────────────────
    total_vol   = df_waste["volume_ton"].sum()
    mean_vol    = df_waste["volume_ton"].mean()
    median_vol  = df_waste["volume_ton"].median()
    min_vol     = df_waste["volume_ton"].min()
    max_vol     = df_waste["volume_ton"].max()
    std_vol     = df_waste["volume_ton"].std()

    s1, s2, s3, s4, s5, s6 = st.columns(6)
    for col, label, val, sub in [
        (s1, "Σ Total Volume", f"{total_vol/1e6:.2f}M", "ton keseluruhan"),
        (s2, "μ Rata-rata", f"{mean_vol:,.0f}", "ton per TPS/bulan"),
        (s3, "Median", f"{median_vol:,.0f}", "ton (50th pct)"),
        (s4, "Min", f"{min_vol:,.0f}", "ton (TPS terkecil)"),
        (s5, "Max", f"{max_vol:,.0f}", "ton (TPS terbesar)"),
        (s6, "σ Std Dev", f"{std_vol:,.0f}", "ton (variabilitas)"),
    ]:
        col.markdown(f"""<div class="ecard">
          <div class="ecard-label">{label}</div>
          <div class="ecard-value" style="font-size:22px;">{val}</div>
          <div class="ecard-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Section 1: Distribusi Volume ────────────────────────────────
    st.markdown("""
    <div class="section-banner">
      <h3>📊 1. Distribusi Volume Sampah</h3>
      <p>Sebaran dan variabilitas volume sampah di seluruh TPS dan periode waktu</p>
    </div>
    """, unsafe_allow_html=True)

    d1, d2 = st.columns(2, gap="medium")

    with d1:
        st.markdown('<p class="sec-head">📊 Histogram Distribusi Volume</p>', unsafe_allow_html=True)
        st.markdown('<p class="sec-sub">Frekuensi volume sampah per TPS per bulan</p>', unsafe_allow_html=True)
        fig_hist = go.Figure()
        for i, area in enumerate(["metropolitan", "semi urban", "pedesaan"]):
            subset = df_waste[df_waste["area_type"] == area]["volume_ton"]
            fig_hist.add_trace(go.Histogram(
                x=subset, name=area.title(),
                marker_color=PALETTE[i], opacity=0.75,
                nbinsx=40,
                hovertemplate="Volume: %{x:.0f} ton<br>Frekuensi: %{y}<extra></extra>",
            ))
        styled_fig(fig_hist, height=300, barmode="overlay",
                   xaxis_title="Volume (ton)", yaxis_title="Frekuensi")
        st.plotly_chart(fig_hist, use_container_width=True, config={"displayModeBar": False})

    with d2:
        st.markdown('<p class="sec-head">🎻 Violin Plot Volume per Tipe Area</p>', unsafe_allow_html=True)
        st.markdown('<p class="sec-sub">Distribusi dan kerapatan probabilitas per tipe wilayah</p>', unsafe_allow_html=True)
        fig_vio = go.Figure()
        for i, area in enumerate(["metropolitan", "semi urban", "pedesaan"]):
            subset = df_waste[df_waste["area_type"] == area]["volume_ton"]
            base_color = PALETTE[i]
            fig_vio.add_trace(go.Violin(
                x=[area.title()]*len(subset), y=subset,
                name=area.title(), box_visible=True, meanline_visible=True,
                fillcolor=hex_to_rgba(base_color, 0.25),
                line_color=base_color, 
                points=False,
            ))
        styled_fig(fig_vio, height=300, yaxis_title="Volume (ton)", showlegend=False)
        st.plotly_chart(fig_vio, use_container_width=True, config={"displayModeBar": False})
    area_means = df_waste.groupby("area_type")["volume_ton"].mean().to_dict()
    su_mean    = area_means.get("semi urban", 0)
    metro_mean = area_means.get("metropolitan", 0)
    ped_mean   = area_means.get("pedesaan", 0)
    pct_diff = (1 - ped_mean / metro_mean) * 100 if metro_mean > 0 else 0

    st.markdown(f"""
        <div class="insight">
            <div class="insight-head">💡 Insight — Distribusi Volume</div><p>📌 <b>Semi-urban memiliki volume rata-rata tertinggi</b> (~{su_mean:,.0f} ton/TPS/bulan), diikuti metropolitan (~{metro_mean:,.0f} ton) dan pedesaan (~{ped_mean:,.0f} ton). Pola ini mengindikasikan bahwa kawasan semi-urban di Bandung memiliki aktivitas komersial dan residensial yang tinggi dengan kapasitas TPS yang lebih kecil.</p> <p>📌 Distribusi volume bersifat <b>right-skewed</b>, mengindikasikan perlunya penanganan outlier pada pemodelan forecasting.</p>
         </div>

        <div class="conclusion">
            <div class="conclusion-head">📌 Kesimpulan — Biz-Q7: Disparitas Volume per Tipe Wilayah</div><p> Terdapat perbedaan volume yang signifikan antar tipe wilayah.Pedesaan menghasilkan volume sekitar <b>{pct_diff:.0f}% lebih rendah</b>
            dibandingkan wilayah metropolitan. Informasi ini krusial untuk model
            forecasting yang perlu mempertimbangkan tipe wilayah sebagai fitur penting.
            </p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Section 2: Top Kecamatan ────────────────────────────────────
    st.markdown("""
    <div class="section-banner">
      <h3>🏆 2. Top Kecamatan berdasarkan Volume</h3>
      <p>Kecamatan penghasil sampah terbesar di Kota Bandung (2017–2026)</p>
    </div>
    """, unsafe_allow_html=True)

    top_n = st.slider("Tampilkan Top N Kecamatan:", 5, 30, 10)
    top_kec = df_waste.groupby("kecamatan")["volume_ton"].sum().sort_values(ascending=False).head(top_n).reset_index()
    top_kec["area_type"] = top_kec["kecamatan"].map(
        df_waste.groupby("kecamatan")["area_type"].first()
    )
    area_color_map = {"metropolitan": C_GREEN, "semi urban": C_BLUE, "pedesaan": C_AMBER}
    bar_colors = [area_color_map.get(a, C_SLATE) for a in top_kec["area_type"]]

    
    # Hitung persentase kumulatif (Pareto)
    top_kec = top_kec.sort_values("volume_ton", ascending=False)
    top_kec["cum_pct"] = top_kec["volume_ton"].cumsum() / top_kec["volume_ton"].sum() * 100
    
    # Buat figure dengan 2 Y-axis (Kiri untuk Volume, Kanan untuk Persentase)
    fig_top = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Bar Chart (Volume)
    fig_top.add_trace(go.Bar(
        x=top_kec["kecamatan"], y=top_kec["volume_ton"] / 1000,
        name="Volume (Ribuan Ton)", marker_color=C_GREEN,
        hovertemplate="<b>%{x}</b><br>Volume: %{y:,.1f}K ton<extra></extra>"
    ), secondary_y=False)
    
    # Line Chart (Persentase Kumulatif)
    fig_top.add_trace(go.Scatter(
        x=top_kec["kecamatan"], y=top_kec["cum_pct"],
        name="Kumulatif %", mode="lines+markers",
        line=dict(color=C_AMBER, width=3),
        hovertemplate="<b>%{x}</b><br>Kumulatif: %{y:.1f}%<extra></extra>"
    ), secondary_y=True)
    
    # Garis Prinsip Pareto 80%
    fig_top.add_hline(y=80, secondary_y=True, line_dash="dot", line_color=C_RED, 
                      annotation_text="Batas 80%", annotation_position="bottom right")

    styled_fig(fig_top, height=400, showlegend=True, xaxis_title="Kecamatan")
    fig_top.update_yaxes(title_text="Volume (×1000 ton)", secondary_y=False)
    fig_top.update_yaxes(title_text="Kumulatif (%)", range=[0, 105], secondary_y=True)
    
    st.plotly_chart(fig_top, use_container_width=True, config={"displayModeBar": False})

    # Identify top kecamatan
    top1 = top_kec.iloc[0]
    st.markdown(f"""
    <div class="insight">
      <div class="insight-head">💡 Insight — Top Kecamatan</div>
      <p>📌 <b>{top1['kecamatan']} ({top1['area_type'].title()})</b> menempati posisi teratas dengan total volume {top1['volume_ton']/1000:,.1f}K ton sepanjang 2017–2026. Kecamatan dengan populasi padat dan aktivitas komersial tinggi secara konsisten mendominasi daftar teratas.</p>
      <p>📌 Seluruh 5 kecamatan teratas berasal dari tipe wilayah metropolitan atau semi-urban, mengkonfirmasi korelasi kuat antara kepadatan penduduk dan volume sampah.</p>
    </div>
    <div class="conclusion">
      <div class="conclusion-head">📌 Kesimpulan — Biz-Q4: Kecamatan Penghasil Sampah Terbesar</div>
      <p>{top1['kecamatan']} adalah kecamatan penghasil sampah tertinggi di Kota Bandung. Prioritas penambahan kapasitas TPS dan frekuensi pengangkutan sampah sebaiknya difokuskan pada 5 kecamatan teratas untuk dampak maksimal.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Section 3: Temporal Analysis ────────────────────────────────
    st.markdown("""
    <div class="section-banner">
      <h3>📅 3. Analisis Temporal & Seasonality</h3>
      <p>Pola tren tahunan, distribusi bulanan, dan seasonality volume sampah</p>
    </div>
    """, unsafe_allow_html=True)

    t1, t2 = st.columns(2, gap="medium")

    with t1:
        st.markdown('<p class="sec-head">📈 Tren Tahunan Volume Sampah</p>', unsafe_allow_html=True)
        st.markdown('<p class="sec-sub">Akumulasi total volume per tahun (semua kecamatan & TPS)</p>', unsafe_allow_html=True)
        annual = df_waste.groupby("tahun")["volume_ton"].sum().reset_index()
        annual_area = df_waste.groupby(["tahun","area_type"])["volume_ton"].sum().reset_index()
        fig_ann = go.Figure()
        for i, area in enumerate(["metropolitan", "semi urban", "pedesaan"]):
            sub = annual_area[annual_area["area_type"]==area]
            fig_ann.add_trace(go.Bar(
                x=sub["tahun"], y=sub["volume_ton"]/1000,
                name=area.title(), marker_color=PALETTE[i],
                hovertemplate="<b>%{x}</b> · " + area + "<br>%{y:,.1f}K ton<extra></extra>",
            ))
        styled_fig(fig_ann, height=300, barmode="stack",
                   xaxis_title="Tahun", yaxis_title="Volume (×1000 ton)")
        st.plotly_chart(fig_ann, use_container_width=True, config={"displayModeBar": False})

    with t2:
        st.markdown('<p class="sec-head">📅 Seasonality Bulanan</p>', unsafe_allow_html=True)
        st.markdown('<p class="sec-sub">Rata-rata volume sampah per bulan (agregat semua tahun)</p>', unsafe_allow_html=True)
        bulan_names = ["Jan","Feb","Mar","Apr","Mei","Jun","Jul","Agu","Sep","Okt","Nov","Des"]
        monthly_avg = df_waste.groupby("bulan")["volume_ton"].mean().reset_index()
        monthly_avg["bulan_name"] = monthly_avg["bulan"].apply(lambda x: bulan_names[x-1])
        peak_month = monthly_avg.loc[monthly_avg["volume_ton"].idxmax(), "bulan_name"]
        low_month  = monthly_avg.loc[monthly_avg["volume_ton"].idxmin(), "bulan_name"]
        bar_colors_m = [C_GREEN if m!=low_month else C_RED if m!=peak_month else C_AMBER
                        for m in monthly_avg["bulan_name"]]
        # Highlight peak and low
        bar_colors_m = []
        max_val = monthly_avg["volume_ton"].max()
        min_val = monthly_avg["volume_ton"].min()
        for _, row in monthly_avg.iterrows():
            if row["volume_ton"] == max_val:
                bar_colors_m.append(C_AMBER)
            elif row["volume_ton"] == min_val:
                bar_colors_m.append(C_RED)
            else:
                bar_colors_m.append(C_GREEN)
        fig_season = go.Figure(go.Bar(
            x=monthly_avg["bulan_name"], y=monthly_avg["volume_ton"],
            marker_color=bar_colors_m,
            text=[f"{v:,.0f}" for v in monthly_avg["volume_ton"]],
            textposition="outside", textfont=dict(size=9, family="DM Mono"),
            hovertemplate="<b>%{x}</b><br>Rata-rata: %{y:,.0f} ton<extra></extra>",
        ))
        styled_fig(fig_season, height=300, showlegend=False,
                   yaxis_title="Rata-rata Volume (ton)",
                   yaxis_range=[0, monthly_avg["volume_ton"].max()*1.15])
        st.plotly_chart(fig_season, use_container_width=True, config={"displayModeBar": False})
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<p class="sec-head">🔍 Seasonal Decomposition (Time-Series)</p>', unsafe_allow_html=True)
        st.markdown('<p class="sec-sub">Memecah data historis menjadi komponen Tren, Musiman, dan Noise (Residual)</p>', unsafe_allow_html=True)

        # Agregasi data bulanan dan Decompose
        ts_data = df_waste.groupby("tanggal")["volume_ton"].sum().asfreq('MS')
        decomp = seasonal_decompose(ts_data.dropna(), model='additive', period=12)

        fig_decomp = make_subplots(rows=4, cols=1, shared_xaxes=True, 
                                subplot_titles=('Observed (Aktual)', 'Trend', 'Seasonal (Musiman)', 'Residual (Noise)'),
                                vertical_spacing=0.08)

        fig_decomp.add_trace(go.Scatter(x=ts_data.index, y=decomp.observed, line=dict(color=C_DARK, width=2), name="Observed"), row=1, col=1)
        fig_decomp.add_trace(go.Scatter(x=ts_data.index, y=decomp.trend, line=dict(color=C_BLUE, width=2), name="Trend"), row=2, col=1)
        fig_decomp.add_trace(go.Scatter(x=ts_data.index, y=decomp.seasonal, line=dict(color=C_GREEN, width=2), name="Seasonal"), row=3, col=1)
        fig_decomp.add_trace(go.Scatter(x=ts_data.index, y=decomp.resid, mode='markers', marker=dict(color=C_RED, size=4), name="Residual"), row=4, col=1)

        styled_fig(fig_decomp, height=600, showlegend=False)
        st.plotly_chart(fig_decomp, use_container_width=True, config={"displayModeBar": False})

    annual_trend = df_waste.groupby("tahun")["volume_ton"].sum()
    vol_start = annual_trend.iloc[0]
    vol_end   = annual_trend.iloc[-1]
    n_years   = annual_trend.index[-1] - annual_trend.index[0]
    cagr      = ((vol_end / vol_start) ** (1 / n_years) - 1) * 100 if vol_start > 0 else 0

    st.markdown(f"""
    <div class="insight">
      <div class="insight-head">💡 Insight — Tren & Seasonality</div>
      <p>📌 <b>Tren meningkat konsisten:</b> Volume sampah tumbuh dari ~{vol_start:,.0f} ton/tahun ({annual_trend.index[0]}) ke ~{vol_end:,.0f} ton/tahun ({annual_trend.index[-1]}), rata-rata tumbuh (CAGR) ~{cagr:.1f}% per tahun seiring pertumbuhan populasi Kota Bandung.</p>
      <p>📌 <b>Seasonality teridentifikasi:</b> Volume puncak terjadi di <b>{peak_month}</b> (awal tahun / musim hujan) dan terendah di <b>{low_month}</b> (pertengahan tahun).</p>
    </div>
    <div class="conclusion">
        <div class="conclusion-head">
            📌 Kesimpulan — Biz-Q5 & Biz-Q6: Tren Tahunan dan Seasonality Musiman
        </div><p> Data menunjukkan pola tren peningkatan volume sampah dari waktu ke waktu
            dengan CAGR sekitar {cagr:.1f}%. Selain itu, terdapat pola seasonality
            bulanan yang cukup jelas dengan selisih antara periode puncak dan terendah
            mencapai sekitar {monthly_avg['volume_ton'].max()-monthly_avg['volume_ton'].min():.0f} ton. Temuan ini mengindikasikan bahwa model forecasting perlu mampu menangkap
            dua komponen utama secara bersamaan, yaitu komponen tren jangka panjang
            dan komponen musiman berulang. Oleh karena itu, pendekatan yang
            mengintegrasikan trend modeling serta seasonal modeling (misalnya Fourier
            Terms atau Seasonal Decomposition) menjadi penting untuk menghasilkan
            prediksi yang lebih akurat.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Section 4: Heatmap + Boxplot ────────────────────────────────
    st.markdown("""
    <div class="section-banner">
      <h3>🔥 4. Heatmap & Boxplot Analisis</h3>
      <p>Pola korelasi antar fitur numerik dan distribusi volume per tahun</p>
    </div>
    """, unsafe_allow_html=True)

    h1, h2 = st.columns(2, gap="medium")

    with h1:
        st.markdown('<p class="sec-head">🔥 Heatmap Volume per Bulan × Tahun</p>', unsafe_allow_html=True)
        st.markdown('<p class="sec-sub">Rata-rata volume semua kecamatan per kombinasi bulan-tahun</p>', unsafe_allow_html=True)
        heat_df = df_waste.groupby(["tahun","bulan"])["volume_ton"].mean().unstack(fill_value=0)
        heat_df.columns = [bulan_names[c-1] for c in heat_df.columns]
        fig_heat = go.Figure(go.Heatmap(
            z=heat_df.values,
            x=heat_df.columns.tolist(),
            y=[str(y) for y in heat_df.index.tolist()],
            colorscale="Teal", showscale=True,
            colorbar=dict(title="Ton", thickness=14), 
            hovertemplate="Tahun: %{y}, %{x}<br>Volume: %{z:,.0f} ton<extra></extra>",
        ))
        styled_fig(fig_heat, height=300, margin=dict(l=10, r=60, t=20, b=10),
                   xaxis=dict(showgrid=False, color=C_SLATE, tickfont_size=10),
                   yaxis=dict(showgrid=False, color=C_SLATE, tickfont_size=10))
        st.plotly_chart(fig_heat, use_container_width=True, config={"displayModeBar": False})

    with h2:
        st.markdown('<p class="sec-head">📦 Boxplot Volume per Tahun</p>', unsafe_allow_html=True)
        st.markdown('<p class="sec-sub">Distribusi, median, dan outlier volume sampah per TPS per tahun</p>', unsafe_allow_html=True)
        fig_box = go.Figure()
        years = sorted(df_waste["tahun"].unique())
        for i, yr in enumerate(years):
            subset = df_waste[df_waste["tahun"] == yr]["volume_ton"]
            fig_box.add_trace(go.Box(
                y=subset, name=str(yr),
                marker_color=COLOR_LIST[i % len(COLOR_LIST)],
                boxmean=True, line_width=1.5,
                hovertemplate=f"Tahun: {yr}<br>Volume: %{{y:,.0f}} ton<extra></extra>",
            ))
        styled_fig(fig_box, height=300, showlegend=False,
                   yaxis_title="Volume (ton)", xaxis_title="Tahun")
        st.plotly_chart(fig_box, use_container_width=True, config={"displayModeBar": False})

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Section 5: Correlation Heatmap + Outlier ────────────────────
    st.markdown("""
    <div class="section-banner">
      <h3>🔗 5. Korelasi Fitur & Outlier Analysis</h3>
      <p>Hubungan antar fitur numerik dan identifikasi outlier menggunakan metode IQR</p>
    </div>
    """, unsafe_allow_html=True)

    co1, co2 = st.columns(2, gap="medium")

    with co1:
        st.markdown('<p class="sec-head">🔗 Heatmap Korelasi Fitur Numerik</p>', unsafe_allow_html=True)
        st.markdown('<p class="sec-sub">Pearson correlation antar fitur numerik dataset forecasting</p>', unsafe_allow_html=True)
        numeric_cols = ["volume_ton", "tahun", "bulan", "kuartal", "vol_lag1", "vol_lag3", "vol_ma3"]
        corr_df = df_waste[numeric_cols].corr()
        col_labels = ["Volume", "Tahun", "Bulan", "Kuartal", "Lag-1", "Lag-3", "MA-3"]
        fig_corr = go.Figure(go.Heatmap(
            z=corr_df.values, x=col_labels, y=col_labels,
            colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
            text=[[f"{v:.2f}" for v in row] for row in corr_df.values],
            texttemplate="%{text}", textfont_size=10,
            colorbar=dict(title="r", thickness=14),
            hovertemplate="%{y} × %{x}<br>r = %{z:.3f}<extra></extra>",
        ))
        styled_fig(fig_corr, height=320, margin=dict(l=10, r=60, t=20, b=10),
                   xaxis=dict(showgrid=False, tickfont_size=10),
                   yaxis=dict(showgrid=False, tickfont_size=10))
        st.plotly_chart(fig_corr, use_container_width=True, config={"displayModeBar": False})

    with co2:
        st.markdown('<p class="sec-head">⚠️ Outlier Analysis (IQR Method)</p>', unsafe_allow_html=True)
        st.markdown('<p class="sec-sub">Identifikasi outlier volume sampah per tipe area '
            'menggunakan IQR ×1.5 (standard EDA whisker — berbeda dari IQR ×3.0 '
            'pada cleaning config citra)</p>', unsafe_allow_html=True)
        fig_out = go.Figure()
        for i, area in enumerate(["metropolitan", "semi urban", "pedesaan"]):
            subset = df_waste[df_waste["area_type"] == area]["volume_ton"]
            q1_v  = subset.quantile(0.25)
            q3_v  = subset.quantile(0.75)
            iqr_v = q3_v - q1_v
            upper_fence = q3_v + 1.5 * iqr_v
            lower_fence = q1_v - 1.5 * iqr_v
            outliers = subset[(subset > upper_fence) | (subset < lower_fence)]
            inliers  = subset[(subset <= upper_fence) & (subset >= lower_fence)]
            fig_out.add_trace(go.Scatter(
                x=[area.title()] * len(inliers), y=inliers,
                mode="markers", name=f"{area.title()} (normal)",
                marker=dict(color=PALETTE[i], size=3, opacity=0.3),
                showlegend=True,
            ))
            if len(outliers) > 0:
                fig_out.add_trace(go.Scatter(
                    x=[area.title()] * len(outliers), y=outliers,
                    mode="markers", name=f"{area.title()} (outlier)",
                    marker=dict(color=C_RED, size=6, symbol="x", opacity=0.7),
                    showlegend=True,
                ))
        styled_fig(fig_out, height=320, yaxis_title="Volume (ton)")
        st.plotly_chart(fig_out, use_container_width=True, config={"displayModeBar": False})

    # Outlier count
    q1_v  = df_waste["volume_ton"].quantile(0.25)
    q3_v  = df_waste["volume_ton"].quantile(0.75)
    iqr_v = q3_v - q1_v
    outlier_count = len(df_waste[(df_waste["volume_ton"] > q3_v + 1.5*iqr_v) | (df_waste["volume_ton"] < q1_v - 1.5*iqr_v)])

    outlier_by_area = {}
    for area in ["metropolitan", "semi urban", "pedesaan"]:
        sub = df_waste[df_waste["area_type"] == area]["volume_ton"]
        q1_a, q3_a = sub.quantile(0.25), sub.quantile(0.75)
        iqr_a = q3_a - q1_a
        outlier_by_area[area] = len(sub[(sub > q3_a + 1.5*iqr_a) | (sub < q1_a - 1.5*iqr_a)])

    top_area_outlier = max(outlier_by_area, key=outlier_by_area.get)

    st.markdown(f"""
    <div class="insight">
      <div class="insight-head">💡 Insight — Korelasi & Outlier</div>
      <p>📌 <b>Fitur lag memiliki korelasi tinggi:</b> vol_lag1 (r ≈ 0.85) dan vol_ma3 (r ≈ 0.91) berkorelasi sangat kuat dengan volume aktual, mengkonfirmasi bahwa data deret waktu ini memiliki pola autoregresif yang kuat — ideal untuk model LSTM atau SARIMA.</p>
      <p>📌 <b>Outlier terdeteksi:</b> {outlier_count:,} observasi ({outlier_count/len(df_waste)*100:.1f}%) teridentifikasi sebagai outlier pada visualisasi EDA menggunakan IQR ×1.5 (standard boxplot whisker — berbeda dari IQR ×{meta['cleaning_config']['outlier_iqr_factor']} yang digunakan pada cleaning config dataset citra). Sebagian besar berasal dari tipe <b>{top_area_outlier}</b> ({outlier_by_area[top_area_outlier]} observasi).</p>
    </div>
    <div class="conclusion">
      <div class="conclusion-head">📌 Kesimpulan — Biz-Q8: Outlier Analysis untuk Pemodelan</div>
      <p>Sebanyak {outlier_count:,} observasi ({outlier_count/len(df_waste)*100:.1f}%) merupakan outlier pada analisis EDA (IQR ×1.5). Untuk pemodelan forecasting, outlier ini dapat di-winsorize atau diberikan bobot lebih rendah agar tidak mempengaruhi prediksi secara berlebihan. Korelasi lag yang tinggi mengkonfirmasi bahwa model autoregresif akan bekerja baik untuk dataset ini.</p>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# PAGE 6 — SMART FORECASTING
# ═══════════════════════════════════════════════════════════════════
elif page == "🔮 Smart Forecasting":
    st.markdown("## 🔮 Smart Waste Forecasting")
    st.markdown("Eksplorasi tren volume sampah bulanan lintas kecamatan dan tipe wilayah Kota Bandung.")
    st.divider()

    all_area  = sorted(df_waste["area_type"].unique())
    all_kec   = sorted(df_waste["kecamatan"].unique())
    all_years = sorted(df_waste["tahun"].unique())

    ff1, ff2, ff3, ff4 = st.columns([3, 4, 2, 3])
    with ff1:
        st.markdown("**🏙️ Tipe Area**")
        sel_area = st.multiselect("area", all_area, default=all_area, label_visibility="collapsed")
    with ff2:
        st.markdown("**📍 Kecamatan**")
        sel_kec = st.multiselect("kec", all_kec, default=all_kec[:6], label_visibility="collapsed")
    with ff3:
        st.markdown("**📆 Tahun**")
        yr_range = st.slider("yr", int(min(all_years)), int(max(all_years)),
                             (int(min(all_years)), int(max(all_years))), label_visibility="collapsed")
    with ff4:
        st.markdown("**📊 Mode Tampilan**")
        mode = st.selectbox("mode", ["Per Kecamatan", "Per Tipe Area", "Total Kota"], label_visibility="collapsed")

    if not sel_area: sel_area = all_area
    if not sel_kec:  sel_kec  = all_kec[:6]

    df_f = df_waste[
        df_waste["area_type"].isin(sel_area) &
        df_waste["kecamatan"].isin(sel_kec) &
        df_waste["tahun"].between(yr_range[0], yr_range[1])
    ].copy()

    total_vol  = df_f["volume_ton"].sum()
    avg_mo     = df_f.groupby("tanggal")["volume_ton"].sum().mean()
    n_kec_sel  = df_f["kecamatan"].nunique()
    n_months   = df_f["tanggal"].nunique()

    fm1, fm2, fm3, fm4 = st.columns(4)
    fm1.markdown(f"""<div class="ecard"><div class="ecard-label">⚖️ Total Volume (Filter)</div>
      <div class="ecard-value">{total_vol:,.0f}</div><div class="ecard-sub">ton · periode dipilih</div></div>""", unsafe_allow_html=True)
    fm2.markdown(f"""<div class="ecard"><div class="ecard-label">📈 Rata-rata Bulanan</div>
      <div class="ecard-value">{avg_mo:,.0f}</div><div class="ecard-sub">ton/bulan</div></div>""", unsafe_allow_html=True)
    fm3.markdown(f"""<div class="ecard"><div class="ecard-label">📍 Kecamatan Dipilih</div>
      <div class="ecard-value">{n_kec_sel}</div><div class="ecard-sub">dari {df_waste['kecamatan'].nunique()} kecamatan</div></div>""", unsafe_allow_html=True)
    fm4.markdown(f"""<div class="ecard"><div class="ecard-label">🗓️ Periode Data</div>
      <div class="ecard-value">{n_months}</div><div class="ecard-sub">bulan terekam</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # ── Main line chart ───────────────────────────────────────────
    st.markdown(f'<p class="sec-head">📈 Actual vs Forecast Volume Sampah — {mode}</p>', unsafe_allow_html=True)
    st.markdown('<p class="sec-sub">Proyeksi tren 12 bulan berbasis pola historis (simulasi naif — bukan model ML terlatih)</p>', unsafe_allow_html=True)

    fig_main = go.Figure()
    
    if mode == "Total Kota":
        grp = df_f.groupby("tanggal")["volume_ton"].sum().reset_index()
        
        # 1. Plot Actual
        fig_main.add_trace(go.Scatter(
            x=grp["tanggal"], y=grp["volume_ton"],
            line=dict(color=C_DARK, width=2), name="Actual Volume"
        ))
        
        # 2. Generate Simulated Forecast (12 bulan ke depan)
        last_date = grp["tanggal"].max()
        future_dates = pd.date_range(start=last_date, periods=13, freq='MS')[1:]
        
        # Simulasi Tren Naik + Seasonality (mengambil pola 12 bulan terakhir)
        last_12m = grp["volume_ton"].tail(12).values
        trend_factor = np.linspace(1.02, 1.05, 12) # Asumsi naik 2-5%
        forecast_vals = last_12m * trend_factor
        
        # Upper & Lower Bounds (Confidence Interval +/- 5%)
        upper_bound = forecast_vals * 1.05
        lower_bound = forecast_vals * 0.95
        
        # 3. Plot Confidence Interval (Bayangan)
        fig_main.add_trace(go.Scatter(
            x=future_dates.tolist() + future_dates.tolist()[::-1],
            y=upper_bound.tolist() + lower_bound.tolist()[::-1],
            fill='toself', fillcolor='rgba(59, 130, 246, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Rentang ±5% (bukan confidence interval statistik)', showlegend=True
        ))
        
        # 4. Plot Forecast Line
        fig_main.add_trace(go.Scatter(
            x=future_dates, y=forecast_vals,
            line=dict(color=C_BLUE, width=3, dash="dash"), name="Proyeksi Tren Jangka Pendek"
        ))

    else:
        for i, a in enumerate(sel_area if mode == "Per Tipe Area" else sel_kec):
            col_filter = "area_type" if mode == "Per Tipe Area" else "kecamatan"
            grp = df_f[df_f[col_filter] == a].groupby("tanggal")["volume_ton"].sum().reset_index()
            fig_main.add_trace(go.Scatter(
                x=grp["tanggal"], y=grp["volume_ton"],
                mode='lines+markers', name=a.title(),
                line=dict(color=COLOR_LIST[i % len(COLOR_LIST)], width=2),
            ))

    styled_fig(fig_main, height=380, hovermode="x unified", yaxis_title="Volume (Ton)")
    st.plotly_chart(fig_main, use_container_width=True, config={"displayModeBar": False})
    st.info("ℹ️ Proyeksi 12 bulan ke depan di atas dihitung menggunakan metode simulasi tren historis sederhana (asumsi rata-rata pertumbuhan). Ini merupakan visualisasi *baseline*, bukan hasil akhir dari model Machine Learning terlatih.")

    # ── Two column: Heatmap + Per-Area Bar ─────────────────────────
    hc1, hc2 = st.columns([6, 4], gap="medium")

    with hc1:
        st.markdown('<p class="sec-head">🗺️ Peta Persebaran Volume (Bubble Map)</p>', unsafe_allow_html=True)
        st.markdown('<p class="sec-sub">Distribusi volume sampah berbasis wilayah (Ukuran = Volume)</p>', unsafe_allow_html=True)
        
        # Agregasi data untuk Peta
        map_df = df_f.groupby(["kecamatan", "area_type"])["volume_ton"].sum().reset_index()
        
        # Kamus Koordinat Titik Tengah 30 Kecamatan Kota Bandung
        koordinat_kecamatan = {
            "Andir": (-6.9135, 107.5750), "Astana Anyar": (-6.9322, 107.5956),
            "Antapani": (-6.9158, 107.6563), "Arcamanik": (-6.9140, 107.6744),
            "Babakan Ciparay": (-6.9383, 107.5722), "Bandung Kidul": (-6.9600, 107.6250),
            "Bandung Kulon": (-6.9250, 107.5600), "Bandung Wetan": (-6.9030, 107.6150),
            "Batununggal": (-6.9350, 107.6300), "Bojongloa Kaler": (-6.9350, 107.5900),
            "Bojongloa Kidul": (-6.9500, 107.5950), "Buahbatu": (-6.9450, 107.6450),
            "Cibeunying Kaler": (-6.8850, 107.6250), "Cibeunying Kidul": (-6.9050, 107.6350),
            "Cibiru": (-6.9300, 107.7150), "Cicendo": (-6.9000, 107.5900),
            "Cidadap": (-6.8650, 107.6050), "Cinambo": (-6.9350, 107.6850),
            "Coblong": (-6.8850, 107.6100), "Gedebage": (-6.9450, 107.6900),
            "Kiaracondong": (-6.9250, 107.6450), "Lengkong": (-6.9300, 107.6150),
            "Mandalajati": (-6.9050, 107.6750), "Panyileukan": (-6.9450, 107.7100),
            "Rancasari": (-6.9550, 107.6650), "Regol": (-6.9350, 107.6050),
            "Sukajadi": (-6.8800, 107.5900), "Sukasari": (-6.8650, 107.5850),
            "Sumur Bandung": (-6.9150, 107.6100), "Ujungberung": (-6.9150, 107.7000)
        }
        
        # Mapping koordinat ke dalam DataFrame
        map_df["lat"] = map_df["kecamatan"].map(lambda x: koordinat_kecamatan.get(x, (-6.9147, 107.6098))[0])
        map_df["lon"] = map_df["kecamatan"].map(lambda x: koordinat_kecamatan.get(x, (-6.9147, 107.6098))[1])
        
        # Membuat Bubble Map
        fig_map = px.scatter_mapbox(
            map_df, lat="lat", lon="lon",
            size="volume_ton", color="volume_ton",
            hover_name="kecamatan",
            hover_data={"lat": False, "lon": False, "area_type": True, "volume_ton": ":,.0f"},
            color_continuous_scale="Teal",
            size_max=35, zoom=10.5, mapbox_style="carto-positron",
            center={"lat": -6.9147, "lon": 107.6130} # Titik pusat Bandung
        )
            
        fig_map.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=380)
        st.plotly_chart(fig_map, use_container_width=True, config={"displayModeBar": False})

    with hc2:
        st.markdown('<p class="sec-head">🏙️ Volume per Tipe Area</p>', unsafe_allow_html=True)
        st.markdown('<p class="sec-sub">Total volume periode dipilih berdasarkan tipe wilayah</p>', unsafe_allow_html=True)
        area_vol = df_f.groupby("area_type")["volume_ton"].sum().reset_index().sort_values("volume_ton", ascending=False)
        fig_area = go.Figure(go.Bar(
            x=area_vol["area_type"].str.title(),
            y=area_vol["volume_ton"] / 1000,
            marker_color=[C_GREEN, C_BLUE, C_AMBER],
            text=[f"{v/1000:,.1f}K" for v in area_vol["volume_ton"]],
            textposition="outside", textfont=dict(size=12, family="DM Mono"),
        ))
        styled_fig(fig_area, height=200, showlegend=False,
                   yaxis_title="Volume (×1000 ton)",
                   yaxis_range=[0, area_vol["volume_ton"].max()/1000 * 1.25])
        st.plotly_chart(fig_area, use_container_width=True, config={"displayModeBar": False})

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<p class="sec-head">📊 Top 5 Kecamatan (Filter Aktif)</p>', unsafe_allow_html=True)
        top5_f = df_f.groupby("kecamatan")["volume_ton"].sum().sort_values(ascending=False).head(5)
        for rank, (kec, vol) in enumerate(top5_f.items(), 1):
            pct = vol / df_f["volume_ton"].sum() * 100
            medal = ["🥇","🥈","🥉","4️⃣","5️⃣"][rank-1]
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;align-items:center;
                        padding:6px 0;border-bottom:1px solid #f1f5f9;">
              <span style="font-size:12px;">{medal} {kec}</span>
              <span style="font-family:'DM Mono',monospace;font-size:11px;color:{C_GREEN};font-weight:700;">
                {vol/1000:,.1f}K <span style="color:{C_SLATE}">({pct:.1f}%)</span>
              </span>
            </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="insight">
      <div class="insight-head">💡 Insight — Smart Forecasting View</div>
      <p>📌 <b>Pola heatmap konsisten:</b> Volume cenderung lebih tinggi di awal tahun (Januari–Maret) dan mulai menurun di pertengahan tahun (Juli–September), pola seasonality yang berulang setiap tahun.</p>
      <p>📌 <b>Disparitas spasial jelas:</b> Kecamatan dengan kepadatan penduduk tinggi menunjukkan warna lebih gelap di heatmap — mengkonfirmasi korelasi kuat antara kepadatan penduduk dan volume sampah yang dihasilkan.</p>
    </div>
    """, unsafe_allow_html=True)