"""
SPECTRA-Sum: Interactive Research Dashboard
Hardware-Aware Hierarchical Spectral Summarization & 3D Manifold Visualizer.
"""

import os
import sys
import time
import numpy as np

# Ensure root directory is on sys.path
root_dir = os.path.abspath(os.path.dirname(__file__))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import nltk
# Add workspace nltk_data if available
ws_nltk = os.path.abspath(os.path.join(root_dir, '..', '..', '.venv', 'nltk_data'))
if os.path.exists(ws_nltk) and ws_nltk not in nltk.data.path:
    nltk.data.path.insert(0, ws_nltk)

import streamlit as st
import plotly.graph_objects as go
from hierarchical_summarize import SpectraSumPipeline

# Set Page Config
st.set_page_config(
    page_title="SPECTRA-Sum | Hierarchical Spectral Summarizer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title { font-size: 2.2rem; font-weight: 800; color: #1E293B; margin-bottom: 0.2rem; }
    .subtitle { font-size: 1.05rem; color: #64748B; margin-bottom: 1.5rem; }
    .metric-card { background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 16px; margin-bottom: 12px; }
    .metric-value { font-size: 1.5rem; font-weight: 700; color: #0F172A; }
    .metric-label { font-size: 0.85rem; color: #64748B; text-transform: uppercase; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# App Header
st.markdown('<div class="main-title">⚡ SPECTRA-Sum</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Hardware-Aware Hierarchical Spectral Summarization & Submodular Graph Optimization</div>', unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.header("🛠️ Engine Configuration")

model_choice = st.sidebar.selectbox(
    "Transformer Backend",
    options=["distill-bart", "bart", "pegasus", "roberta"],
    index=0,
    help="Select the sequence-to-sequence model architecture."
)

precision_mode = st.sidebar.selectbox(
    "Hardware Precision",
    options=["fp32", "fp16", "int8"],
    index=0,
    help="INT8 uses dynamic QNNPACK quantization to minimize memory bandwidth on edge/CPU hardware."
)

st.sidebar.markdown("---")
st.sidebar.header("📐 Mathematical Hyperparameters")

submodular_lambda = st.sidebar.slider(
    "Submodular Redundancy Penalty (λ)",
    min_value=0.0,
    max_value=1.0,
    value=0.35,
    step=0.05,
    help="Penalizes semantic overlap between selected sentences in each cluster."
)

graph_threshold = st.sidebar.slider(
    "Affinity Edge Cutoff (τ)",
    min_value=0.05,
    max_value=0.5,
    value=0.15,
    step=0.05,
    help="Minimum cosine similarity required to establish a graph edge."
)

max_length = st.sidebar.slider("Max Summary Length", 50, 400, 200, 25)
min_length = st.sidebar.slider("Min Summary Length", 20, 150, 50, 10)

# Sample Document Preset
DEFAULT_ARTICLE = (
    "Natural language processing has undergone fundamental transformations with the rise of deep transformer architectures. "
    "Pretrained sequence-to-sequence models such as BART and Pegasus excel at capturing nuanced discourse patterns and generating "
    "concise abstractive summaries. However, deploying these models on extensive technical manuals, legal corpora, and multi-document "
    "archives introduces significant inference-time bottlenecks. Fixed positional embeddings enforce strict context window constraints, "
    "typically limiting processing to 1024 tokens. Standard industry workarounds frequently resort to naive truncation, inadvertently "
    "omitting critical domain context.\n\n"
    "To resolve this limitation without expensive fine-tuning or retraining, spectral graph theory offers a mathematically sound framework. "
    "By modeling sentences as vertices in a dense semantic manifold, the affinity matrix captures both semantic proximity and narrative flow. "
    "Computing the Normalized Symmetric Graph Laplacian enables spectral partitioning into harmonic eigen-subspaces. "
    "Furthermore, submodular optimization via facility location guarantees a bounded (1 - 1/e) approximation for maximizing topic coverage "
    "while penalizing redundant formulations. This creates a multi-scale hierarchy where sentence clusters are summarized locally before "
    "synthesizing an executive document abstract.\n\n"
    "From an embedded and systems engineering perspective, execution latency and memory bandwidth dominate inference operational costs. "
    "Dynamic INT8 quantization converts transformer linear projection weights into 8-bit integer operations, reducing peak RAM demand "
    "and doubling arithmetic throughput on resource-constrained compute nodes."
)

# Main Input Section
article_input = st.text_area(
    "Document Input (or enter custom text below):",
    value=DEFAULT_ARTICLE,
    height=240,
    placeholder="Paste long document or multi-document text here..."
)

col_run, col_reset = st.columns([1, 6])
execute_pipeline = col_run.button("🚀 Run SPECTRA-Sum", type="primary")

if execute_pipeline and article_input.strip():
    with st.spinner("Executing Hierarchical Spectral Pipeline & Telemetry..."):
        try:
            pipeline = SpectraSumPipeline(
                model_name=model_choice,
                precision=precision_mode,
                submodular_redundancy_weight=submodular_lambda,
                similarity_threshold=graph_threshold,
            )
            results = pipeline.process_document(
                article_input,
                max_summary_length=max_length,
                min_summary_length=min_length,
            )
            st.session_state["spectra_results"] = results
            st.success("Hierarchical synthesis complete!")
        except Exception as e:
            st.error(f"Execution Error: {e}")

if "spectra_results" in st.session_state:
    res = st.session_state["spectra_results"]
    hw = res["hardware_metrics"]
    ev = res["evaluation_metrics"]
    graph = res["spectral_graph_data"]

    # Top Telemetry Row
    st.markdown("### 📊 Systems & Faithfulness Telemetry")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    col1.metric("Latency", f"{hw.get('wall_time_sec', 0):.2f} s")
    col2.metric("Throughput", f"{hw.get('throughput_tokens_per_sec', 0):.1f} tok/s")
    col3.metric("Peak RAM", f"{hw.get('peak_ram_mb', 0):.1f} MB")
    col4.metric("Faithfulness", f"{ev.get('faithfulness_score', 0) * 100:.1f} %")
    col5.metric("Compression", f"{ev.get('compression_ratio', 0):.2f}x")

    st.markdown("---")

    # Tabs Layout
    tab_summary, tab_tree, tab_graph, tab_ece = st.tabs([
        "📄 Executive Summary",
        "🌲 Multi-Scale Hierarchy",
        "🌐 3D Spectral Manifold",
        "⚡ Hardware Systems Profile"
    ])

    with tab_summary:
        st.markdown("#### Final Synthesized Abstract")
        st.info(res["final_summary"])
        
        st.markdown("##### Grounding & Verification")
        if ev.get("hallucination_detected", False):
            st.warning(f"⚠️ Flagged {len(ev['unsupported_sentences'])} sentence(s) with low source grounding.")
            for s in ev["unsupported_sentences"]:
                st.caption(f"- *{s}*")
        else:
            st.success("✅ All summary propositions verified against source document (Zero Hallucination Flagged).")

    with tab_tree:
        st.markdown("#### Hierarchical Multi-Scale Decomposition (RAPTOR-Style)")
        st.caption("Inspect individual sub-clusters synthesized prior to meta-abstract generation.")
        
        clusters = res.get("level_1_extractive_clusters", {})
        abstracts = res.get("level_2_cluster_abstracts", [])

        for idx, (lbl, sents) in enumerate(clusters.items()):
            with st.expander(f"Topic Partition {lbl + 1} ({len(sents)} Submodular Sentences)", expanded=True):
                st.markdown("**Submodular Extractive Representatives:**")
                for s in sents:
                    st.markdown(f"- {s}")
                
                if idx < len(abstracts):
                    st.markdown("**Cluster Abstract:**")
                    st.success(abstracts[idx])

    with tab_graph:
        st.markdown("#### 3D Spectral Manifold Embedding")
        st.caption("Visualizing sentence vertices embedded via the bottom harmonic eigenvectors of the Normalized Graph Laplacian.")

        coords = graph.get("spectral_coords", np.zeros((1, 3)))
        labels = graph.get("labels", np.zeros(len(coords), dtype=int))
        centrality = graph.get("centrality", np.ones(len(coords)))

        if len(coords) >= 2:
            fig = go.Figure()

            # Scatter points
            fig.add_trace(go.Scatter3d(
                x=coords[:, 0],
                y=coords[:, 1],
                z=coords[:, 2] if coords.shape[1] >= 3 else np.zeros(len(coords)),
                mode='markers+text',
                marker=dict(
                    size=np.clip(centrality * 40, 6, 20),
                    color=labels,
                    colorscale='Viridis',
                    opacity=0.85,
                    showscale=True,
                    colorbar=dict(title="Cluster ID")
                ),
                text=[f"S{i}" for i in range(len(coords))],
                hoverinfo='text'
            ))

            fig.update_layout(
                margin=dict(l=0, r=0, b=0, t=30),
                scene=dict(
                    xaxis_title="Eigenvector 1",
                    yaxis_title="Eigenvector 2",
                    zaxis_title="Eigenvector 3",
                ),
                height=550
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Input document must have at least 2 sentences for spectral manifold projection.")

    with tab_ece:
        st.markdown("#### Hardware & Systems Diagnostics")
        st.caption("Low-level memory bandwidth, arithmetic intensity, and compute profiling.")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""
            - **Device:** `{hw.get('device_name', 'Unknown')}`
            - **Precision Mode:** `{hw.get('precision_mode', 'fp32').upper()}`
            - **Tokens Produced:** `{hw.get('tokens_generated', 0)}`
            - **Throughput:** `{hw.get('throughput_tokens_per_sec', 0):.2f} tokens/second`
            """)
        with c2:
            st.markdown(f"""
            - **Arithmetic Intensity:** `{hw.get('arithmetic_intensity_estimate', 0):.2f} FLOPs/Byte`
            - **Peak Host Memory:** `{hw.get('peak_ram_mb', 0):.2f} MB`
            - **RAM Delta:** `{hw.get('ram_delta_mb', 0):.2f} MB`
            - **CPU Utilization:** `{hw.get('cpu_utilization_pct', 0):.1f} %`
            """)
