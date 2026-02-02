import streamlit as st
import numpy as np
import pandas as pd
import time
import plotly.graph_objects as go
import io
from PIL import Image
import matplotlib.pyplot as plt

# --- CONFIGURATION DE LA FORGE ---
st.set_page_config(page_title="KADJED-AI Forge", layout="wide", page_icon="⚒️")

# CSS Personnalisé pour l'ambiance Forge
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #e0e0e0; }
    .stMetric { background-color: #1c1c1c; padding: 10px; border-radius: 10px; border: 1px solid #ffd700; }
    </style>
    """, unsafe_allow_html=True)

# Initialisation des variables de session
if "messages" not in st.session_state:
    st.session_state.messages = []
if "master_vector" not in st.session_state:
    st.session_state.master_vector = [0.33, 0.33, 0.34]
if "k_curvature" not in st.session_state:
    st.session_state.k_curvature = 1.0
if "attractor_type" not in st.session_state:
    st.session_state.attractor_type = "Djed (Point Fixe)"

# --- FONCTIONS LOGIQUES ---

def calculate_master_vector(prompt):
    prompt_lower = prompt.lower()
    if len(prompt) > 80:
        phi_c, phi_m, phi_d = 0.5, 0.3, 0.2
    elif any(word in prompt_lower for word in ["imagine", "crée", "nouveau"]):
        phi_c, phi_d, phi_m = 0.3, 0.5, 0.2
    elif any(word in prompt_lower for word in ["qui", "quand", "fait"]):
        phi_m, phi_c, phi_d = 0.6, 0.3, 0.1
    else:
        phi_m, phi_c, phi_d = 0.4, 0.4, 0.2
    
    # Ajout d'une légère variation aléatoire (entropie de la forge)
    vector = np.array([phi_m, phi_c, phi_d]) + np.random.uniform(-0.05, 0.05, 3)
    vector = np.clip(vector, 0.01, 1.0)
    return list(vector / vector.sum())

def generate_response(prompt, phi_m, phi_c, phi_d, attractor_type, k_val):
    responses = {
        "Djed (Point Fixe)": "Analyse cristalline terminée. La structure est stable.",
        "Ankh (Cycle Limite)": "Le flux harmonique est établi. Les cycles se rejoignent.",
        "Oudjat (Étrange)": "L'imprévisible s0'est manifesté. Les motifs émergent du chaos."
    }
    base = responses[attractor_type]
    mod = f" Alliage : Fer {phi_m:.1%} | Or {phi_c:.1%} | Erbium {phi_d:.1%}."
    stase = " [STASE TOTALE]" if 0.98 <= k_val <= 1.02 else ""
    return f"{base}{mod}{stase}\n\n*Le verbe a été transmuté selon les paramètres de la Forge.*"

# --- INTERFACE ---

st.title("🏛️ KADJED-AI : La Forge du Verbe Triadique")
st.caption("Système Expert TTU–MC³ | Fréquence de Résonance : 113.0 Hz")

with st.sidebar:
    st.header("🛡️ Console de Contrôle")
    st.session_state.k_curvature = st.slider("Courbure K (Santé)", 0.8, 1.2, st.session_state.k_curvature)
    
    if 0.95 <= st.session_state.k_curvature <= 1.05:
        st.success(f"🎯 STASE PRIMAIRE : K={st.session_state.k_curvature:.3f}")
    else:
        st.error("🚨 DÉRIVE DÉTECTÉE")

    st.session_state.attractor_type = st.selectbox("Mode Cognitif", ["Djed (Point Fixe)", "Ankh (Cycle Limite)", "Oudjat (Étrange)"])
    
    st.divider()
    # Graphique Radar Temps Réel
    fig_side = go.Figure(data=go.Scatterpolar(
        r=st.session_state.master_vector + [st.session_state.master_vector[0]],
        theta=['Fer', 'Or', 'Erbium', 'Fer'],
        fill='toself', fillcolor='rgba(255, 215, 0, 0.3)', line=dict(color='gold')
    ))
    fig_side.update_layout(polar=dict(radialaxis=dict(visible=False, range=[0, 1])), showlegend=False, height=200, margin=dict(l=20,r=20,t=20,b=20))
    st.plotly_chart(fig_side, use_container_width=True)

# Zone de Chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Introduisez votre pensée dans la forge..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.status("⚡ Transmutation en cours...") as status:
            st.session_state.master_vector = calculate_master_vector(prompt)
            time.sleep(0.8)
            status.update(label="✅ Alliage Forgé", state="complete")
        
        full_res = generate_response(prompt, *st.session_state.master_vector, st.session_state.attractor_type, st.session_state.k_curvature)
        st.write(full_res)
        
        # Boutons d'action
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔊 Synthèse 113Hz"):
                st.info("Résonance à 113Hz activée...")
                st.audio("https://www.soundjay.com/buttons/beep-01a.mp3")
        with col2:
            if st.button("🖼️ Visualiser l'Alliage"):
                fig, ax = plt.subplots()
                ax.pie(st.session_state.master_vector, labels=['Fer', 'Or', 'Erbium'], colors=['#7f8c8d', '#f1c40f', '#9b59b6'])
                st.pyplot(fig)

    st.session_state.messages.append({"role": "assistant", "content": full_res})