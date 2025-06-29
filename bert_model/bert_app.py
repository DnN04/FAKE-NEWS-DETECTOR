import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import time
import os

# ===== CONFIGURATION =====
MODEL_PATH = "./bert_model/model"
APP_NAME = "Veritas AI"
VERSION = "2.1.0"

# ===== PAGE SETUP =====
st.set_page_config(
    page_title=f"{APP_NAME} | News Authenticator",
    page_icon="🔮",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ===== MODEL LOADING =====
@st.cache(allow_output_mutation=True, show_spinner=False)
def load_model():
    try:
        if not os.path.exists(MODEL_PATH):
            st.error(f"Model directory not found at: {MODEL_PATH}")
            return None, None
            
        with st.spinner("🦋 Initializing neural architecture..."):
            tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
            model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
            model.eval()
            return tokenizer, model
            
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None, None

tokenizer, model = load_model()

# ===== PREMIUM UI =====
def inject_css():
    st.markdown(f"""
    <link href='https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Inter:wght@400;500;600&display=swap' rel='stylesheet'>
    <style>
    :root {{
        --primary: #8b5cf6;
        --primary-dark: #7c3aed;
        --secondary: #10b981;
        --accent: #d4af37;
        --dark-bg: #0f0a1a;
        --dark-card: #1a142d;
        --text-primary: #f8fafc;
        --text-secondary: #b8b8d9;
    }}

    .stApp {{
        background: var(--dark-bg) !important;
        background-image: radial-gradient(circle at 25% 25%, rgba(123, 97, 255, 0.15) 0%, transparent 50%) !important;
        color: var(--text-primary) !important;
        font-family: 'Inter', sans-serif;
    }}
    </style>
    """, unsafe_allow_html=True)

inject_css()

# ===== HEADER =====
st.markdown(f"""
<div style="text-align: center; padding: 2rem 0; position: relative; margin-bottom: 2rem;">
    <div style="
        font-family: 'Playfair Display', serif;
        font-size: 3.5rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin-bottom: 0.5rem;
        background: linear-gradient(135deg, #8b5cf6, #d4af37);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
        text-shadow: 0 2px 10px rgba(139, 92, 246, 0.3);
    ">{APP_NAME}</div>
    <div style="font-size: 1.2rem; color: #b8b8d9; font-style: italic; margin-bottom: 1rem; opacity: 0.8;">
        In veritate victoria
    </div>
    <div style="color: #b8b8d9; letter-spacing: 0.05em; font-size: 0.9rem;">
        BERT-Powered Semantic Forensics | v{VERSION}
    </div>
</div>
""", unsafe_allow_html=True)

# ===== INPUT SECTION =====
st.markdown("""
<div style="margin-bottom: 1.5rem;">
    <div style="font-size: 1.1rem; margin-bottom: 0.75rem; display: block; color: #d4af37;">
        ↳ Textual Analysis Portal
    </div>
""", unsafe_allow_html=True)

user_input = st.text_area(
    "Enter text to analyze:",
    height=220,
    key="text-input",
    placeholder="Paste news content here...",
    help="The system will perform deep semantic analysis"
)

# ===== ANALYSIS BUTTON =====
if st.button("⟣ Initiate Quantum Analysis", key="analyze-btn"):
    if not tokenizer or not model:
        st.error("Model not loaded properly - check console for errors")
    elif not user_input.strip():
        st.warning("✖ Please enter text to analyze")
    else:
        with st.spinner("▸ Conducting holographic assessment..."):
            try:
                start_time = time.time()
                
                inputs = tokenizer(
                    user_input, 
                    return_tensors="pt", 
                    truncation=True, 
                    padding=True, 
                    max_length=512
                )
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                    fake_prob = probs[0][0].item() * 100
                    real_prob = probs[0][1].item() * 100
                
                # Results Display
                st.markdown("""
                <div style="
                    background-color: #1a142d;
                    border-radius: 16px;
                    padding: 2rem;
                    margin: 1.5rem 0;
                    border: 1px solid #2a2342;
                    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
                ">
                    <div style="font-family: 'Playfair Display', serif; font-size: 1.5rem; margin-bottom: 1.5rem; color: #f8fafc;">
                        Authenticity Report
                    </div>
                """, unsafe_allow_html=True)
                
                # Prediction Badge
                if real_prob > fake_prob:
                    st.markdown(f"""
                    <div style="
                        background: rgba(16, 185, 129, 0.15);
                        color: #10b981;
                        padding: 1rem;
                        border-radius: 12px;
                        border: 1px solid rgba(16, 185, 129, 0.3);
                        margin: 1rem 0;
                        text-align: center;
                        font-weight: bold;
                        font-size: 1.2rem;
                        box-shadow: 0 0 15px rgba(16, 185, 129, 0.2);
                    ">
                        ✓ Authentic Content ({(real_prob - fake_prob):.1f}% confidence on data)
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="
                        background: rgba(239, 68, 68, 0.15);
                        color: #ef4444;
                        padding: 1rem;
                        border-radius: 12px;
                        border: 1px solid rgba(239, 68, 68, 0.3);
                        margin: 1rem 0;
                        text-align: center;
                        font-weight: bold;
                        font-size: 1.2rem;
                        box-shadow: 0 0 15px rgba(239, 68, 68, 0.2);
                    ">
                        ✗ Disinformation Detected ({(fake_prob - real_prob):.1f}% confidence on data)
                    </div>
                    """, unsafe_allow_html=True)
                
                # Confidence Bars
                st.markdown(f"""
                <div style="margin: 1.5rem 0;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                        <span style="color: #ef4444;">Deception Indicators <span style="font-weight: 600; color: #f8fafc;">{fake_prob:.1f}%</span></span>
                        <span style="color: #10b981;">Veracity Markers <span style="font-weight: 600; color: #f8fafc;">{real_prob:.1f}%</span></span>
                    </div>
                    <div style="height: 10px; border-radius: 5px; margin: 1rem 0 2rem; background: #2a2342; overflow: hidden;">
                        <div style="height: 100%; background: linear-gradient(90deg, #ef4444, #f87171); width:{fake_prob}%;"></div>
                    </div>
                    <div style="height: 10px; border-radius: 5px; margin-bottom: 0.5rem; background: #2a2342; overflow: hidden;">
                        <div style="height: 100%; background: linear-gradient(90deg, #10b981, #34d399); width:{real_prob}%;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div style="font-size: 0.8rem; color: #b8b8d9; text-align: right; margin-top: 1rem;">
                    Analysis completed in {time.time() - start_time:.2f}s
                </div>
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")

# ===== FOOTER =====
st.markdown(f"""
<div style="
    margin-top: 4rem;
    text-align: center;
    color: #b8b8d9;
    font-size: 0.9rem;
    padding: 1.5rem;
    border-top: 1px solid rgba(139, 92, 246, 0.1);
">
    {APP_NAME} | Cognitive Authentication System<br>
    <div style="margin-top: 0.5rem; font-size: 0.8rem; opacity: 0.7;">
        © 2025 Durgesh Narayan Nayak | v{VERSION}
    </div>
</div>
<div style="position: fixed; bottom: 10px; right: 10px; font-size: 0.8rem; color: rgba(180, 180, 220, 0.3); letter-spacing: 0.05em;">
    ⋆⋅☆⋅⋆
</div>
""", unsafe_allow_html=True)
