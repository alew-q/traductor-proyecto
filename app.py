import torch
import streamlit as st
from translator import load_model, translate

# ============================================================
# CONFIG — carpeta con todos los archivos del modelo Marian
# ============================================================
MODEL_DIR = "checkpoints"   # debe contener .pth/.bin + tokenizer_config.json, etc.

# ============================================================
# CARGA DEL MODELO (cacheado)
# ============================================================
@st.cache_resource(show_spinner="Cargando modelo...")
def get_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_model(MODEL_DIR, device=device)
    return model, tokenizer, device


# ============================================================
# UI
# ============================================================
st.set_page_config(
    page_title="Traductor ES → EN",
    page_icon="🌐",
    layout="centered",
)

st.title("🌐 Traductor Español → Inglés")

try:
    model, tokenizer, device = get_model()
    device_label = "GPU (CUDA)" if device == "cuda" else "CPU"
    st.sidebar.success(f"Modelo cargado · {device_label}")
except Exception as e:
    st.error(f"❌ Error al cargar el modelo: {e}")
    st.stop()



# ── Área principal ──────────────────────────────────────────
col1, col2 = st.columns(2, gap="medium")

with col1:
    st.subheader("🇪🇸 Español")
    input_text = st.text_area(
        label="input",
        placeholder="Escribe o pega el texto aquí...",
        height=380,
        label_visibility="collapsed",
    )

with col2:
    st.subheader("🇬🇧 Inglés")
    result_box = st.empty()

translate_btn = st.button("Traducir ▶", type="primary", use_container_width=True)

if translate_btn:
    text = input_text.strip()
    if not text:
        st.warning("Escribe algo antes de traducir.")
    else:
        with st.spinner("Traduciendo..."):
            result = translate(
                text=text,
                model=model,
                tokenizer=tokenizer,
                device=device,
            )

        with result_box.container():
            st.markdown(
                f"""
                <div style="
                    background-color: #1e2130;
                    border: 1px solid #3a3f5c;
                    border-radius: 8px;
                    padding: 12px 16px;
                    min-height: 380px;
                    font-size: 15px;
                    line-height: 1.6;
                    white-space: pre-wrap;
                    word-wrap: break-word;
                    color: #e8eaf6;
                ">
{result}
                </div>
                """,
                unsafe_allow_html=True,
            )