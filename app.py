import time
import streamlit as st
from inference import load_artifacts, greedy_translate

st.set_page_config(
    page_title="Traductor ES → EN",
    page_icon="🌍",
    layout="centered",
)

st.title("🌍 Traductor neuronal español → inglés")
st.caption("Demo de inferencia con Transformer pequeño en PyTorch")

with st.sidebar:
    st.header("Configuración")
    checkpoint_path = st.text_input("Checkpoint", value="checkpoints/best_transformer.pth")
    tokenizer_path = st.text_input("Tokenizer JSON", value="tokenizer_es_en/tokenizer_bpe8k.json")
    max_source_len = st.slider("Longitud máxima de entrada", 32, 256, 128, 16)
    max_new_tokens = st.slider("Máximo de tokens generados", 10, 150, 80, 10)
    force_cpu = st.checkbox("Forzar CPU", value=True)

@st.cache_resource
def cached_load(checkpoint_path: str, tokenizer_path: str, force_cpu: bool):
    return load_artifacts(checkpoint_path, tokenizer_path, force_cpu)

try:
    artifacts = cached_load(checkpoint_path, tokenizer_path, force_cpu)
    st.success(f"Modelo cargado en: {artifacts['device']}")
except Exception as e:
    st.error(f"No se pudo cargar el modelo o tokenizer: {e}")
    st.stop()

text_input = st.text_area(
    "Texto en español",
    value="Hola, este es un proyecto universitario de traducción automática neuronal.",
    height=150,
)

col1, col2 = st.columns(2)
translate_btn = col1.button("Traducir", use_container_width=True)
clear_btn = col2.button("Limpiar", use_container_width=True)

if clear_btn:
    st.rerun()

if translate_btn:
    if not text_input.strip():
        st.warning("Ingresa un texto.")
        st.stop()

    start = time.perf_counter()

    try:
        translation, output_ids = greedy_translate(
            text=text_input.strip(),
            model=artifacts["model"],
            tokenizer=artifacts["tokenizer"],
            device=artifacts["device"],
            pad_id=artifacts["pad_id"],
            bos_id=artifacts["bos_id"],
            eos_id=artifacts["eos_id"],
            max_source_len=max_source_len,
            max_new_tokens=max_new_tokens,
        )

        elapsed = time.perf_counter() - start

        st.subheader("Traducción al inglés")
        st.write(translation if translation.strip() else "[salida vacía]")

        with st.expander("Detalles"):
            st.write(f"Tiempo: {elapsed:.4f} s")
            st.write(f"IDs generados: {output_ids}")

    except Exception as e:
        st.error(f"Error durante la inferencia: {e}")

st.markdown("---")
st.subheader("Ejemplos")
examples = [
    "Buenos días, necesito ayuda con mi tarea.",
    "La traducción automática neuronal puede fallar en frases ambiguas.",
    "Nosotros entrenamos un Transformer pequeño para español e inglés."
]

for i, example in enumerate(examples, start=1):
    if st.button(f"Usar ejemplo {i}"):
        st.session_state["example_text"] = example

if "example_text" in st.session_state:
    st.info(st.session_state["example_text"])