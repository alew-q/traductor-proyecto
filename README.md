# Traductor neuronal ES -> EN

Aplicación en Streamlit para cargar un modelo Transformer pequeño entrenado en PyTorch y traducir texto de español a inglés.

## Estructura

- `app.py`: interfaz Streamlit
- `model.py`: arquitectura del modelo
- `inference.py`: carga de artifacts e inferencia
- `checkpoints/best_model.pth`: checkpoint del modelo
- `tokenizer_es_en/`: tokenizer guardado

## Instalación local

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py