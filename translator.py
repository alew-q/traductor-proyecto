import re
import os
import glob
import torch
from transformers import MarianMTModel, MarianTokenizer, MarianConfig


# ============================================================
# HELPER: cargar pesos desde .pth
# ============================================================
def _load_from_pth(model_dir: str, weights_path: str, device: str) -> MarianMTModel:
    config = MarianConfig.from_pretrained(model_dir)
    model = MarianMTModel(config)

    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state = checkpoint["state_dict"]
        elif "model_state" in checkpoint:
            state = checkpoint["model_state"]
        else:
            state = checkpoint
    else:
        state = checkpoint

    model.load_state_dict(state, strict=False)
    return model


# ============================================================
# CARGA DEL MODELO MARIANMT
# ============================================================
def load_model(model_dir: str, device: str = "cpu"):
    """
    Carga un modelo MarianMT desde una carpeta local.
    Soporta pesos en pytorch_model.bin, model.safetensors, o cualquier .pth
    """
    tokenizer = MarianTokenizer.from_pretrained(model_dir)

    # Buscar archivo de pesos estándar primero
    standard = ["pytorch_model.bin", "model.safetensors"]
    weights_path = None
    for name in standard:
        p = os.path.join(model_dir, name)
        if os.path.exists(p):
            weights_path = p
            break

    # Si no hay nombre estándar, buscar cualquier .pth
    if weights_path is None:
        pth_files = glob.glob(os.path.join(model_dir, "*.pth"))
        if pth_files:
            weights_path = pth_files[0]

    if weights_path is None:
        raise FileNotFoundError(
            f"No se encontró ningún archivo de pesos (.bin, .safetensors, .pth) en {model_dir}"
        )

    # Cargar según tipo de archivo
    if weights_path.endswith((".bin", ".safetensors")):
        model = MarianMTModel.from_pretrained(model_dir)
    else:
        model = _load_from_pth(model_dir, weights_path, device)

    model = model.to(device)
    model.eval()
    return model, tokenizer


# ============================================================
# DIVIDIR PÁRRAFO EN ORACIONES
# ============================================================
def _split_sentences(paragraph: str) -> list:
    partes = re.split(r'(?<=[.!?])\s+', paragraph.strip())
    return [p.strip() for p in partes if p.strip()]


# ============================================================
# TRADUCCIÓN PÚBLICA
# ============================================================
def translate(
    text: str,
    model: MarianMTModel,
    tokenizer: MarianTokenizer,
    device: str = "cpu",
    max_len: int = 128,
) -> str:
    """
    Traduce texto ES -> EN preservando párrafos.
    """
    parrafos = re.split(r'\n+', text.strip())
    parrafos_traducidos = []

    for parrafo in parrafos:
        parrafo = parrafo.strip()
        if not parrafo:
            parrafos_traducidos.append("")
            continue

        oraciones = _split_sentences(parrafo)
        inputs = tokenizer(
            oraciones,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,
        ).to(device)

        translated = model.generate(**inputs, max_new_tokens=max_len)
        decoded = tokenizer.batch_decode(translated, skip_special_tokens=True)

        parrafos_traducidos.append(" ".join(decoded))

    return "\n\n".join(parrafos_traducidos)