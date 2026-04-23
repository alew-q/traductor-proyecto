import torch
from tokenizers import Tokenizer
from model import Seq2SeqTransformer


def load_artifacts(checkpoint_path: str, tokenizer_path: str, force_cpu: bool = True):
    device = torch.device("cpu" if force_cpu or not torch.cuda.is_available() else "cuda")

    tokenizer = Tokenizer.from_file(tokenizer_path)

    pad_id = tokenizer.token_to_id("<pad>")
    bos_id = tokenizer.token_to_id("<sos>")
    eos_id = tokenizer.token_to_id("<eos>")

    if pad_id is None or bos_id is None or eos_id is None:
        raise ValueError("No se encontraron <pad>, <sos> o <eos> en el tokenizer.")

    vocab_size = tokenizer.get_vocab_size()

    model = Seq2SeqTransformer(
        vocab_size=vocab_size,
        pad_idx=pad_id,
        d_model=256,
        nhead=8,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dim_feedforward=512,
        dropout=0.1,
        max_len=512,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "model_state" not in checkpoint:
        raise ValueError("El checkpoint no contiene la clave 'model_state'.")

    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    return {
        "model": model,
        "tokenizer": tokenizer,
        "device": device,
        "pad_id": pad_id,
        "bos_id": bos_id,
        "eos_id": eos_id,
    }


@torch.inference_mode()
def greedy_translate(
    text: str,
    model,
    tokenizer,
    device,
    pad_id: int,
    bos_id: int,
    eos_id: int,
    max_source_len: int = 128,
    max_new_tokens: int = 80,
):
    src_enc = tokenizer.encode(text)
    src_ids_list = src_enc.ids[:max_source_len]

    src_ids = torch.tensor([src_ids_list], dtype=torch.long, device=device)
    src_key_padding_mask = src_ids.eq(pad_id)

    generated = torch.tensor([[bos_id]], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        tgt_key_padding_mask = generated.eq(pad_id)

        logits = model(
            src=src_ids,
            tgt=generated,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )

        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)

        if next_token.item() == eos_id:
            break

    output_ids = generated[0].tolist()
    clean_ids = [tok for tok in output_ids if tok not in {bos_id, eos_id, pad_id}]
    translated_text = tokenizer.decode(clean_ids)

    return translated_text, output_ids