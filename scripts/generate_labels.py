import os
import random
import ast
import pandas as pd
import warnings
from tqdm import tqdm

from omegaconf import OmegaConf
from transformers import pipeline, AutoTokenizer, GenerationConfig
from transformers import logging as hf_logging

# Files
INPUT_FILE = "datasets/clean/filtered_tribert.csv"
OUTPUT_FILE = "datasets/clean/boundary_dataset.csv"
DEBUG_FILE = "datasets/clean/debug_first_10.txt"

# Model
MODEL_NAME = "Qwen/Qwen3.5-2B" #0.8,2,4,9,27  

FIM_PREFIX = "<|fim_prefix|>"
FIM_SUFFIX = "<|fim_suffix|>"
FIM_MIDDLE = "<|fim_middle|>"


def load_context_tokens_default():
    try:
        cfg = OmegaConf.load("conf/config.yaml")
        return cfg.get("generation", {}).get("context_tokens", 100)
    except Exception:
        return 100


def split_last_sentence(text: str):
    sentences = [s.strip() for s in text.split(".") if s.strip()]
    if not sentences:
        return text, ""
    last_sent = sentences[-1]
    mid = len(last_sent) // 2
    return text.replace(last_sent, "").strip(), last_sent[:mid].strip()


def split_first_sentence(text: str):
    sentences = [s.strip() for s in text.split(".") if s.strip()]
    if not sentences:
        return "", text
    first_sent = sentences[0]
    mid = len(first_sent) // 2
    return first_sent[mid:].strip(), text.replace(first_sent, "").strip()


def get_context_tokens(tokenizer, text: str, n_tokens: int, from_end: bool = True) -> str:
    if not text:
        return ""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    sliced = tokens[-n_tokens:] if from_end else tokens[:n_tokens]
    return tokenizer.decode(sliced)


def prepare_document(row_dict, tokenizer, n_tokens: int):
    """
    Fase 1: Prepara il documento. Calcola rand, fate e, se necessario, 
    costruisce il prompt da mandare in batch alla GPU.
    """
    chunks = ast.literal_eval(row_dict["chunks"])
    original_hybrid_text = str(row_dict.get("hybrid_text", ""))

    h_to_m_indices = []
    m_to_h_indices = []
    
    for i in range(1, len(chunks)):
        if chunks[i-1]["label"] == 0 and chunks[i]["label"] == 1:
            h_to_m_indices.append(i)
        elif chunks[i-1]["label"] == 1 and chunks[i]["label"] == 0:
            m_to_h_indices.append(i)

    fate = "none"
    target_mod_idx = -1 
    
    rand = random.random()
    if rand < 0.50:
        fate = "none"
    elif rand < 0.75 and h_to_m_indices:
        fate = "h_to_m"
        target_mod_idx = random.choice(h_to_m_indices)
    elif m_to_h_indices:
        fate = "m_to_h"
        target_mod_idx = random.choice(m_to_h_indices)

    # 🚀 FAST PATH: Nessuna modifica FIM necessaria
    if fate == "none":
        return {
            "status": "fast", 
            "text": original_hybrid_text.strip(), 
            "chunks": chunks, 
            "rand": rand, 
            "fate": fate
        }

    # 🐌 SLOW PATH: Prepariamo i dati per il prompt
    new_chunks = [dict(c) for c in chunks] 
    
    def build_text_up_to(chunk_list, end_idx):
        return " ".join([c["text"] for c in chunk_list[:end_idx]])
        
    def build_text_from(chunk_list, start_idx):
        return " ".join([c["text"] for c in chunk_list[start_idx:]])

    prompt = ""
    prep_data = {
        "status": "slow", 
        "fate": fate, 
        "target_mod_idx": target_mod_idx,
        "new_chunks": new_chunks, 
        "rand": rand
    }

    if fate == "h_to_m":
        h_idx = target_mod_idx - 1
        m_idx = target_mod_idx
        
        h_text_full = new_chunks[h_idx]["text"]
        m_text_full = new_chunks[m_idx]["text"]
        
        h_text, h_first_half = split_last_sentence(h_text_full)
        
        text_before = (build_text_up_to(new_chunks, h_idx) + " " + h_text + " " + h_first_half).strip()
        text_after = (m_text_full + " " + build_text_from(new_chunks, m_idx + 1)).strip()
        
        context_before = get_context_tokens(tokenizer, text_before, n_tokens, from_end=True)
        context_after = get_context_tokens(tokenizer, text_after, n_tokens, from_end=False)
        
        prompt = f"{FIM_PREFIX}{context_before}{FIM_SUFFIX}{context_after}{FIM_MIDDLE}"
        
        prep_data["prompt"] = prompt
        prep_data["h_text"] = h_text
        prep_data["h_first_half"] = h_first_half

    elif fate == "m_to_h":
        m_idx = target_mod_idx - 1
        h_idx = target_mod_idx
        
        m_text_full = new_chunks[m_idx]["text"]
        h_text_full = new_chunks[h_idx]["text"]
        
        h_second_half, h_rest = split_first_sentence(h_text_full)
        
        text_before = (build_text_up_to(new_chunks, m_idx) + " " + m_text_full).strip()
        text_after = (h_second_half + " " + h_rest + " " + build_text_from(new_chunks, h_idx + 1)).strip()
        
        context_before = get_context_tokens(tokenizer, text_before, n_tokens, from_end=True)
        context_after = get_context_tokens(tokenizer, text_after, n_tokens, from_end=False)
        
        prompt = f"{FIM_PREFIX}{context_before}{FIM_SUFFIX}{context_after}{FIM_MIDDLE}"
        
        prep_data["prompt"] = prompt
        prep_data["h_second_half"] = h_second_half
        prep_data["h_rest"] = h_rest

    return prep_data


def finalize_document(prep_data, raw_generated_text):
    """
    Fase 2: Prende l'output della GPU e ricostruisce i chunk definitivi.
    """
    new_chunks = prep_data["new_chunks"]
    fate = prep_data["fate"]
    target_mod_idx = prep_data["target_mod_idx"]

    # Pulizia output LLM
    clean_text = raw_generated_text.split("<|endoftext|>")[0].strip()
    ai_generated = clean_text.replace(FIM_SUFFIX, "").replace(FIM_MIDDLE, "")

    if fate == "h_to_m":
        h_idx = target_mod_idx - 1
        m_idx = target_mod_idx
        h_text = prep_data["h_text"]
        h_first_half = prep_data["h_first_half"]
        m_text_full = new_chunks[m_idx]["text"]

        new_chunks[h_idx]["text"] = (h_text + " " + h_first_half).strip()
        new_chunks[m_idx]["text"] = (ai_generated + " " + m_text_full).strip()

    elif fate == "m_to_h":
        m_idx = target_mod_idx - 1
        h_idx = target_mod_idx
        m_text_full = new_chunks[m_idx]["text"]
        h_second_half = prep_data["h_second_half"]
        h_rest = prep_data["h_rest"]

        new_chunks[m_idx]["text"] = (m_text_full + " " + ai_generated).strip()
        new_chunks[h_idx]["text"] = (h_second_half + " " + h_rest).strip()

    final_text = " ".join([c["text"] for c in new_chunks]).strip()
    return final_text, new_chunks


def main():

    n_tokens = load_context_tokens_default()

    if not os.path.exists(INPUT_FILE):
        print(f"❌ File {INPUT_FILE} non trovato. Lancia prima lo script di filtering!")
        return

    print(f"🤖 Caricamento Tokenizer e Pipeline FIM ({MODEL_NAME}) | Context: {n_tokens} token...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    gen_cfg = GenerationConfig(
        do_sample=True,
        temperature=0.7,
        max_new_tokens=40,
        pad_token_id=tokenizer.eos_token_id,
    )

    hf_logging.set_verbosity_error()
    warnings.filterwarnings("ignore")

    generator = pipeline(
        "text-generation",
        model=MODEL_NAME,
        tokenizer=tokenizer,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"🚀 Generazione dataset da: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)

    processed_data = []
    debug_logs = []
    
    BATCH_SIZE = 16
    
    for i in tqdm(range(0, len(df), BATCH_SIZE), desc="Processing Batches", unit="batch"):
        batch_rows = df.iloc[i : i + BATCH_SIZE]
        
        batch_preps = []
        batch_prompts = []
        
        # 1. Preparazione
        for _, row in batch_rows.iterrows():
            row_dict = row.to_dict()
            prep = prepare_document(row_dict, tokenizer, n_tokens)
            prep["row_dict"] = row_dict # Conserviamo la riga per dopo
            
            batch_preps.append(prep)
            if prep["status"] == "slow":
                batch_prompts.append(prep["prompt"])
        
        # 2. Generazione Parallela sulla GPU (Se ci sono prompt)
        gen_results = []
        if batch_prompts:
            # Passiamo l'intera lista alla pipeline, la GPU gestisce tutto in parallelo!
            res = generator(batch_prompts, batch_size=BATCH_SIZE, generation_config=gen_cfg, return_full_text=False)
            # Estrarre il testo (la pipeline restituisce una lista di liste di dizionari)
            gen_results = [r[0]["generated_text"] for r in res]
            
        # 3. Ricostruzione e Salvataggio
        gen_idx = 0
        for prep in batch_preps:
            row_dict = prep["row_dict"]
            original_chunks = ast.literal_eval(row_dict["chunks"])
            
            if prep["status"] == "fast":
                full_text = prep["text"]
                updated_chunks = prep["chunks"]
            else:
                raw_ai_text = gen_results[gen_idx]
                gen_idx += 1
                full_text, updated_chunks = finalize_document(prep, raw_ai_text)
                
            # Logica di Debug per i primi 10 documenti estratti
            if len(debug_logs) < 10:
                log_str = f"=== ESSAY ID: {row_dict['essay_id']} ===\n"
                log_str += f"RAND: {prep['rand']:.4f} | FATE: {prep['fate']}\n\n"
                log_str += f"[PRIMA - CHUNKS]\n{original_chunks}\n\n"
                log_str += f"[DOPO - CHUNKS]\n{updated_chunks}\n\n"
                log_str += f"[TESTO FINALE]\n{full_text}\n"
                log_str += "="*60 + "\n"
                debug_logs.append(log_str)
            elif (len(debug_logs)== 10):
                with open(DEBUG_FILE, "w", encoding="utf-8") as f:
                    f.write("\n".join(debug_logs))

            
            # Aggiorniamo le colonne
            row_dict["text"] = full_text
            row_dict["chunks"] = updated_chunks
            
            if "boundaries" in row_dict:
                del row_dict["boundaries"]
                
            processed_data.append(row_dict)

    # Salvataggio CSV
    out_df = pd.DataFrame(processed_data)
    out_df.to_csv(OUTPUT_FILE, index=False)
    
    # Salvataggio Debug Text

        
    print(f"\n✅ Fatto! Salvato dataset in: {OUTPUT_FILE}")
    print(f"📝 File di debug (Primi 10) salvato in: {DEBUG_FILE}")


if __name__ == "__main__":
    main()