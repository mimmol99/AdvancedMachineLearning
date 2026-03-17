import pandas as pd
import os
import ast
import hydra
from omegaconf import DictConfig

def normalize_label(label):
    """
    Converte label testuali in interi:
    human -> 0
    machine -> 1
    """
    if label is None:
        return None

    label = str(label).strip().lower()

    if label in {"human", "h", "0"}:
        return 0
    if label in {"machine", "m", "ai", "1"}:
        return 1

    return None

def parse_sent_and_label(value):
    """
    Parsea la colonna sent_and_label che contiene una lista di tuple:
    [("frase 1", "human"), ("frase 2", "machine"), ...]
    """
    if pd.isna(value):
        return []
    if isinstance(value, list):
        return value
    try:
        return ast.literal_eval(str(value))
    except Exception as e:
        print(f"⚠️ Impossibile parsare sent_and_label: {e}")
        return []

def build_chunks_from_sent_and_label(row):
    """
    Costruisce la lista di chunk nel formato:
    [
        {"text": "...", "label": 0},
        {"text": "...", "label": 1},
    ]
    """
    items = parse_sent_and_label(row.get("sent_and_label", None))
    chunks = []

    for item in items:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue

        sentence, raw_label = item
        sentence = str(sentence).strip()
        label = normalize_label(raw_label)

        if not sentence or label is None:
            continue

        chunks.append({
            "text": sentence,
            "label": label
        })

    return chunks

def build_hybrid_text_from_chunks(chunks):
    """
    Ricostruisce il testo finale concatenando i chunk nell'ordine dato.
    """
    return " ".join(chunk["text"].strip() for chunk in chunks if str(chunk["text"]).strip()).strip()

def print_dataset_stats(df, stage_name):
    """
    Calcola e stampa le statistiche del dataset utili per la tabella LaTeX.
    """
    print(f"\n{'='*15} STATISTICHE {stage_name} {'='*15}")
    
    total_samples = len(df)
    print(f"Total Samples:    {total_samples}")
    
    if "train_ix" in df.columns:
        splits = df["train_ix"].value_counts().to_dict()
        print(f"Splits:           Train: {splits.get('train', 0)} | Val: {splits.get('val', 0)} | Test: {splits.get('test', 0)}")
    
    if "essay_id" in df.columns:
        dupes = df.duplicated(subset=["essay_id"]).sum()
        print(f"Duplicate Essays: {dupes} (Rimosse se > 0)")
        
    if "boundary_num" in df.columns:
        mean_b = df["boundary_num"].mean()
        print(f"Mean Boundaries:  {mean_b:.2f}")
        
    if "author_seq" in df.columns:
        top_transitions = df["author_seq"].value_counts().head(5).index.tolist()
        print(f"Transition Types: {', '.join(str(x) for x in top_transitions)}")
        
    print("="*50 + "\n")

# Inizializziamo Hydra puntando alla cartella conf
@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    # Leggiamo i parametri dal config di Hydra
    proc_cfg = cfg.dataset.processing
    input_file = proc_cfg.input_file
    output_file = proc_cfg.output_file
    min_boundaries = proc_cfg.min_boundaries
    keep_highest_boundaries = proc_cfg.keep_highest_boundaries

    print(f"🚀 Lettura dataset grezzo: {input_file}")
    print(f"⚙️ Configurazione: Min Boundaries = {min_boundaries} | Deduplicazione = {keep_highest_boundaries}")
    
    df = pd.read_excel(input_file)

    if "boundary_num" not in df.columns:
        raise ValueError("❌ Colonna 'boundary_num' non trovata nel dataset!")

    # Normalizzazione split anticipata
    if "train_ix" in df.columns:
        df["train_ix"] = df["train_ix"].astype(str).str.lower().str.strip()
        df["train_ix"] = df["train_ix"].replace("valid", "val")
    else:
        df["train_ix"] = "train"

    # --- STAMPA STATISTICHE PRE-FILTERING ---
    print_dataset_stats(df, "ORIGINAL (PRE-FILTERING)")

    # 1 & 2. Deduplicazione (controllata da config)
    if keep_highest_boundaries and "essay_id" in df.columns:
        print("🧹 Esecuzione deduplicazione per essay_id (mantenendo il massimo numero di boundaries)...")
        df = df.sort_values(by=["essay_id", "boundary_num"], ascending=[True, False])
        df_filtered = df.drop_duplicates(subset=["essay_id"], keep="first").copy()
    else:
        print("⏭️ Deduplicazione saltata come da configurazione.")
        df_filtered = df.copy()

    # 3. Filtro minimo sui boundary (controllato da config)
    print(f"🔍 Filtraggio documenti con meno di {min_boundaries} boundaries...")
    df_filtered = df_filtered[df_filtered["boundary_num"] >= min_boundaries].copy()

    # 4. Costruzione chunks
    df_filtered["chunks"] = df_filtered.apply(build_chunks_from_sent_and_label, axis=1)

    # Rimuovi righe senza chunk validi
    df_filtered = df_filtered[df_filtered["chunks"].apply(len) > 0].copy()

    # 5. Costruzione hybrid_text
    df_filtered["hybrid_text"] = df_filtered["chunks"].apply(build_hybrid_text_from_chunks)

    # --- STAMPA STATISTICHE POST-FILTERING ---
    print_dataset_stats(df_filtered, "FILTERED (POST-FILTERING)")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    cols_to_save = [
        "essay_id", "author_seq", "boundary_num", "chunks", 
        "train_ix", "hybrid_text", "sent_and_label"
    ]
    final_cols = [c for c in cols_to_save if c in df_filtered.columns]

    df_filtered[final_cols].to_csv(output_file, index=False)

    print(f"✅ Fatto! Mantenuti {len(df_filtered)} essay.")
    print(f"💾 Salvato in: {output_file}")

if __name__ == "__main__":
    main()