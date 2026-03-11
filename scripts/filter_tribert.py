import pandas as pd
import os
import ast

INPUT_FILE = "datasets/raw/raw_tribert.xlsx"
OUTPUT_FILE = "datasets/clean/filtered_tribert.csv"
MIN_BOUNDARIES = 2  # Es. H_M_H -> 2 transizioni (boundaries)


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

    Restituisce una lista Python valida.
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
        ...
    ]
    usando la colonna sent_and_label.
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
    
    # 1. Total Samples
    total_samples = len(df)
    print(f"Total Samples:    {total_samples}")
    
    # 2. Splits
    if "train_ix" in df.columns:
        splits = df["train_ix"].value_counts().to_dict()
        print(f"Splits:           Train: {splits.get('train', 0)} | Val: {splits.get('val', 0)} | Test: {splits.get('test', 0)}")
    
    # 3. Duplicate Essays
    if "essay_id" in df.columns:
        dupes = df.duplicated(subset=["essay_id"]).sum()
        print(f"Duplicate Essays: {dupes} (Rimosse se > 0)")
        
    # 4. Mean Boundaries
    if "boundary_num" in df.columns:
        mean_b = df["boundary_num"].mean()
        print(f"Mean Boundaries:  {mean_b:.2f}")
        
    # 5. Transition Types (Top 5)
    if "author_seq" in df.columns:
        # Prende i 5 pattern di transizione più comuni
        top_transitions = df["author_seq"].value_counts().head(5).index.tolist()
        print(f"Transition Types: {', '.join(str(x) for x in top_transitions)}")
        
    print("="*50 + "\n")


def main():
    print(f"🚀 Lettura dataset grezzo: {INPUT_FILE}")
    df = pd.read_excel(INPUT_FILE)

    if "boundary_num" not in df.columns:
        raise ValueError("❌ Colonna 'boundary_num' non trovata nel dataset!")

    # Normalizzazione split anticipata per avere le statistiche PRE-filtering corrette
    if "train_ix" in df.columns:
        df["train_ix"] = df["train_ix"].astype(str).str.lower().str.strip()
        df["train_ix"] = df["train_ix"].replace("valid", "val")
    else:
        df["train_ix"] = "train"

    # --- STAMPA STATISTICHE PRE-FILTERING ---
    print_dataset_stats(df, "ORIGINAL (PRE-FILTERING)")

    # 1. Ordina per essay_id e per numero di boundary descrescente
    if "essay_id" in df.columns:
        df = df.sort_values(by=["essay_id", "boundary_num"], ascending=[True, False])
        # 2. Mantieni una sola riga per essay_id: quella con più boundary
        df_filtered = df.drop_duplicates(subset=["essay_id"], keep="first").copy()
    else:
        df_filtered = df.copy()

    # 3. Filtro minimo sui boundary
    df_filtered = df_filtered[df_filtered["boundary_num"] >= MIN_BOUNDARIES].copy()

    # 4. Costruzione chunks da sent_and_label
    df_filtered["chunks"] = df_filtered.apply(build_chunks_from_sent_and_label, axis=1)

    # Rimuovi righe senza chunk validi
    df_filtered = df_filtered[df_filtered["chunks"].apply(len) > 0].copy()

    # 5. Costruzione hybrid_text coerente con i chunk
    df_filtered["hybrid_text"] = df_filtered["chunks"].apply(build_hybrid_text_from_chunks)

    # --- STAMPA STATISTICHE POST-FILTERING ---
    print_dataset_stats(df_filtered, "FILTERED (POST-FILTERING)")

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    cols_to_save = [
        "essay_id",
        "author_seq",
        "boundary_num",
        "chunks",
        "train_ix",
        "hybrid_text",
        "sent_and_label",
    ]

    final_cols = [c for c in cols_to_save if c in df_filtered.columns]

    df_filtered[final_cols].to_csv(OUTPUT_FILE, index=False)

    print(f"✅ Fatto! Mantenuti {len(df_filtered)} essay unici e complessi.")
    print(f"💾 Salvato in: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()