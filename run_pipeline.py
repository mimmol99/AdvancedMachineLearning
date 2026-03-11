# run_pipeline.py
import os
import hydra
import torch
import mlflow
import logging
import subprocess
import ast
import socket
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
import pandas as pd

# Importiamo i nostri moduli
from src.models.architecture import DebertaCRFBoundaryDetector
from src.data.dataset import BoundaryDataset
from src.training.trainer import Trainer
from scripts.visualize_samples import generate_visualization

log = logging.getLogger(__name__)

def is_port_in_use(port):
    """Checks if a port is already occupied."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def start_mlflow_ui(port=5000):
    """Starts the MLflow UI only if it's not already running."""
    if is_port_in_use(port):
        log.info(f"ℹ️ MLflow UI already active on http://localhost:{port}. Skipping startup.")
        return None

    log.info(f"🌐 Starting MLflow UI server on http://localhost:{port}...")
    try:
        ui_process = subprocess.Popen(
            ["mlflow", "ui", "--host", "0.0.0.0", "--port", str(port)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        log.info("✅ MLflow UI started successfully in the background.")
        return ui_process
    except Exception as e:
        log.error(f"❌ Failed to start MLflow UI: {e}")
        return None

def analyze_datasets(train_ds, val_ds, test_ds, tokenizer, chart_dir):
    """Analizza il numero di documenti, i chunk Human vs AI e le distribuzioni delle lunghezze."""
    log.info(f"📊 Generazione dei grafici aggregati nella cartella {chart_dir} ...")
    splits = {"Train": train_ds, "Val": val_ds, "Test": test_ds}
    
    bins = list(range(0, 110, 10))
    
    doc_counts_data = [] 
    counts_data = [] # Struttura ripristinata per i chunk
    lengths_data = {"Train": {"H": [], "A": []}, 
                    "Val": {"H": [], "A": []}, 
                    "Test": {"H": [], "A": []}}

    ### NUOVO: Struttura per tracciare la lunghezza totale del documento
    doc_lengths_data = {"Train": [], "Val": [], "Test": []}

    for split_name, ds in splits.items():
        # 1. Salviamo il numero totale di documenti per questo split
        doc_counts_data.append({"Split": split_name, "Document Count": len(ds.data)})
        
        h_count, a_count = 0, 0
        
        # 2. Raccogliamo i dati per le lunghezze dei chunk e il loro conteggio
        for _, row in ds.data.iterrows():
            chunks = ast.literal_eval(str(row.get("chunks", "[]")))
            
            ### NUOVO: Inizializziamo il contatore per la lunghezza del documento
            total_doc_length = 0 
            
            for c in chunks:
                label = int(c.get("label", 0))
                text = c.get("text", "")
                
                # Tokenizziamo il testo del chunk per calcolarne la lunghezza
                tokens = tokenizer.encode(text, add_special_tokens=False)
                chunk_length = len(tokens)
                
                ### NUOVO: Aggiungiamo la lunghezza del chunk al totale del documento
                total_doc_length += chunk_length 
                
                if chunk_length > 100:
                    chunk_length = 100 
                
                if label == 0:
                    h_count += 1
                    lengths_data[split_name]["H"].append(chunk_length)
                else:
                    a_count += 1
                    lengths_data[split_name]["A"].append(chunk_length)
                    
            ### NUOVO: Salviamo la lunghezza totale del documento per lo split corrente
            # Limitiamo a 512 per il grafico (dato che DeBERTa taglia a 512)
            if total_doc_length > 512:
                total_doc_length = 512
            doc_lengths_data[split_name].append(total_doc_length)

        # Aggiungiamo i conteggi dei chunk per il grafico Human vs AI
        counts_data.append({"Split": split_name, "Label": "Human", "Count": h_count})
        counts_data.append({"Split": split_name, "Label": "AI", "Count": a_count})

    # --- 1. GRAFICO: CONTEGGIO DOCUMENTI TOTALI PER SPLIT ---
    df_docs = pd.DataFrame(doc_counts_data)
    plt.figure(figsize=(7, 5))
    ax_docs = sns.barplot(data=df_docs, x="Split", y="Document Count", hue="Split", palette="viridis", legend=False)
    plt.title("Total Documents per Split")
    plt.ylabel("Number of Documents")
    plt.xlabel("Dataset Split")
    for container in ax_docs.containers:
        ax_docs.bar_label(container, padding=3, fmt='%d', fontweight='bold')
        
    path_doc_cnt = os.path.join(chart_dir, "aggregated_document_count.png")
    plt.savefig(path_doc_cnt, facecolor='white', transparent=False)
    if mlflow.active_run():
        mlflow.log_artifact(path_doc_cnt)
    plt.close()

    # --- 2. GRAFICO: CONTEGGIO CHUNK HUMAN VS AI ---
    df_counts = pd.DataFrame(counts_data)
    plt.figure(figsize=(8, 5))
    ax_chunks = sns.barplot(data=df_counts, x="Split", y="Count", hue="Label", palette=["blue", "orange"])
    plt.title("Human vs AI Samples Count per Split (Chunk-level)")
    plt.ylabel("Number of Chunks")
    plt.xlabel("Dataset Split")
    for container in ax_chunks.containers:
        ax_chunks.bar_label(container, padding=3, fmt='%d')
        
    path_cnt = os.path.join(chart_dir, "aggregated_samples_count.png")
    plt.savefig(path_cnt, facecolor='white', transparent=False)
    if mlflow.active_run():
        mlflow.log_artifact(path_cnt)
    plt.close()

    # --- 3. GRAFICO: LUNGHEZZE AGGREGATE DEI CHUNK (Subplots) ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False) 
    fig.suptitle("Chunk Lengths Distribution (Tokens 0-100)", fontsize=16)

    for i, split_name in enumerate(["Train", "Val", "Test"]):
        ax = axes[i]
        ax.hist(
            [lengths_data[split_name]["H"], lengths_data[split_name]["A"]], 
            bins=bins, 
            label=['Human', 'AI'], 
            color=['blue', 'orange'], 
            stacked=False,
            edgecolor='black'
        )
        
        ax.set_title(split_name)
        ax.set_xlabel("Tokens")
        ax.set_xticks(bins)
        if i == 0:
            ax.set_ylabel("Frequency")
        ax.legend()

    plt.tight_layout()
    path_len = os.path.join(chart_dir, "aggregated_lengths.png")
    plt.savefig(path_len, facecolor='white', transparent=False)
    if mlflow.active_run():
        mlflow.log_artifact(path_len)
    plt.close()
    
    #4. GRAFICO: DISTRIBUZIONE LUNGHEZZA INTERO DOCUMENTO ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
    fig.suptitle("Total Document Lengths Distribution (Tokens 0-512)", fontsize=16)
    
    # Bins per i documenti (es. salti da 50 token)
    doc_bins = list(range(0, 600, 50))
    
    for i, split_name in enumerate(["Train", "Val", "Test"]):
        ax = axes[i]
        sns.histplot(
            doc_lengths_data[split_name], 
            bins=doc_bins, 
            color='purple', 
            edgecolor='black', 
            ax=ax,
            kde=True # Aggiunge una linea di densità morbida
        )
        
        ax.set_title(split_name)
        ax.set_xlabel("Total Document Tokens")
        ax.set_xticks(doc_bins)
        if i == 0:
            ax.set_ylabel("Number of Documents")
        else:
            ax.set_ylabel("")
            
    plt.tight_layout()
    path_doc_len = os.path.join(chart_dir, "aggregated_document_lengths.png")
    plt.savefig(path_doc_len, facecolor='white', transparent=False)
    if mlflow.active_run():
        mlflow.log_artifact(path_doc_len)
    plt.close()
    
    log.info("✅ Grafici aggregati generati e caricati su MLflow!") 

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    start_mlflow_ui()

    log.info(f"🚀 Starting Experiment: {cfg.experiment_name}")
    
    # 0. Setup Cartella Output Grafici
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    chart_dir = os.path.join("charts", timestamp)
    os.makedirs(chart_dir, exist_ok=True)
    
    # 1. Setup Base & MLflow
    mlflow.set_experiment(cfg.experiment_name)
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    log.info(f"⚙️ Dispositivo in uso: {device}")
    
    crf_status = "CRF" if cfg.model.get("use_crf", True) else "NoCRF"

    run_name = f"{crf_status}_LR-{cfg.training.lr}_WD-{cfg.training.weight_decay}_DO-{cfg.model.head.dropout}_SC-{cfg.model.custom_ce.short_chunk_alpha}_AT-{cfg.model.loss_weights.alpha_transition}"    
    
    with mlflow.start_run(run_name=run_name):  

        mlflow.log_params(OmegaConf.to_container(cfg, resolve=True))

        # 2. Inizializza Modello e Tokenizer
        model = DebertaCRFBoundaryDetector(cfg).to(device)
        tokenizer = model.get_tokenizer()
        
        # 3. Caricamento dei Dataset
        common_params = {
            "path": cfg.dataset.path,
            "tokenizer": tokenizer,
            "max_length": cfg.dataset.max_length
        }

        train_ds = BoundaryDataset(**common_params, split='train', ratio=cfg.training.training_data_ratio)
        val_ds   = BoundaryDataset(**common_params, split='val',   ratio=cfg.training.validation_data_ratio)
        test_ds  = BoundaryDataset(**common_params, split='test',  ratio=cfg.test.test_data_ratio)

        log.info(f"📊 Dataset loaded: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")

        # 4. DataLoaders
        train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size, shuffle=True, num_workers=0, pin_memory=False)
        val_loader   = DataLoader(val_ds,   batch_size=cfg.training.batch_size, shuffle=False, num_workers=0, pin_memory=False)
        test_loader  = DataLoader(test_ds,  batch_size=cfg.training.batch_size, shuffle=False, num_workers=0, pin_memory=False)
        
        analyze_datasets(train_ds, val_ds, test_ds, tokenizer, chart_dir)
        
        # 5. Ottimizzatore
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=cfg.training.lr, 
            weight_decay=cfg.training.weight_decay
        )
        
        # 6. Inizializzazione Trainer & Training Loop
        trainer = Trainer(model, train_loader, val_loader, optimizer, device, chart_dir)
        best_model_path = os.path.join(chart_dir, "best_model.pth")
        
        trainer.fit(
            epochs=cfg.training.epochs, 
            es_cfg=cfg.training.get("early_stopping", {}), 
            best_model_path=best_model_path
        )
        
        # 7. Valutazione Finale sul Test Set
        log.info(f"\n🧪 Caricamento miglior modello per il TEST SET...")
        model.load_state_dict(torch.load(best_model_path))
        
        test_metrics = trainer.evaluate(test_loader, epoch=None, prefix="test")
        mlflow.log_metric("final_test_f1", test_metrics["f1"])

        trainer.evaluate_boundaries(test_loader, tokenizer, prefix="test")
        
        # 8. Generazione Report e Immagine
        df_report, report_csv_path = trainer.generate_report(test_loader, tokenizer, prefix="test")
        
        if df_report is not None and report_csv_path is not None:
            output_png_path = os.path.join(chart_dir, "test_samples_visualization.png")
            success = generate_visualization(report_csv_path, output_png_path, num_samples=5)
            if success and mlflow.active_run():
                mlflow.log_artifact(output_png_path)

if __name__ == "__main__":
    main()