# src/training/trainer.py
import os
import torch
import mlflow
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix

log = logging.getLogger(__name__)

class Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer, device, chart_dir):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.chart_dir = chart_dir

    def _decode_preds(self, emissions, mask):
        """Metodo helper per decodificare correttamente con o senza CRF"""
        if self.model.use_crf:
            return self.model.crf.decode(emissions, mask=mask)
        else:
            raw_preds = torch.argmax(emissions, dim=-1)
            preds = []
            for i in range(raw_preds.size(0)):
                seq_len = mask[i].sum().item()
                preds.append(raw_preds[i][:seq_len].cpu().numpy().tolist())
            return preds

    def fit(self, epochs, es_cfg, best_model_path="best_model.pth"):
        """Gestisce il loop di training e validazione con Early Stopping."""
        monitor_metric = es_cfg.get("monitor", "f1")
        mode = es_cfg.get("mode", "max")
        patience = es_cfg.get("patience", 3)
        min_delta = es_cfg.get("min_delta", 0.001)

        best_metric_val = -float("inf") if mode == "max" else float("inf")
        patience_counter = 0

        torch.save(self.model.state_dict(), best_model_path)

        for epoch in range(1, epochs + 1):
            # Fase di Addestramento
            self.train_epoch(epoch)
            
            # Fase di Validazione
            metrics = self.evaluate(self.val_loader, epoch=epoch, prefix="val")
            current_val = metrics.get(monitor_metric)

            # Controllo Early Stopping
            is_best = False
            if mode == "max" and current_val > (best_metric_val + min_delta):
                is_best = True
            elif mode == "min" and current_val < (best_metric_val - min_delta):
                is_best = True

            if is_best:
                best_metric_val = current_val
                torch.save(self.model.state_dict(), best_model_path)
                log.info(f"⭐ Nuovo modello migliore salvato! ({monitor_metric}: {best_metric_val:.4f})")
                patience_counter = 0  
            else:
                patience_counter += 1
                log.info(f"⚠️ Nessun miglioramento significativo per '{monitor_metric}'. Patience: {patience_counter}/{patience}")

            if patience_counter >= patience:
                log.info(f"🛑 Early stopping innescato all'epoca {epoch}! Interruzione addestramento.")
                break

    def _save_confusion_matrix(self, true_labels, pred_labels, prefix, epoch=None):
        """Salva e carica su MLflow la matrice di confusione."""
        cm = confusion_matrix(true_labels, pred_labels, labels=[0, 1])
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Human', 'AI'], yticklabels=['Human', 'AI'])
        plt.ylabel('Vero Label')
        plt.xlabel('Label Predetto')
        
        plt.title(f'Confusion Matrix - {prefix.upper().replace("_", " ")}')
        
        cm_filename = f"confusion_matrix_{prefix}.png"
        cm_path = os.path.join(self.chart_dir, cm_filename)
        
        plt.savefig(cm_path)
        plt.close()
        
        if mlflow.active_run():
            mlflow.log_artifact(cm_path)

    def _evaluate_chunk_level(self, true_seqs, pred_seqs, prefix="test", epoch=None):
        """Valuta le metriche a livello di CHUNK usando il Majority Voting."""
        chunk_true = []
        chunk_pred = []

        for t_seq, p_seq in zip(true_seqs, pred_seqs):
            if not t_seq:
                continue
                
            current_chunk_true_label = t_seq[0]
            current_chunk_preds = []

            for t, p in zip(t_seq, p_seq):
                if t == current_chunk_true_label:
                    current_chunk_preds.append(p)
                else:
                    chunk_true.append(current_chunk_true_label)
                    maj_vote = 1 if sum(current_chunk_preds) > len(current_chunk_preds) / 2 else 0
                    chunk_pred.append(maj_vote)

                    current_chunk_true_label = t
                    current_chunk_preds = [p]

            if current_chunk_preds:
                chunk_true.append(current_chunk_true_label)
                maj_vote = 1 if sum(current_chunk_preds) > len(current_chunk_preds) / 2 else 0
                chunk_pred.append(maj_vote)

        acc = accuracy_score(chunk_true, chunk_pred)
        prec = precision_score(chunk_true, chunk_pred, average='macro', zero_division=0)
        rec = recall_score(chunk_true, chunk_pred, average='macro', zero_division=0)
        f1 = f1_score(chunk_true, chunk_pred, average='macro', zero_division=0)

        log.info(f"CHUNK-LEVEL {prefix.upper()} | Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1 Macro: {f1:.4f}")

        if mlflow.active_run():
            metrics_to_log = {
                f"{prefix}_chunk_acc": acc,
                f"{prefix}_chunk_precision": prec,
                f"{prefix}_chunk_recall": rec,
                f"{prefix}_chunk_f1": f1
            }
            mlflow.log_metrics(metrics_to_log, step=epoch if epoch is not None else 0)

        self._save_confusion_matrix(chunk_true, chunk_pred, prefix=f"{prefix}_chunk", epoch=epoch)

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        all_preds, all_labels = [], []
        pbar = tqdm(self.train_loader, desc=f"Train Epoch {epoch}")
        
        for batch in pbar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(input_ids, attention_mask, labels=labels)
            loss = outputs['loss']
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
            with torch.no_grad():
                preds = self._decode_preds(outputs['emissions'].detach(), attention_mask.bool())                
                for i in range(len(preds)):
                    seq_len = len(preds[i])
                    all_preds.extend(preds[i])
                    all_labels.extend(labels[i][:seq_len].cpu().numpy().tolist())
            
        avg_loss = total_loss / len(self.train_loader)
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

        log.info(f"Epoch {epoch} | TRAIN | Loss: {avg_loss:.4f} | Acc: {acc:.4f} | F1: {f1:.4f}")
        
        if mlflow.active_run():
            mlflow.log_metric("train_loss", avg_loss, step=epoch)
            mlflow.log_metric("train_acc", acc, step=epoch)
            mlflow.log_metric("train_f1", f1, step=epoch)

    def evaluate(self, loader, epoch=None, prefix="val"):
        self.model.eval()
        total_loss = 0
        all_preds, all_labels = [], []
        
        all_true_seqs = []
        all_pred_seqs = []

        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Evaluating ({prefix})"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(input_ids, attention_mask, labels=labels)
                loss = outputs['loss']
                total_loss += loss.item()

                preds = self._decode_preds(outputs['emissions'], attention_mask.bool())
                
                for i in range(len(preds)):
                    seq_len = len(preds[i])
                    t_seq = labels[i][:seq_len].cpu().numpy().tolist()
                    p_seq = preds[i]
                    
                    all_true_seqs.append(t_seq)
                    all_pred_seqs.append(p_seq)
                    
                    all_preds.extend(p_seq)
                    all_labels.extend(t_seq)

        avg_loss = total_loss / len(loader)
        
        acc = accuracy_score(all_labels, all_preds)
        prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

        log_msg = f"TOKEN-LEVEL {prefix.upper()} | Loss: {avg_loss:.4f} | Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f}"
        if epoch is not None:
            log_msg = f"Epoch {epoch} | " + log_msg
            
        log.info(log_msg)

        if mlflow.active_run():
            metrics_to_log = {
                f"{prefix}_loss": avg_loss,
                f"{prefix}_token_acc": acc,
                f"{prefix}_token_precision": prec,
                f"{prefix}_token_recall": rec,
                f"{prefix}_token_f1": f1
            }
            # Usa 0 se epoch è None (es. durante il test finale)
            mlflow.log_metrics(metrics_to_log, step=epoch if epoch is not None else 0)
            
        if prefix == "test":
            self._save_confusion_matrix(all_labels, all_preds, prefix=f"{prefix}_token", epoch=epoch)
            self._evaluate_chunk_level(all_true_seqs, all_pred_seqs, prefix=prefix, epoch=epoch)
            
        return {"loss": avg_loss, "acc": acc, "f1": f1}

    def evaluate_boundaries(self, loader, tokenizer, prefix="test"):
        self.model.eval()
        log.info(f"🔎 Valutazione Boundaries e Acc by Length su {prefix.upper()}...")
        
        bins = list(range(0, 110, 10))
        bin_names = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins)-1)]
        acc_by_len = {b: {"correct": 0, "total": 0} for b in bin_names}

        real_h2a, real_a2h = 0, 0
        pred_h2a, pred_a2h = 0, 0
        
        dist_correct_h2a, dist_correct_a2h = [], []
        dist_any_boundary = []
        
        num_docs = 0

        def get_boundaries(seq):
            h_a, a_h = [], []
            for j in range(1, len(seq)):
                if seq[j-1] == 0 and seq[j] == 1: h_a.append(j)
                elif seq[j-1] == 1 and seq[j] == 0: a_h.append(j)
            return h_a, a_h

        def get_min_dist(true_idx, pred_indices):
            if not pred_indices: return None
            return min(abs(true_idx - p) for p in pred_indices)

        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                preds = outputs['predictions']
                num_docs += len(preds)

                for i in range(len(preds)):
                    seq_len = len(preds[i])
                    true_seq = labels[i][:seq_len].cpu().numpy().tolist()
                    pred_seq = preds[i]

                    t_h2a, t_a2h = get_boundaries(true_seq)
                    p_h2a, p_a2h = get_boundaries(pred_seq)
                    p_any = p_h2a + p_a2h  
                    
                    real_h2a += len(t_h2a); real_a2h += len(t_a2h)
                    pred_h2a += len(p_h2a); pred_a2h += len(p_a2h)
                    
                    for t in t_h2a:
                        dist_corr = get_min_dist(t, p_h2a)
                        if dist_corr is not None: dist_correct_h2a.append(dist_corr)
                        
                        dist_any = get_min_dist(t, p_any)
                        if dist_any is not None: dist_any_boundary.append(dist_any)

                    for t in t_a2h:
                        dist_corr = get_min_dist(t, p_a2h)
                        if dist_corr is not None: dist_correct_a2h.append(dist_corr)
                        
                        dist_any = get_min_dist(t, p_any)
                        if dist_any is not None: dist_any_boundary.append(dist_any)
                        
                    chunk_start = 0
                    for j in range(1, len(true_seq) + 1):
                        if j == len(true_seq) or true_seq[j] != true_seq[j-1]:
                            c_len = j - chunk_start
                            c_true = true_seq[chunk_start:j]
                            c_pred = pred_seq[chunk_start:j]
                            
                            c_corr = sum(1 for t, p in zip(c_true, c_pred) if t == p)
                            
                            plot_len = min(c_len, 99)
                            b_idx = min(plot_len // 10, len(bins) - 2)
                            b_name = bin_names[b_idx]
                            
                            acc_by_len[b_name]["correct"] += c_corr
                            acc_by_len[b_name]["total"] += c_len
                            
                            chunk_start = j

        avg_real_total = (real_h2a + real_a2h) / num_docs if num_docs > 0 else 0
        avg_pred_total = (pred_h2a + pred_a2h) / num_docs if num_docs > 0 else 0
        avg_real_h2a = real_h2a / num_docs if num_docs > 0 else 0
        avg_pred_h2a = pred_h2a / num_docs if num_docs > 0 else 0
        avg_real_a2h = real_a2h / num_docs if num_docs > 0 else 0
        avg_pred_a2h = pred_a2h / num_docs if num_docs > 0 else 0

        labels_chart = ['Total Bounds', 'H -> AI', 'AI -> H']
        real_means = [avg_real_total, avg_real_h2a, avg_real_a2h]
        pred_means = [avg_pred_total, avg_pred_h2a, avg_pred_a2h]

        x = np.arange(len(labels_chart))
        width = 0.35

        fig, ax = plt.subplots(figsize=(8, 6))
        rects1 = ax.bar(x - width/2, real_means, width, label='Real', color='blue')
        rects2 = ax.bar(x + width/2, pred_means, width, label='Predicted', color='orange')

        ax.set_ylabel('Average count per Document')
        ax.set_title(f'Average Boundaries per Document - {prefix.upper()}')
        ax.set_xticks(x)
        ax.set_xticklabels(labels_chart)
        ax.legend()
        ax.bar_label(rects1, padding=3, fmt='%.2f')
        ax.bar_label(rects2, padding=3, fmt='%.2f')

        fig.tight_layout()
        chart_path = os.path.join(self.chart_dir, f"avg_boundaries_{prefix}.png")
        plt.savefig(chart_path)
        plt.close()

        mean_dist_correct_h2a = np.mean(dist_correct_h2a) if dist_correct_h2a else 0
        mean_dist_correct_a2h = np.mean(dist_correct_a2h) if dist_correct_a2h else 0
        mean_dist_any = np.mean(dist_any_boundary) if dist_any_boundary else 0

        labels_dist = ['Nearest Correct\n(H -> AI)', 'Nearest Correct\n(AI -> H)', 'Nearest Any\nBoundary']
        dist_means = [mean_dist_correct_h2a, mean_dist_correct_a2h, mean_dist_any]

        fig, ax = plt.subplots(figsize=(8, 6))
        bars = ax.bar(labels_dist, dist_means, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        
        ax.set_ylabel('Average Distance (Tokens)')
        ax.set_title(f'Average Distance to Boundaries - {prefix.upper()}')
        ax.bar_label(bars, padding=3, fmt='%.2f')

        fig.tight_layout()
        dist_chart_path = os.path.join(self.chart_dir, f"avg_boundary_distances_{prefix}.png")
        plt.savefig(dist_chart_path)
        plt.close()

        acc_vals = []
        for b in bin_names:
            tot = acc_by_len[b]["total"]
            if tot > 0:
                acc_vals.append(acc_by_len[b]["correct"] / tot)
            else:
                acc_vals.append(0.0)

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(bin_names, acc_vals, color='mediumpurple', edgecolor='black')
        ax.set_ylabel('Token-level Accuracy')
        ax.set_xlabel('Chunk Length (Tokens)')
        ax.set_title(f'Token Accuracy by Chunk Length - {prefix.upper()}')
        ax.set_ylim(0, 1.1)
        ax.bar_label(bars, padding=3, fmt='%.2f')
        plt.xticks(rotation=45)
        
        fig.tight_layout()
        acc_chart_path = os.path.join(self.chart_dir, f"accuracy_by_length_{prefix}.png")
        plt.savefig(acc_chart_path)
        plt.close()

        if mlflow.active_run():
            mlflow.log_artifact(chart_path)
            mlflow.log_artifact(dist_chart_path)
            mlflow.log_artifact(acc_chart_path)
            
            mlflow.log_metrics({
                "real_h2a_boundaries": real_h2a, "pred_h2a_boundaries": pred_h2a,
                "real_a2h_boundaries": real_a2h, "pred_a2h_boundaries": pred_a2h,
                "mean_nearest_correct_dist_h2a": mean_dist_correct_h2a,
                "mean_nearest_correct_dist_a2h": mean_dist_correct_a2h,
                "mean_nearest_any_boundary_dist": mean_dist_any,
                f"avg_real_total_bounds_{prefix}": avg_real_total,
                f"avg_pred_total_bounds_{prefix}": avg_pred_total
            })
            
            for b, a in zip(bin_names, acc_vals):
                if acc_by_len[b]["total"] > 0:
                    mlflow.log_metric(f"acc_len_{b}", a)

        log.info("\n🚧 Statistiche Boundaries (Tokens):")
        log.info(f"   Media Boundaries per doc: Real={avg_real_total:.2f}, Pred={avg_pred_total:.2f}")
        log.info(f"   Human->AI: Real={real_h2a}, Pred={pred_h2a} | Nearest Correct Dist = {mean_dist_correct_h2a:.2f}")
        log.info(f"   AI->Human: Real={real_a2h}, Pred={pred_a2h} | Nearest Correct Dist = {mean_dist_correct_a2h:.2f}")
        log.info(f"   Distanza Media Nearest Any Boundary = {mean_dist_any:.2f}\n")
        
    def generate_report(self, loader, tokenizer, prefix="test"):
        self.model.eval()
        report_data = []

        log.info(f"📝 Generazione debug report testuale per: {prefix}")
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(loader, desc=f"Reporting ({prefix})")):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(input_ids, attention_mask, labels=labels)
                preds = self._decode_preds(outputs['emissions'], attention_mask.bool())
                
                for i in range(len(preds)):
                    seq_len = len(preds[i])
                    true_labels = labels[i][:seq_len].cpu().numpy().tolist()
                    pred_labels = preds[i]
                    
                    token_ids = input_ids[i][:seq_len].cpu().numpy().tolist()
                    tokens = tokenizer.convert_ids_to_tokens(token_ids)
                    
                    doc_id = (batch_idx * loader.batch_size) + i
                    for tok, t_lbl, p_lbl in zip(tokens, true_labels, pred_labels):
                        report_data.append({
                            "doc_id": doc_id,
                            "token": tok,
                            "true_label": t_lbl,
                            "pred_label": p_lbl
                        })

        if report_data:
            df_report = pd.DataFrame(report_data)
            csv_path = os.path.join(self.chart_dir, f"{prefix}_predictions_report.csv")
            df_report.to_csv(csv_path, index=False)
            
            if mlflow.active_run():
                mlflow.log_artifact(csv_path)
            return df_report, csv_path
        return None, None