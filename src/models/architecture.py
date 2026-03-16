import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from torchcrf import CRF
import logging

log = logging.getLogger(__name__)

class DebertaCRFBoundaryDetector(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.num_labels = cfg.model.get("num_labels", 2)
        self.crf_reduction = cfg.model.get("crf_reduction", "token_mean")
        
        self.use_crf = cfg.model.get("use_crf", True)

        # Loss Weights
        loss_cfg = cfg.model.get("loss_weights", {})
        self.alpha_crf = loss_cfg.get("alpha_crf", 1.0)
        self.alpha_ce = loss_cfg.get("alpha_ce", 1.0)
        self.alpha_transition = loss_cfg.get("alpha_transition", 0.5)
        
        # Custom CE Parameters
        ce_cfg = cfg.model.get("custom_ce", {})
        self.short_chunk_alpha = ce_cfg.get("short_chunk_alpha", 2.0)

        log.info(f"Inizializzazione DeBERTaV3 + CRF (num_labels={self.num_labels})")
        log.info(f"Loss weights -> CRF: {self.alpha_crf} | CE (Short Chunks): {self.alpha_ce} | Transition Penalty: {self.alpha_transition}")

        config = AutoConfig.from_pretrained(cfg.model.backbone.path)
        self.backbone = AutoModel.from_pretrained(
            cfg.model.backbone.path,
            config=config,
            ignore_mismatched_sizes=True,
            torch_dtype=torch.float32 
        )

        if cfg.model.backbone.get("freeze", False):
            for param in self.backbone.embeddings.parameters():
                param.requires_grad = False
            for layer in self.backbone.encoder.layer[:6]:
                for param in layer.parameters():
                    param.requires_grad = False

        self.dropout = nn.Dropout(cfg.model.head.get("dropout", 0.1))
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.crf = CRF(self.num_labels, batch_first=True)

    def _compute_transition_penalty(self, emissions, mask):
        """
        Penalizza i cambiamenti rapidi nelle probabilità predette tra token adiacenti.
        Forza il modello a produrre chunk meno frammentati e più lunghi.
        """
        # Convertiamo i logits in probabilità
        probs = F.softmax(emissions, dim=-1) # [B, SeqLen, NumClasses]
        
        # Calcoliamo la differenza assoluta tra il token T e il token T+1
        diff = torch.abs(probs[:, 1:, :] - probs[:, :-1, :]) # [B, SeqLen-1, NumClasses]
        
        # Consideriamo solo le transizioni valide (non il padding)
        valid_mask = mask[:, 1:] & mask[:, :-1] # [B, SeqLen-1]
        
        # Sommiamo la differenza, applichiamo la maschera e facciamo la media
        penalty = (diff.sum(dim=-1) * valid_mask.float()).sum() / (valid_mask.float().sum() + 1e-8)
        return penalty

    def _compute_custom_ce(self, emissions, labels, mask):
        """
        Calcola la Cross Entropy a livello di token dando maggior peso ai chunk corti
        in base alla loro lunghezza (inversamente proporzionale).
        """
        batch_size, seq_len, num_labels = emissions.shape
        
        ce_loss = F.cross_entropy(
            emissions.view(-1, num_labels), 
            labels.view(-1), 
            reduction='none'
        ).view(batch_size, seq_len)
        
        weights = torch.ones_like(labels, dtype=torch.float32)
        
        for i in range(batch_size):
            seq_mask = mask[i].bool()
            seq_labels = labels[i][seq_mask]
            actual_len = seq_labels.size(0)
            
            if actual_len == 0:
                continue
                
            chunk_start = 0
            for j in range(1, actual_len + 1):
                # Se label cambia o siamo a fine documento, chiudiamo il chunk
                if j == actual_len or seq_labels[j] != seq_labels[j-1]:
                    chunk_len = j - chunk_start
                    
                    # Inverse Length Weighting: I chunk corti ricevono boost maggiore
                    len_weight = 1.0 + self.short_chunk_alpha * (1.0 / (chunk_len ** 0.5))
                    weights[i, chunk_start:j] = len_weight
                    chunk_start = j
                
        weighted_loss = ce_loss * weights * mask.float()
        return weighted_loss.sum() / (mask.float().sum() + 1e-8)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        emissions = self.classifier(sequence_output)

        mask = attention_mask.bool()

        if labels is not None:
            loss = 0.0
            
            # 1. Standard CRF Loss (Sequence likelihood)
            if self.alpha_crf > 0:
                crf_nll = -self.crf(emissions, labels, mask=mask, reduction=self.crf_reduction)
                loss += self.alpha_crf * crf_nll
                
            # 2. Short Chunk Weighted Cross Entropy
            if self.alpha_ce > 0:
                custom_ce = self._compute_custom_ce(emissions, labels, mask)
                loss += self.alpha_ce * custom_ce
                
            # 3. Transition Penalty (Smoothing) per ridurre le transizioni frammentate
            if self.alpha_transition > 0:
                tv_penalty = self._compute_transition_penalty(emissions, mask)
                loss += self.alpha_transition * tv_penalty
                
            return {"loss": loss, "emissions": emissions}
        else:
            if self.use_crf:
                preds = self.crf.decode(emissions, mask=mask)
            else:
                # Senza CRF, prendiamo semplicemente l'argmax delle emissioni e applichiamo la maschera
                raw_preds = torch.argmax(emissions, dim=-1)
                preds = []
                for i in range(raw_preds.size(0)):
                    seq_len = mask[i].sum().item()
                    preds.append(raw_preds[i][:seq_len].cpu().numpy().tolist())

            return {"predictions": preds, "emissions": emissions}
    def get_tokenizer(self):
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(
            self.cfg.model.backbone.path,
            use_fast=True
        )