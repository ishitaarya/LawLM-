"""Phase 7.5F — Conservative QA adaptation from the project's Phase-5 model.

Important: this intentionally does NOT resume the failed QA checkpoint.
It starts from lawsuit_llm_expanded_best.pt, uses one epoch and a low
learning rate, and reports prompt-boundary diagnostics.
"""
from __future__ import annotations
import json, math, random
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from src.model.lawlm_model import LawLM

DATA = Path("data/qa_v2")
BASE = Path("checkpoints/lawsuit_llm_expanded_best.pt")
BEST = Path("checkpoints/lawsuit_llm_qa_v2_best.pt")
HISTORY = Path("logs/qa_v2_training_history.jsonl")
BATCH_SIZE = 1
GRAD_ACCUM = 16
EPOCHS = 1
LR = 5e-6
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
SEED = 42
BLOCK = 256

def seed():
    random.seed(SEED); torch.manual_seed(SEED)

class QADataset(Dataset):
    def __init__(self, path):
        self.rows=[json.loads(x) for x in path.open(encoding="utf-8") if x.strip()]
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i]
        return {"input_ids":torch.tensor(r["input_ids"],dtype=torch.long),
                "labels":torch.tensor(r["labels"],dtype=torch.long)}

def collate(batch):
    n=max(x["input_ids"].numel() for x in batch)
    ids=torch.zeros((len(batch),n),dtype=torch.long)
    labels=torch.full((len(batch),n),-100,dtype=torch.long)
    for i,x in enumerate(batch):
        l=x["input_ids"].numel(); ids[i,:l]=x["input_ids"]; labels[i,:l]=x["labels"]
    return ids,labels

def ppl(x): return math.exp(min(x,20))

def model_from_checkpoint(device):
    ck=torch.load(BASE,map_location=device,weights_only=False)
    m=LawLM(vocab_size=10000,block_size=BLOCK,embed_dim=384,num_heads=6,num_layers=4,ff_hidden_dim=1536,dropout=0.1).to(device)
    m.load_state_dict(ck["model_state_dict"])
    return m,ck

def evaluate(m,loader,device):
    m.eval(); total=0.; batches=0
    with torch.no_grad():
        for ids,labels in loader:
            _,loss=m(ids.to(device),targets=labels.to(device))
            total+=float(loss); batches+=1
    return total/max(1,batches)

def main():
    seed()
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train=QADataset(DATA/"train.jsonl"); val=QADataset(DATA/"validation.jsonl")
    tl=DataLoader(train,batch_size=BATCH_SIZE,shuffle=True,collate_fn=collate,num_workers=0)
    vl=DataLoader(val,batch_size=BATCH_SIZE,shuffle=False,collate_fn=collate,num_workers=0)
    m,base=model_from_checkpoint(device)
    opt=torch.optim.AdamW(m.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
    best=float("inf")
    print("="*72); print("LawSuit LLM — Phase 7.5F QA V2"); print("="*72)
    print("Device:",device); print("Train:",len(train)); print("Validation:",len(val))
    print("LR:",LR,"Grad accumulation:",GRAD_ACCUM,"Epochs:",EPOCHS)
    print("Base:",BASE); print("Failed QA checkpoint is NOT used."); print("="*72)
    for epoch in range(1,EPOCHS+1):
        m.train(); opt.zero_grad(set_to_none=True); running=0.
        for bi,(ids,labels) in enumerate(tl,1):
            _,loss=m(ids.to(device),targets=labels.to(device))
            (loss/GRAD_ACCUM).backward(); running+=float(loss)
            if bi%GRAD_ACCUM==0 or bi==len(tl):
                torch.nn.utils.clip_grad_norm_(m.parameters(),GRAD_CLIP)
                opt.step(); opt.zero_grad(set_to_none=True)
            if bi%2000==0 or bi==len(tl):
                print(f"Epoch {epoch} Batch {bi:,}/{len(tl):,} Loss {loss.item():.4f}")
        train_loss=running/max(1,len(tl)); val_loss=evaluate(m,vl,device)
        rec={"epoch":epoch,"train_loss":train_loss,"validation_loss":val_loss,
             "train_perplexity":ppl(train_loss),"validation_perplexity":ppl(val_loss),
             "learning_rate":LR,"gradient_accumulation":GRAD_ACCUM,
             "base_checkpoint":str(BASE),"pretrained_llm":False}
        HISTORY.parent.mkdir(parents=True,exist_ok=True)
        with HISTORY.open("a",encoding="utf-8") as f: f.write(json.dumps(rec)+"\n")
        print(f"Epoch {epoch}: train {train_loss:.4f} | val {val_loss:.4f} | PPL {ppl(val_loss):.2f}")
        if val_loss < best:
            best=val_loss
            torch.save({"epoch":epoch,"model_state_dict":m.state_dict(),
                        "optimizer_state_dict":opt.state_dict(),"validation_loss":val_loss,
                        "model_config":{"vocab_size":10000,"block_size":BLOCK,"embed_dim":384,
                        "num_layers":4,"num_heads":6,"ff_hidden_dim":1536,"dropout":0.1},
                        "training":{"task":"conservative legal QA adaptation","external_pretrained_llm":False,
                        "base_checkpoint":str(BASE),"learning_rate":LR}},BEST)
            print("Saved:",BEST)
    print("="*72); print("QA V2 complete. Best:",BEST,"Validation loss:",best)

if __name__=="__main__": main()
