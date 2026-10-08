"""Phase 7.5G — QA V2 evaluation with token metrics and free generation."""
from __future__ import annotations
import argparse,json,math
from pathlib import Path
import sentencepiece as spm
import torch
from src.model.lawlm_model import LawLM
from src.training.train_qa_v2 import QADataset,collate

CKPT=Path("checkpoints/lawsuit_llm_qa_v2_best.pt")
TEST=Path("data/qa_v2/test.jsonl")
TOK=Path("data/tokenizer/lawsuit_bpe.model")
EOS=3; BLOCK=256

def ppl(x): return math.exp(min(x,20))
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--checkpoint",default=str(CKPT)); ap.add_argument("--test",default=str(TEST))
    ap.add_argument("--tokenizer",default=str(TOK)); ap.add_argument("--count",type=int,default=10)
    ap.add_argument("--max-new-tokens",type=int,default=80); ap.add_argument("--temperature",type=float,default=0.7)
    ap.add_argument("--top-k",type=int,default=40); ap.add_argument("--output",default="logs/qa_v2_evaluation.json")
    a=ap.parse_args(); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck=torch.load(a.checkpoint,map_location=device,weights_only=False); c=ck["model_config"]
    m=LawLM(vocab_size=c["vocab_size"],block_size=c["block_size"],embed_dim=c["embed_dim"],
            num_heads=c["num_heads"],num_layers=c["num_layers"],ff_hidden_dim=c["ff_hidden_dim"],dropout=c["dropout"]).to(device)
    m.load_state_dict(ck["model_state_dict"]); m.eval()
    ds=QADataset(Path(a.test)); sp=spm.SentencePieceProcessor(model_file=a.tokenizer)
    total=0.; n=0
    with torch.no_grad():
        for i in range(len(ds)):
            ids=torch.tensor([ds[i]["input_ids"]],dtype=torch.long,device=device)
            labels=torch.tensor([ds[i]["labels"]],dtype=torch.long,device=device)
            _,loss=m(ids,targets=labels); total+=float(loss); n+=1
    loss=total/max(1,n)
    samples=[]
    for i in range(min(a.count,len(ds))):
        r=ds.rows[i]; prompt=r["input_ids"][:next(j for j,x in enumerate(r["labels"]) if x!=-100)]
        x=torch.tensor([prompt],dtype=torch.long,device=device)
        y=m.generate(x,max_new_tokens=a.max_new_tokens,temperature=a.temperature,top_k=a.top_k,repetition_penalty=1.15,eos_token_id=EOS)
        new=y[0].tolist()[len(prompt):]
        samples.append({"index":i,"question":r["question"],"expected":r["answer"],
                        "generated":sp.decode(new).strip(),"generated_ids":new,
                        "first_generated_id":new[0] if new else None})
    report={"phase":"7.5G","checkpoint":str(a.checkpoint),"test_examples":len(ds),
            "test_loss":loss,"test_perplexity":ppl(loss),"samples":samples,
            "pretrained_llm":False}
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    Path(a.output).write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print("="*72); print("QA V2 EVALUATION"); print("="*72)
    print("Test:",len(ds),"Loss:",round(loss,4),"PPL:",round(ppl(loss),2))
    for s in samples:
        print(f"\n#{s['index']} Section {s['question']}"); print("Expected:",s["expected"]); print("Generated:",s["generated"])
    print("Report:",a.output)

if __name__=="__main__": main()
