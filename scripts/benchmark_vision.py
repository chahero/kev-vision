"""Local-only latency probe; prepared PIL input to CPU probability output."""
import json, math, statistics, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from kev_runtime import engine_options
from experiment_vision import OUT
from prepare_kev_vllm import load_pinned_kev
from PIL import Image
from transformers import AutoTokenizer
from vllm import LLM,PoolingParams


def stats(values):
    ordered=sorted(values)
    return {'samples':len(values),'p50_ms':statistics.median(values)*1000,'p95_ms':ordered[math.ceil(.95*len(ordered))-1]*1000,'min_ms':min(values)*1000,'max_ms':max(values)*1000}


def main():
    load_pinned_kev();km=sys.modules['pinned_kev.model']
    tok=AutoTokenizer.from_pretrained(OUT/'merged-vision',local_files_only=True)
    case={'state':'Inspect the supplied image.','questions':[{'instr':'What color is the large shape?','options':['Red','Green','Blue'],'label':0}]}
    ids=km.encode(tok,case,max_state=1024,max_branch=2048,strict=True)['ids']
    marker=[tok.convert_tokens_to_ids(t) for t in ('<|vision_start|>','<|image_pad|>','<|vision_end|>')]
    image_ids=ids[:1]+marker+ids[1:]
    options=engine_options(); options.update(model=str(OUT/'merged-vision'),limit_mm_per_prompt={'image':1,'video':0},mm_processor_cache_gb=0,mm_processor_kwargs={'min_pixels':65536,'max_pixels':589824})
    start=time.perf_counter(); model=LLM(**options);startup=time.perf_counter()-start
    serial=0
    source=Image.open(OUT/'images/color-red.png').convert('RGB')
    def prompt(size):
        nonlocal serial
        serial+=1
        if size==0:return {'prompt_token_ids':ids}
        im=source.resize((size,size))
        # A different RGB corner pixel makes each image's content hash unique,
        # so repeated-image encoder reuse cannot make the measurement look faster.
        im.putpixel((0,0),(serial%256,(serial//256)%256,0))
        return {'prompt_token_ids':image_ids,'multi_modal_data':{'image':im}}
    def run(prompts):
        start=time.perf_counter()
        outputs=model.encode(prompts,pooling_task='token_embed',pooling_params=PoolingParams(task='token_embed'),use_tqdm=False)
        probabilities=[out.outputs.data.detach().cpu().flatten().tolist() for out in outputs]
        elapsed=time.perf_counter()-start
        assert all(len(p)==3 and all(math.isfinite(v) for v in p) for p in probabilities)
        return elapsed, [len(out.prompt_token_ids) for out in outputs]
    reports=[]
    for size in (0,256,512,768):
        for _ in range(5):run([prompt(size)])
        times=[];lengths=[]
        for _ in range(30):
            elapsed,lens=run([prompt(size)]);times.append(elapsed);lengths+=lens
        for _ in range(3):run([prompt(size),prompt(size)])
        batches=[]
        for _ in range(20):
            elapsed,_=run([prompt(size),prompt(size)]);batches.append(elapsed)
        row={'resolution':size,'tokens':sorted(set(lengths)),'single':stats(times),'batch2':stats(batches),'batch2_items_per_second':40/sum(batches),'raw_single_seconds':times,'raw_batch_seconds':batches}
        reports.append(row);print(json.dumps(row),flush=True)
    report={'startup_seconds':startup,'device':'NVIDIA GB10 Linux aarch64','dtype':'BF16','vision_attention':'default FLASH_ATTN','context_limit':2048,'max_num_seqs':2,'mm_processor_cache_gb':0,'unique_images':True,'timing_boundary':'Prepared PIL images and token IDs through LLM.encode to CPU probabilities; excludes file decoding, Kev text encoding, HTTP, model startup and warmup','warmup_single':5,'warmup_batch':3,'rows':reports,'limitations':'Single local run with other services resident; synthetic images; no service SLA or production throughput claim'}
    (OUT/'latency-result.json').write_text(json.dumps(report,indent=2)+'\n')
    print('BENCHMARK COMPLETE',flush=True)


if __name__=='__main__':main()
