"""Compare native vLLM image pooling with the no-training Transformers reference."""
import argparse
import json
import sys
from pathlib import Path
from kev_runtime import AUDIT, engine_options
from experiment_vision import OUT, encode_inputs
from prepare_kev_vllm import load_pinned_kev


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vision-attention',choices=['FLASH_ATTN','TORCH_SDPA'])
    parser.add_argument('--vision-precision', choices=['native', 'reference'], default='native')
    parser.add_argument('--all-singletons', action='store_true', help='Also compare every singleton and reversed batch')
    parser.add_argument('--benchmark', action='store_true', help='Warm 256px CAT latency with unique images, excluding image preparation')
    parser.add_argument('--output', type=Path, help='Write report to a separate path without replacing an earlier run')
    args=parser.parse_args()
    import torch
    from PIL import Image
    from transformers import AutoTokenizer, AutoImageProcessor
    from vllm import LLM, PoolingParams
    load_pinned_kev(); km=sys.modules['pinned_kev.model']
    tok=AutoTokenizer.from_pretrained(OUT/'merged-vision',local_files_only=True)
    processor=AutoImageProcessor.from_pretrained(OUT/'merged-vision',local_files_only=True)
    reference=json.loads((OUT/'reference-result.json').read_text())
    options=engine_options()
    options.update(model=str(OUT/'merged-vision'),limit_mm_per_prompt={'image':1,'video':0},
                   mm_processor_kwargs={'min_pixels':65536,'max_pixels':65536})
    if args.vision_attention:
        options['mm_encoder_attn_backend']=args.vision_attention
    if args.vision_precision != 'native':
        options['hf_overrides'] = {'kev_vision_precision': args.vision_precision}
    if args.benchmark:
        options['mm_processor_cache_gb'] = 0
    model=LLM(**options)
    prompts=[];refs=[]
    for case in reference['rows']:
        record={'state':case['state'],'questions':[{'instr':case['question'],'options':case['options'],'label':0}]}
        ids=km.encode(tok,record,max_state=1024,max_branch=2048,strict=True)['ids']
        for condition in ('image','blank','no_image'):
            prompt={'prompt_token_ids':ids}
            image=None
            if condition!='no_image':
                path=OUT/'images'/(case['image'] if condition=='image' else 'blank.png')
                marker=[tok.convert_tokens_to_ids(t) for t in ('<|vision_start|>','<|image_pad|>','<|vision_end|>')]
                image=Image.open(path).convert('RGB')
                prompt={'prompt_token_ids':ids[:1]+marker+ids[1:], 'multi_modal_data':{'image':image}}
            prompts.append(prompt)
            refs.append({'name':case['name'],'condition':condition,'expected':case['outputs'][condition]['probabilities'],'label':case['options'].index(case['expected']), 'input_ids':encode_inputs(km,tok,processor,case,image)['input_ids'][0].tolist()})
    text_path=AUDIT/'expanded-reference.json'
    if not text_path.exists(): text_path=AUDIT/'reference.json'
    text=json.loads(text_path.read_text())
    for i,row in enumerate(text['rows']):
        prompts.append({'prompt_token_ids':row['input_ids']})
        refs.append({'name':'text-'+str(i),'condition':'text_regression','expected':row['probabilities'],'label':row.get('expected_label',0),'input_ids':row['input_ids']})
    def encode(items):
        return model.encode(items, pooling_task='token_embed',
                            pooling_params=PoolingParams(task='token_embed'), use_tqdm=False)

    def check(ref, out):
        assert out.prompt_token_ids == ref['input_ids'], ref['name']
        p=out.outputs.data.detach().cpu()
        assert tuple(p.shape)==(len(ref['expected']),1)
        p=p.flatten(); expected=torch.tensor(ref['expected'])
        assert torch.isfinite(p).all() and (p>=0).all() and (p<=1).all() and abs(float(p.sum())-1)<1e-5
        return {**{k:v for k,v in ref.items() if k!='input_ids'},
                'probabilities':p.tolist(),'max_error':float((p-expected).abs().max()),
                'same_argmax':int(p.argmax())==int(expected.argmax()),'correct':int(p.argmax())==ref['label']}

    cat_index=next(i for i,r in enumerate(refs) if r['name']=='word-cat' and r['condition']=='image') if any(r['name']=='word-cat' for r in refs) else 0
    first_cat = check(refs[cat_index], encode([prompts[cat_index]])[0])
    print('CAT FIRST', json.dumps(first_cat), flush=True)
    outputs=encode(prompts)
    rows=[check(ref, out) for ref,out in zip(refs,outputs,strict=True)]
    for row in rows:
        print(json.dumps(row),flush=True)
    diagnostic={'name':first_cat['name'],'probabilities':first_cat['probabilities'],
                'reference_error':first_cat['max_error'], 'same_argmax':first_cat['same_argmax'],
                'batch_difference':max(abs(a-b) for a,b in zip(first_cat['probabilities'],rows[cat_index]['probabilities'],strict=True))}
    modes = {'batch': rows}
    if args.all_singletons:
        reversed_outputs = encode(list(reversed(prompts)))
        modes['reversed_batch'] = [check(ref,out) for ref,out in zip(reversed(refs),reversed_outputs,strict=True)]
        modes['singletons'] = [check(ref,encode([prompt])[0]) for ref,prompt in zip(refs,prompts,strict=True)]
    summaries = {}
    for name, mode_rows in modes.items():
        summaries[name] = {}
        for condition in ('image','blank','no_image','text_regression'):
            subset = [r for r in mode_rows if r['condition']==condition]
            summaries[name][condition] = {
                'rows':len(subset), 'same_argmax':sum(r['same_argmax'] for r in subset),
                'max_error':max(r['max_error'] for r in subset),
                'failures':[{'name':r['name'],'max_error':r['max_error'],'same_argmax':r['same_argmax']}
                            for r in subset if r['max_error']>=0.02 or not r['same_argmax']],
            }
            # Text labels are not guaranteed to be ground truth; count image
            # fixture/control accuracy only, not invented text label defaults.
            if condition != 'text_regression':
                summaries[name][condition]['correct'] = sum(r['correct'] for r in subset)
    report={'vision_attention':args.vision_attention or 'default', 'vision_precision':args.vision_precision,
            'mm_processor_cache':'disabled' if args.benchmark else 'vllm default',
            'input_ids_match':True,'singleton_diagnostic':diagnostic,
            'passed':first_cat['max_error']<0.02 and first_cat['same_argmax'] and
                     all(r['max_error']<0.02 and r['same_argmax'] for mode_rows in modes.values() for r in mode_rows),
            'tolerance':0.02,'rows':rows,'modes':modes,'summaries':summaries}
    if args.benchmark:
        import statistics
        import time
        source = prompts[cat_index]['multi_modal_data']['image']
        serial = 0

        def fresh_prompt():
            nonlocal serial
            serial += 1
            image = source.copy()
            image.putpixel((0,0), (serial%256,(serial//256)%256,0))
            return {'prompt_token_ids':prompts[cat_index]['prompt_token_ids'], 'multi_modal_data':{'image':image}}

        def timed(count):
            items = [fresh_prompt() for _ in range(count)]
            start = time.perf_counter()
            result = encode(items)
            for output in result:
                p = output.outputs.data.detach().cpu().flatten()
                assert len(p)==len(refs[cat_index]['expected']) and torch.isfinite(p).all()
            return (time.perf_counter()-start)*1000

        timings = {}
        for count, samples in ((1,30),(2,20)):
            for _ in range(5):
                timed(count)
            times = [timed(count) for _ in range(samples)]
            timings[str(count)] = {'samples':samples, 'p50_ms':statistics.median(times),
                                  'raw_ms':times}
        report['benchmark'] = {'resolution':list(source.size),'unique_images':True,'mm_processor_cache_gb':0,
                               'timing_boundary':'Prepared PIL images/token IDs through LLM.encode and CPU probability readback; excludes preparation, loading, HTTP and warmup',
                               'limitations':'One local run; other resident services; not a production SLA',
                               'warmup_per_batch_size':5,'by_batch_size':timings}
    print('SUMMARY', json.dumps(summaries), flush=True)
    suffix = '' if args.vision_precision=='native' else '-'+args.vision_precision
    if args.all_singletons or args.benchmark:
        suffix += '-extended'
    output_path = args.output or OUT/('vllm-'+(args.vision_attention or 'default').lower()+suffix+'-result.json')
    output_path.write_text(json.dumps(report,indent=2)+'\n')
    print('PASS' if report['passed'] else 'FAIL',len(rows),flush=True)
    if not report['passed']: raise SystemExit(1)


if __name__=='__main__': main()
