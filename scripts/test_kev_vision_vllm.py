"""Compare native vLLM image pooling with the no-training Transformers reference."""
import argparse
import json
import sys
from kev_runtime import AUDIT, engine_options
from experiment_vision import OUT, encode_inputs
from prepare_kev_vllm import load_pinned_kev


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vision-attention',choices=['FLASH_ATTN','TORCH_SDPA'])
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
    outputs=model.encode(prompts,pooling_task='token_embed',pooling_params=PoolingParams(task='token_embed'),use_tqdm=False)
    rows=[]
    for ref,out in zip(refs,outputs,strict=True):
        assert out.prompt_token_ids == ref.pop('input_ids'), ref['name']
        p=out.outputs.data.detach().cpu()
        assert tuple(p.shape)==(len(ref['expected']),1)
        p=p.flatten(); expected=torch.tensor(ref['expected'])
        assert torch.isfinite(p).all() and (p>=0).all() and (p<=1).all() and abs(float(p.sum())-1)<1e-5
        row={**ref,'probabilities':p.tolist(),'max_error':float((p-expected).abs().max()),'same_argmax':int(p.argmax())==int(expected.argmax()),'correct':int(p.argmax())==ref['label']}
        rows.append(row);print(json.dumps(row),flush=True)
    cat_index=next(i for i,r in enumerate(refs) if r['name']=='word-cat' and r['condition']=='image') if any(r['name']=='word-cat' for r in refs) else 0
    single=model.encode([prompts[cat_index]],pooling_task='token_embed',pooling_params=PoolingParams(task='token_embed'),use_tqdm=False)[0].outputs.data.detach().cpu().flatten()
    diagnostic={'name':refs[cat_index]['name'],'probabilities':single.tolist(),'reference_error':float((single-torch.tensor(refs[cat_index]['expected'])).abs().max()), 'batch_difference':float((single-torch.tensor(rows[cat_index]['probabilities'])).abs().max())}
    report={'vision_attention':args.vision_attention or 'default','input_ids_match':True,'singleton_diagnostic':diagnostic,'passed':diagnostic['reference_error']<0.02 and all(r['max_error']<0.02 and r['same_argmax'] for r in rows),'tolerance':0.02,'rows':rows}
    (OUT/('vllm-'+(args.vision_attention or 'default').lower()+'-result.json')).write_text(json.dumps(report,indent=2)+'\n')
    print('PASS' if report['passed'] else 'FAIL',len(rows),flush=True)
    if not report['passed']: raise SystemExit(1)


if __name__=='__main__': main()
