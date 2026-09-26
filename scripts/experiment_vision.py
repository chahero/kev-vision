"""No-training vision feasibility experiment; does not modify the text export.

Loads the exact full Qwen base, replaces only its language weights with the
validated Kev BF16 merge, and applies the original fp32 pointer head.
"""
import argparse
import json
import math
import time
from pathlib import Path

from kev_runtime import AUDIT, ROOT
from prepare_kev_vllm import base_path, load_pinned_kev

OUT = ROOT / 'artifacts/vision-experiment'


def fixtures():
    from PIL import Image, ImageDraw, ImageFont
    images = OUT / 'images'
    images.mkdir(parents=True, exist_ok=True)
    cases = []
    def add(name, im, question, options, answer):
        im.save(images / (name + '.png'))
        cases.append(dict(name=name, image=name+'.png', state='Inspect the supplied image.',
                          question=question, options=options, expected=answer))
    for name, rgb in [('red',(230,25,25)), ('green',(0,180,0)), ('blue',(20,50,230))]:
        im=Image.new('RGB',(256,256),'white'); d=ImageDraw.Draw(im)
        d.rectangle((40,40,216,216),fill=rgb)
        add('color-'+name,im,'What color is the large shape?', ['Red','Green','Blue'],name.capitalize())
    for shape in ['Circle','Square','Triangle']:
        im=Image.new('RGB',(256,256),'white'); d=ImageDraw.Draw(im)
        if shape=='Circle': d.ellipse((40,40,216,216),fill='black')
        elif shape=='Square': d.rectangle((40,40,216,216),fill='black')
        else: d.polygon([(128,32),(32,224),(224,224)],fill='black')
        add('shape-'+shape.lower(),im,'Which geometric shape is shown?', ['Circle','Square','Triangle'],shape)
    for count in [1,2,3]:
        im=Image.new('RGB',(256,256),'white'); d=ImageDraw.Draw(im)
        for i in range(count):
            x=int(256*(i+1)/(count+1)); d.ellipse((x-22,106,x+22,150),fill='black')
        add('count-'+str(count),im,'How many separate black circles are visible?', ['One','Two','Three'],['One','Two','Three'][count-1])
    for side,x in [('Left',48),('Center',128),('Right',208)]:
        im=Image.new('RGB',(256,256),'white'); d=ImageDraw.Draw(im)
        d.ellipse((x-28,100,x+28,156),fill='red')
        add('position-'+side.lower(),im,'Where is the red circle horizontally?', ['Left','Center','Right'],side)
    # System font used for optional text fixtures; record the actual case count.
    font_path=Path('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf')
    if font_path.exists():
        font=ImageFont.truetype(str(font_path),64)
        for word in ['CAT','DOG']:
            im=Image.new('RGB',(256,256),'white'); d=ImageDraw.Draw(im)
            d.text((128,128),word,font=font,fill='black',anchor='mm')
            add('word-'+word.lower(),im,'Which word is printed?', ['CAT','DOG','BIRD'],word)
    Image.new('RGB',(256,256),'white').save(images/'blank.png')
    (OUT/'cases.json').write_text(json.dumps(cases,indent=2)+'\n')
    return cases


def encode_inputs(km, tokenizer, image_processor, case, image=None):
    import torch
    record={'state':case['state'],'questions':[{'instr':case['question'],'options':case['options'],'label':0}]}
    encoded=km.encode(tokenizer,record,max_state=1024,max_branch=2048,strict=True)
    ids=encoded['ids']
    batch={}
    if image is not None:
        batch=dict(image_processor(images=image,return_tensors='pt'))
        image_tokens=int(batch['image_grid_thw'][0].prod())//(image_processor.merge_size**2)
        image_id=tokenizer.convert_tokens_to_ids('<|image_pad|>')
        inserted=[tokenizer.convert_tokens_to_ids('<|vision_start|>')]+[image_id]*image_tokens+[tokenizer.convert_tokens_to_ids('<|vision_end|>')]
        if len(km.rows_of(encoded)[0])+len(inserted)>1024:
            raise km.ContextOverflow('Image plus state exceeds 1024 tokens')
        ids=ids[:1]+inserted+ids[1:]
        if len(ids)>2048:
            raise km.ContextOverflow('Image plus question row exceeds 2048 tokens')
        batch['mm_token_type_ids']=torch.tensor([[int(i==image_id) for i in ids]])
    batch['input_ids']=torch.tensor([ids])
    batch['attention_mask']=torch.ones_like(batch['input_ids'])
    return batch


def load_reference():
    import torch
    from transformers import AutoModel, AutoTokenizer, AutoImageProcessor
    from safetensors.torch import load_file
    from huggingface_hub import snapshot_download
    load_pinned_kev()
    import sys
    km=sys.modules['pinned_kev.model']
    base=base_path()
    snapshot_download('Qwen/Qwen3.5-4B-Base',revision=base.name,allow_patterns=['preprocessor_config.json'])
    tokenizer=AutoTokenizer.from_pretrained(AUDIT/'merged-text',local_files_only=True)
    processor=AutoImageProcessor.from_pretrained(base,local_files_only=True)
    print('Loading full pinned base, including vision tower',flush=True)
    model=AutoModel.from_pretrained(base,dtype=torch.bfloat16,device_map='cuda',local_files_only=True,attn_implementation='sdpa').eval()
    merged=load_file(AUDIT/'merged-text/model.safetensors',device='cpu')
    weights={k.removeprefix('model.'):v for k,v in merged.items()}
    model.language_model.load_state_dict(weights,strict=True)
    del weights,merged
    head=torch.load(AUDIT/'kev-checkpoint/head.pt',map_location='cuda',weights_only=True)
    return km,tokenizer,processor,model,head


def score(model,head,batch,tokenizer):
    import torch
    from torch.nn.functional import linear
    batch={k:v.to('cuda') for k,v in batch.items()}
    with torch.inference_mode():
        hidden=model(**batch,use_cache=False).last_hidden_state[0].float()
        ids=batch['input_ids'][0]
        opts=(ids==tokenizer.convert_tokens_to_ids('<|box_end|>')).nonzero(as_tuple=True)[0]
        decide=(ids==tokenizer.convert_tokens_to_ids('<|fim_suffix|>')).nonzero(as_tuple=True)[0]
        assert len(decide)==1 and int(decide[0])==len(ids)-1
        h=head['head']
        q=linear(hidden[decide[0]],h['q.weight'],h['q.bias'])
        k=linear(hidden[opts],h['k.weight'],h['k.bias'])
        logits=(k@q)/math.sqrt(q.numel())/head['temperature']
        return logits.softmax(-1).cpu().tolist()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export',action='store_true',help='Also write a full multimodal checkpoint for vLLM experiments')
    args=parser.parse_args()
    if args.export and (OUT/'merged-vision/model.safetensors').exists():
        parser.error('Vision export already exists; run without --export or archive it before rebuilding')
    import torch
    from PIL import Image
    torch.set_num_threads(4)
    OUT.mkdir(parents=True,exist_ok=True)
    cases=fixtures()
    km,tok,processor,model,head=load_reference()
    rows=[]
    for case in cases:
        row=dict(case,outputs={})
        for condition in ('image','blank','no_image'):
            path=OUT/'images'/(case['image'] if condition=='image' else 'blank.png')
            image=Image.open(path).convert('RGB') if condition!='no_image' else None
            inputs=encode_inputs(km,tok,processor,case,image)
            start=time.perf_counter(); probabilities=score(model,head,inputs,tok)
            row['outputs'][condition]={'probabilities':probabilities,'selected':case['options'][max(range(len(probabilities)),key=probabilities.__getitem__)],
                                       'tokens':inputs['input_ids'].shape[1],'seconds':time.perf_counter()-start}
        rows.append(row)
        print(case['name'],{k:v['selected'] for k,v in row['outputs'].items()},flush=True)
    reference=json.loads((AUDIT/'reference.json').read_text())
    text=[]
    for row in reference['rows']:
        inputs={'input_ids':torch.tensor([row['input_ids']]),'attention_mask':torch.ones((1,len(row['input_ids'])),dtype=torch.long)}
        p=score(model,head,inputs,tok)
        error=max(abs(a-b) for a,b in zip(p,row['probabilities'],strict=True))
        same=max(range(len(p)),key=p.__getitem__)==max(range(len(p)),key=row['probabilities'].__getitem__)
        text.append({'error':error,'same_argmax':same})
    summary={condition:sum(r['outputs'][condition]['selected']==r['expected'] for r in rows) for condition in ('image','blank','no_image')}
    report={'scope':'Synthetic feasibility only; no visual task training or calibration','dtype':'bfloat16','cases':len(rows),'correct':summary,'rows':rows,'text_regression':text}
    (OUT/'reference-result.json').write_text(json.dumps(report,indent=2)+'\n')
    print('SUMMARY',summary,'text',text,flush=True)
    if args.export:
        from safetensors.torch import save_file
        import shutil
        out=OUT/'merged-vision';out.mkdir(exist_ok=True)
        cfg=model.config.to_dict();cfg['architectures']=['KevQwen3_5VisionForPooling']
        cfg.update(kev_head_dim=head['head_dim'],kev_temperature=head['temperature'],
                   kev_option_end_id=tok.convert_tokens_to_ids('<|box_end|>'),kev_decide_id=tok.convert_tokens_to_ids('<|fim_suffix|>'))
        (out/'config.json').write_text(json.dumps(cfg,indent=2))
        tok.save_pretrained(out);processor.save_pretrained(out)
        shutil.copy2(AUDIT/'kev-checkpoint/head.pt',out/'head.pt')
        print('Exporting full multimodal weights',flush=True)
        weights={'model.'+k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()}
        temporary=out/'model.safetensors.partial'
        save_file(weights,temporary,metadata={'format':'pt'})
        temporary.replace(out/'model.safetensors')
        (out/'export-manifest.json').write_text(json.dumps({'base_revision':base_path().name, 'text_export':json.loads((AUDIT/'merged-text/export-manifest.json').read_text()), 'tensors':len(weights), 'dtype':'bfloat16', 'experimental':True},indent=2)+'\n')
        print('Exported',len(weights),'tensors',flush=True)
    if not all(r['same_argmax'] and r['error']<0.02 for r in text):
        raise SystemExit('Text regression failed')


if __name__=='__main__':
    main()
