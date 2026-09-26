"""GPU-free image-token checks, using locally prepared pinned model inputs."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))


class VisionInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from kev_runtime import AUDIT
        if not (AUDIT/'merged-text/tokenizer.json').exists():
            raise unittest.SkipTest('Prepare pinned model inputs first')
        from experiment_vision import OUT, load_pinned_kev
        if not (OUT/'merged-vision/preprocessor_config.json').exists():
            raise unittest.SkipTest('Prepare the experimental vision export first')
        from transformers import AutoTokenizer, AutoImageProcessor
        load_pinned_kev()
        cls.km=sys.modules['pinned_kev.model']
        cls.tokenizer=AutoTokenizer.from_pretrained(AUDIT/'merged-text',local_files_only=True)
        cls.processor=AutoImageProcessor.from_pretrained(OUT/'merged-vision',local_files_only=True)

    def encode(self, case, image):
        from experiment_vision import encode_inputs
        return encode_inputs(self.km,self.tokenizer,self.processor,case,image)

    def case(self):
        return {'state':'Inspect the image.','question':'What color?','options':['Red','Blue']}

    def test_text_path_matches_upstream(self):
        c=self.case()
        row=self.km.encode(self.tokenizer,{'state':c['state'],'questions':[{'instr':c['question'],'options':c['options'],'label':0}]},max_state=1024,max_branch=2048,strict=True)
        actual=self.encode(c,None)
        self.assertEqual(actual['input_ids'][0].tolist(),row['ids'])
        self.assertNotIn('pixel_values',actual)

    def test_image_expansion_preserves_controls(self):
        from PIL import Image
        c=self.case();c['state']+=' <|image_pad|> <|fim_suffix|>'
        b=self.encode(c,Image.new('RGB',(256,256),'red'));ids=b['input_ids'][0].tolist()
        convert=self.tokenizer.convert_tokens_to_ids
        self.assertEqual(ids.count(convert('<|image_pad|>')),64)
        self.assertEqual(ids.count(convert('<|fim_suffix|>')),1)
        self.assertEqual(ids[-1],convert('<|fim_suffix|>'))
        self.assertEqual(ids.count(convert('<|box_end|>')),2)
        self.assertEqual(int(b['mm_token_type_ids'].sum()),64)
        self.assertEqual(b['image_grid_thw'].tolist(),[[1,16,16]])

    def test_image_budget_counts_toward_state(self):
        from PIL import Image
        c=self.case();c['state']='word '*980
        self.encode(c,None)
        with self.assertRaises(self.km.ContextOverflow):
            self.encode(c,Image.new('RGB',(256,256),'red'))


if __name__=='__main__': unittest.main()
