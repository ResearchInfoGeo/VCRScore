#!pip install -q git+https://github.com/huggingface/transformers.git
from transformers import ViltProcessor, ViltForImageAndTextRetrieval
import torch
from PIL import Image

class vitl:
    
    def __init__(self,):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-coco")
        self.model = ViltForImageAndTextRetrieval.from_pretrained("dandelin/vilt-b32-finetuned-coco")
        self.model.to(device)
    
    def score_vitl(self, caption, img):
        im = img
        width, height = im.size
        if width<333 or height<333:
            newsize = (500, 500)
            im = im.resize(newsize)
            im = im.convert("RGB")

        scores = dict()
        # encode
        oracion=caption
        encoding = self.processor(im, oracion, return_tensors="pt",max_length= 39,truncation=True)
        # forward pass
        outputs = self.model(**encoding)
        # get score
        score = outputs.logits[:,0].item()
        return score
