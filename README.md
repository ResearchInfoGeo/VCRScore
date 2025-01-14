# VCRScore

Image captioning has become an essential Vision & Language research task. It is about predicting the most accurate caption given a specific image or video. The research community has achieved impressive results by continuously proposing new models and approaches to improve the overall model's performance. Nevertheless, despite increasing proposals, the performance metrics used to measure their advances have remained practically untouched through the years. A probe of that, nowadays metrics like BLEU, METEOR, CIDEr, and ROUGE are still very used, aside from more sophisticated metrics such as BertScore and ClipScore. 
    Hence, it is essential to adjust how are measure the advances, limitations, and scopes of the new image captioning proposals, as well as to adapt new metrics to these new advanced image captioning approaches.
    This work proposes a new evaluation metric for the image captioning problem. To do that, first, it was generated a human-labeled dataset to assess to which degree the captions correlate with the image's content. Taking these human scores as ground truth, we propose a new metric, and compare it with several well-known metrics, from classical to newer ones. Outperformed results were also found, and interesting insights were presented and discussed. 

<img width="827" alt="image" src="https://github.com/user-attachments/assets/04572bb0-c87d-4e54-8ecd-191c99ba9858" />


# Install

## ViLT Transformer

!pip install -q git+https://github.com/huggingface/transformers.git

## sklearn version 1.3.0

!pip install sklearn = 1.3.0

## More libraries

tensorflow

transformers

torch

pandas

# Download

## YOLO weights

## Retrieval Optimmized CLIP Models

https://visual-computing.com/files/MCIP/MCIP-ViT-L-14-336.pth

# Use our metric (example)

```
import requests
from PIL import Image
import module_VCRScore
url = "http://images.cocodataset.org/val2017/000000184384.jpg"
image = Image.open(requests.get(url, stream=True).raw)
ef = ['a large piece of blueberry cake on a plate',
 'a plate of food attractively arranged on a table',
 'a plate of blueberry coffee cake with butter and an orange slice on a table with breakfast foods']
caption = 'a bluebery cake is on a plate and is topped with butter'
vcr = module_VCRScore.module_VCRScore()
value_VCRScore,Clip_Score,score_clip_ref,score_vitl = vcr.VCRScore(ref,caption,image)



