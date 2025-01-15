# VCRScore

Image captioning has become an essential Vision & Language research task. It is about predicting the most accurate caption given a specific image or video. The research community has achieved impressive results by continuously proposing new models and approaches to improve the overall model's performance. Nevertheless, despite increasing proposals, the performance metrics used to measure their advances have remained practically untouched through the years. A probe of that, nowadays metrics like BLEU, METEOR, CIDEr, and ROUGE are still very used, aside from more sophisticated metrics such as BertScore and ClipScore. 
    Hence, it is essential to adjust how are measure the advances, limitations, and scopes of the new image captioning proposals, as well as to adapt new metrics to these new advanced image captioning approaches.
    This work proposes a new evaluation metric for the image captioning problem. To do that, first, it was generated a human-labeled dataset to assess to which degree the captions correlate with the image's content. Taking these human scores as ground truth, we propose a new metric, and compare it with several well-known metrics, from classical to newer ones. Outperformed results were also found, and interesting insights were presented and discussed. 

<img width="827" alt="image" src="https://github.com/user-attachments/assets/04572bb0-c87d-4e54-8ecd-191c99ba9858" />


# Install

## ViLT Transformer

!pip install -q git+https://github.com/huggingface/transformers.git

## sklearn version 1.6.0

!pip install scikit-learn==1.6.0

## open clip version 2.26.1

!pip install open_clip_torch==2.26.1

## More libraries

jupyterlab

numpy

pandas

pillow

scikit-image

struct

tensorflow

timm

torch


# Download

## YOLO weights

https://drive.google.com/file/d/1Tnsgs_JHWfLXIX8hHBq_H-sVZX9Wg_pD/view?usp=sharing

## Retrieval Optimmized CLIP Models

https://visual-computing.com/files/MCIP/MCIP-ViT-L-14-336.pth

## Files

Load all the *.py modules and the .pickle model in a folder

# Use our metric (example)

```python
import requests
from PIL import Image
import module_VCRScore
url = "http://images.cocodataset.org/val2017/000000184384.jpg"
image = Image.open(requests.get(url, stream=True).raw)
ref = ['a large piece of blueberry cake on a plate',
 'a plate of food attractively arranged on a table',
 'a plate of blueberry coffee cake with butter and an orange slice on a table with breakfast foods']
caption = 'a bluebery cake is on a plate and is topped with butter'
vcr = module_VCRScore.module_VCRScore()
value_VCRScore,Clip_Score,score_clip_ref,score_vitl = vcr.VCRScore(ref,caption,image)
print(value_VCRScore,clip_score,score_clip_ref,score_vitl)
```
# Values of the example

```python
value_VCRScore = 0.7892486441901005 

clip_score = 0.5580947995185852 

score_clip_ref = 0.6823944449424744 

score_vitl = 0.44036316871643066
```

# Outputs

The first output is the value for our metric proposal; the second output is the value of CLIP as metric; the third is the MCIP as metric; and finally, ViLT as value.


