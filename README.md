# VCRScore

import requests
from PIL import Image

import module_VCRScore

url = "http://images.cocodataset.org/val2017/000000184384.jpg"
image = Image.open(requests.get(url, stream=True).raw)

display(image)

ref = ['a large piece of blueberry cake on a plate',
 'a plate of food attractively arranged on a table',
 'a plate of blueberry coffee cake with butter and an orange slice on a table with breakfast foods']
caption = 'a bluebery cake is on a plate and is topped with butter'


vcr = module_VCRScore.module_VCRScore()
value_VCRScore,sc,score_clipsc,score_vitl = vcr.VCRScore(ref,caption,image)

print(value_VCRScore,sc,score_clipsc,score_vitl)
