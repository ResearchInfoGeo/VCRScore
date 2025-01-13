import open_clip
import torch
from scipy import stats
from PIL import Image

class clipsc:
    
    def __init__(self):
        
        self.model, _, self.preprocess=open_clip.create_model_and_transforms("ViT-L-14-336", pretrained="openai")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        checkpoint_path = 'MCIP-ViT-L-14-336.pth'
        mcip_state_dict = torch.load(checkpoint_path,map_location=torch.device('cpu'))
        self.model.load_state_dict(mcip_state_dict, strict=True)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer('ViT-L-14-336')

    def scoreclip(self,caption,img,ref):
        
        img = self.preprocess(img).unsqueeze(0).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        with torch.no_grad():
            images_features= self.model.encode_image(img).cpu()
        images_features = torch.tensor(images_features, dtype=torch.float32)
        images_features /= images_features.norm(dim=-1, keepdim=True)
        
        pred_captions = [caption]
        feat = []
        bs = 32
        for i in range(0, len(pred_captions), bs):
            t = self.tokenizer(pred_captions[i:i+bs]).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))        
            with torch.no_grad():
                feat.append(self.model.encode_text(t).cpu())
        features = torch.cat(feat)
        pred_features = features / features.norm(dim=-1, keepdim=True) 

        #pred_features = cap_features(pred_captions)
        
        c_score = images_features * pred_features
        c_score = torch.sum(c_score, -1)
        c_score = torch.maximum(c_score, torch.zeros_like(c_score))
        c_score = c_score * 2.5
        r_score = []
        
        ref_cap_features = []

        feat1 = []
        bs = 32
        for i in range(0, len(ref), bs):
            t1 = self.tokenizer(ref[i:i+bs]).to(self.device)        
            with torch.no_grad():
                feat1.append(self.model.encode_text(t1).cpu())
        features1 = torch.cat(feat1)
        cap_features = features1 / features1.norm(dim=-1, keepdim=True) 

        ref_cap_features.append(cap_features)
    
        ref_cap_features = torch.cat(ref_cap_features)  
        ref_cap_features = torch.reshape(ref_cap_features,(-1, 3, 768))
                                                
        for c, r in zip(pred_features, ref_cap_features):
            s = r @ c
            m = torch.max(s)
            r_score.append(torch.maximum(m, torch.zeros_like(m)))
        r_score = torch.stack(r_score)
        score =  1/((1/c_score + 1/r_score)/2)
        return c_score.item(), score.item()
  
