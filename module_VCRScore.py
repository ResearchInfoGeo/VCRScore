import yolo
import detr
import PrRe
import vitl
import clipsc
import pickle
import numpy as np
import sys
import warnings


class module_VCRScore:
    
    def __init__(self):
        def fxn():
            warnings.warn("deprecated", DeprecationWarning)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fxn()

    def VCRScore (self,ref,caption,img):
        model_yolo = yolo.YOLO("YOLO-model.h5")
        _, yolo_classes, _ = model_yolo.detect(img)
        
        model_det =detr.DETR()
        detr_classes = model_det.detect(img)
        
        ref_lista = ' '.join(ref)
        pr = PrRe.PrRe(ref_lista,caption,detr_classes,yolo_classes)
        recall_own,precision_own=pr.calcula()
        
        model_vitl =vitl.vitl()
        score_vitl = model_vitl.score_vitl(caption,img)
        
        model_clipsc = clipsc.clipsc()
        sc,score_clipsc = model_clipsc.scoreclip(caption,img,ref)
        
        model_VCRscore = pickle.load(open('model_GBR_VCR.pickle', 'rb'))
        #'recall_own','distancia','clip_ref_score'
        values = np.array([recall_own[0],score_vitl,score_clipsc])
        values = values.reshape(1, -1)
        value_VCRScore = model_VCRscore.predict(values)
        return value_VCRScore[0],sc,score_clipsc,score_vitl
        
