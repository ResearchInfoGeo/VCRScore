import pandas as pd

# Estimation precision-recall 
class PrRe:
    
    def __init__(self,ref,caption,detr_classes,yolo_classes):
        self.pool = [ref+' '+' '.join(detr_classes)+' '  + ' '.join(yolo_classes)]
        self.caption = [caption]
        
    def calcula(self):
        Dpred= self.caption
        Dreal= self.pool
        #Calculo 1: precision y 2
        precisions = []
        Recalls = []
    
        for i in range(len(Dpred)):
            r = intersection_precision(Dpred[i], Dreal[i])
            b = intersection_recall(Dpred[i], Dreal[i])
       
            
        precisions.append(intersection_precision(Dpred[i], Dreal[i]))
        Recalls.append(intersection_recall(Dpred[i], Dreal[i]))
        return(precisions, Recalls)

def intersection_precision(list_pred, list_pool):

    if(isinstance(list_pred, list) & isinstance(list_pool, list)):
        list_pred = list(set(list_pred))
        list_pool = list(set(list_pool))
        r = list(set(list_pred) & set(list_pool))
        
        p = (len(r) * 100) / (len(list_pred))
        
        return p

    if(isinstance(list_pred, str) & isinstance(list_pool, str)):
        lista_p = list_pred.split()
       
        lista_po = list_pool.split()
       

        list_pred = list(set(lista_p))
        
        list_pool = list(set(lista_po))
       

        r = list(set(list_pred) & set(list_pool))
        

        p = (len(r) * 100) / (len(list_pred))
        

        return p
def intersection_recall(list_pred, list_pool):

    if(isinstance(list_pred, list) & isinstance(list_pool, list)):
        list_pred = list(set(list_pred))
        list_pool = list(set(list_pool))
        r = list(set(list_pred) & set(list_pool))
        p = (len(r) * 100) / (len(list_pool))
        return p

    if(isinstance(list_pred, str) & isinstance(list_pool, str)):
        list_pred = list_pred.split()
        list_pool = list_pool.split()

        list_pred = list(set(list_pred))
        list_pool = list(set(list_pool))

        r = list(set(list_pred) & set(list_pool))
        p = (len(r) * 100) / (len(list_pool))
        return p

