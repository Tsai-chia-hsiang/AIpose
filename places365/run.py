import torch
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import os
import os.path as osp
import sys
import json
from PIL import Image

p365dir = os.path.dirname(os.path.realpath(__file__))
class_fname = osp.join(p365dir ,'categories_places365.txt')
label_clustering_file = osp.join(p365dir, "label_with_topic.json")

class Place365_Pretrianed_ResNet50():

    def __init__(self, modelat:os.PathLike, classespath:os.PathLike, label_clustering_file:os.PathLike)->tuple:
        
        self.model = self.load_pretrain(modeldir=modelat)
        self.classes = self.load_classes(filepath=classespath)
        self.label_clustering = self.load_label_topic(label_clustering_file=label_clustering_file)
        self.argumentation = trn.Compose([
            trn.Resize((256,256)),trn.CenterCrop(224),trn.ToTensor(), 
            trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def load_label_topic(self, label_clustering_file)->list:
        ret = {}
        with open(label_clustering_file, "r") as j:
            ret = json.load(j)
        return ret

    def load_pretrain(self,modeldir)->torch.nn.Module:
        arch='resnet50'
        model_file = osp.join(modeldir, f'{arch}_places365.pth.tar')
        model = models.__dict__[arch](num_classes=365)  
        checkpoint = torch.load(model_file, map_location='cpu') 
        sd = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
        model.load_state_dict(sd)
        model.eval()
        return model

    def load_classes(self, filepath)->list:
        classes = []
        with open(filepath) as class_file:
            for line in class_file:
                classes.append(line.strip().split(' ')[0][3:])           
        return classes

    def predict(self,scene_img, topk:int=5)->list:
        img = Image.open(scene_img) if isinstance(scene_img, str) else scene_img
        input_img = self.argumentation(img).unsqueeze(0)
        logit = self.model.forward(input_img)
        h_x = F.softmax(logit, 1).data.squeeze()
        probs, idx = h_x.sort(0, True)
        topk_id_with_label_prob = []
        for i in range(0,topk):
            topk_id_with_label_prob.append(
                {
                    'id':int(idx[i].item()),  
                    'label':self.classes[idx[i]], 
                    'topic':self.label_clustering[idx[i]],
                    'prob':float(probs[i].item())
                }
            )
        return topk_id_with_label_prob


def predict():
    
    place365rn50 = Place365_Pretrianed_ResNet50(
        modelat=p365dir, classespath=class_fname,
        label_clustering_file=label_clustering_file
    )
    r = place365rn50.predict(sys.argv[1], topk=1)
    return r

if __name__ == "__main__":
    predict()
    
    
    
    