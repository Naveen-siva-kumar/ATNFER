'''
Aum Sri Sai Ram
                
Naveen

'''

import torch
from model.resnet import *
import os

def resModel(args,device): #resnet18

        """
        This function gives back the model where the backbone is resnet18 with MSCeleb-1M weights are loaded into it.
        The resnet18 function is tweaked in its forward function so that we get both the predictions and class activation maps.
        
        """
   
        model = torch.nn.DataParallel(resnet18(num_classes=args.num_classes,end2end= False,pretrained= False)).to(device)
    
        if args.pretrained:
       
            checkpoint = torch.load(args.pretrained)
            pretrained_state_dict = checkpoint['state_dict']
            model_state_dict = model.state_dict()
         
            for key in pretrained_state_dict:
                if  ((key == 'module.fc.weight') | (key=='module.fc.bias') | (key=='module.feature.weight') | (key=='module.feature.bias') ) :
                    #print(key) 
                    pass
                else:
                    #print(key)
                    model_state_dict[key] = pretrained_state_dict[key]

            model.load_state_dict(model_state_dict, strict = False)
            print('Model loaded from Msceleb pretrained')
            
            

        else:
            print('No pretrained resent18 model built.')
        return model 

 