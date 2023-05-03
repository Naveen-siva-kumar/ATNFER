'''
Aum Sri Sai Ram


'''

# -*- coding:utf-8 -*-
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from model.cnn import resModel

from common.utils import accuracy

from algorithm.loss import * 




class noisyfer:

    """
    This class gets the args from main file model from cnn file has training function evaluate function.
    
    """

    def __init__(self, args, train_dataset, device, input_channel, num_classes):

        # Hyper Parameters
        self.batch_size = args.batch_size
        learning_rate = args.lr
        self.relabel_epochs = args.relabel_epochs       
        self.relabled_count = 0
        self.eps = args.eps
        self.warmup_epochs = args.warmup_epochs
        self.alpha = args.alpha 
        self.device = device
        self.num_iter_per_epoch = args.num_iter_per_epoch
        self.print_freq = args.print_freq        
        self.n_epoch = args.n_epoch
        self.train_dataset = train_dataset
        self.co_lambda_max = args.co_lambda_max
        self.beta = args.beta
        self.num_classes  = args.num_classes
        self.max_epochs = args.n_epoch
        self.w = args.w
        self.h = args.h
        if  args.model_type=="res":               
            self.model = resModel(args,self.device)
            #self.ema_model = resModel(args)
            

        self.model = self.model.to(device)
        
        self.weighted_CCE =  DCE(num_class=args.num_classes, reduction='mean',device=self.device,epsilon=self.eps)
        self.optimizer = torch.optim.Adam(self.model.parameters() , lr=0.0001, weight_decay=1e-4)        
                                             
        print('\n Initial learning rate is:')
        for param_group in self.optimizer.param_groups:
            print(  param_group['lr'])                              
        
        if args.resume:
           pretrained = torch.load(args.resume)
           pretrained_state_dict1 = pretrained['model']   
           model_state_dict =  self.model.state_dict()
           loaded_keys = 0
           total_keys = 0
           for key in pretrained_state_dict1:                
               if  ((key=='module.fcx.weight')|(key=='module.fcx.bias')):
                   #print(key)
                   pass
               else:    
                   model_state_dict[key] = pretrained_state_dict1[key]
                   total_keys+=1
                   if key in model_state_dict :
                      loaded_keys+=1
           
           print("Loaded params num:", loaded_keys)
           print("Total params num:", total_keys)
           self.model.load_state_dict(model_state_dict) 
            
           print('Model loaded from path=',args.resume)
           
        
        self.ce_loss = torch.nn.CrossEntropyLoss().to(device)
        self.m1_statedict =  self.model.state_dict()
        self.o_statedict = self.optimizer.state_dict()  
        self.adjust_lr = args.adjust_lr
    
    def generate_flip_grid(self,w, h):
        # used to flip attention maps  for flipped version of the image
        x_ = torch.arange(w).view(1, -1).expand(h, -1)
        y_ = torch.arange(h).view(-1, 1).expand(-1, w)
        grid = torch.stack([x_, y_], dim=0).float().to(self.device)
        grid = grid.unsqueeze(0).expand(1, -1, -1, -1)
        grid[:, 0, :, :] = 2 * grid[:, 0, :, :] / (w - 1) - 1
        grid[:, 1, :, :] = 2 * grid[:, 1, :, :] / (h - 1) - 1
        grid[:, 0, :, :] = -grid[:, 0, :, :]
        return grid
    
    #Attention consistency loss
    def ACLoss(self,att_map1, att_map2, grid_l, output):
        flip_grid_large = grid_l.expand(output.size(0), -1, -1, -1)
        flip_grid_large = Variable(flip_grid_large, requires_grad = False)
        flip_grid_large = flip_grid_large.permute(0, 2, 3, 1)
        att_map2_flip = F.grid_sample(att_map2, flip_grid_large, mode = 'bilinear', padding_mode = 'border', align_corners=True)
        flip_loss_l = F.mse_loss(att_map1, att_map2_flip)
        return flip_loss_l

    # Evaluate the Model
    def evaluate(self, test_loader):
        print('Evaluating ...')
        self.model.eval()  
        correct1 = 0
        total1 = 0
        correct  = 0
        with torch.no_grad():
            for images,_, labels, _ in test_loader:
                images = (images).to(self.device)
                logits1,_ = self.model(images)
                outputs1 = F.softmax(logits1, dim=1)
                _, pred1 = torch.max(outputs1.data, 1)
                total1 += labels.size(0)
                correct1 += (pred1.cpu() == labels).sum()
                _, avg_pred = torch.max(outputs1, 1)
                correct += (avg_pred.cpu() == labels).sum()
            acc1 = 100 * float(correct1) / float(total1)
        return acc1
       
    def save_model(self, epoch, acc, noise,args):
        #print('Models saving at '+'checkpoints/'+args.cp_file+"/epoch_"+str(epoch)+'_noise_'+noise+"_warmup="+str(args.warmup_epochs)+"_acc_"+str(acc)[:6]+".pth") 
        torch.save({' epoch':  epoch,
                    'model': self.m1_statedict,
                    'optimizer':self.o_statedict,},                          
                    'checkpoints/'+args.cp_file+ "/epoch_"+str(epoch)+'_noise_'+noise+"_warmup_"+str(args.warmup_epochs)+"_"+"_"+args.dataset+"_"+"_acc_"+str(acc)[:6]+".pth") 
        print('Models saved at'+'checkpoints/'+args.cp_file+ "/epoch_"+str(epoch)+'_noise_'+noise+"_warmup_"+str(args.warmup_epochs)+"_acc_"+str(acc)[:6]+".pth")
    
    
    
               
    # Train the Model
    def train(self, train_loader, epoch,clean,tau=1/8):
        print('Training ...')
        self.model.train() 
        eps = self.eps
        
        train_total = 0
        train_correct = 0
        
        if epoch < self.warmup_epochs:
            print('\n Warm up stage using supervision loss based on easy samples')
        elif epoch == self.warmup_epochs:
            print('\n Robust learning stage using consistency loss combined with supervision loss based on selected clean samples using dynamic threshold')
        
        
        for i, (images1, images2, labels, indexes) in enumerate(train_loader):
        
            
            images1 = images1.to(self.device)
            images2 = images2.to(self.device)
            labels = labels.to(self.device)
            
            # Forward + Backward + Optimize
            
            logits1,hm1 = self.model(images1,attention=True)
            logits2,hm2 = self.model(images2,attention=True)
            prec1 = max(accuracy(logits1, labels, topk=(1,)),accuracy(logits2, labels, topk=(1,)))
            
            grid_l = self.generate_flip_grid(self.w,self.h,)

            train_total += 1
            train_correct += prec1
            
            
            if epoch < self.warmup_epochs:               
                loss = (self.ce_loss(logits1,labels) + self.ce_loss(logits2, labels))/2.0
                  
            else:
                loss1 = self.weighted_CCE(logits1,labels,clean)
                loss2 = self.weighted_CCE(logits2,labels,clean)
                
                loss_c = (loss1 + loss2)/2.0 #Superivison loss
               
                loss_o = self.ACLoss(hm1, hm2, grid_l, logits1)
                   
               
                loss =  loss_c + loss_o #+ model_loss
                  
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            #self.ema.update_params(self.model)
            #self.ema.apply_shadow(self.ema_model)
            
            
            if (i + 1) % self.print_freq == 0:
                print(
                    'Epoch [%d/%d], Iter [%d/%d] Training Accuracy1: %.4F, Loss1: %.4f'
                    % (epoch + 1, self.n_epoch, i + 1, len(self.train_dataset) // self.batch_size, prec1, loss.data.item() ))

        train_acc1 = float(train_correct) / float(train_total)
        
        
        
        return train_acc1
            
            
            
    
    def adjust_learning_rate(self, optimizer, epoch):
        
        for param_group in optimizer.param_groups:
           param_group['lr'] *= 0.95
          
    

    
    
    
    
    
    
