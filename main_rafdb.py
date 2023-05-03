


# -*- coding:utf-8 -*-
'''
Aum Sri Sai Ram
Naveen

'''
import os
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import argparse
import time

import pandas as pd
import cv2

from algorithm.noisyfer_aug import noisyfer
from algorithm.randaugument import RandomAugment
from algorithm import transform as T

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--result_dir', type=str, help='dir to save result txt files', default='results')
parser.add_argument('--raf_path', type=str, default='../data/RAFDB', help='Raf-DB dataset path.')
parser.add_argument('--resume', type=str, default='', help='resume from saved model')   
parser.add_argument('--pretrained', type=str, default='../DarshanNoisyProject22_submitted/pretrained/res18_naive.pth_MSceleb.tar',  help='Pretrained weights')
parser.add_argument('--dataset', type=str, help='rafdb, ferplus, affectnet', default='rafdb')
parser.add_argument('--noise_file', type=str, help='EmoLabel/0.1noise_train.txt train_label.txt/', default='EmoLabel/train_label.txt')

parser.add_argument('--comment', type=str, default="usingwce_flippedimagestrongaugment_nosteplr",help="")

parser.add_argument('--noise_type', type=str, help='[pairflip, symmetric, asymmetric]', default='symmetric')
parser.add_argument('--num_gradual', type=int, default=10,help='how many epochs for linear drop rate, can be 5, 10, 15. This parameter is equal to Tk for R(T) in Co-teaching paper.')
parser.add_argument('--beta', type=float, default=0.25,  help='..based on ')
parser.add_argument('--alpha', type=float, default=0.5,  help='..based on ')
parser.add_argument('--eps', type=float, default=0.35,  help='..based on ')
parser.add_argument('--co_lambda_max', type=float, default=.9,   help='..based on ')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--print_freq', type=int, default=20)
parser.add_argument('--num_workers', type=int, default=4, help='how many subprocesses to use for data loading')
parser.add_argument('--num_iter_per_epoch', type=int, default=400)
parser.add_argument('--epoch_decay_start', type=int, default=80)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjust_lr', type=int, default=1)
parser.add_argument('--num_models', type=int, default=1)
parser.add_argument('--relabel_epochs', type=int, default=40)

parser.add_argument('--log_file', type=str, default="mar/raf/all",help="path after logs/ to save log file")
parser.add_argument('--cp_file', type=str, default="mar/raf/all",help="path after checkpoints/ to save checkpoints")
parser.add_argument('--device', type=str, default="")
parser.add_argument('--model_type', type=str, help='[mlp,cnn,res]', default='res')
parser.add_argument('--w', type=int, default=7, help='width of the attention map')
parser.add_argument('--h', type=int, default=7, help='height of the attention map')
parser.add_argument('--n_epoch', type=int, default=40)
parser.add_argument('--num_classes', type=int, default=7)
parser.add_argument('--batch_size', type=int, default=150, help='batch_size')
parser.add_argument('--warmup_epochs', type=int, default=4)

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Seed
torch.manual_seed(args.seed)
if args.gpu is not None:
    device = torch.device('cuda:{}'.format(args.gpu))
    args.device =device
    torch.cuda.manual_seed(args.seed)

else:
    device = torch.device('cpu')
    torch.manual_seed(args.seed)

# Hyper Parameters
batch_size = args.batch_size
learning_rate = args.lr

                         
class RafDataSet(data.Dataset):
    def __init__(self, raf_path, phase, transform = None):
        self.phase = phase
        self.transform = transform
        self.raf_path = raf_path
        self.clean=False
        
        NAME_COLUMN = 0
        LABEL_COLUMN = 1
        df_train_clean = pd.read_csv(os.path.join(self.raf_path, 'EmoLabel/train_label.txt'), sep=' ', header=None)
        df_train_noisy = pd.read_csv(os.path.join(self.raf_path, args.noise_file), sep=' ', header=None)
        
        df_test = pd.read_csv(os.path.join(self.raf_path, 'EmoLabel/test_label.txt'), sep=' ', header=None)
        if phase == 'train':
            dataset_train_noisy = df_train_noisy[df_train_noisy[NAME_COLUMN].str.startswith('train')]
            dataset_train_clean = df_train_clean[df_train_clean[NAME_COLUMN].str.startswith('train')]
            self.clean_label = dataset_train_clean.iloc[:, LABEL_COLUMN].values - 1 # 0:Surprise, 1:Fear, 2:Disgust, 3:Happiness, 4:Sadness, 5:Anger, 6:Neutral
            self.noisy_label = dataset_train_noisy.iloc[:, LABEL_COLUMN].values - 1 # 0:Surprise, 1:Fear, 2:Disgust, 3:Happiness, 4:Sadness, 5:Anger, 6:Neutral
            self.label = self.noisy_label
            file_names = dataset_train_noisy.iloc[:, NAME_COLUMN].values
            self.noise_or_not = (self.noisy_label == self.clean_label) #By DG
            if(args.noise_file == 'EmoLabel/test_label.txt'):
                self.clean =True
            else:
                self.clean =False
            
        
        else:             
            dataset = df_test[df_test[NAME_COLUMN].str.startswith('test')]
            self.label = dataset.iloc[:, LABEL_COLUMN].values - 1 # 0:Surprise, 1:Fear, 2:Disgust, 3:Happiness, 4:Sadness, 5:Anger, 6:Neutral            
            file_names = dataset.iloc[:, NAME_COLUMN].values
        
        self.new_label = [] 
        class_dict={0:0,1:0,2:0,3:0,4:0,5:0,6:0}
        for label in self.label:
            self.new_label.append(self.change_emotion_label_same_as_affectnet(label))
            class_dict[self.change_emotion_label_same_as_affectnet(label)]+=1
            
        self.label = self.new_label
        #print(class_dict)
        
        
        self.file_paths = []
        # use raf aligned images for training/testing
        for f in file_names:
            f = f.split(".")[0]
            f = f +"_aligned.jpg"
            path = os.path.join(self.raf_path, 'Image/aligned', f)
            self.file_paths.append(path)
        
    def __clean__(self):
        return self.clean    
        
       
    def change_emotion_label_same_as_affectnet(self, emo_to_return):
        """
        Parse labels to make them compatible with AffectNet.  
        #https://github.com/siqueira-hc/Efficient-Facial-Feature-Learning-with-Wide-Ensemble-based-Convolutional-Neural-Networks/blob/master/model/utils/udata.py
        """

        if emo_to_return == 0:
            emo_to_return = 3
        elif emo_to_return == 1:
            emo_to_return = 4
        elif emo_to_return == 2:
            emo_to_return = 5
        elif emo_to_return == 3:
            emo_to_return = 1
        elif emo_to_return == 4:
            emo_to_return = 2
        elif emo_to_return == 5:
            emo_to_return = 6
        elif emo_to_return == 6:
            emo_to_return = 0

        return emo_to_return   
         
    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        org_image = cv2.imread(path)
        image = org_image[:, :, ::-1] # BGR to RGB
        label = self.label[idx]
        flip_image = cv2.flip(image,1)
        
        image1 =  self.transform[0](image)
        
        if(self.clean):
            flip_image = T.Addg()(flip_image)
        image2 = self.transform[2](RandomAugment(2, 10)(flip_image))
        
        return image1,image2, label, idx                         


def main():
    
    print('\n\t\t\tAum Sri Sai Ram\n\n\n')
    
    print('\n\nNoise level: {} and warmupepochs {} are'.format(args.noise_file, args.warmup_epochs))
    print(args)
    t=time.localtime()
    time_stamp=time.strftime('%d-%m-%Y-%H-%M-%S',t)
    
    
    if(args.log_file):
        txtfile = 'logs/'+args.log_file+'/log_'+args.dataset+'__'+time_stamp+"__"+args.noise_file.split('/')[-1]+"_warmup_"+str(args.warmup_epochs)+".txt"
    else:
        txtfile = "temp.txt"   
    
    if  args.dataset == 'rafdb':   
        input_channel = 3
        num_classes = args.num_classes
        
        args.epoch_decay_start = 100    
        
        args.model_type = "res"
        
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
        
        trans_weak = T.Compose([
                 
                T.Resize((224, 224)),
                T.PadandRandomCrop(border=4, cropsize=(224, 224)),
                T.Normalize(mean, std),
                T.ToTensor(),
            ])
        trans_strong = T.Compose([
                T.Resize((224, 224)),
                T.PadandRandomCrop(border=4, cropsize=(224, 224)),
                T.RandomHorizontalFlip(p=0.5),
                RandomAugment(2, 10),
                T.Normalize(mean, std),
                T.ToTensor(),
            ])

        

        train_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(scale=(0.02, 0.25)) ])
        
        
        data_transforms_val = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])
                                     
        train_dataset = RafDataSet(args.raf_path, phase = 'train', transform = [trans_weak, trans_strong,train_transforms])    
        
        print('Train set size:', train_dataset.__len__())        
                                                                            
        test_dataset = RafDataSet(args.raf_path, phase = 'test', transform = [data_transforms_val,data_transforms_val,data_transforms_val])    
        print('Validation set size:', test_dataset.__len__())
        
    else:
        print('Invalid dataset')
        
    
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size = batch_size,
                                               num_workers = args.num_workers,
                                               drop_last=True,
                                               shuffle = True,  
                                               pin_memory = True) 
    
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                               batch_size = batch_size,
                                               num_workers = args.num_workers,
                                               shuffle = False,  
                                               pin_memory = True)    
                                               
                                                                            
    
    model= noisyfer(args, train_dataset, device, input_channel, num_classes)
    
    with open(txtfile, "a") as myfile:
        myfile.write('epoch         train_acc           test_acc\n')
    
    best_test_acc   = 0.0   
    # training
    continue_epoch =0
    if(args.resume):
        continue_epoch = int(args.resume.split('_')[1]) + 1
    best_epoch=0
    for epoch in range(continue_epoch, args.n_epoch):
        
        train_acc = model.train(train_loader, epoch,train_dataset.__clean__())
        
        
        test_acc =  model.evaluate(test_loader)
        
        if best_test_acc <   test_acc:
            best_test_acc = test_acc
            best_epoch=epoch+1
            if(args.cp_file):
                model.save_model(epoch, test_acc, args.noise_file.split('/')[-1],args)  
            
             
                
        print(  'Epoch [%d/%d] Test Accuracy on the %s test images: Accuracy %.4f' % (
                    epoch + 1, args.n_epoch, len(test_dataset), test_acc))
        
        with open(txtfile, "a") as myfile:
            myfile.write(str(int(epoch)+1) + '          '  + str(train_acc) +'          '  + str(test_acc) +"\n")
                
        
        
        

    print('\n\n \t Best Test acc for {} at epoch {} is {}: '.format(args.noise_file,best_epoch, best_test_acc))    


if __name__ == '__main__':
    main()
    
    