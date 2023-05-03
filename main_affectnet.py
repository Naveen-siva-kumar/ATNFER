# -*- coding:utf-8 -*-
'''
Aum Sri Sai Ram

Naveen

'''
import os
import torch
import torchvision.transforms as transforms
import torch.utils.data as data

import time
import cv2
import argparse,random

import pandas as pd

from algorithm.noisyfer_aug import noisyfer

from algorithm.randaugument import RandomAugment
from algorithm import transform as T


parser = argparse.ArgumentParser()



parser.add_argument('--lr', type=float, default=0.0001)

parser.add_argument('--result_dir', type=str, help='dir to save result txt files', default='results')

parser.add_argument('--affectnet_path', type=str, default='../data/AffectNetdataset/Manually_Annotated_Images_aligned/', help='Affectnet dataset path.')
    
parser.add_argument('--pretrained', type=str, default='../DarshanNoisyProject22_submitted/pretrained/res18_naive.pth_MSceleb.tar', help='Pretrained weights')
                        
parser.add_argument('--resume', type=str, default='', help='Use FEC trained models')                  

parser.add_argument('--forget_rate', type=float, help='forget rate', default=None)

parser.add_argument('--dataset', type=str, help='rafdb, ferplus, affectnet', default='affectnet7')

                    
parser.add_argument('--num_gradual', type=int, default=10,
                    help='how many epochs for linear drop rate, can be 5, 10, 15. This parameter is equal to Tk for R(T) in Co-teaching paper.')
                    
parser.add_argument('--exponent', type=float, default=1,
                    help='exponent of the forget rate, can be 0.5, 1, 2. This parameter is equal to c in Tc for R(T) in Co-teaching paper.')
                    
parser.add_argument('--beta', type=float, default= 0.65, help='..based on ')
                    

parser.add_argument('--eps', type=float, default=0.35,  help='..based on ')

parser.add_argument('--alpha', type=float, default=0.5,  help='..based on ')                    
                    
parser.add_argument('--noise_file', type=str, default='Noisy/train_affectnet8_fullpath_list.txt', help='Noisy/0.3noise_train_affectnet8_list.txt')

parser.add_argument('--noise_rate', type=float, help='corruption rate, should be less than 1', default=0.1)                    

parser.add_argument('--co_lambda_max', type=float, default=.9,                   help='..based on ')
                    
parser.add_argument('--batch_size', type=int, default=128, help='batch_size')                    

parser.add_argument('--n_epoch', type=int, default=15)

parser.add_argument('--num_classes', type=int, default=8)

parser.add_argument('--seed', type=int, default=1)

parser.add_argument('--print_freq', type=int, default=200)

parser.add_argument('--num_workers', type=int, default=4, help='how many subprocesses to use for data loading')

parser.add_argument('--num_iter_per_epoch', type=int, default=400)

parser.add_argument('--epoch_decay_start', type=int, default=80)

parser.add_argument('--gpu', type=int, default=0)

parser.add_argument('--co_lambda', type=float, default=0.1)

parser.add_argument('--adjust_lr', type=int, default=1)

parser.add_argument('--relabel_epochs', type=int, default=25)

parser.add_argument('--warmup_epochs', type=int, default= 3)

parser.add_argument('--margin', type=float, default=0.4)

parser.add_argument('--log_file', type=str, default="mar/aff/all",help="feb/raf/30--for february rafdb 30%noise")

parser.add_argument('--cp_file', type=str, default="mar/aff/all")

parser.add_argument('--model_type', type=str, help='[mlp,cnn,res]', default='res')

parser.add_argument('--save_model', type=str, help='save model?', default="False")

parser.add_argument('--save_result', type=str, help='save result?', default="True")

parser.add_argument('--drop_rate', type=float, default=0, help='Drop out rate.')

parser.add_argument('--w', type=int, default=7, help='width of the attention map')

parser.add_argument('--h', type=int, default=7, help='height of the attention map')




args = parser.parse_args()

# args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
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


                            
                         
class AffectNetDataSet(data.Dataset):
    def __init__(self, affectnet_path, phase, transform = None, basic_aug = False):
        self.phase = phase
        self.transform = transform
        self.affectnet_path = affectnet_path
        self.clean = False

        NAME_COLUMN = 0
        LABEL_COLUMN = 1
        df_train_clean = pd.read_csv(os.path.join('../data/Affectnetmetadata', 'Noisy/train_affectnet8_fullpath_list.txt'), sep=' ', header=None)
        df_train_noisy = pd.read_csv(os.path.join('../data/Affectnetmetadata', args.noise_file), sep=' ', header=None)
       
        os.path.join(self.affectnet_path, args.noise_file)
        
        df_test = pd.read_csv(os.path.join('../data/Affectnetmetadata', 'Noisy/val_affectnet8_fullpath_list.txt'), sep=' ', header=None)
        
        options = [7,8]   #for 8 classes i need to put 8 for 7 classes i need to put 7 and 8 in list
        
        if phase == 'train':
            dataset_train_clean = df_train_clean.loc[~df_train_clean[1].isin(options)]
            dataset_train_noisy = df_train_noisy.loc[~df_train_noisy[1].isin(options)]
            self.clean_label = dataset_train_clean.iloc[:, LABEL_COLUMN].values 
            self.noisy_label = dataset_train_noisy.iloc[:, LABEL_COLUMN].values
            #print(dataset_train_noisy[1].value_counts())
            self.label = self.noisy_label
            file_names = dataset_train_noisy.iloc[:, NAME_COLUMN].values
            self.noise_or_not = (self.noisy_label == self.clean_label) #By DG
            if(args.noise_file == 'Noisy/train_affectnet8_fullpath_list.txt'):
                self.clean = True
            else:
                self.clean = False
        else:             
            
            dataset = df_test.loc[~df_test[1].isin(options)]  
            self.label = dataset.iloc[:, LABEL_COLUMN].values             
            file_names = dataset.iloc[:, NAME_COLUMN].values
        
        self.file_paths = []        
        
        for f in file_names:
            
            if phase == 'train':            
               path = os.path.join(self.affectnet_path,  f)
            else:
               path = os.path.join(self.affectnet_path, f)
            self.file_paths.append(path)
     
        
        
        
    def __len__(self):
        return len(self.file_paths)
    
    def __clean__(self):
        return self.clean

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = cv2.imread(path)
        image = image[:, :, ::-1] # BGR to RGB
        label = self.label[idx]
        flip_image = cv2.flip(image,1)
        
        image1 =  self.transform[0](image)
        if(self.clean):
            flip_image = T.Addg()(flip_image)
        
        image2 = self.transform[2](RandomAugment(2, 10)(flip_image))
        
        return image1,image2, label, idx                         
                            
class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
    """

    def __init__(self, dataset, indices=None, num_samples=None):
                
        
        self.indices = list(range(len(dataset))) \
            if indices is None else indices
        
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples
        
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1
                
        
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        dataset_type = type(dataset)
        
        if dataset_type is AffectNetDataSet:
            return dataset.label[idx]#[1]
        else:
            raise NotImplementedError
                
    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples  
     



def main():
    
    print('\n\t\tAum Sri Sai Ram')
    print('\n\tFER with noisy annotations\n')
    print('\n\nNoise level: {} and alpha {} are'.format(args.noise_file, args.alpha))
    t=time.localtime()
    time_stamp=time.strftime('%d-%m-%Y-%H-%M-%S',t) 
    
    if(args.log_file):
        txtfile = 'logs/'+args.log_file+'/log_'+args.dataset+'_'+time_stamp+"_"+args.noise_file.split('/')[-1]+"num_classes_"+str(args.num_classes)+"_warmup_"+str(args.warmup_epochs)+".txt"
    else:
        txtfile = "temp.txt"  
     
    
    
    print('\n \n')
    #print(args)
    
    input_channel = 3
    num_classes = args.num_classes
    init_epoch = 5
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    
    data_transforms_val = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std )])
                                 
    trans_weak = T.Compose([
                 
                T.Resize((224, 224)),
                T.PadandRandomCrop(border=4, cropsize=(224, 224)),
                #T.RandomHorizontalFlip(p=0.5),
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
            #transforms.RandomHorizontalFlip(p=1),
            transforms.ToTensor(),
            transforms.Normalize(mean,
                             std),
            transforms.RandomErasing(scale=(0.02, 0.25)) ])
    
    train_dataset = AffectNetDataSet(args.affectnet_path, phase = 'train', transform = [trans_weak, trans_strong,train_transforms])    
    
    print('\n Train set size:', train_dataset.__len__())                                                                            
    test_dataset = AffectNetDataSet(args.affectnet_path, phase = 'test', transform = [data_transforms_val,data_transforms_val,data_transforms_val])      
    print('\n Validation set size:', test_dataset.__len__())
    
    


   
    train_sampler = ImbalancedDatasetSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=(train_sampler is None),
                                                   num_workers=args.num_workers, pin_memory=True, sampler=train_sampler)  
    
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                               batch_size = batch_size,
                                               num_workers = args.num_workers,
                                               shuffle = False,  
                                               pin_memory = True)                                    
    # Define models
    print('building model...')
    model= noisyfer(args, train_dataset, device, input_channel, num_classes)
    
    with open(txtfile, "a") as myfile:
        myfile.write('epoch         train_acc           test_acc\n')
    
    best_test_acc   = 0.0   
    # training
    continue_epoch =0
    if(args.resume):
        continue_epoch = int(args.resume.split('_')[1]) + 1
    best_epoch=0
    
    epoch = 0
    

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
            myfile.write(str(int(epoch)+1) + ' '  + str(train_acc) +' '  + str(test_acc)+"\n")
           
        
        
    print('\n\n \t Best Test acc for {} at epoch {} is {}: '.format(args.noise_file,best_epoch, best_test_acc)) 
if __name__ == '__main__':
    main()
   