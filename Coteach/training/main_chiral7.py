# -*- coding:utf-8 -*-
import sys
sys.path.append('/global/home/users/cgroschner/Co-teaching')
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random
from data.cifar import CIFAR10, CIFAR100
from data.mnist import MNIST
from data.chiral3 import CHIRAL
from model3 import CNN
import argparse
import numpy as np
import datetime
import shutil
import h5py


from loss_v2 import loss_coteaching

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type = float, default = 0.001)
parser.add_argument('--result_dir', type = str, help = 'dir to save result txt files', default = '/global/scratch/cgroschner/bohan_coteach_test/results/')
parser.add_argument('--noise_rate', type = float, help = 'corruption rate, should be less than 1', default = 0.2)
parser.add_argument('--forget_rate', type = float, help = 'forget rate', default = None)
parser.add_argument('--noise_type', type = str, help='[pairflip, symmetric, chiral_flip]', default='chiral_flip')
parser.add_argument('--num_gradual', type = int, default = 10, help='how many epochs for linear drop rate, can be 5, 10, 15. This parameter is equal to Tk for R(T) in Co-teaching paper.')
parser.add_argument('--exponent', type = float, default = 1, help='exponent of the forget rate, can be 0.5, 1, 2. This parameter is equal to c in Tc for R(T) in Co-teaching paper.')
parser.add_argument('--top_bn', action='store_true')
parser.add_argument('--dataset', type = str, help = 'chiral', default = 'chiral')
parser.add_argument('--n_epoch', type=int, default=40)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--print_freq', type=int, default=50)
parser.add_argument('--num_workers', type=int, default=4, help='how many subprocesses to use for data loading')
parser.add_argument('--num_iter_per_epoch', type=int, default=1000)
parser.add_argument('--epoch_decay_start', type=int, default=15)
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--version_number', default=None)

args = parser.parse_args()


# Seed
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# Hyper Parameters
batch_size = args.batch_size
learning_rate = args.lr

# load dataset
class RotationTransform:
    """Rotate by one of the given angles."""

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)



if args.dataset=='chiral':
    input_channel=1
    num_classes=2
    args.top_bn = False
    args.epoch_decay_start = args.epoch_decay_start
    args.n_epoch = args.n_epoch
    torch.manual_seed(17)
    random.seed(17)
    train_dataset = CHIRAL(root='/global/scratch/cgroschner/bohan_coteach_test/',
                                train=True,
                                transform = transforms.Compose([transforms.RandomRotation(5),transforms.RandomResizedCrop(128,scale=(0.7,1.3),ratio=(1,1)),RotationTransform([0,180]),transforms.ToTensor()]),
                                noise_rate=args.noise_rate
                         )

    test_dataset = CHIRAL(root='/global/scratch/cgroschner/bohan_coteach_test/',
                               train=False,
                               transform = transforms.Compose([transforms.RandomRotation(5),transforms.RandomResizedCrop(128,scale=(0.7,1.3),ratio=(1,1)),RotationTransform([0,180]),transforms.ToTensor()]),
                               noise_rate=args.noise_rate
                        )


if args.forget_rate is None:
    forget_rate=args.noise_rate
else:
    forget_rate=args.forget_rate

noise_or_not = train_dataset.noise_or_not

# Adjust learning rate and betas for Adam Optimizer
mom1 = 0.9
mom2 = 0.1
alpha_plan = [learning_rate] * args.n_epoch
beta1_plan = [mom1] * args.n_epoch
for i in range(args.epoch_decay_start, args.n_epoch):
    alpha_plan[i] = float(args.n_epoch - i) / (args.n_epoch - args.epoch_decay_start) * learning_rate
    beta1_plan[i] = mom2

def adjust_learning_rate(optimizer, epoch):
    for param_group in optimizer.param_groups:
        param_group['lr']=alpha_plan[epoch]
        param_group['betas']=(beta1_plan[epoch], 0.999) # Only change beta1

# define drop rate schedule
rate_schedule = np.ones(args.n_epoch)*forget_rate
rate_schedule[:args.num_gradual] = np.linspace(0, forget_rate**args.exponent, args.num_gradual)

save_dir = args.result_dir +'/' +args.dataset+'/coteaching/'

if not os.path.exists(save_dir):
    os.system('mkdir -p %s' % save_dir)

if args.version_number != None:
    model_str=args.dataset+'_coteaching_'+args.noise_type+'_'+str(100*args.noise_rate)+'p_v'+str(args.version_number)
else:
    model_str=args.dataset+'_coteaching_'+args.noise_type+'_'+str(100*args.noise_rate)+'p'

txtfile=save_dir+"/"+model_str+".txt"
h5file = save_dir+"/"+model_str+".h5"
h5file = h5py.File(h5file,'w')
nowTime=datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
if os.path.exists(txtfile):
    os.system('mv %s %s' % (txtfile, txtfile+".bak-%s" % nowTime))


def accuracy(logit, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    output = F.softmax(logit, dim=1)
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

# Train the Model
def train(train_loader,epoch, model1, optimizer1, model2, optimizer2, h5file):
    print('Training %s...' % model_str)
    pure_ratio_list=[]
    pure_ratio_1_list=[]
    pure_ratio_2_list=[]

    train_total=0
    train_correct=0
    train_total2=0
    train_correct2=0

    ind1_list = []
    ind2_list = []
    for i, (images, labels, indexes) in enumerate(train_loader):
        ind=indexes.cpu().numpy().transpose()
        if i>args.num_iter_per_epoch:
            break

        images = Variable(images).cuda()
        labels = Variable(labels).cuda()
        if i == 0:
            print('label shape',labels.size())
        # Forward + Backward + Optimize
        logits1=model1(images)
        prec1, _ = accuracy(logits1, labels, topk=(1, 2)) #originally: rec1, _ = accuracy(logits1, labels, topk=(1, 5))
        train_total+=1
        train_correct+=prec1

        logits2 = model2(images)
        prec2, _ = accuracy(logits2, labels, topk=(1, 2))
        train_total2+=1
        train_correct2+=prec2
        loss_1, loss_2, pure_ratio_1, pure_ratio_2,num_remember, ind_1_forget, ind_2_forget = loss_coteaching(logits1, logits2, labels, rate_schedule[epoch], ind, noise_or_not)

        ind1_list.append(ind_1_forget)
        ind2_list.append(ind_2_forget)

        pure_ratio_1_list.append(100*pure_ratio_1)
        pure_ratio_2_list.append(100*pure_ratio_2)

        optimizer1.zero_grad()
        loss_1.backward()
        optimizer1.step()
        optimizer2.zero_grad()
        loss_2.backward()
        optimizer2.step()
        if (i+1) % args.print_freq == 0:
            # print ('Epoch [%d/%d], Iter [%d/%d] Training Accuracy1: %.4F, Training Accuracy2: %.4f, Loss1: %.4f, Loss2: %.4f, Pure Ratio1: %.4f, Pure Ratio2 %.4f'
            #       %(epoch+1, args.n_epoch, i+1, len(train_dataset)//batch_size, prec1, prec2, loss_1.data[0], loss_2.data[0], np.sum(pure_ratio_1_list)/len(pure_ratio_1_list), np.sum(pure_ratio_2_list)/len(pure_ratio_2_list)))
            print ('Epoch [%d/%d], Iter [%d/%d] Training Accuracy1: %.4F, Training Accuracy2: %.4f, Loss1: %.4f, Loss2: %.4f, Pure Ratio1: %.4f, Pure Ratio2 %.4f'
                  %(epoch+1, args.n_epoch, i+1, len(train_dataset)//batch_size, prec1, prec2, loss_1.item(), loss_2.item(), np.sum(pure_ratio_1_list)/len(pure_ratio_1_list), np.sum(pure_ratio_2_list)/len(pure_ratio_2_list)))

    train_acc1=float(train_correct)/float(train_total)
    train_acc2=float(train_correct2)/float(train_total2)

    h5file.create_dataset('ind1_epoch'+str(epoch),data=ind1_list)
    h5file.create_dataset('ind2_epoch'+str(epoch),data=ind2_list)
    return train_acc1, train_acc2, pure_ratio_1_list, pure_ratio_2_list, loss_1, loss_2, num_remember

# Evaluate the Model
def evaluate(test_loader, model1, model2):
    print('Evaluating %s...' % model_str)
    model1.eval()    # Change model to 'eval' mode.
    correct1 = 0
    total1 = 0
    idx = 0
    loss_1 = []
    for images, labels, _ in test_loader:
        images = Variable(images).cuda()
        labelsCUDA = Variable(labels).cuda()
        logits1 = model1(images)
        # if idx == 0:
        #     print('logits: ',logits1)
        loss = F.cross_entropy(logits1, labelsCUDA, reduce = False)
        loss_1 = np.concatenate([loss_1,loss.cpu().data.numpy().flatten()],axis=0)

        outputs1 = F.softmax(logits1, dim=1)
        # if idx == 0:
        #     print('outputs: ',outputs1)
        _, pred1 = torch.max(outputs1.data, 1)
        # if idx == 0:
        #     print('pred: ',pred1)
        # if idx == 0:
        #     print('labels: ',labels)
        total1 += labels.size(0)
        correct1 += (pred1.cpu() == labels).sum()
        idx += 1

    model2.eval()    # Change model to 'eval' mode
    correct2 = 0
    total2 = 0
    loss_2 = []
    for images, labels, _ in test_loader:
        images = Variable(images).cuda()
        labelsCUDA = Variable(labels).cuda()
        logits2 = model2(images)
        loss = F.cross_entropy(logits2, labelsCUDA, reduce = False)
        loss_2 = np.concatenate([loss_2,loss.cpu().data.numpy().flatten()],axis=0)
        outputs2 = F.softmax(logits2, dim=1)
        _, pred2 = torch.max(outputs2.data, 1)
        total2 += labels.size(0)
        correct2 += (pred2.cpu() == labels).sum()
    loss_1 = np.mean(loss_1)
    loss_2 = np.mean(loss_2)
    acc1 = 100*float(correct1)/float(total1)
    acc2 = 100*float(correct2)/float(total2)
    return acc1, acc2,loss_1,loss_2


def main():
    # Data Loader (Input Pipeline)
    print('loading dataset...')
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               num_workers=args.num_workers,
                                               drop_last=True,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              num_workers=args.num_workers,
                                              drop_last=True,
                                              shuffle=True)
    # Define models
    print('building model...')
    cnn1 = CNN(input_channel=input_channel, n_outputs=num_classes)
    cnn1.cuda()
    print(cnn1.parameters)
    optimizer1 = torch.optim.Adam(cnn1.parameters(), lr=learning_rate)

    cnn2 = CNN(input_channel=input_channel, n_outputs=num_classes)
    cnn2.cuda()
    print(cnn2.parameters)
    optimizer2 = torch.optim.Adam(cnn2.parameters(), lr=learning_rate)

    mean_pure_ratio1=0
    mean_pure_ratio2=0

    with open(txtfile, "a") as myfile:
        myfile.write('epoch: train_acc1 train_acc2 test_acc1 test_acc2 pure_ratio1 pure_ratio2 loss1 loss2 num_remember \n')

    epoch=0
    train_acc1=0
    train_acc2=0
    testl1 = 1e9
    testl2 = 1e9
    # evaluate models with random weights
    test_acc1a, test_acc2a, test_loss1a, test_loss2a=evaluate(test_loader, cnn1, cnn2)
    print('Epoch [%d/%d] Test Accuracy on the %s test images: Model1 %.4f %% Model2 %.4f %% Pure Ratio1 %.4f %% Pure Ratio2 %.4f %%' % (epoch+1, args.n_epoch, len(test_dataset), test_acc1a, test_acc2a, mean_pure_ratio1, mean_pure_ratio2))
    # save results
    with open(txtfile, "a") as myfile:
        myfile.write(str(int(epoch)) + ': '  + str(train_acc1) +' '  + str(train_acc2) +' '  + str(test_acc1a) + " " + str(test_acc2a) + ' '  + str(mean_pure_ratio1) +' ' +str(mean_pure_ratio2)+ ' '  + str(test_loss1a)+ ' '+ str(test_loss2a)+ ' '  + 'N/A' + "\n")

    # training
    for epoch in range(1, args.n_epoch):
        # train models
        cnn1.train()
        adjust_learning_rate(optimizer1, epoch)
        cnn2.train()
        adjust_learning_rate(optimizer2, epoch)
        train_acc1, train_acc2, pure_ratio_1_list, pure_ratio_2_list, loss_1, loss_2, num_remember=train(train_loader, epoch, cnn1, optimizer1, cnn2, optimizer2,h5file)
        # evaluate models
        test_acc1, test_acc2, test_loss1, test_loss2=evaluate(test_loader, cnn1, cnn2)
        # save results
        model_name = model_str+nowTime
        if  test_loss1.item() < testl1:
            torch.save(cnn1.state_dict(), '/global/scratch/cgroschner/bohan_coteach_test/%s_cnn1.pth' %
                       model_name)
            train_loss1 = loss_1.item()
            print("=> saved best cnn1")
        if  test_loss2.item() < testl2:
            torch.save(cnn2.state_dict(), '/global/scratch/cgroschner/bohan_coteach_test/%s_cnn2.pth' %
                       model_name)
            train_loss2 = loss_2.item()
            print("=> saved best cnn2")
        mean_pure_ratio1 = sum(pure_ratio_1_list)/len(pure_ratio_1_list)
        mean_pure_ratio2 = sum(pure_ratio_2_list)/len(pure_ratio_2_list)
        print('Epoch [%d/%d] Test Accuracy on the %s test images: Model1 %.4f %% Model2 %.4f %%, Pure Ratio 1 %.4f %%, Pure Ratio 2 %.4f %%' % (epoch+1, args.n_epoch, len(test_dataset), test_acc1, test_acc2, mean_pure_ratio1, mean_pure_ratio2))
        with open(txtfile, "a") as myfile:
            myfile.write(str(int(epoch)) + ': '  + str(train_acc1) +' '  + str(train_acc2) +' '  + str(test_acc1) + " " + str(test_acc2) + ' ' + str(mean_pure_ratio1) + ' ' + str(mean_pure_ratio2) + ' '+ str(test_loss1) + ' '+str(test_loss2)+ ' '+ str(num_remember) +"\n")
    h5file.close()

if __name__=='__main__':
    main()
