import numpy as np
import torch
from Coteach.training.model3 import CNN
import torch.nn.functional as F
from math import sqrt

def incorrect_labels(truth,pred):
    """Returns the stack of incorrect labels"""
    incorrects = []
    for idx,gt in enumerate(truth):
        if pred.shape[1] > 1:
            if pred[idx][0]!=gt[0]:
                incorrects.append(int(pred[idx][0]))
        else:
            if pred[idx]!=gt:
                incorrects.append(int(pred[idx]))

    return incorrects

def hand_ratios(Y,pred):
    """return fraction of incorrect labels that were incorrectly left handed"""
    hand_dist = np.array(incorrect_labels(Y,pred))
    left = sum(hand_dist == 0)
    right = sum(hand_dist==1)
    print('right: ',right,' left: ',left)
    if right+left > 1:
        return left/(right+left), left, right
    else:
        return 0.5, left, right


def skew_val_set_left(right_val,left_val,frac_left):
    if frac_left == 0:
        right_new = np.expand_dims(right_val,axis=-1)
        valX = np.concatenate([right_new,right_new,right_new],axis=-1)
        valY = np.array([[1,0] for i in np.arange(0,len(right_new))])
    else:
        num_left = int(np.round(frac_left*len(right_val)))
        labels_left = [[0,1] for i in np.arange(0,num_left)]
        indices = np.random.choice(len(left_val),size=num_left,replace=False)
        right_new = np.expand_dims([right_val[idx] for idx in np.arange(0,len(right_val)) if idx not in indices],axis=-1)
        labels_right = [[1,0] for i in np.arange(0,len(right_new))]
        left_new = np.expand_dims([left_val[idx] for idx in indices],axis=-1)
        print('number of rights =',len(right_new),' should be: ',len(right_val)-num_left, ' check: ',len(right_val)-len(indices))
        valX = np.concatenate([right_new,left_new],axis=0)
        valX = np.concatenate([valX,valX,valX],axis=-1)
        valY = np.concatenate([labels_right,labels_left],axis=0)
    return valX, valY

def detected_fraction_left(right_val,left_val,error_rates,files,model):
    """Calculate the fraction of left particles as predicted by the input model"""
    frac_dict = {}
    for e in error_rates:
        frac = float(e)/100
        valX, valY = skew_val_set_left(right_val,left_val,frac)
        frac_dict[e] = []
        error_files = [f for f in files if e+'p' in f]
        for f in error_files:
            print(f.split('/')[-1])
            model.load_weights(f)
            pred= model.predict(valX)
            num_left = 0
            for p in pred>0.5:
                if p[0] == 0:
                    num_left+=1
            print(num_left/len(pred))
            frac_dict[e].append(num_left/len(pred))
    fracs = []
    errors=[]
    frac_errors = []
    for i in frac_dict:
        if len(frac_dict[i]) == 0:
            pass
        elif len(frac_dict[i]) == 1:
            errors.append(int(float(i)))
            fracs.append(frac_dict[i][0])
            frac_errors.append(np.std(frac_dict[i])/sqrt(len(frac_dict[i])))
        else:
            errors.append(int(float(i)))
            fracs.append(np.mean(frac_dict[i]))
            frac_errors.append(np.std(frac_dict[i])/sqrt(len(frac_dict[i])))
    return errors, fracs, frac_errors

def skew_val_set_left_MNIST(right_val,left_val,frac_left):
    if frac_left == 0:
        valX = right_val
        valY = np.array([1 for i in np.arange(0,len(right_val))])
    else:
        num_left = int(np.round(frac_left*len(right_val)))
        labels_left = [0 for i in np.arange(0,num_left)]
        indices = np.random.choice(len(left_val),size=num_left,replace=False)
        right_new = np.array([right_val[idx] for idx in np.arange(0,len(right_val)) if idx not in indices])
        labels_right = [1 for i in np.arange(0,len(right_new))]
        left_new = np.array([left_val[idx] for idx in indices])
        print('number of rights =',len(right_new),' should be: ',len(right_val)-num_left, ' check: ',len(right_val)-len(indices))
        valX = np.concatenate([right_new,left_new],axis=0)
        valY = np.concatenate([labels_right,labels_left],axis=0)
    return valX, valY

def detected_fraction_left_MNIST(right_val,left_val,error_rates,files,model):
    """Calculate the fraction of left particles as predicted by the input model"""
    frac_dict = {}
    for e in error_rates:
        frac = float(e)/100
        valX, valY = skew_val_set_left_MNIST(right_val,left_val,frac)
        frac_dict[e] = []
        error_files = [f for f in files if e+'p' in f]
        for f in error_files:
            print(f.split('/')[-1])
            model.load_weights(f)
            pred= model.predict(valX)
            num_left = 0
            for p in pred>0.5:
                if p == 0:
                    num_left+=1
            print(num_left/len(pred))
            frac_dict[e].append(num_left/len(pred))
    fracs = []
    errors=[]
    frac_errors = []
    for i in frac_dict:
        if len(frac_dict[i]) == 0:
            pass
        elif len(frac_dict[i]) == 1:
            errors.append(int(float(i)))
            fracs.append(frac_dict[i][0])
            frac_errors.append(np.std(frac_dict[i])/sqrt(len(frac_dict[i])))
        else:
            errors.append(int(float(i)))
            fracs.append(np.mean(frac_dict[i]))
            frac_errors.append(np.std(frac_dict[i])/sqrt(len(frac_dict[i])))
    return errors, fracs, frac_errors

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
#     output = F.softmax(logit, dim=1)
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

def skew_val_set_left_coteach(right_val,left_val,frac_left):
    num_left = int(np.round(frac_left*len(right_val)))
    labels_left = [[0] for i in np.arange(0,num_left)]
    indices = np.random.choice(len(left_val),size=num_left,replace=False)
    right_new = np.array([right_val[idx] for idx in np.arange(0,len(right_val)) if idx not in indices])
    labels_right = [[1] for i in np.arange(0,len(right_new))]
    left_new = np.array([left_val[idx] for idx in indices])
    print('number of rights =',len(right_new),' should be: ',len(right_val)-num_left, ' check: ',len(right_val)-len(indices))
    valX = np.concatenate([right_new,left_new],axis=0)
    valY = np.concatenate([labels_right,labels_left],axis=0)
    return valX, valY

def detected_fraction_left_coteach(right_val,left_val,error_rates,files):
    """Calculate the fraction of left particles as predicted by the input model"""
    device = torch.device('cpu')
    frac_dict = {}
    for e in error_rates:
        frac = float(e)/100
        valX, valY = skew_val_set_left_coteach(right_val,left_val,frac)
        frac_dict[e] = []
        error_files = [f for f in files if e+'p' in f]
        for f in error_files:
            print(f.split('/')[-1])
            torch.manual_seed(1)
            torch.cuda.manual_seed(1)
            cnn1 = CNN(input_channel=1, n_outputs=2)
            cnn1.load_state_dict(torch.load(f, map_location=device))
            cnn1.eval()
            with torch.no_grad():
                pred = cnn1(torch.from_numpy(valX).float())
            pred = F.softmax(pred, dim=1)
            _, plabel = pred.topk(1,1)
            num_left = 0
            for p in plabel.numpy():
                if p[0] == 0:
                    num_left+=1
            print(num_left/len(pred))
            frac_dict[e].append(num_left/len(pred))
    fracs = []
    errors=[]
    frac_errors = []
    for i in frac_dict:
        if len(frac_dict[i]) == 0:
            pass
        elif len(frac_dict[i]) == 1:
            errors.append(int(float(i)))
            fracs.append(frac_dict[i][0])
            frac_errors.append(np.std(frac_dict[i])/sqrt(len(frac_dict[i])))
        else:
            errors.append(int(float(i)))
            fracs.append(np.mean(frac_dict[i]))
            frac_errors.append(np.std(frac_dict[i])/sqrt(len(frac_dict[i])))
    return errors, fracs, frac_errors
