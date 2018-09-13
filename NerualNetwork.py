import torch
import torch.nn as nn
import numpy as np
import random
import collections
from torch.utils import data
from torch.autograd import Variable
import torch.nn.functional as F
from Measure import Measure
from sklearn.metrics import roc_auc_score,precision_recall_curve
from imblearn.over_sampling import SMOTE


class Dataset(data.Dataset):
    def __init__(self,split='train'):
        self.files = collections.defaultdict(list)
        self.split=split

        train_data=np.load('data/train_data.npy')[:,:-1]
        train_flag=np.load('data/train_data.npy')[:,-1]
        test_data=np.load('data/test_data.npy')[:,:-1]
        test_flag=np.load('data/test_data.npy')[:,-1]
        val_data = np.load('data/val_data.npy')[:, :-1]
        val_flag = np.load('data/val_data.npy')[:, -1]

        smo = SMOTE(ratio={1: 10000}, random_state=42)
        train_data, train_flag = smo.fit_sample(train_data, train_flag)

        train_data = torch.from_numpy(train_data).float().unsqueeze(2).unsqueeze(3)
        train_flag = torch.from_numpy(train_flag).long()
        val_data = torch.from_numpy(val_data).float().unsqueeze(2).unsqueeze(3)
        val_flag = torch.from_numpy(val_flag).long()
        test_data = torch.from_numpy(test_data).float().unsqueeze(2).unsqueeze(3)
        test_flag = torch.from_numpy(test_flag).long()

        self.files["train"].append({
            "feature":train_data,
            "label":train_flag
        })
        self.files["val"].append({
            "feature": val_data,
            "label": val_flag
        })
        self.files["test"].append({
            "feature":test_data,
            "label":test_flag
        })

    def __len__(self):
        return len(self.files[self.split][0]["feature"])

    def __getitem__(self, idx):
        feature=self.files[self.split][0]["feature"][idx]
        label=self.files[self.split][0]["label"][idx]
        return feature,label


class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(torch.Tensor([1,3]).cuda())
        #self.nll_loss = nn.NLLLoss2d()

    def forward(self, inputs, targets):
        #print(inputs.size())
        return self.nll_loss(F.log_softmax(inputs,dim=1), targets)

class Classifier(nn.Module):
    def __init__(self,n_features=30,out_class=2):
        super(Classifier,self).__init__()
        self.classifier=nn.Sequential(
            nn.Linear(n_features,n_features),
            nn.ReLU(),
            nn.Linear(n_features,out_class)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x=self.classifier(x)
        return x

def train():
    dst = Dataset(split="train")
    model=Classifier().cuda()
    loss_func=CrossEntropyLoss2d().cuda()
    opt_SGD = torch.optim.SGD(model.parameters(), lr=1e-2,momentum=0.99)
    trainloader = data.DataLoader(dst, batch_size=2048)
    for epoch in range(2000):
        running_loss=0.0
        for i,_data in enumerate(trainloader):
            model.zero_grad()
            feature, flag = _data
            feature=Variable(feature).cuda()
            flag=Variable(flag).cuda()
            output=model(feature)
            loss=loss_func(output,flag)
            running_loss+=loss
            loss.backward()
            opt_SGD.step()
        print "epoch:%d loss:%f"%(epoch,running_loss/i)
    torch.save(model.state_dict(), 'params/NN/params3.pkl')

def test():
    dst = Dataset(split="test")
    valdst=Dataset(split="val")
    model=Classifier().cuda()
    testloader = data.DataLoader(dst, batch_size=1)
    valloader = data.DataLoader(valdst, batch_size=1)
    model.load_state_dict(torch.load('params/NN/params2.pkl'))
    model.eval()

    val_flag = []
    val_output_prob = []
    for i, _data in enumerate(valloader):
        feature, flag = _data
        feature = Variable(feature).cuda()
        output = model(feature)
        output_prob = F.softmax(output, dim=1).detach().cpu().numpy()
        val_output_prob.append(output_prob[0][1])
        val_flag.append(flag.numpy()[0])
    val_flag = np.array(val_flag)
    val_output_prob = np.array(val_output_prob)
    _,_,thresold=Measure().get_pr_curve(val_flag,val_output_prob)

    test_flag=[]
    test_output_prob=[]
    for i, _data in enumerate(testloader):
        feature, flag = _data
        feature = Variable(feature).cuda()
        output = model(feature)
        output_prob=F.softmax(output,dim=1).detach().cpu().numpy()
        test_output_prob.append(output_prob[0][1])
        test_flag.append(flag.numpy()[0])
    test_flag=np.array(test_flag)
    test_output_prob=np.array(test_output_prob)
    test_output = np.zeros(test_output_prob.shape)
    test_output[test_output_prob > thresold] = 1
    
    precision = Measure().Precision(test_flag, test_output)
    recall = Measure().Recall(test_flag, test_output)
    f1 = Measure().F1_score(test_flag, test_output)
    acc = Measure().Accuracy(test_flag, test_output)
    print "precision:%f\nrecall:%f\nf1:%f\nacc:%f\n" % (precision, recall, f1, acc)
    print "auc:%f"%roc_auc_score(test_flag, test_output_prob)
    #Measure().get_pr_curve(test_flag,test_output_prob)


if __name__=="__main__":
    #train()
    test()
