import numpy as np
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
class Measure:
    def _calConfusion(self,GT, pred):
        TP = np.sum(pred[GT==1] == 1).astype(np.float)
        FP = np.sum(pred[GT==0] == 1).astype(np.float)
        TN = np.sum(pred[GT==0] == 0).astype(np.float)
        FN = np.sum(pred[GT==1] == 0).astype(np.float)
        return TP, FP, TN, FN

    def Precision(self,GT,pred,eps=1e-8):
        TP, FP, TN, FN=self._calConfusion(GT,pred)
        return TP/(TP+FP+eps)

    def Recall(self,GT,pred,eps=1e-8):
        TP, FP, TN, FN=self._calConfusion(GT,pred)
        return TP/(TP+FN+eps)

    def F1_score(self,GT,pred,eps=1e-8):
        p,r=self.Precision(GT,pred),self.Recall(GT,pred)
        return 2*p*r/(p+r+eps)

    def Accuracy(self,GT,pred,eps=1e-8):
        TP, FP, TN, FN = self._calConfusion(GT, pred)
        return (TP+TN) / (TP+ TN+ FN+FP+eps)

    def get_roc_curve(self,GT,pred):
        fpr, tpr, thresholds = roc_curve(GT,pred)
        plt.figure()
        plt.title('SVM-ROC Curve')
        plt.xlabel('false positive rate')
        plt.ylabel('true positive rate')
        plt.plot(fpr,tpr)
        plt.show()

    def get_pr_curve(self,GT,pred,beta=2,eps=1e-8):
        p=[]
        r=[]
        boundarys=[]
        f1=[]
        boundary=0
        for i in range(200):
            boundary+=1.0/200
            pred_binary=pred.copy()
            pred_binary[pred>=boundary]=1
            pred_binary[pred<boundary]=0
            if self.Precision(GT,pred_binary)!=0 or self.Recall(GT,pred_binary)!=0:
                p.append(self.Precision(GT,pred_binary))
                r.append(self.Recall(GT,pred_binary))
                boundarys.append(boundary)
                f1.append((1+beta*beta)*(p[-1]*r[-1])/(beta*beta*p[-1]+r[-1]+eps))

        p=np.array(p)
        r=np.array(r)
        boundarys=np.array(boundarys)
        f1=np.array(f1)
        max_pos=np.argmax(f1)
        return p,r,boundarys[max_pos]