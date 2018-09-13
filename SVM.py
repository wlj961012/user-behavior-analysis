import numpy as np
from sklearn.svm import SVC
import os
from sklearn.metrics import roc_auc_score
from Measure import Measure
from Data import get_downsampling_data,get_upsampling_data,get_data
from sklearn.externals import joblib
def SVM(modelpath=None):#modelpath="params/SVM/svm_balanced.model"):
    train_data,train_flag,val_data,val_flag,test_data,test_flag=get_data()
    train_data=train_data[:40000]
    train_flag=train_flag[:40000]
    if modelpath is not None:
        svm_ = joblib.load(modelpath)
    else:
        svm_=SVC(kernel='rbf',probability=True)
        svm_.fit(train_data,train_flag)
        joblib.dump(svm_, "params/SVM/svm_balanced.model")

    #val
    val_output_prob = svm_.predict_proba(val_data)
    val_output_prob = np.array(val_output_prob)[:, 1]
    _, _, thresold = Measure().get_pr_curve(val_flag, val_output_prob)

    #test
    test_output_prob = svm_.predict_proba(test_data)
    test_output_prob=np.array(test_output_prob)[:,1]
    test_output=np.zeros(test_output_prob.shape)
    test_output[test_output_prob>thresold]=1

    precision=Measure().Precision(test_flag,test_output)
    recall = Measure().Recall(test_flag, test_output)
    f1 = Measure().F1_score(test_flag, test_output)
    acc = Measure().Accuracy(test_flag, test_output)
    print "model:SVM:\nprecision:%.2f\nrecall:%.2f\nf1:%.2f\nacc:%.2f\n"%(precision,recall,f1,acc)
    print "auc:%.2f"%roc_auc_score(test_flag, test_output_prob)
    #Measure().get_pr_curve(test_flag,test_output_prob)

def model_agg_svm(model_path_dir="params/SVM/Agg_svm"):
    train_data, train_flag,val_data, val_flag,test_data, test_flag = get_downsampling_data()
    model_li=[]
    for i in range(10):
        model_name="svm-%d.model"%i
        path=os.path.join(model_path_dir,model_name)
        if os.path.exists(path):
            svm_ = joblib.load(os.path.join(model_path_dir,model_name))
        else:
            print "train submodel-%d)"%(i)
            svm_ =  SVC(kernel='rbf',probability=True,class_weight={0:0.45,1:0.55})
            svm_.fit(train_data[i], train_flag[i])
            joblib.dump(svm_, os.path.join(model_path_dir,model_name))
        model_li.append(svm_)

    #val
    val_output_li=[]
    for i in range(10):
        val_output=model_li[i].predict_proba(val_data)
        val_output_li.append(val_output)
    val_output_li = np.array(val_output_li)
    val_output_li = val_output_li.sum(0) / 10
    val_output_proba = val_output_li[:, 1]
    _,_,thresold=Measure().get_pr_curve(val_flag,val_output_proba)

    #test
    test_output_li=[]
    for i in range(10):
        test_output=model_li[i].predict_proba(test_data)
        test_output_li.append(test_output)
    test_output_li=np.array(test_output_li)
    test_output_li=test_output_li.sum(0)/10
    test_output_proba = test_output_li[:, 1]
    test_output=np.zeros(test_output_proba.shape)
    test_output[test_output_proba>thresold]=1

    precision = Measure().Precision(test_flag, test_output)
    recall = Measure().Recall(test_flag, test_output)
    f1 = Measure().F1_score(test_flag, test_output)
    acc = Measure().Accuracy(test_flag, test_output)
    print "model:multi-SVM:\nprecision:%.2f\nrecall:%.2f\nf1:%.2f\nacc:%.2f\n" % (precision, recall, f1, acc)
    print "auc:%.2f" % roc_auc_score(test_flag, test_output_proba)
    #Measure().get_roc_curve(test_flag,test_output_proba)
    #Measure().get_pr_curve(test_flag,test_output_proba)
if __name__=="__main__":
    model_agg_svm()