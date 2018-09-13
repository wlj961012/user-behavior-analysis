import numpy as np
from sklearn.metrics import roc_auc_score
import os
from sklearn.linear_model import LogisticRegression
from Measure import Measure
from Data import get_downsampling_data,get_upsampling_data,get_data
from sklearn.externals import joblib

def Logistic_regression(modelpath=None):
    train_data, train_flag,val_data,val_flag, test_data, test_flag = get_data()
    if modelpath is not None:
        lr = joblib.load(modelpath)
    else:
        lr = LogisticRegression(penalty='l2', solver='liblinear', class_weight="balanced")
        lr.fit(train_data, train_flag)
        joblib.dump(lr, "params/LR/LR.model")

    # val
    val_output_prob = lr.predict_proba(val_data)
    val_output_prob = np.array(val_output_prob)[:, 1]
    _, _, thresold = Measure().get_pr_curve(val_flag, val_output_prob)

    test_output_proba=lr.predict_proba(test_data)
    test_output_proba = np.array(test_output_proba)[:, 1]
    test_output = np.zeros(test_output_proba.shape)
    test_output[test_output_proba > thresold] = 1

    precision = Measure().Precision(test_flag, test_output)
    recall = Measure().Recall(test_flag, test_output)
    f1 = Measure().F1_score(test_flag, test_output)
    acc = Measure().Accuracy(test_flag, test_output)
    print "precision:%.2f\nrecall:%.2f\nf1:%.2f\nacc:%.2f\n" % (precision, recall, f1, acc)
    print "auc:%.2f"%roc_auc_score(test_flag, test_output_proba)
    #Measure().get_pr_curve(test_flag,test_output_proba)


def model_agg_lr(model_path_dir="params/LR/Agg_lr"):
    train_data, train_flag,val_data,val_flag, test_data, test_flag = get_downsampling_data()
    model_li=[]

    for i in range(10):
        model_name = "lr-%d.model" % i
        path=os.path.join(model_path_dir,model_name)
        if os.path.exists(path):
            lr = joblib.load(os.path.join(model_path_dir, model_name))
        else:
            print "train submodel-%d)" % (i)
            lr = LogisticRegression(solver='liblinear', class_weight="balanced")
            lr.fit(train_data[i], train_flag[i])
            joblib.dump(lr, os.path.join(model_path_dir, model_name))
        model_li.append(lr)
    # val
    val_output_li = []
    for i in range(10):
        val_output = model_li[i].predict_proba(val_data)
        val_output_li.append(val_output)
    val_output_li = np.array(val_output_li)
    val_output_li = val_output_li.sum(0) / 10
    val_output_proba = val_output_li[:, 1]
    _, _, thresold = Measure().get_pr_curve(val_flag, val_output_proba)

    #test
    test_output_li=[]
    for i in range(10):
        test_output=model_li[i].predict_proba(test_data)
        test_output_li.append(test_output)
    test_output_li=np.array(test_output_li)
    test_output_li=test_output_li.sum(0)/10
    test_output_proba = test_output_li[:, 1]
    test_output = np.zeros(test_output_proba.shape)
    test_output[test_output_proba > thresold] = 1

    precision = Measure().Precision(test_flag, test_output)
    recall = Measure().Recall(test_flag, test_output)
    f1 = Measure().F1_score(test_flag, test_output)
    acc = Measure().Accuracy(test_flag, test_output)
    print "precision:%.2f\nrecall:%.2f\nf1:%.2f\nacc:%.2f\n" % (precision, recall, f1, acc)
    print "auc:%.2f" % roc_auc_score(test_flag, test_output_proba)
    #Measure().get_pr_curve(test_flag,test_output_proba)

if __name__=="__main__":
    Logistic_regression()