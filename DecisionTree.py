import numpy as np
from sklearn import tree
from Measure import Measure
import os
from sklearn.externals import joblib
from Data import get_upsampling_data,get_data,get_downsampling_data
from sklearn.metrics import roc_auc_score

def Decision_tree():
    train_data, train_flag,val_data,val_flag, test_data, test_flag = get_data()
    dt = tree.DecisionTreeClassifier(max_depth=6,class_weight="balanced")
    dt = dt.fit(train_data, train_flag)
    # val
    val_output_prob = dt.predict_proba(val_data)
    val_output_prob = np.array(val_output_prob)[:, 1]
    _, _, thresold = Measure().get_pr_curve(val_flag, val_output_prob)

    # test
    test_output_prob = dt.predict_proba(test_data)
    test_output_prob = np.array(test_output_prob)[:, 1]
    test_output = np.zeros(test_output_prob.shape)
    test_output[test_output_prob > thresold] = 1

    precision = Measure().Precision(test_flag, test_output)
    recall = Measure().Recall(test_flag, test_output)
    f1 = Measure().F1_score(test_flag, test_output)
    acc = Measure().Accuracy(test_flag, test_output)
    print "precision:%.2f\nrecall:%.2f\nf1:%.2f\nacc:%.2f\n" % (precision, recall, f1, acc)
    print "auc:%.2f"%roc_auc_score(test_flag, test_output_prob)

def model_agg_dt(model_path_dir="params/DecisionTree/Agg_DT"):
    train_data, train_flag,val_data, val_flag,test_data, test_flag = get_downsampling_data()
    model_li=[]
    for i in range(10):
        model_name="DT-%d.model"%i
        path=os.path.join(model_path_dir,model_name)
        if os.path.exists(path):
            dt = joblib.load(os.path.join(model_path_dir,model_name))
        else:
            print "train submodel-%d)"%(i)
            dt = tree.DecisionTreeClassifier(max_depth=6, class_weight="balanced")
            dt.fit(train_data[i], train_flag[i])
            joblib.dump(dt, os.path.join(model_path_dir,model_name))
        model_li.append(dt)

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

if __name__=="__main__":
    model_agg_dt()
