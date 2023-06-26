from deep_forest import deep_forest
from deep_forest import generate_combinations
import numpy as np
import copy
import numpy as np
from numpy import genfromtxt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from aug import one_DF
from aug import aug_DF

def train_augDF(dataset="arrhythmia",model="df",mode="trans",random_state=42):
    processed=0
    print("dataset is "+dataset)
    if dataset=="arrhythmia":
        processed=1
        num_classes=13
        aug_schedule = [['layer: 0', 0.1, 0.05], ['layer: 1', 0.1, 0.05], ['layer: 2', 0.05, 0.05], ['layer: 3', 0.0, 0.05], ['layer: 4', 0.0, 0.05], ['layer: 5', 0.0, 0.05], ['layer: 6', 0.0, 0.05], ['layer: 7', 0.05, 0.1], ['layer: 8', 0.05, 0.2], ['layer: 9', 0.05, 0.3], ['layer: 10', 0.05, 0.3], ['layer: 11', 0.0, 0.3], ['layer: 12', 0.0, 0.3], ['layer: 13', 0.0, 0.3], ['layer: 14', 0.05, 0.3]]
        data = genfromtxt('./arrhythmia', delimiter=',',dtype=float, missing_values='?', filling_values=0.0 )
        trainset, testset = train_test_split(data, random_state=42)
        print(trainset.shape, testset.shape)
        classes, counts = np.unique(trainset[:, -1], return_counts=True)

        classes_to_augment = classes[np.where(counts < 10)]

        new_trainset = trainset.copy()
        for cls in classes_to_augment:
            cls_indices = np.where(trainset[:, -1] == cls)[0]
            num_samples = 10 - len(cls_indices)
            new_samples = np.random.choice(cls_indices, size=num_samples, replace=True)
            new_trainset = np.concatenate([new_trainset, trainset[new_samples]], axis=0)
            
        classes, counts = np.unique(new_trainset[:, -1], return_counts=True)

        X_train = new_trainset[:,:-1]
        y_train = new_trainset[:,-1]
        X_test = testset[:,:-1]
        y_test = testset[:,-1]
        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        y_test = le.transform(y_test)
    else:
        print("new dataset to preprocess!!!")
    if processed==1:
        print("sucess load data!")
        print("the model is "+model)
        if mode=="trans":
            add=1
            aug=1
            print("we use policy that has been searched and saved")
        if mode=="ori":
            add=0
            aug=0
            #it has 5 fold cross validation outside, better than original model)
            print("we run original model without any augment")
        if mode=="search":
            add=1
            aug=1
            print("we run the augDF and search its policy")    
        acc_aug=[]
        for i in range(5): 
            if mode=="trans":
                one=one_DF(X_train, y_train, X_test, y_test, num_classes=num_classes,random_state=random_state+i,add=add,model=model)
                acc=one.train_by_policy(aug_schedule)
                acc_aug.append(acc[2])
            if mode=="ori":
                one=one_DF(X_train, y_train, X_test, y_test, num_classes=num_classes,random_state=random_state+i,add=add,model=model)
                acc=one.train()
                acc_aug.append(acc[2])
            if mode=="search":
                aug=aug_DF(X_train, y_train, X_test, y_test, num_classes=num_classes, max_layer=15,random_state=random_state+i,add=add,model=model)
                acc = aug.train()
                acc_aug.append(acc[2])
                policy=aug.get_best_policy()
        print(acc_aug)
        print("test acc mean:{} \ntest acc std: {}".format( str(np.mean(acc_aug)), str(np.std(acc_aug))) )