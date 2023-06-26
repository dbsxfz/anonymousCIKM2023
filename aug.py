from deep_forest import deep_forest
from deep_forest import generate_combinations
from gcForestcs import gcForestcs
import numpy as np
import copy

class one_DF:
    def __init__(self,X_train, y_train, X_test, y_test, num_classes,max_layer=15,random_state=42,add=1,search=0,model="df"):
        self.num_classes=num_classes
        self.max_layer=max_layer
        self.current_layer=0
        self.best_policy=[]
        if model=="df":        
            self.df = deep_forest(num_estimator=64, num_forests=1, num_classes=self.num_classes, random_state=random_state,n_fold=5, max_layer=15, extend=1, aug_type='erase', aug_prob=0.0, aug_mag = 0.0,bool_add=add)
        elif model=="cs":
            self.df=gcForestcs(num_estimator=64, num_forests=1, num_classes=self.num_classes, random_state=random_state,n_fold=5, max_layer=15, extend=1, aug_type='erase', aug_prob=0.0, aug_mag = 0.0,threshold=0.9,bool_add=add)
        self.df.load_data(X_train, y_train, X_test, y_test)
        if search==1:
            self.best_policy=self.df.get_best_policy(X_train, y_train, X_test, y_test)
        else:
            # this means no augment since eprobs=0
            self.best_policy=['erase',0.0,0.05]
    def train_layer(self):
        temp_acc=self.df.train_by_layer(self.best_policy)
        # print("average val acc "+str(temp_acc))
        self.current_layer=self.current_layer+1
    def train(self):
        for i in range(self.max_layer):
            print("layer " +str(i))
            self.train_layer()
        best_acc=self.df.get_best_acc_of_all_layer()
        return best_acc
    def train_by_policy(self,aug_schedule):
        for i in range(self.max_layer):
            print("layer " +str(i))
            # print(aug_schedule[i])
            self.best_policy=aug_schedule[i]
            self.train_layer()
        best_acc=self.df.get_best_acc_of_all_layer()
        return best_acc
    def predict(self,X_test, y_test):
        return self.df.predict(X_test, y_test)

class aug_DF:
    def __init__(self,X_train, y_train, X_test, y_test, num_classes, max_layer=15,random_state=42,add=1,search=1,model="df"):
        
        self.num_classes=num_classes
        
        self.df_list=[]
        self.max_layer=max_layer
        self.current_layer=0
        self.best_policy=[]
        self.acc_list=[]
        self.policy_list=[]
        self.best_df_index=0
        
        for i in range(8):   
            if model=="df":        
                df = deep_forest(num_estimator=64, num_forests=1, num_classes=self.num_classes, random_state=random_state,n_fold=5, max_layer=15, extend=1, aug_type='erase', aug_prob=0.0, aug_mag = 0.0,bool_add=add)
            elif model=="cs":
                df=gcForestcs(num_estimator=64, num_forests=1, num_classes=self.num_classes, random_state=random_state,n_fold=5, max_layer=15, extend=1, aug_type='erase', aug_prob=0.0, aug_mag = 0.0,threshold=0.9,bool_add=add)
            df.load_data(X_train, y_train, X_test, y_test)
            self.df_list.append(df)      
        self.best_policy=self.df_list[0].get_best_policy(X_train, y_train, X_test, y_test)
        self.policy_list.append(self.best_policy)

    def generate_policy(self,policy_count):
        generated_policy=generate_combinations(self.best_policy[1],self.best_policy[2],policy_count)
        new_policy=[]
        for i in range(policy_count):
            temp_policy = []
            temp_policy.append('erase')
            temp_policy.append(generated_policy[i][0])
            temp_policy.append(generated_policy[i][1])
            new_policy.append(temp_policy)
        return new_policy
    
    def renew_policy(self):
        if self.current_layer==0:
            new_policy=self.generate_policy(7)
            self.policy_list.extend(new_policy)
        else:
            new_policy=self.generate_policy(4)
            ind = np.argpartition(self.acc_list, 4)[:4]
            for i in range(4):
                self.policy_list[ind[i]]=new_policy[i]
                self.df_list[ind[i]]=copy.deepcopy(self.df_list[self.best_df_index])
                
    def train_8_augDF_by_layer(self):
        self.renew_policy()
        temp_acc_list=[]
        temp_test_list=[]
        for i in range(8): 
            print("model " + str(i))
            temp_acc=self.df_list[i].train_by_layer(self.policy_list[i])
            temp_acc_list.append(temp_acc[0])
            temp_test_list.append(temp_acc[1])
        self.acc_list=temp_acc_list
        
        index=len(temp_acc_list)-1-temp_acc_list[::-1].index(max(temp_acc_list))

        self.best_df_index=index
        self.best_policy=self.policy_list[index]
        print("best model index "+str(index))
        print("corrsponding policy "+str(self.best_policy))
        print("best val acc "+str(temp_acc_list[index]))
        print("corrsponding test acc "+str(temp_test_list[index]))
        self.current_layer=self.current_layer+1
        
    def train(self):
        for i in range(self.max_layer):
            print("layer " +str(i))
            self.train_8_augDF_by_layer()
        self.df_list[self.best_df_index].get_best_acc_of_all_layer()
        print(self.df_list[self.best_df_index].aug_policy_schedule)
        return self.df_list[self.best_df_index].get_best_acc_of_all_layer()
    def get_best_policy(self):
        return self.df_list[self.best_df_index].aug_policy_schedule
    
    def predict(self,X_test, y_test):
        return self.df_list[self.best_df_index].predict(X_test, y_test)