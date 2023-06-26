from sklearn.model_selection import KFold, StratifiedKFold
from layer import *
import random

types = ['erase', 'noise']
eprobs = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
emags = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
nprobs = [0.0, 0.05, 0.1, 0.2, 0.3]
nmags = [0.05, 0.1, 0.2, 0.3, 0.5]

def generate_combinations(j,k,count):
    index1 = eprobs.index(j)  
    index2 = emags.index(k)  
    
    combinations = set()  
    combinations.add((j, k))  
    while len(combinations) < count+1:
        weights = []
        for j_index in range(len(eprobs)):
            for k_index in range(len(emags)):
                distance1 = abs(j_index - index1)
                distance2 = abs(k_index - index2)
                weight = 1 / (distance1 + distance2 + 1)  
                weights.append(weight)
        
        random_index = random.choices(range(len(weights)), weights=weights)[0]
        random_index1 = random_index // len(emags)  
        random_index2 = random_index % len(emags)  
        
        combination = (eprobs[random_index1], emags[random_index2])  
        combinations.add(combination)  
    
    combinations.remove((j, k))  
    return list(combinations)

def adjust_selection(categories, selection):
    category_counts = {category: 0 for category in np.unique(categories)}  

    for category, selected in zip(categories, selection):
        if selected == 0:  
            category_counts[category] += 1
    for i in range(len(selection)):
        category = categories[i]
        selected = selection[i]

        if category_counts[category] < 10:  
            if selected == 1:  
                selection[i] = 0  
                category_counts[category] += 1
    return selection
    
class gcForestcs:
    def __init__(self, num_estimator, num_forests, num_classes, max_layer=100, max_depth=31, max_features=0.1, n_fold=5, min_samples_leaf=1, \
        sample_weight=None, random_state=42, purity_function="gini" , bootstrap=True, parallel=True, num_threads=-1, extend=1, aug_type='erase', aug_prob=0.0, aug_mag = 0.0 ,threshold=0.9,bool_add=1):
        
        
        self.num_estimator = num_estimator
        self.num_forests = num_forests
        self.num_classes = num_classes
        self.n_fold = n_fold
        self.max_depth = max_depth
        self.max_features = max_features
        self.max_layer = max_layer 
        self.min_samples_leaf = min_samples_leaf
        self.sample_weight = sample_weight
        self.random_state = random_state
        self.purity_function = purity_function
        self.bootstrap = bootstrap
        self.parallel = parallel
        self.num_threads = num_threads
        
        self.extend = extend
        
        self.aug_type = aug_type
        
        self.aug_prob = aug_prob
        self.aug_mag = aug_mag
        ##################################################
        self.model = []
        self.current_layer_index = 0
        self.current_train_data = 0
        self.current_train_label = 0
        self.current_test_data = 0
        self.current_test_label = 0
        self.current_ensemble_prob=0
        self.best_layer_index = 0
        self.best_val_acc = 0
        self.best_test_acc = 0
        self.kf_val = StratifiedKFold( 5, shuffle=True, random_state=42)
        
        self.val_index_list = []
        self.train_index_list = []
        self.aug_policy_schedule = []
        
        # cs extend 
        self.threshold=threshold
        self.train_pre_prob=0
        self.test_pre_prob=0
        self.train_pass_index=0
        self.test_pass_index=0
        self.current_train_stack=[]
        self.current_test_stack=[]
        self.init_train_data=0
        self.init_test_data=0
        self.left_test_index=[]
        self.add=bool_add
        
                      
    def load_data(self, train_data, train_label, X_test, y_test):
        self.current_train_data = train_data
        self.init_train_data=train_data
        self.current_train_label = train_label
        self.current_test_data = X_test
        self.init_test_data=X_test
        self.current_test_label = y_test
        
        self.train_pre_prob=np.zeros((self.current_train_data.shape[0],self.num_classes))
        self.train_pass_index=np.zeros(self.current_train_data.shape[0])

        self.test_pre_prob=np.zeros((self.current_test_data.shape[0],self.num_classes))
        self.test_pass_index=np.zeros(self.current_test_data.shape[0])
        
        self.left_test_index = np.where(self.test_pass_index == 0)[0]
        for train_index, val_index in self.kf_val.split(train_data, train_label):
            self.val_index_list.append(val_index)
            self.train_index_list.append(train_index)
        num_classes=int(np.max(self.current_train_label) + 1)
    
    def get_best_policy(self,train_data, train_label, X_test, y_test):
        
        num_classes = int(np.max(train_label) + 1)
        
        self.means = np.mean(train_data,axis=0)
        self.stds = np.std(train_data,axis=0)
        
        if( num_classes != self.num_classes ):
            raise Exception("init num_classes not equal to actual num_classes")

        # basis process
        train_data_new = train_data.copy()
        test_data_new = X_test.copy()
        test_label_new = y_test.copy()

        layer_index = self.current_layer_index

        kf = StratifiedKFold( self.n_fold, shuffle=True, random_state=self.random_state + layer_index)

        print("\n--------------\nlayer {},   X_train shape:{}, X_test shape:{}...\n ".format(str(layer_index), train_data_new.shape, test_data_new.shape) )

        rand_val_acc = 0.0
        rand_test_acc = 0.0
        best_rand_val_acc = 0.0
        best_rand_policy = []
        
        def try_one_time(train_data,train_label,val_data,val_label,test_data,test_label,prob,mag):
            layer = KfoldWarpper(self.num_forests, self.num_estimator, self.num_classes, self.n_fold, kf,\
                                    layer_index, self.max_depth, self.max_features, self.min_samples_leaf, self.sample_weight, self.random_state + layer_index, \
                                    self.purity_function, self.bootstrap,  self.parallel, self.num_threads, self.extend, 'erase', prob, mag, self.means, self.stds  )
            train_prob, train_stack = layer.train(train_data, train_label)
            val_prob, val_stack = layer.predict( val_data )
            test_prob, test_stack = layer.predict(test_data)
            temp_val_acc = compute_accuracy(val_label, val_prob)
            temp_test_acc = compute_accuracy(test_label, test_prob)
            return temp_val_acc,temp_test_acc,val_prob,val_stack,test_prob,test_stack,layer
        
        for j in eprobs:
            for k in emags:
                if k > j + 0.05:
                    continue
                rand_val_acc = 0.0
                rand_test_acc = 0.0
                
                for train_index, val_index in zip(self.train_index_list, self.val_index_list ):
                    y_train = train_label[train_index]
                    y_train = np.tile(y_train, self.extend)
                    
                    X_train = train_data_new[train_index, :]
                    X_train = np.tile(X_train, (self.extend, 1))
                    
                    X_val = train_data_new[val_index, :]
                    
                    y_val = train_label[val_index]

                    temp=try_one_time(X_train, y_train, X_val, y_val, test_data_new, test_label_new, prob=j, mag=k)
                        
                    rand_val_acc += temp[0]
                    rand_test_acc += temp[1]
                    
                rand_val_acc /= 5
                rand_test_acc /= 5
                print('erase; ',j,' ',k,' ', rand_val_acc,' ',rand_test_acc,' ')
                
                if rand_val_acc >= best_rand_val_acc:
                    best_rand_val_acc = rand_val_acc
                    best_rand_policy_tmp = []
                    best_rand_policy_tmp.append('erase')
                    best_rand_policy_tmp.append(j)
                    best_rand_policy_tmp.append(k)
                    best_rand_policy = best_rand_policy_tmp
        
        print('final best policy, ', best_rand_policy)   
        return best_rand_policy       
      
    def train_by_layer(self,best_policy):
        
        self.aug_policy_schedule.append(['layer: '+str(self.current_layer_index), best_policy[1], best_policy[2]])
        
        num_classes = int(np.max(self.current_train_label) + 1)
        
        self.means = np.mean(self.current_train_data,axis=0)
        self.stds = np.std(self.current_train_data,axis=0)
        
        if( num_classes != self.num_classes ):
            raise Exception("init num_classes not equal to actual num_classes")

        # basis process
        train_data_new = self.current_train_data.copy()
        
        test_data_new = self.current_test_data.copy()
        layer_index=self.current_layer_index
        
        rand_val_acc = 0
        rand_test_acc = 0
        
        def try_one_time(train_data,train_label,val_data,val_label,test_data,test_label):
            kf = StratifiedKFold( self.n_fold, shuffle=True, random_state=self.random_state + layer_index)
            layer = KfoldWarpper(self.num_forests, self.num_estimator, self.num_classes, self.n_fold, kf,\
                                    layer_index, self.max_depth, self.max_features, self.min_samples_leaf, self.sample_weight, self.random_state + layer_index, \
                                    self.purity_function, self.bootstrap,  self.parallel, self.num_threads, self.extend, best_policy[0], best_policy[1], best_policy[2], self.means, self.stds  )
            train_prob, train_stack = layer.train(train_data, train_label)
            val_prob, val_stack = layer.predict( val_data )
            test_prob, test_stack = layer.predict(test_data)
            temp_val_acc = compute_accuracy(val_label, val_prob)
            temp_test_acc = compute_accuracy(test_label, test_prob)
            return temp_val_acc,temp_test_acc,train_prob,train_stack,val_prob,val_stack,test_prob,test_stack,layer
        
        train_result=0
        best_val_acc=0    
        best_train_index = None
        best_val_index = None
        
        for train_index, val_index in zip(self.train_index_list, self.val_index_list ):

            y_train = self.current_train_label[train_index]
            y_train = np.tile(y_train, self.extend)
            
            X_train = self.current_train_data[train_index, :]
            X_train = np.tile(X_train, (self.extend, 1))
            
            X_val = self.current_train_data[val_index, :]          
            y_val = self.current_train_label[val_index]

            temp=try_one_time(X_train, y_train, X_val, y_val, self.current_test_data, self.current_test_label)
            # if temp[0]>best_val_acc:
            train_result=temp
            best_val_acc=temp[0]
            best_train_index = train_index
            best_val_index = val_index
                
            rand_val_acc += temp[0]
            rand_test_acc += temp[1]

        rand_val_acc /= 5
        rand_test_acc /= 5
        
        
        temp_val_acc=train_result[0]
        temp_test_acc=train_result[1]
        
        train_prob=train_result[2]
        train_stack=train_result[3]
        
        val_prob=train_result[4]
        val_stack=train_result[5]
        test_prob=train_result[6]
        test_stack=train_result[7]
        layer=train_result[8]
        
        #The general idea is to replace the current result row with each round of training results, 
        # because entering the training means that it cannot be a pass from the previous round
        train_prob_max= np.max(train_prob, axis=1)
        train_indices = np.where(train_prob_max > self.threshold)
        train_pass_index=best_train_index[train_indices]
        val_prob_max=np.max(val_prob,axis=1)
        val_indices = np.where(val_prob_max > self.threshold)
        val_pass_index=best_val_index[val_indices]
        
        print("the layer pass train "+str(len(train_pass_index)+len(val_pass_index)))
        self.train_pass_index[train_pass_index]=1
        self.train_pass_index[val_pass_index]=1
        self.train_pre_prob[best_train_index]=train_prob
        self.train_pre_prob[best_val_index]=val_prob
        print("total pass train "+str(self.train_pass_index.sum()))
        
        #Because the test requires ensmble, it is all tested and the results are replaced according to the rules
        zero_indices = np.where(self.test_pass_index == 0)[0]
        test_prob_max=np.max(test_prob,axis=1)
        test_indices = np.where(test_prob_max > self.threshold)
        test_pass_index= np.intersect1d(zero_indices, test_indices)
        self.test_pass_index[test_pass_index]=1
        self.test_pre_prob[zero_indices]=test_prob[zero_indices]
        print("the layer pass test "+str(len(test_pass_index)))
        
        # for class less than 10 examples
        self.train_pass_index=adjust_selection(self.current_train_label, self.train_pass_index)
        
        if layer_index == 1:
            self.current_ensemble_prob = test_prob
        if layer_index > 1:
            self.current_ensemble_prob += test_prob
            

        if layer_index > 0:
            print('ensemble acc: ',compute_accuracy(self.current_test_label, self.current_ensemble_prob / (layer_index + 1)))
        
        if num_classes == 2:
            train_stack = train_stack[:, 0::2]
            val_stack = val_stack[:, 0::2]
            test_stack = test_stack[:, 0::2]
            
        train_prob_max= np.max(train_prob, axis=1)
        train_indices = np.where(train_prob_max <= self.threshold)
        train_unpass_index=best_train_index[train_indices]
        val_prob_max=np.max(val_prob,axis=1)
        val_indices = np.where(val_prob_max <= self.threshold)
        val_unpass_index=best_val_index[val_indices]
        temp_train_stack=np.zeros((self.current_train_data.shape[0],self.num_classes))
        if self.num_classes==2:
           temp_train_stack=temp_train_stack[:, 0::2]
        temp_test_stack=test_stack
        temp_train_stack[train_unpass_index]=train_stack[train_indices]
        temp_train_stack[val_unpass_index]=val_stack[val_indices]
        if self.add==1:
            if layer_index==0:
                self.current_test_stack=temp_test_stack
                self.current_train_stack=temp_train_stack   
            else:
                self.current_test_stack=np.concatenate([self.current_test_stack,temp_test_stack], axis=1 )
                self.current_train_stack=np.concatenate([self.current_train_stack,temp_train_stack], axis=1 )
        else:
            self.current_test_stack=temp_test_stack
            self.current_train_stack=temp_train_stack
            
        self.current_train_data = np.concatenate([self.init_train_data,self.current_train_stack],axis=1)
        self.current_test_data = np.concatenate([self.init_test_data, self.current_test_stack], axis=1 )
        
        zero_indices = np.where(self.train_pass_index == 0)[0]
        zero_label =self.current_train_label[zero_indices]

        # this also not good, since we will not have a balanced train and valid
        # temp_train_index_list=[]
        # temp_val_index_list=[]
        # for train_index, val_index in zip(self.train_index_list, self.val_index_list ):
        #     train_index = train_index[np.logical_not(self.train_pass_index[train_index])]
        #     val_index = val_index[np.logical_not(self.train_pass_index[val_index])]
        #     temp_train_index_list.append(train_index)
        #     temp_val_index_list.append(val_index)
        # self.train_index_list=temp_train_index_list
        # self.val_index_list=temp_val_index_list
        
        
        # Because the 5 fold cross validation is used again here, 
        # there is a possibility of overfitting the validation set in the next layer, which requires a better writing method
        kf = StratifiedKFold( 5, shuffle=True, random_state=42+self.current_layer_index)
        self.train_index_list=[]
        self.val_index_list=[]
        for train_index, test_index in kf.split(zero_label,zero_label):
            train_data = zero_indices[train_index]
            self.train_index_list.append(train_data)
            test_data = zero_indices[test_index]
            self.val_index_list.append(test_data)
            
            
        self.left_test_index = np.where(self.test_pass_index == 0)[0]

        temp_val_acc = compute_accuracy(self.current_train_label, self.train_pre_prob)
        temp_test_acc = compute_accuracy( self.current_test_label, self.test_pre_prob )
        print("val  acc:{} \nTest acc: {}".format( str(temp_val_acc), str(temp_test_acc)) )
        
        if temp_val_acc >= self.best_val_acc:
            self.best_val_acc = temp_val_acc
            self.best_layer_index = layer_index
            self.best_test_acc = temp_test_acc
            
        self.current_layer_index = self.current_layer_index + 1
        
        self.model.append(layer)
        
        return rand_val_acc, rand_test_acc
    
    def get_best_acc_of_all_layer(self):
        print("best val acc " +str(self.best_val_acc))
        print("best layer index "+str(self.best_layer_index))
        print("best test acc "+str(self.best_test_acc))
        return self.best_val_acc,self.best_layer_index,self.best_test_acc
    
    
    # todo  
    def predict(self, test_data, y_test, ensemble_layer=1):
        test_data_new = test_data.copy()     
        ensemble_prob = []
        ensemble_acc=[]
        test_acc=[]
        layer_index = 0
        test_pass_index=np.zeros(test_data.shape[0])
        test_pre_prob=np.zeros((test_data.shape[0],self.num_classes))
        for layer in self.model:
            print('layer ',layer_index)
            test_prob, test_stack = layer.predict(test_data_new)
            zero_indices = np.where(test_pass_index == 0)[0]
            test_prob_max=np.max(test_prob,axis=1)
            test_indices = np.where(test_prob_max > self.threshold)
            temp_test_pass_index= np.intersect1d(zero_indices, test_indices)
            test_pass_index[temp_test_pass_index]=1
            test_pre_prob[zero_indices]=test_prob[zero_indices]
            
            if self.num_classes == 2:
                test_stack = test_stack[:, 0::2]
            
            print('test acc: ',compute_accuracy(y_test, test_pre_prob))
            test_acc.append(compute_accuracy(y_test, test_pre_prob))
            if self.add==1:
                test_data_new = np.concatenate([test_data_new, test_stack], axis=1)
            else:
                test_data_new = np.concatenate([test_data, test_stack], axis=1)
            if layer_index == 1:
                ensemble_prob = test_pre_prob
            if layer_index > 1:
                ensemble_prob += test_pre_prob
            if layer_index > 0:
                print('ensemble acc: ',compute_accuracy(y_test, ensemble_prob / (layer_index + 1)))
                ensemble_acc.append(compute_accuracy(y_test, ensemble_prob / (layer_index + 1)))
            layer_index = layer_index + 1
        print("test acc of all layers:")
        print(test_acc)    
        print("best valid layer index: {} and test acc: {}".format( str(self.best_layer_index), str(test_acc[self.best_layer_index])) )
        print("best test layer index: {} and test acc: {}".format( str(np.argmax(test_acc)), str(np.max(test_acc)) ))
        return np.argmax(test_prob, axis=1)
