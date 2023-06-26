from sklearn.model_selection import KFold, StratifiedKFold
from layer import *
import random
# noise is to do and not in our paper
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

    
class deep_forest:
    def __init__(self, num_estimator, num_forests, num_classes, max_layer=100, max_depth=31, max_features=0.1, n_fold=5, min_samples_leaf=1, \
        sample_weight=None, random_state=42, purity_function="gini" , bootstrap=True, parallel=True, num_threads=-1, extend=1, aug_type='erase', aug_prob=0.0, aug_mag = 0.0,bool_add=1 ):       
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
        
        #add things
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
        # for choose adding or replacing
        self.add=bool_add
        self.original_train_data=[]
        self.original_test_data=[]
                      
    def load_data(self, train_data, train_label, X_test, y_test):
        self.current_train_data = train_data
        self.current_train_label = train_label
        self.current_test_data = X_test
        self.current_test_label = y_test
        if self.add==0:
            self.original_train_data=train_data
            self.original_test_data= X_test
        for train_index, val_index in self.kf_val.split(train_data, train_label):
            self.val_index_list.append(val_index)
            self.train_index_list.append(train_index)
    
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
                # only use valid result to choose
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
        # train_data_new = self.current_train_data.copy()        
        # test_data_new = self.current_test_data.copy()
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
        # The last one is chosen instead of the best verification set    
        best_train_index = None
        best_val_index = None
        
        for train_index, val_index in zip(self.train_index_list, self.val_index_list ):

            y_train = self.current_train_label[train_index]
            y_train = np.tile(y_train, self.extend)
            
            X_train = self.current_train_data[train_index, :]
            X_train = np.tile(X_train, (self.extend, 1))
            
            # no enhance for valid data
            X_val = self.current_train_data[val_index, :]
            y_val = self.current_train_label[val_index]
            
            temp=try_one_time(X_train, y_train, X_val, y_val, self.current_test_data, self.current_test_label)
            # here we do not choose the best of valid result 
            #if temp[0]>best_val_acc:
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

        train_new_stack = np.empty((self.current_train_data.shape[0],val_stack.shape[1]))
        # since we split ori train into train and val train_new_stack has no empty row in reality
        for i,j in enumerate(best_val_index):
            train_new_stack[j,:] = val_stack[i]
        for i,j in enumerate(best_train_index):
            train_new_stack[j,:] = train_stack[i]
        if self.add==1:
            self.current_train_data = np.concatenate([self.current_train_data,train_new_stack],axis=1)
            self.current_test_data = np.concatenate([self.current_test_data, test_stack], axis=1 )
        else:
            self.current_train_data = np.concatenate([self.original_train_data,train_new_stack],axis=1)
            self.current_test_data = np.concatenate([self.original_test_data, test_stack], axis=1 )
            
        print("val  acc:{} \ntest acc: {}".format( str(temp_val_acc), str(temp_test_acc)) )
        # not use average valid to choose test
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
        
    def predict(self, test_data, y_test):
        test_data_new = test_data.copy()     
        ensemble_prob = []
        test_acc=[]
        ensemble_acc=[]
        layer_index = 0
        for layer in self.model:
            print('layer ',layer_index)
            test_prob, test_stack = layer.predict(test_data_new)
            
            if self.num_classes == 2:
                test_stack = test_stack[:, 0::2]
            
            print('test acc: ',compute_accuracy(y_test, test_prob))
            test_acc.append(compute_accuracy(y_test, test_prob))
            if self.add==1:
                test_data_new = np.concatenate([test_data_new, test_stack], axis=1)
            else:
                test_data_new = np.concatenate([test_data, test_stack], axis=1)
            if layer_index == 1:
                ensemble_prob = test_prob
            if layer_index > 1:
                ensemble_prob += test_prob
            if layer_index > 0:
                print('ensemble acc: ',compute_accuracy(y_test, ensemble_prob / (layer_index + 1)))
                ensemble_acc.append(compute_accuracy(y_test, ensemble_prob / (layer_index + 1)))
            layer_index = layer_index + 1
        print("test acc of all layers:")
        print(test_acc)    
        print("best valid layer index: {} and test acc: {}".format( str(self.best_layer_index), str(test_acc[self.best_layer_index])) )
        print("best test layer index: {} and test acc: {}".format( str(np.argmax(test_acc)), str(np.max(test_acc)) ))
        return np.argmax(test_prob, axis=1)