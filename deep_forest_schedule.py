from sklearn.model_selection import KFold, StratifiedKFold
from layer import *

class deep_forest:
    def __init__(self, num_estimator, num_forests, num_classes, max_layer=100, max_depth=31, max_features=0.1, n_fold=5, min_samples_leaf=1, \
        sample_weight=None, random_state=42, purity_function="gini" , bootstrap=True, parallel=True, num_threads=-1, extend=1, aug_schedule=None ):
        
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
        
        self.aug_schedule = aug_schedule
        self.model = []

    def train(self,train_data, train_label, X_test, y_test):
         # basis information of dataset 
        num_classes = int(np.max(train_label) + 1)
        
        self.means = np.mean(train_data,axis=0)
        self.stds = np.std(train_data,axis=0)
        
        if( num_classes != self.num_classes ):
            raise Exception("init num_classes not equal to actual num_classes")

        num_samples, num_features = train_data.shape

        # basis process
        train_data_new = train_data.copy()
        test_data_new = X_test.copy()
        

        # return value
        val_p = []
        val_acc = []
        best_train_acc = 0.0
        best_val_acc=0.0
        best_test_acc=0.0
        layer_index = 0
        best_layer_index = 0
        bad = 0


        kf = StratifiedKFold( self.n_fold, shuffle=True, random_state=self.random_state)  ##  KFold / StratifiedKFold
        
        # ensemble_prob = []

        while layer_index < self.max_layer:

            print("\n--------------\nlayer {},   X_train shape:{}, X_test shape:{}...\n ".format(str(layer_index), train_data_new.shape, test_data_new.shape) )

            layer = KfoldWarpper(self.num_forests, self.num_estimator, self.num_classes, self.n_fold, kf,\
                                layer_index, self.max_depth, self.max_features, self.min_samples_leaf, self.sample_weight, self.random_state, \
                                self.purity_function, self.bootstrap,  self.parallel, self.num_threads, self.extend, 'erase', self.aug_schedule[layer_index][1], self.aug_schedule[layer_index][2], self.means, self.stds  )

            val_prob, val_stack = layer.train(train_data_new, train_label)
            test_prob, test_stack = layer.predict( test_data_new )
            
            if layer_index == 1:
                ensemble_prob = test_prob
            if layer_index > 1:
                ensemble_prob  += test_prob
                
            if layer_index > 0:
                print('ensemble acc: ',compute_accuracy(y_test, ensemble_prob / (layer_index + 1)))
            
            if num_classes == 2:
                val_stack = val_stack[:, 0::2]
            train_data_new = np.concatenate([train_data_new, val_stack], axis=1)
            
            self.means = np.mean(train_data_new, axis=0)
            self.stds = np.std(train_data_new, axis=0)   
            
            
            if num_classes == 2:
                test_stack = test_stack[:, 0::2]
            test_data_new = np.concatenate([test_data_new, test_stack], axis=1 )

            temp_val_acc = compute_accuracy(train_label, val_prob)
            temp_test_acc = compute_accuracy( y_test, test_prob )
            print("val  acc:{} \nTest acc: {}".format( str(temp_val_acc), str(temp_test_acc)) )
            
            if temp_val_acc >= best_val_acc:
                best_val_acc = temp_val_acc
                best_layer_index = layer_index
                best_test_acc = temp_test_acc
            
            layer_index = layer_index + 1
            
            self.model.append(layer)

        
        print( 'best layer index: {}, its\' test acc: {} '.format(best_layer_index, best_test_acc) )

        return best_test_acc



    def predict_proba(self, test_data):
        test_data_new = test_data.copy()
        test_prob = []
        ###############
        layer_index = 0
        for layer in self.model:
            test_prob, test_stack = layer.predict(test_data_new)
            
            if self.num_classes == 2:
                test_stack = test_stack[:, 0::2]
            
            test_data_new = np.concatenate([test_data, test_stack], axis=1)
            layer_index = layer_index + 1

        return test_prob

    def predict(self, test_data, y_test, ensemble_layer):
        test_data_new = test_data.copy()
        #test_prob = []
        
        ensemble_prob = np.zeros((y_test.shape[0],y_test.shape[1]))
        #################
        layer_index = 0
        for layer in self.model:
            print('layer ',layer_index)
            test_prob, test_stack = layer.predict(test_data_new)
            
            if self.num_classes == 2:
                test_stack = test_stack[:, 0::2]
            
            print('test acc: ',compute_accuracy(y_test, test_prob))
            test_data_new = np.concatenate([test_data_new, test_stack], axis=1)
            if layer_index >= ensemble_layer:
                ensemble_prob = (ensemble_prob * (layer_index - ensemble_layer - 1) + test_prob) / (layer_index)
                print('ensemble acc: ',compute_accuracy(y_test, ensemble_prob))
            layer_index = layer_index + 1

        print(test_prob)
        
        return np.argmax(test_prob, axis=1)