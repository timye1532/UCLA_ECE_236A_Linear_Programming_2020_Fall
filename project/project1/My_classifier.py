import matplotlib.pyplot as plt
import numpy as np
import cvxpy as cp

def select_class(image, label, class_list=[1, 7]):
    '''
    inputs:
    image: 2-dim numpy array, (60000 x 784)
    label: 2-dim numpy array, (60000 x 10)
    class_list: a list contains the class that we want to select 
    ex. if we want "1" and "7", class_list = [1, 7]
    
    outputs:
    selected_image: 2-dim numpy array (# examples x 784)
    selected_label: 2-dim numpy array (# examples x 784)
    
    '''
    selected_image = []
    selected_label = []
    m = image.shape[0]
    for i in range(m):
        for cls in class_list:
            if label[i][cls] == 1:
                selected_image.append(image[i])
                selected_label.append(label[i])
                break     
    return np.array(selected_image), np.array(selected_label)

def apply_erasure(image, label, p=0.6):
    '''
    inputs:
    image: 2-dim training example, has shape (#examples x 784)
    label: 2-dim label, has shape (#examples x 784)
    p: erasure ratio
    
    outputs:
    erased_image: has same shape with the original image
    label: just copy original label
    
    '''
    size = image.shape
    random_mat = np.random.uniform(low=0, high=1, size=size)
    cond = np.less(random_mat, 1-p)
    erased_image = np.where(cond, image, 0)
    return erased_image, label

class MyClassifier:
    def __init__(self,K=2,M=784,class_dict=[1, 7]):
        self.K = K  #Number of classes
        self.M = M  #Number of features
        
        self.class_dict = class_dict
        self.W = {}
        self.w = {}
    
    def onehot_2_binary(self, y_onehot, cls_1, cls_2):
        m = y_onehot.shape[0]
        y_binary = np.zeros((m, 1))
        for i in range(m):
            if y_onehot[i][cls_1] == 1:
                y_binary[i] = 1
            elif y_onehot[i][cls_2] == 1:
                y_binary[i] = -1
                
        return y_binary
    
    def onehot_2_scalar(self, y_onehot):
        m, n = y_onehot.shape
        y_scalar = np.zeros((m, 1))
        for i in range(m):
            for j in range(n):
                if y_onehot[i][j] == 1:
                    y_scalar[i] = j
                    break
        return y_scalar
    
    def scalar_2_binary(self, y_scalar, cls_1, cls_2):
        m = y_scalar.shape[0]
        y_binary = np.zeros((m, 1))
        for i in range(m):
            if y_scalar[i] == cls_1:
                y_binary[i] = 1
            elif y_scalar[i] == cls_2:
                y_binary[i] = -1
        
        return y_binary
    
    def select_class_by_scalar(self, X_data, Y_data, cls_1, cls_2):
        X_select = []
        Y_select = []
        m = X_data.shape[0]
        for i in range(m):
            if Y_data[i] == cls_1 or Y_data[i] == cls_2:
                X_select.append(X_data[i])
                Y_select.append(Y_data[i])
                
        return np.array(X_select), np.array(Y_select)
    
    def select_class_by_onehot(self, X_data, Y_data, cls_1, cls_2):
        X_select = []
        Y_select = []
        m = X_data.shape[0]
        for i in range(m):
            if Y_data[i][cls_1] == 1 or Y_data[i][cls_2] == 1:
                X_select.append(X_data[i])
                Y_select.append(Y_data[i])
        
        return np.array(X_select), np.array(Y_select)
        
    def train(self, p, train_data, train_label, lambd=0, add_erasure=True):
        
        # THIS IS WHERE YOU SHOULD WRITE YOUR TRAINING FUNCTION
        #
        # The inputs to this function are:
        #
        # self: a reference to the classifier object.
        # train_data: a matrix of dimesions N_train x M, where N_train
        # is the number of inputs used for training. Each row is an
        # input vector.
        # trainLabel: a vector of length N_train. Each element is the
        # label for the corresponding input column vector in trainData.
        #
        # Make sure that your code sets the classifier parameters after
        # training. For example, your code should include a line that
        # looks like "self.W = a" and "self.w = b" for some variables "a"
        # and "b".
        
        assert train_data.shape[0] == train_label.shape[0]
        print("Start training!")
        num_class = self.K
        for cls_1 in range(num_class):
            for cls_2 in range(cls_1+1, num_class):
                label_1 = self.class_dict[cls_1]
                label_2 = self.class_dict[cls_2]
                X_train, Y_select = self.select_class_by_onehot(train_data, train_label, label_1, label_2)
                y_train = self.onehot_2_binary(Y_select, label_1, label_2)
                if add_erasure == True:
                    X_train_erase = self.apply_erasure(X_train, p)
                    X_train = np.concatenate((X_train, X_train_erase), axis=0)
                    y_train = np.concatenate((y_train, y_train), axis=0)
                
                assert X_train.shape[0] == y_train.shape[0]
                m, n = X_train.shape
                W = cp.Variable((n, 1))
                w = cp.Variable()

                loss = cp.sum(cp.pos(1 - cp.multiply(y_train, X_train @ W + w)))
                # l1-regularization
                reg = cp.norm(W, 1)
                obj = cp.Minimize(loss/m + lambd*reg)
                prob = cp.Problem(obj)
                prob.solve()
                self.W[(cls_1, cls_2)] = W.value
                self.w[(cls_1, cls_2)] = w.value 
                print("finish training class {} vs class {}".format(cls_1, cls_2))
        print("End training!")
        

    def f(self,input):
        # THIS IS WHERE YOU SHOULD WRITE YOUR CLASSIFICATION FUNCTION
        #
        # The inputs of this function are:
        #
        # input: the input to the function f(*), equal to g(y) = W^T y + w
        #
        # The outputs of this function are:
        #
        # s: this should be a scalar equal to the class estimated from
        # the corresponding input data point, equal to f(W^T y + w)
        # You should also check if the classifier is trained i.e. self.W and
        # self.w are nonempty
        for i in range(self.K):
            for j in range(i+1, self.K):
                assert (i, j) in self.W
                assert (i, j) in self.w
                
        votes = np.zeros((self.K, 1))
        for (cls_1, cls_2) in self.W:
            W = self.W[(cls_1, cls_2)]
            w = self.w[(cls_1, cls_2)]
            y = np.sign(np.dot(W.T, input) + w)
            if y >= 0:
                votes[cls_1] += 1
            else:
                votes[cls_2] += 1
                
        return np.argmax(votes)
        
    def classify(self,test_data):
        # THIS FUNCTION OUTPUTS ESTIMATED CLASSES FOR A DATA MATRIX
        # 
        # The inputs of this function are:
        # self: a reference to the classifier object.
        # test_data: a matrix of dimesions N_test x M, where N_test
        # is the number of inputs used for training. Each row is an
        # input vector.
        #
        #
        # The outputs of this function are:
        #
        # test_results: this should be a vector of length N_test,
        # containing the estimations of the classes of all the N_test
        # inputs.
        m = test_data.shape[0]
        test_results = np.zeros((m, 1))
        for i in range(m):
            test_image = test_data[i]
            result = self.f(test_image)
            test_results[i] = self.class_dict[result]
        
        return test_results
    
    def apply_erasure(self, image, p=0.6):
        '''
            inputs:
            image: 2-dim training example, has shape (#examples x 784)
            label: 2-dim label, has shape (#examples x 784)
            p: erasure ratio
    
            outputs:
            erased_image: has same shape with the original image
            label: just copy original label
    
        '''
        size = image.shape
        random_mat = np.random.uniform(low=0, high=1, size=size)
        cond = np.less(random_mat, 1-p)
        erased_image = np.where(cond, image, 0)
        return erased_image
    
    def TestCorrupted(self,p,test_data):
        # THIS FUNCTION OUTPUTS ESTIMATED CLASSES FOR A DATA MATRIX
        #
        #
        # The inputs of this function are:
        #
        # self: a reference to the classifier object.
        # test_data: a matrix of dimesions N_test x M, where N_test
        # is the number of inputs used for training. Each row is an
        # input vector.
        #
        # p:erasure probability
        #
        #
        # The outputs of this function are:
        #
        # test_results: this should be a vector of length N_test,
        # containing the estimations of the classes of all the N_test
        # inputs.
        
        test_data_erase = self.apply_erasure(test_data, p)
        results = self.classify(test_data_erase)
        return results
    
    def accuracy_cal(self, y_true, y_pred):
        m = y_true.shape[0]
        correct = 0
        for i in range(m):
            if y_true[i] == y_pred[i]:
                correct += 1
        return correct / m


test_X = np.load("./data/test_X.npy")
test_Y = np.load("./data/test_Y.npy")

train_X = np.load("./data/train_X.npy")
train_Y = np.load("./data/train_Y.npy")

class_list = [1, 7]
train_X_17, train_Y_17 = select_class(train_X, train_Y, class_list)
print("-- shape for training data -- ")
print(train_X_17.shape)
print(train_Y_17.shape)

test_X_17, test_Y_17 = select_class(test_X, test_Y, class_list)
print("-- shape for testing data -- ")
print(test_X_17.shape)
print(test_Y_17.shape)

# erasure probability
p = 0.6
my_cls = MyClassifier(K=2, M=784, class_dict=class_list)
my_cls.train(p, train_X_17, train_Y_17, lambd=0.01, add_erasure=True)
y_pred = my_cls.classify(test_X_17)
y_pred_era = my_cls.TestCorrupted(p,test_X_17)
y_true = my_cls.onehot_2_scalar(test_Y_17)

acc = my_cls.accuracy_cal(y_true, y_pred)
print(acc)
acc_erasure = my_cls.accuracy_cal(y_true, y_pred_era)
print(acc_erasure)