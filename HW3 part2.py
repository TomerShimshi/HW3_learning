from distutils.log import error
import numpy as np
from matplotlib import pyplot as plt


# Decision stump used as weak classifier
class DecisionStump:
    def __init__(self):
        self.polarity = 1
        self.feature_idx = None
        self.threshold = None
        

    def predict(self, X):
        if X.ndim>1:
            r=1
            n_samples = X.shape[0]
            X_column = X[:, self.feature_idx]
        else:
            n_samples=1
            X_column=X[self.feature_idx]
        predictions = np.ones(n_samples)
        if self.polarity == 1:
            predictions[X_column < self.threshold] = -1
        else:
            predictions[X_column > self.threshold] = -1

        return predictions
#load the data
train = np.loadtxt("data/MNIST_train_images.csv", delimiter=',')
train=np.asarray(train)
train_labels = np.loadtxt("data/MNIST_train_labels.csv", delimiter=',')
train_labels=np.asarray(train_labels)
#load the test data
test = np.loadtxt("data/MNIST_test_images.csv", delimiter=',')
test=np.asarray(test)
test_labels = np.loadtxt("data/MNIST_test_labels.csv", delimiter=',')
test_labels=np.asarray(test_labels)
temp = train[0].ndim



#plot 1 example from the train set 
plt.imshow(np.asarray(np.reshape(train[6, :], (28, 28))),cmap='gray', vmin=0, vmax=255)
plt.title('1 train image')
plt.savefig('1 train image.png')
# adjustable T
T= 30
m,n_featurs= train.shape
alpha = np.zeros(T)
p= (1/m)*np.ones((T+1,m))
errors_train = np.zeros(T)
errors_test = np.zeros(T)
clfs=[]
n_clf=m+1
for t in range(T):
    ty=2
    
    min_error = float("inf")
    clf = DecisionStump()
    #now we use a greedy al to find the  find best threshold and feature
    for featur_i in range(n_featurs):
        x_col = train[:,featur_i]
        thresholds = np.unique(x_col)
        for threshold in thresholds:
            #prdict pos polarity
            w=1
            predictions = np.ones(m)
            predictions[x_col<threshold]=-1
            missclasified = p[t][train_labels != predictions]
            error = np.sum(missclasified)
            if (error> 0.5):
                error = 1-error
                w=-1
            if error < min_error:
                min_error = error
                clf.polarity = w
                clf.threshold =threshold
                clf.feature_idx = featur_i
        #now we have the best h
    EPS = 1e-10 #so we wont devide by zero
    alpha[t] = 0.5*np.log((1.0-min_error+EPS)/(min_error+EPS))
    preds = clf.predict(train)
    #not sure about this
    sum=0
    for i in range(m):
        sum+=p[t][i]*np.exp(-alpha[t]*train_labels[i]*clf.predict(train[i]))
    #p[t+1][clf.feature_idx]=(p[t][clf.feature_idx]*np.exp(-alpha[t]*train_labels*preds))/sum
    for i in range(m):
        p[t+1][i]=(p[t][i]*np.exp(-alpha[t]*train_labels[i]*preds[i]))/sum
    
    #not sure about this
    #p[t+1]/=np.sum(p[t+1])
    clfs.append(clf)
    clf_preds = [alpha[t] * clf.predict(train) for clf in clfs]
    y_pred = np.sum(clf_preds, axis=0)
    y_pred = np.sign(y_pred)
    missclasified_train = [train_labels != y_pred]
    error_train = np.sum(missclasified_train)
    #add the error for the plot
    errors_train[t]=error_train
    
    #now for the test
    clf_preds = [alpha[t] * clf.predict(test) for clf in clfs]
    y_pred = np.sum(clf_preds, axis=0)
    y_pred = np.sign(y_pred)
    missclasified_test = [test_labels != y_pred]
    error_test = np.sum(missclasified_test)
    errors_test[t]=error_test

    print('error in iteration {} for the train is ={}'.format(t,error_train))
    print('error in iteration {} for the test is ={}'.format(t,error_test))
   

              

                

                #now we find the error

clf_preds = [alpha[t] * clf.predict(test) for clf in clfs]
y_pred = np.sum(clf_preds, axis=0)
y_pred = np.sign(y_pred)
missclasified = [test_labels != y_pred]
error = np.sum(missclasified)
accurecy = 1.0 - error/len(test_labels)
print('on the test we got {} errors out of {} and accurecy of {}'.format(error,len(test_labels),accurecy))
t=[i for i in range(T)]
plt.figure()

plt.plot(t,errors_test,  color='g', label='test errors')
plt.plot(t,errors_train, color='r', label='train errors')
# Naming the x-axis, y-axis and the whole graph
plt.xlabel("Iteration")
plt.ylabel("Errors")
plt.title("train and test errors per iteration")
plt.legend()
plt.savefig('part2 train and test errors.png')
temp=1