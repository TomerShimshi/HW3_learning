import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics.pairwise import polynomial_kernel
import scipy

T_rbf=50
T_poly = 100
sigma_max= 50
def RBF (x1,x2,sigma):
    k= np.zeros((147,147))
    for i in range(len(x1)):
        for j in range(len(x2)):
            temp1 =x1[i]
            temp=  scipy.spatial.distance.euclidean(x1[i],x2[j])
            k[i][j]=np.exp(-(temp/(2*sigma**2)))
    return k
q=0.2
# add a var to decide wether to run the rbf or the poly kernal

# get the train data an plot it 
train = np.loadtxt("data/train.csv", delimiter=',')
train=np.asarray(train)
best_sigma =4
min_errors = 80
for sigma in range(1,sigma_max):
    sigma = float(sigma/10.0)
    #gamma = 0.5*sigma**2
    #plt.show()
    m= len(train)

    alpha_rbf = np.zeros(m)
    # create the K matrix

    #K_rbf = rbf_kernel(train[:,0:2],train[:,0:2],gamma)#np.zeros((m,m))
    K_rbf = RBF(train[:,0:2],train[:,0:2],sigma)#rbf_kernel(train[:,0:2],train[:,0:2],gamma)#np.zeros((m,m))


    # Test the 2 alogorithems

    #first we load the data
    test = np.loadtxt("data/test.csv", delimiter=',')
    test=np.asarray(test)

    # now we check the test
    n = len(test)
    error_count_rbf =0
    error_count_poly =0
    pos_labels_rbf=[]
    neg_labels_rbf=[]

    pos_labels_poly=[]
    neg_labels_poly=[]
    #optimize for rbf
    for t in range(T_rbf):
        error_count_rbf =0
        for j in range(m):
            y_hat= np.sign(np.dot(alpha_rbf,K_rbf[:,j]))
            alpha_rbf[j]+=0.5*(train[j,2]-y_hat)
            if y_hat != train[j,2]:
                error_count_rbf +=1
 
    #start with rbf
    for j in range(n):
        y_hat= np.sign(np.dot(alpha_rbf,K_rbf[:,j]))
        shuff = test[j,2]
        if y_hat != test[j,2]:
            error_count_rbf +=1
    if error_count_rbf <min_errors:
        min_errors = error_count_rbf
        best_sigma=sigma

print('the min number of erros is {} and the best sigma is {} '.format(min_errors,best_sigma))
