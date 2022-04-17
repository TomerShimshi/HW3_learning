import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics.pairwise import polynomial_kernel
#contants to change
T_rbf=100
T_poly = 100
sigma= 0.5
gamma = 0.5*sigma**2
q=1
# get the train data an plot it 
train = np.loadtxt("data/train.csv", delimiter=',')
train=np.asarray(train)
pos_train =np.asarray([i for i in train if i[2]==1] )
neg_train =np.asarray([i for i in train if i[2]==-1] )
temp= np.asarray(pos_train)#pos_train[:, 0:1]
temp2= temp[:, 0:1]
# plot the data
#plt.scatter(pos_train[:, 0:1],pos_train[:, 1:2],color='red')
#plt.scatter(neg_train[:, 0:1],neg_train[:, 1:2],color='blue')
#plt.show()
m= len(train)

alpha_rbf = np.zeros(m)
alpha_poly = np.zeros(m)
# create the K matrix

K_rbf = rbf_kernel(train[:,0:2],train[:,0:2],gamma)#np.zeros((m,m))
k_poly = polynomial_kernel(train[:,0:2],train[:,0:2],coef0=q)

#optimize for rbf
for t in range(T_rbf):
    error_count_rbf =0
    for j in range(m):
        y_hat= np.sign(np.sum(alpha_rbf*K_rbf[j]))
        alpha_rbf[j]+=0.5*(train[j,2]-y_hat)
        if y_hat != train[j,2]:
            error_count_rbf +=1

    print('after iter num {} we got {} errors in rbf'.format(t,error_count_rbf))  


#optimize for poly
for t in range(T_poly):
    error_count_rbf =0
    error_count_poly =0
    for j in range(m):
        y_hat= np.sign(np.sum(alpha_poly*K_rbf[j]))
        alpha_poly[j]+=0.5*(train[j,2]-y_hat)
        if y_hat != train[j,2]:
            error_count_rbf +=1

    print('after iter num {} we got {} errors in poly'.format(t,error_count_rbf))  
temp=1