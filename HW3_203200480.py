import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics.pairwise import polynomial_kernel
#contants to change
T_rbf=100
T_poly = 100
sigma= 0.5
gamma = 0.5*sigma**2
q=0.2
# add a var to decide wether to run the rbf or the poly kernal
run = 'poly'
# get the train data an plot it 
train = np.loadtxt("data/train.csv", delimiter=',')
train=np.asarray(train)
pos_train =np.asarray([i for i in train if i[2]==1] )
neg_train =np.asarray([i for i in train if i[2]==-1] )
temp= np.asarray(pos_train)#pos_train[:, 0:1]
temp2= temp[:, 0:1]
# plot the data
plt.scatter(pos_train[:, 0:1],pos_train[:, 1:2],color='blue')
plt.scatter(neg_train[:, 0:1],neg_train[:, 1:2],color='orange')
plt.title('the train data')
#plt.show()

plt.savefig('train_data.png')

#plt.show()
m= len(train)

alpha_rbf = np.zeros(m)
alpha_poly = np.zeros(m)
# create the K matrix

K_rbf = rbf_kernel(train[:,0:2],train[:,0:2],gamma)#np.zeros((m,m))
k_poly = polynomial_kernel(train[:,0:2],train[:,0:2],coef0=q)





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
if run == 'rbf':
    temp=1
    #optimize for rbf
    for t in range(T_rbf):
        error_count_rbf =0
        for j in range(m):
            y_hat= np.sign(np.sum(alpha_rbf*K_rbf[j]))
            alpha_rbf[j]+=0.5*(train[j,2]-y_hat)
            if y_hat != train[j,2]:
                error_count_rbf +=1
            if t == T_rbf-1: #meaning this is the final round
                #add sample to plot
                if y_hat != train[j,2]:
                    plt.scatter(train[j, 0:1],train[j, 1:2],color='green') 
                elif y_hat == -1:
                    plt.scatter(train[j, 0:1],train[j, 1:2],color='orange') 
                elif y_hat == 1:
                    plt.scatter(train[j, 0:1],train[j, 1:2],color='blue') 

        print('after iter num {} we got {} errors in rbf'.format(t,error_count_rbf))  
    #start with rbf
    for j in range(n):
        y_hat= np.sign(np.sum(alpha_rbf*K_rbf[j]))
        shuff = test[j,2]
        if y_hat != test[j,2]:
            error_count_rbf +=1
            plt.scatter(test[j, 0:1],test[j, 1:2],marker= '*',color='green') 
        elif y_hat == -1:
            plt.scatter(test[j, 0:1],test[j, 1:2],marker= '*',color='orange') 
        elif y_hat == 1:
            plt.scatter(test[j, 0:1],test[j, 1:2],marker= '*',color='blue') 


    print('the number of erros of the rbf kernal is {}'.format(error_count_rbf))
    plt.title('RBF')
    plt.show()
    
    plt.savefig('rbf.png')
else:
    #optimize for poly
    for t in range(T_poly):
        error_count =0
        
        for j in range(m):
            y_hat= np.sign(np.sum(alpha_poly*k_poly[j]))
            alpha_poly[j]+=0.5*(train[j,2]-y_hat)
            if y_hat != train[j,2]:
                error_count +=1
            if t == T_poly-1: #meaning this is the final round
                #add sample to plot
                if y_hat != train[j,2]:
                    plt.scatter(train[j, 0:1],train[j, 1:2],color='green') 
                elif y_hat == -1:
                    plt.scatter(train[j, 0:1],train[j, 1:2],color='orange') 
                elif y_hat == 1:
                    plt.scatter(train[j, 0:1],train[j, 1:2],color='blue') 

        print('after iter num {} we got {} errors in poly'.format(t,error_count))  

    #now poly
    error_count_poly =0
    for j in range(n):
        y_hat= np.sign(np.sum(alpha_poly*k_poly[j]))
        shuff = test[j,2]
        if y_hat != test[j,2]:
            error_count_poly +=1
            plt.scatter(test[j, 0:1],test[j, 1:2],marker= '*',color='green') 
        elif y_hat == -1:
            plt.scatter(test[j, 0:1],test[j, 1:2],marker= '*',color='orange') 
        elif y_hat == 1:
            plt.scatter(test[j, 0:1],test[j, 1:2],marker= '*',color='blue') 

    print('the number of erros of the poly kernal is {}'.format(error_count_poly))
    plt.title('polynomial q degree')
    #plt.show()
    plt.savefig('poly.png')
    temp=1