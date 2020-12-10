import time
start_time = time.time() #add taimer
import xlsxwriter
workbook = xlsxwriter.Workbook('arrays.xlsx')
worksheet = workbook.add_worksheet()
import numpy as np
np.random.seed(7)                                      # control the random initializatioon


                                       # Number of neurons in the hidden layer
alpha = 0.025                                          # Learning rate
n_iterations = 10000
iii = 0
x=[]
array=np.zeros((10,10))
   
for iii in range(0,4): 
    x=[]                 #creating an array for x(k)
#     for i in range(0,10):
#         x.append(0)
        
    n_hidden = 10  
    if iii ==1: 
        n_iterations = 20000
    elif iii==2: 
        n_iterations = 50000
    elif iii==3:
        n_iterations = 750000
    elif iii==4:
        n_iterations = 100000
    else:
        n_iterations = 10000   
    
#         
#     col = 0
#     newrow = np.array([x])
#     array = np.vstack((array,newrow))
#                                 
#                        
#     for row, data in enumerate(array):
#         worksheet.write_row(row, col, data)
#         
    
        ### Load Dataset
    dataset = np.loadtxt("pima-indians-diabetes.csv", delimiter=",")
    X_raw = dataset[:,0:8]
    y_raw = dataset[:,8]
        
        # Prepare the data
    X = X_raw.T
    y = y_raw.T
        
        # Data information
    m = X.shape[1]
    num_features = X.shape[0]
    num_classes = np.size(np.unique(y))
        
    print ("Number of training examples: " + str(m))
    print ("Number of features = " + str(num_features))
    print ("Number of classes = " + str(num_classes))
        
        
        # Sigmoid Function
    def sigmoid(z):
        s = 1 / (1 + np.exp(-z) )
        return s
    
    while (n_hidden<=160):       
        # initialize parameters randomly
        W1 = 0.01 * np.random.randn(n_hidden, num_features)
        b1 = np.zeros((n_hidden, 1))
        W2 = 0.01 * np.random.randn(1, n_hidden)
        b2 = np.zeros((1, 1))
        
        
        for i in range(n_iterations):
          
            # First Layer
            ################## Please add two equations ##############################
            Z1 = np.dot(W1,X)+b1
            A1 = sigmoid(Z1)
            ################# End of your Code #######################################
            
            # Second layer
            ################# Please add two layers #################################
            Z2 = np.dot(W2,A1)+b2
            A2 = sigmoid(Z2)
            ################# End of Code ###########################################
            
            loss = (-1/m)* np.sum(y*np.log(A2) + (1-y)*np.log(1-A2))
        
#             if i % 1000 == 0:
#                 print("iteration %d: loss %f" % (i, loss))
                
            ################# Please add six equations for Backpropagation ##########
            dZ2 = A2-y
            dW2 = (1/m)*np.dot(dZ2,A1.T)
            db2 = (1/m)*np.sum(dZ2,axis=1,keepdims=True)
            dZ1 = np.dot(W2.T,dZ2)*(A1*(1-A1))
            dW1 = (1/m)*np.dot(dZ1,X.T)
            db1 = (1/m)*np.sum(dZ1,axis=1,keepdims=True) 
            ########################## End of your code ##############################
            
            # perform a parameter update
            W1 = W1 - alpha * dW1
            b1 = b1 - alpha * db1
            W2 = W2 - alpha * dW2
            b2 = b2 - alpha * db2
            
            
        # evaluate training set accuracy
        tZ1 = np.dot(W1, X) + b1
        tA1 = sigmoid(tZ1)
        tZ2  = np.dot(W2, tA1) + b2
        tA2 = sigmoid(tZ2)
        
        predictions = (A2 > 0.5)
        print('hidden layers=', n_hidden)
    
        print('training accuracy: %.2f' % (np.mean(predictions == y)))
        a=(float(round(np.mean(predictions == y),2)))
        x.insert(0,a)
        n_hidden = n_hidden+25
        print('accurasy on iteration', n_iterations,'=', x)

    iii = iii+1
     
print ("My program took", time.time() - start_time, "to run")
