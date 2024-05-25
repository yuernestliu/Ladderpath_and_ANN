#!/usr/bin/env python
# coding: utf-8

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import keras_tuner
import hyperopt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt import rand

#  parameter setting
layer = [8,2,2,2]  #change model network parameters, and enter parameters directly，（Hidden layers are set to a maximum of 4 layers, with the last '2' being the output layer）
path = r'./mod2dim3.xlsx'#data storage path
bin_list = [0.1]  #coarse-grained interval
learning_rate = 0.35 #learning rate
epochs = 10 #training epochs
eta_random_time = 10 #The number of times random was used to calculate eta
input_dim = 3 # the input dimension of data

if len(layer)-1 ==1 :
    from layer1_function_mod import *
  
if len(layer)-1 ==2 :
    from layer2_function_mod import *
   
if len(layer)-1 ==3 :
    from layer3_function_mod import *
    
if len(layer)-1 ==4 :
    from layer4_function_mod import *
def load_data(path = r'./mod2dim3.xlsx'):
    data =  pd.read_excel(path)
    data_array = np.array(data)
    X_data = data_array[:,0:3] #X sample data
    y_data = data_array[:,3] #y label data
    X_data.shape,y_data.shape
    #import random
    np.random.seed(12345)
    permutation = np.random.permutation(y_data.shape[0])
    X_train0 = X_data[permutation, :]
    y_train0 = y_data[permutation]
    X_train = X_train0[0:900, :]
    y_train = y_train0[0:900]
    X_test = X_train0[900:1000, :]
    y_test = y_train0[900:1000]
    #print(y_test)
    return X_train,y_train,X_test,y_test

total_path = []
path_list_total = []
acc_list = []
eta_last_list = []
order_list = []
Ladderpath_list = []
path_list_total_list = []
path = path
X_train,y_train,X_test,y_test = load_data(path = path)
ming_unique = load_ming_dic()
bin_list = bin_list

best_epochs_weight_list,loss_list = create_model(x=X_train,y=y_train,period = epochs,epochs = epochs,lr=learning_rate,layer123=layer,input_dim=input_dim)
train_acc_list,test_acc_list,grouping_weight_list,total_path,path_list_total,Ladderpath,Order = main_process(best_epochs_weight_list,layer123=layer,X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,ming_unique=ming_unique,bin_list=bin_list,input_dim=input_dim)
eta_init_list = []
for i in range(len(path_list_total)):
    list_change1,random_order = change_word(a=path_list_total[i],eta_random_time=eta_random_time)
    list_change2,max_order_all = max_order(a=path_list_total[i])
    eta=(np.array(Order[i])-np.array(random_order))/(np.array(max_order_all)-np.array(random_order))
    eta_init_list.append(eta)

  
print('acc:',train_acc_list)
print('eta:',eta_init_list)

