import numpy as np
############################# supporting functions for Perceptron ################################
# Load train.csv and test.csv
with open('bank-note/train.csv') as f:
    training_data = [];
    for line in f:
        terms = line.strip().split(',')
        training_data.append(terms)

with open('bank-note/test.csv') as f:
    testing_data = [];
    for line in f:
        terms = line.strip().split(',')
        testing_data.append(terms)

# convert data to float
def convert_to_float(input_data):
    new_list = input_data
    for i in range(len(input_data)):
        for j in range(len(new_list[0])):
            new_list[i][j] = float(input_data[i][j])
    return new_list

training_data = convert_to_float(training_data)
testing_data = convert_to_float(testing_data)

# convert (0,1) labels t0 (-1,1)
def convert_to_pm_one(input_data):
    new_list = input_data
    for i in range(len(input_data)):
        if new_list[i][-1] != 1.0:
            new_list[i][-1] = -1.0
    return new_list

training_data = convert_to_pm_one(training_data)
testing_data = convert_to_pm_one(testing_data)

# dim of the feature vector
n = len(training_data[0])-1

############################# Perceptron Algorithm functions ################################

# get prediction with single perceptron
def single_perceptron(input_data,lrn_rate,shuffled_indx,w,b):
    for i in range(len(input_data)):
        perm_coeff = shuffled_indx[i]
        yi = input_data[perm_coeff][-1]
        sgn = np.inner(w,input_data[perm_coeff][0:n])+b
        if yi * sgn <= 0:
            w = w + [lrn_rate * yi * x for x in input_data[perm_coeff][0:n]]
            b = b + lrn_rate * yi
    return [w,b]

def error_counter(input_data,w,b):
    counter = 0
    input_len = len(input_data)
    for i in range(len(input_data)):
        yi = input_data[i][-1]
        sgn = np.inner(w,input_data[i][0:n])+b
        if yi * sgn <=0:
            counter += 1
    avg_error = counter/input_len
    return  avg_error

# repeat prediction for multiple epoaches
def multiple_perceptron(input_data,lrn_rate,w,b,epoch):
    for k in range(epoch):
        shuffled_indx = np.random.permutation(len(input_data))
        [w, b] = single_perceptron(input_data, lrn_rate, shuffled_indx, w, b)
    return [w,b]


################################### Run Standard Perceptron ########################################
def standard_perceptron(train_data,test_data,w0,b0, lrn_rate, epoch):
    [w, b] = multiple_perceptron(train_data, lrn_rate, w0, b0, epoch)
    avg_error = error_counter(test_data, w, b)
    return [w,b,avg_error]

# initialize parameters
w0 = np.zeros(n)
lrn_rate = 1
epoch = 10
b0 = 0

# print results
print('epoch' , ' ' , '                ' , 'weight vector' , '         ' ,'       ', '  bias'  , ' ' , ' error')
for T in range(epoch+1):
    [w,b,avg_error] = standard_perceptron(training_data, testing_data, np.zeros(n), 0, lrn_rate, T)
    print(T , ' ' , ' ' , w , ' ' ,  ' ', b   , '  ' , avg_error)