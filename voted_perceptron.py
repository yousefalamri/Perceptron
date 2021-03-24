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

# convert data float
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

def remove_zeros(Ck):
    new_list = [];
    for j in range(len(Ck)):
        if Ck[j] != 0:
            new_list.append(Ck[j])
    return new_list

################################### voted perceptron functions ########################################


def calculate_vote(input_data, lrn_rate, w, b, epoch):
    input_len = len(input_data)
    shuffled_indx = []
    for i in range(epoch):
        shuffled_indx = shuffled_indx + np.random.permutation(input_len).tolist()
    w_and_b = []
    Ck = np.zeros(epoch * input_len).tolist()
    m = 0
    for i in range(epoch * input_len):
        perm_coeff = shuffled_indx[i]
        yi = input_data[perm_coeff][-1]
        prod = np.inner(w,input_data[perm_coeff][0:n])+b
        if yi * prod <= 0:
            w = w + [lrn_rate * yi * x for x in input_data[perm_coeff][0:n]]
            b = b + lrn_rate * yi
            m = m + 1
            wi_and_bi = np.append(w, b)
            w_and_b.append(wi_and_bi)
            Ck[m] = 1
        elif yi * prod > 0:
            Ck[m] += 1
    return [w_and_b, remove_zeros(Ck)]


def convert_array_to_list(w_and_b,Ck):
    output_list = []
    for k in range(len(w_and_b)):
        output_list.append(w_and_b[k].tolist() + [Ck[k]])
    return output_list

def sgn(input):
    output = 0
    if input > 0:
        output = 1
    else:
        output = -1
    return output

def calculate_prediction_error(input_data,w_and_b,Ck):
    input_len = len(input_data)
    w_and_b_and_Ck = convert_array_to_list(w_and_b, Ck)
    prediction_list = []
    error_count = 0
    for i in range(len(input_data)):
        sign = sgn(sum([(this_iter[-1]) * sgn(np.inner(input_data[i][0:n], this_iter[0:n]) + this_iter[n])
                        for this_iter in w_and_b_and_Ck]))
        prediction_list.append(sign)
    for i in range(len(input_data)):
        if prediction_list[i] != input_data[i][-1]:
            error_count += 1
    return error_count/input_len

################################### Run voted perceptron ########################################

train_data = training_data
test_data = testing_data
w0 = np.zeros(n)
b0 = 0
lrn_rate = 1
epoch = 10

def voted_perceptron(train_data, test_data, w0, b0, lrn_rate, epoch):
    for T in range(epoch+1):
        [w_and_b, Ck] = calculate_vote(train_data, lrn_rate, w0, b0, T)
        print('T=',T)
        print(convert_array_to_list(w_and_b, Ck))
        prediction_error = calculate_prediction_error(test_data, w_and_b, Ck)
        print('Error=',prediction_error)


voted_perceptron(train_data, test_data, w0, b0, lrn_rate, epoch)