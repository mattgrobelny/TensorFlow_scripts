from sklearn import svm

import sys

# Goal: develope a MLA to predict manner of death from Sex Age MaritalStatus info

# define the path to the file
training_dataset = "/home/mgrobelny/Data/Death_data/DeathRecords/DeathRecords_10k_training.tsv"
test_dataset = "/home/mgrobelny/Data/Death_data/DeathRecords/DeathRecords_10k_test.tsv"
MannerOfDeath_translator_file = "/home/mgrobelny/Data/Death_data/DeathRecords/MannerOfDeath.csv"

x_data_train = ()

y_data_train = ()

# open training_dataset
train_dataset_fh = open(training_dataset, "r")

# Set up intial dictionaries
# SEX
sex_dictionary = {"M": 0, "F": 1}

MaritalStatus_dictionary = {"D": 0, "M": 1, "S": 2, "U": 3, "W": 4}
# read each line and remove the newlines
for line in train_dataset_fh:
    line_array = line.split('\t')
    # change SEX
    line_array[1] = sex_dictionary[line_array[1]]
    # Change MaritalStatus
    line_array[3] = MaritalStatus_dictionary[line_array[3]]
    x_data_train.append(line_array[1:4])
    y_data_train.append(line_array[5])


train_dataset_fh.close()

print x_data_train[1:10]
print y_data_train[1:10]

# Build SVM

clf = svm.SVC()
clf.fit(x_data_train, y_data_train)

# SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
#    decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
#    max_iter=-1, probability=False, random_state=None, shrinking=True,
#    tol=0.001, verbose=False)
#

# Open test dataset
test_dataset_fh = open(training_dataset, "r")

x_data_test = ()
y_data_test = ()

for line in test_dataset_fh:
    line_array = l i n e . s p l i t ( ' \ t ' ) 
    # change SEX
    line_array[1] = sex_dictionary[line_array[1]]
    # Change MaritalStatus
    line_array[3] = MaritalStatus_dictionary[line_array[3]]
    x_data_test.append(line_array[1:4])
    y_data_test.append(line_array[5])

test_dataset_fh.close()

print x_data_test[1:10]
print y_data_test[1:10]

acuraccy = clf.score(x_data_test, y_data_test)

print acuraccy
