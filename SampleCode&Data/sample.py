import numpy as np

from sklearn.metrics import accuracy_score

from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest

from sklearn.svm import SVC

import sys


def read_data(data_dir):
    f = open(data_dir)
    data_list = f.readlines()
    return data_list

def fs_SelectKBest(train_X, train_y, feature_num):
	train_X_new = SelectKBest(chi2, k=feature_num).fit(train_X, train_y)
	top_k_idx = np.argsort(train_X_new.scores_.tolist())[-feature_num:]
	
	return top_k_idx

def obtain_data(train_file, test_file, feature_idx):	
	train_X = []
	train_y = []
	train_data = read_data(train_file)
	for i in range(len(train_data)):
		line = train_data[i].strip('\n').split(',')
		train_y.append(int(line[len(line) - 1]))
		tmp = []
		for j in range(len(feature_idx)):
			tmp.append(float(line[feature_idx[j]]))
		train_X.append(tmp)
		
	test_X = []
	test_y = []
	test_data = read_data(test_file)
	for i in range(len(test_data)):
		line = test_data[i].strip('\n').split(',')
		test_y.append(int(line[len(line) - 1]))
		tmp = []
		for j in range(len(feature_idx)):
			tmp.append(float(line[feature_idx[j]]))
		test_X.append(tmp)
	
	return np.array(train_X), train_y, np.array(test_X), test_y
	
def SVM(train_X, train_y, test_X, test_y):
	clf = SVC(kernel = 'linear', gamma='auto')
	clf.fit(train_X, train_y)
	predict = clf.predict(test_X)
	acc = accuracy_score(test_y, predict.tolist())
		
	return predict, acc

def feature_selection(train_file, feature_num):	
	train_data = read_data(train_file)
	train_X = []
	train_y = []
	for i in range(len(train_data)):
		line = train_data[i].strip('\n').split(',')
		train_y.append(int(line[len(line) - 1]))
		tmp = []
		for j in range(0, len(line) - 1):
			tmp.append(float(line[j]))
		train_X.append(tmp)	
	feature_idx_SelectKBest = fs_SelectKBest(train_X, train_y, feature_num) 	
	
	return feature_idx_SelectKBest
	    
if __name__ == "__main__":

	data = read_data("pima-indians-diabetes.csv")
	
	ratio = 0.8
	train = open('train.txt', 'w')
	for i in range(int(ratio * len(data))):
		train.write(data[i])
	train.close()
	
	test = open('test.txt', 'w')
	for i in range(int(ratio * len(data)), len(data)):
		test.write(data[i])
	test.close()

	#Feature selection with chi2 methods
	feature_idx = feature_selection("train.txt", 3)
	print('Index of selected features:', feature_idx)
	
	#Build new data for training and testing with selected features "feature_idx"
	train_X, train_y, test_X, test_y = obtain_data("train.txt", "test.txt", feature_idx)
	
	#Using logistic regression to complete classification and evaluation
	predict, acc = SVM(train_X, train_y, test_X, test_y)
	print("Prediction accuracy is ", acc)
		
	