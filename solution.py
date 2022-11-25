#PLEASE WRITE THE GITHUB URL BELOW!
#https://github.com/SeongEon-Kim/OSS_Project_AutoML
import pandas as pd
from sklearn.model_selection import train_test_split # 데이터 셋 나누기
from sklearn.tree import DecisionTreeClassifier # 결정 트리 분류기 

from sklearn.metrics import accuracy_score 
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from sklearn.ensemble import RandomForestClassifier # 랜덤 포레스트 분류기

from sklearn.svm import SVC # SVM 분류기
from sklearn.pipeline import make_pipeline # 전처리의 각 단계, 모델 생성, 학습 등을 포함하는 여러 단계의 머신러닝 프로세스를 한 번에 처리할 수 있는 클래스
from sklearn.preprocessing import StandardScaler # 전처리 스케일 조정

import sys

def load_dataset(dataset_path):
    
	dataset_df = pd.read_csv(dataset_path)
 
	return dataset_df

def dataset_stat(dataset_df):
    
    n_feats = len(dataset_df.keys())-1
    n_class0 = 0
    n_class1 = 0
    
    for i in range (len(dataset_df)):
        
        if dataset_df['target'][i] == 0:
            n_class0 +=1
            
        else :
            n_class1 +=1
            
    return n_feats, n_class0, n_class1

def split_dataset(dataset_df, testset_size):
	
	X = dataset_df.drop(columns="target", axis = 1)
	y = dataset_df["target"]

	x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = testset_size)

	return x_train, x_test, y_train, y_test
	

def decision_tree_train_test(x_train, x_test, y_train, y_test):
    
	dt_cls = DecisionTreeClassifier()
	dt_cls.fit(x_train, y_train)

	# default_arguments (https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html) 참고
	acc = accuracy_score(dt_cls.predict(x_test), y_test)
	prec = precision_score(dt_cls.predict(x_test), y_test, average='binary')
	recall = recall_score(dt_cls.predict(x_test), y_test)

	return acc, prec, recall

def random_forest_train_test(x_train, x_test, y_train, y_test):

	rf_cls = RandomForestClassifier()
	rf_cls.fit(x_train, y_train)
 
	acc = accuracy_score(rf_cls.predict(x_test), y_test)
	prec = precision_score(rf_cls.predict(x_test), y_test)
	recall = recall_score(rf_cls.predict(x_test), y_test)

	return acc, prec, recall

def svm_train_test(x_train, x_test, y_train, y_test):
	
	svm_pipe = make_pipeline(
    	StandardScaler(),
    	SVC()
	)
	svm_pipe.fit(x_train, y_train)
	
	acc = accuracy_score(y_test, svm_pipe.predict(x_test))
	prec = precision_score(y_test, svm_pipe.predict(x_test))
	recall = recall_score(y_test, svm_pipe.predict(x_test))

	return acc, prec, recall

def print_performances(acc, prec, recall):
	#Do not modify this function!
	print ("Accuracy: ", acc)
	print ("Precision: ", prec)
	print ("Recall: ", recall)

if __name__ == '__main__':
	#Do not modify the main script!
	data_path = sys.argv[1]
	data_df = load_dataset(data_path)

	n_feats, n_class0, n_class1 = dataset_stat(data_df)
	print ("Number of features: ", n_feats)
	print ("Number of class 0 data entries: ", n_class0)
	print ("Number of class 1 data entries: ", n_class1)

	print ("\nSplitting the dataset with the test size of ", float(sys.argv[2]))
	x_train, x_test, y_train, y_test = split_dataset(data_df, float(sys.argv[2]))

	acc, prec, recall = decision_tree_train_test(x_train, x_test, y_train, y_test)
	print ("\nDecision Tree Performances")
	print_performances(acc, prec, recall)

	acc, prec, recall = random_forest_train_test(x_train, x_test, y_train, y_test)
	print ("\nRandom Forest Performances")
	print_performances(acc, prec, recall)

	acc, prec, recall = svm_train_test(x_train, x_test, y_train, y_test)
	print ("\nSVM Performances")
	print_performances(acc, prec, recall)