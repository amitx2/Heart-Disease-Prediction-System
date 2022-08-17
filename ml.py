import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report,accuracy_score, precision_score, recall_score, f1_score, mean_squared_error
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
np.random.seed(42)

def process_data():
    data = pd.read_csv("heart.csv")
    categorical_val = []
    continous_val = []
    for column in data.columns:
        if len(data[column].unique()) <= 10:
            categorical_val.append(column)
        else:
            continous_val.append(column)
    categorical_val.remove('target')
    dataset = pd.get_dummies(data, columns = categorical_val)

    s_sc = StandardScaler()
    col_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    dataset[col_to_scale] = s_sc.fit_transform(dataset[col_to_scale])
    return dataset

models = [LogisticRegression(),
          DecisionTreeClassifier(),
          RandomForestClassifier(), 
          SVC()]

dataset=process_data()
y = dataset['target']
x = dataset.drop(['target'],axis=1)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
model_name=[]
train_acc=[]
test_acc=[]
precision =[]
recall=[]
f1_sc = []
mse = []
for model in models:
    model.fit(x_train,y_train)
    model_name.append(type(model).__name__)
    train_acc.append(accuracy_score(y_train,model.predict(x_train)))
    test_acc.append(accuracy_score(y_test,model.predict(x_test)))
    precision.append(precision_score(y_test,model.predict(x_test)))
    recall.append(recall_score(y_test,model.predict(x_test)))
    f1_sc.append(f1_score(y_test,model.predict(x_test)))
    mse.append(mean_squared_error(y_test,model.predict(x_test)))
result = pd.DataFrame({"Model":model_name,"Train Acc":train_acc,"Test Acc":test_acc,"Precision":precision,"Recall":recall,"f1_score":f1_sc,"MSE":mse})
result

def auc_graph(model, x_train, y_train, x_test, y_test):
    import sklearn.metrics as metrics
    model.fit(x_train,y_train)
    y_pred = (model.predict(x_test)> 0.5)
    fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred)
    roc_auc = metrics.auc(fpr, tpr)

    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

model = DecisionTreeClassifier()
auc_graph(model,x_train,y_train,x_test,y_test)

