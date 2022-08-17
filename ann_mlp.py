from keras.models import Sequential
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report,accuracy_score, precision_score, recall_score, f1_score, mean_squared_error
from sklearn.model_selection import train_test_split, KFold
import numpy as np
np.random.seed(2) 
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt

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
    
def mlp(classes, input_shape):
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(input_shape, activation='relu'))
    model.add(keras.layers.Dense(64, activation='relu'))
    
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(rate=0.25))
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(32, activation='relu'))
    
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(rate=0.25))
    model.add(keras.layers.Dense(32, activation='relu'))
    if classes==1:
        loss_func = "binary_crossentropy"
        actv = 'sigmoid'
    else:
        loss_func = "categorical_crossentropy"
        actv='softmax'
    model.add(keras.layers.Dense(classes,   activation=actv))
    model.compile(loss=loss_func, optimizer="adam",metrics=["accuracy"])
    return model

dataset=process_data()
y = dataset['target']
x = dataset.drop(['target'],axis=1)
input_shape = x.shape[1]
classes = 1
batch_size = 25
epochs = 100
if classes!=1:
        y = keras.utils.to_categorical(y)
x_train,x_valid,y_train,y_valid = train_test_split(x,y,test_size=0.20, random_state = 2)
x_valid,x_test,y_valid,y_test = train_test_split(x_valid,y_valid,test_size=0.5, random_state = 2)
#x_test,y_test= x_valid,y_valid
model = mlp(classes,input_shape)
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_valid, y_valid), verbose=0)
print("Train Acc: ",history.history['accuracy'][-1])
print("Train loss: ",history.history['loss'][-1])
print("Val Acc: ",history.history['val_accuracy'][-1])
print("Val loss: ",history.history['val_loss'][-1])

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['Train_acc', 'Val_acc'])
plt.title("Traing Acc vs Validation Acc")
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Train_loss', 'Val_loss'])
plt.title("Traing Loss vs Validation Loss")
plt.show()

print(model.evaluate(x_test,y_test))
y_pred = (model.predict(x_test)> 0.5)
print("report",classification_report(y_test,y_pred)) 
print("accuracy_score",accuracy_score(y_test,y_pred))
print("precision_score",precision_score(y_test,y_pred)) 
print("recall_score",recall_score(y_test,y_pred))
print("f1_score",f1_score(y_test,y_pred))
print("mse",mean_squared_error(y_test,y_pred))

import sklearn.metrics as metrics
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

warnings.filterwarnings('ignore')
num_folds = 10
no_epochs = 100
batch_size = 25
dataset=process_data()
y = dataset['target'].values
x = dataset.drop(['target'],axis=1).values
input_shape = x.shape[1]
classes = 1
if classes!=1:
        y = keras.utils.to_categorical(y)
acc_per_fold = []
loss_per_fold = []
kfold = KFold(n_splits=num_folds, shuffle=True)
fold_no = 1
for train, test in kfold.split(x,y):
    model = mlp(classes,input_shape)
    history = model.fit(x[train], y[train],
                batch_size=batch_size,
                epochs=no_epochs,
                verbose=0)

    # Generate generalization metrics
    scores = model.evaluate(x[test], y[test], verbose=0)
    print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
    acc_per_fold.append(scores[1] * 100)
    loss_per_fold.append(scores[0])

    # Increase fold number
    fold_no = fold_no + 1
print("Mean Ac: ",np.mean(acc_per_fold))
print("Mean  Loss: ", np.mean(loss_per_fold))

