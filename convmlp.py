import torch.optim as optim
import torch.nn as nn
import torch
from torch.autograd import Variable
from convmlp import convmlp_l,convmlp_m,convmlp_s
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report,accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import numpy as np

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

def iterate_minibatches(inputs, targets, batch_size, shuffle=True):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]
        
def get_accuracy(x_test1,y_test1):
    x_test = x_test1.to_numpy()
    x_test = x_test.reshape(x_test.shape[0],1,x_test.shape[1])
    x_test = x_test.astype(np.float32)
    x_test = torch.from_numpy(x_test)
    y_test = y_test1.to_numpy()
    y_test = torch.from_numpy(y_test)
    with torch.no_grad():
        output = model(x_test)
    y_pred = []
    for i in output:
        y_pred.append((np.argmax(i)))

    y_true = []
    for i in y_test:
        y_true.append((np.argmax(i)))
    return accuracy_score(y_true,y_pred)

def train(model,X,Y,training_epochs,batch_size):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    training_epochs = training_epochs
    batch_size = batch_size   
    train_acc = []
    train_losses = []
    for e in range(training_epochs):
        acc= []
        losses = [] 
        for batch in iterate_minibatches(X,Y, batch_size):
            x, y = batch
            inputs, targets = torch.from_numpy(x), torch.from_numpy(y)
            torch.set_printoptions(threshold=1000)
            #print(inputs.shape)
            #print(inputs)
            #break
            optimizer.zero_grad()   
            output = model(inputs)
                
            loss = criterion(output, targets.float())
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

            output1 = output.detach()
            y_pred = []
            for i in output1:
                y_pred.append((np.argmax(i)))

            y_true = []
            for i in targets:
                y_true.append((np.argmax(i)))
            acc.append(accuracy_score(y_true,y_pred))
        train_losses.append(np.mean(losses))
        train_acc.append(np.mean(acc))
        print("Epoch: {}/{}...".format(e+1, training_epochs),
                    "Train Loss: {:.4f}...".format(np.mean(train_losses)),
                    "Train Acc:{:.4f}".format(np.mean(acc)))
    print("Test Acc: ", get_accuracy(x_test,y_test))
    return train_acc,train_losses

dataset=process_data()
y_cat = pd.get_dummies(dataset['target'], columns = ['target'])
x = dataset.drop(['target'],axis=1)
x_train,x_test,y_train,y_test = train_test_split(x,y_cat,test_size=0.20)
X,Y=x_train.to_numpy(), y_train.to_numpy()
X = X.reshape(X.shape[0],1,X.shape[1])
#Y = Y.reshape(Y.shape[0],1)
X = X.astype(np.float32)

model = convmlp_s()
history = train(model,X,Y,50,25)

import matplotlib.pyplot as plt
plt.plot(history[0])
plt.plot(history[1])
plt.legend(['acc','loss'])
plt.title("Accuracy and loss")
plt.show()