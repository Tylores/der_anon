import numpy as np
import pandas as pd
from time import time
import random
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn import svm, datasets
import matplotlib.pyplot as plt
import numpy as np

def loadData(filepath: str) -> pd.DataFrame:
    data = pd.read_csv(filepath)
    if data.isnull().values.any():
        print('data is missing values')
        exit()
    return data

def simulateDevice(data: pd.DataFrame, type: str) -> pd.DataFrame:
    size = 10
    power = [0]*size
    energy = [0]*size
    duration = [0]*size
    interval = [0]*size

    df = data[data['type'] == type]
    model = ['simulation']*size
    mode = ['normal']*size
    type = [type]*size
    n,m = df.shape
    for i in range(size):
        if n > 1:
            power[i] = random.randint(df['power'].min(), df['power'].max())
            energy[i] = random.randint(df['energy'].min(), df['energy'].max())
            duration[i] = random.randint(df['duration'].min(), df['duration'].max())
            interval[i] = random.randint(df['interval'].min(), df['interval'].max())
        else:
            power[i] = random.randint(
                round(df['power'].max()*0.95), 
                round(df['power'].max()*1.05)
            )
            energy[i] = random.randint(
                round(df['energy'].max()*0.95), 
                round(df['energy'].max()*1.05)
            )
            duration[i] = random.randint(
                round(df['duration'].max()*0.95), 
                round(df['duration'].max()*1.05)
            )
            interval[i] = random.randint(
                round(df['interval'].max()*0.95), 
                round(df['interval'].max()*1.05)
            )
    new_df = pd.DataFrame(
        {
            'model': model,
            'type': type,
            'mode': mode,
            'power': power,
            'energy': energy,
            'duration': duration,
            'interval': interval
        },
        index=range(size)
    )

    return pd.concat([data,new_df],ignore_index=True)

def simulateVariance(data: pd.DataFrame) -> pd.DataFrame:
    data = simulateDevice(data, 'air conditioner')
    data = simulateDevice(data, 'cloth washer')
    data = simulateDevice(data, 'clothes dryer')
    data = simulateDevice(data, 'dishwasher')
    data = simulateDevice(data, 'refrigerator')
    data = simulateDevice(data, 'water heater')
    return data

def runSVM(data: pd.DataFrame, name: str):
    '''https://towardsdatascience.com/multiclass-classification-with-support-vector-machines-svm-kernel-trick-kernel-functions-f9d5377d6f02'''
    X = data.filter(['energy', 'power'], axis=1).to_numpy()
    y, uniques = pd.factorize(data['type'])

    X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.8, random_state=0)

    linear = svm.SVC(kernel='linear', C=1, decision_function_shape='ovo').fit(X_train, y_train)

    #stepsize in the mesh, it alters the accuracy of the plotprint
    #to better understand it, just play with the value, change it and print it
    h = 1
    #create the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))

    # Plot also the training points
    fig = plt.figure()
    sctr = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.PuBuGn, edgecolors='grey')
    cbar = plt.colorbar(sctr)
    cbar.ax.set_yticklabels(uniques)
    plt.xlabel('Energy (Wh)')
    plt.ylabel('Power (W)')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    
    Z = linear.predict(np.c_[xx.ravel(), yy.ravel()])
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    CS = plt.contourf(xx, yy, Z, cmap=plt.cm.PuBuGn, alpha=0.7)
    plt.savefig('outputs/'+name+'_linear_svm.svg', bbox_inches='tight')

    
    # retrieve the accuracy
    accuracy_lin = linear.score(X_test, y_test)
    print('Accuracy Linear Kernel: ', accuracy_lin)

    # creating a confusion matrix
    cm_lin = confusion_matrix(y_test, linear.predict(X_test))
    print(cm_lin)

    fig2 = plt.figure()
    CM = ConfusionMatrixDisplay.from_estimator(
        linear,
        X_test, 
        y_test, 
        display_labels=uniques, 
        cmap=plt.cm.PuBuGn,
        xticks_rotation=45
    )
    plt.savefig('outputs/'+name+'_confusion.svg', bbox_inches='tight')
        

if __name__ == '__main__':
    data = loadData('data/frr.csv')
    data = simulateVariance(data)
    runSVM(data, 'base')

    data2 = loadData('data/frr_split.csv')
    data2 = simulateVariance(data2)
    runSVM(data2, 'split')

