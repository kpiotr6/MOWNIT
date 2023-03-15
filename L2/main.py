import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from random import randint
import scipy.linalg as scp
def draw(data):
    data.hist("texture (mean)", by="Malignant/Benign", grid=True, bins=[0, 5, 10, 15, 20, 25, 30, 35, 40],
                    range=(0, 90))
    axs = plt.gcf().get_axes()
    for i in range(0, 2):
        axs[i].set_xlabel("texture (mean)")
        axs[i].set_ylabel("number")
        axs[i].grid()
        axs[i].set_ylim([0, 90])
    plt.figure()
    p = data.plot(y="smoothness (mean)")
    p.set_ylabel("smoothness (mean)")
    p.set_xlabel("sequence number")
    plt.show()

def to_quadratic(data):
    quadratic_data = np.zeros((len(data), 14))
    for i in range(len(data)):
        for j in range(4):
            quadratic_data[i][j] = data[i][j]
        for j in range(4,8):
            quadratic_data[i][j] = data[i][j-4]**2
        for j in range(8,11):
            quadratic_data[i][j] = data[i][0]*data[i][j-7]
        quadratic_data[i][11] = data[i][1]*data[i][2]
        quadratic_data[i][12] = data[i][1] * data[i][3]
        quadratic_data[i][13] = data[i][2] * data[i][3]
    return quadratic_data

def type(list):
    num_types = np.zeros(len(list))
    for i in range(len(list)):
        if list[i]=="M":
            num_types[i] = 1
        else:
            num_types[i] = -1
    return num_types

def get_Aw(data,b,quadratic=False):
    if quadratic:
        A = np.array(to_quadratic(data.loc[:, ["radius (mean)", "perimeter (mean)", "area (mean)", "symmetry (mean)"]].values))
    else:
        A = np.array(data_train.iloc[:, 2:len(data_train.columns)].values)
    AT= np.transpose(A)
    ATA = np.linalg.inv(np.dot(AT,A))
    w = np.dot(np.dot(ATA,AT),b)
    return A,w

if __name__ == '__main__':
    with open("breast-cancer.labels","r") as f:
        labels = [line.rstrip() for line in f]
    plt.figure()
    data_train = pd.read_csv('breast-cancer-train.dat',names=labels)
    data_validate = pd.read_csv('breast-cancer-validate.dat', names=labels)
    draw(data_train)
    b_train = type(data_train.loc[:,["Malignant/Benign"]].values)
    b_sol = type(data_validate.loc[:, ["Malignant/Benign"]].values)
    Atrain,w = get_Aw(data_train,b_train,True)
    Asol = to_quadratic(data_validate.loc[:, ["radius (mean)", "perimeter (mean)", "area (mean)", "symmetry (mean)"]].values)
    # Asol = np.array(data_validate.iloc[:, 2:len(data_train.columns)].values)
    test = np.dot(Asol,w)
    # test = np.dot(Asol, scp.lstsq(Atrain,b_train)[0])
    for i in range(len(test)):
        if(test[i]<0):
            test[i] = -1
        else:
            test[i] = 1
    ctr = 0
    for i in range(len(test)):
        if(test[i]==b_train[i]):
            ctr+=1
    print(ctr)