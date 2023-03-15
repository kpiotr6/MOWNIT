import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
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
    Atrain_q,w_q = get_Aw(data_train,b_train,True)
    Atrain, w = get_Aw(data_train, b_train,False)
    Asol_q = to_quadratic(data_validate.loc[:, ["radius (mean)", "perimeter (mean)", "area (mean)", "symmetry (mean)"]].values)
    Asol = np.array(data_validate.iloc[:, 2:len(data_train.columns)].values)
    test = np.dot(Asol,w)
    test_q = np.dot(Asol_q,w_q)
    cond = np.dot(np.linalg.norm(Atrain), np.linalg.norm(np.linalg.pinv(Atrain)))
    cond_q = np.dot(np.linalg.norm(Atrain_q,None), np.linalg.norm(np.linalg.pinv(Atrain_q),None)) #Forbenius
    for i in range(len(test)):
        if(test[i]<0):
            test[i] = -1
        else:
            test[i] = 1
        if test_q[i]<0:
            test_q[i] = -1
        else:
            test_q[i] = 1
    fp = 0
    fn = 0
    fp_q = 0
    fn_q = 0
    for i in range(len(test)):
        if(test[i]==1 and b_sol[i]==-1):
            fp+=1
        elif(test[i]==-1 and b_sol[i]==1):
            fn+=1
        if (test_q[i] == 1 and b_sol[i] == -1):
            fp_q += 1
        elif (test_q[i] == -1 and b_sol[i] == 1):
            fn_q += 1
    print("False positive: "+str(fp))
    print("False negative: "+str(fn))
    print("Effficency "+str(1-(fp+fn)/260))
    print("False positive_q: "+str(fp_q))
    print("False negative_q: "+str(fn_q))
    print("Effficency_q "+str(1-(fp_q+fn_q)/260))
    # np.dot(np.dot)