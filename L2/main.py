import pandas as pd
import matplotlib.pyplot as plt

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

if __name__ == '__main__':
    with open("breast-cancer.labels","r") as f:
        labels = [line.rstrip() for line in f]
    plt.figure()
    data_train = pd.read_csv('breast-cancer-train.dat',names=labels)
    data_validate = pd.read_csv('breast-cancer-validate.dat', names=labels)
    draw(data_train)
    
