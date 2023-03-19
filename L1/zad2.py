import mpmath
import matplotlib.pyplot as plt



if __name__ == "__main__":
    n = 60
    x = [1/3,1/12]
    k = [i for i in range(1,n+1)]
    z = [1/3,1/12]
    e = [0,0]
    for i in range(2,n):
        x.append(2.25*x[i-1]-0.5*x[i-2])
        z.append(4**(-i)/3)
        e.append(abs(x[i]-z[i]))
    fig,axs = plt.subplots(2)
    axs[0].plot(k,x,color='r',label='Wzór rekurencyjny',marker="o")
    axs[0].plot(k,z, color='b', label='Wzór ogólny', marker="o")
    axs[1].plot(k,e, color='black',label='Błąd',marker="o")
    for a in axs:
        a.legend()
        a.semilogy()
    axs[0].set_ylabel("Wartości")
    axs[1].set_ylabel("Błąd")
    axs[1].set_xlabel("k")
    print("{:<27} {:<27} {:<27}".format("Rekurencyjny","Ogólny","Błąd"))
    for a,b,c in zip(x,z,e):
        v = "{:<27} {:<27} {:<27}"
        print(v.format(str(a),str(b),str(c)))
    plt.show()

