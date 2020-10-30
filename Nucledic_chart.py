import numpy as np
import matplotlib.pyplot as plt
global c1, c2, c3, c4, c5
c1 = 15.9
c2 = 18.4
c3 = 0.71
c4 = 23.2
c5 = 11.5

def BindingEnergy(A,Z):
    def Sigma(A,Z):
        if Z % 2 == 0:
            if (A - Z)%2 == 0:
                return 1
            else:
                return 0
        elif (A-Z)%2 == 0:
            if Z % 2 == 0:
                return 1
            else:
                return 0
        else:
            return -1
    try:
        return c1 * A - c2 * A ** (2/3) - c3 * Z ** 2/(A ** (1/3)) - c4 * (A - 2 * Z) ** 2/ (A) + c5 * Sigma(A,Z) * A ** (-1/2)
    except Exception as E:
        raise E

def Sep_p(A,Z):
    if Z > 1 and A > 1:
        if (BindingEnergy(A, Z) - BindingEnergy(A - 1, Z - 1)) > 0 :
            return True
        else:
            return False

def Sep_n(A,Z):
    if Z > 1 and A > 1:
        if (BindingEnergy(A, Z) - BindingEnergy(A - 1, Z)) > 0:
            return True
        else:
            return False

print(Sep_n(10,10))
Protons = np.linspace(1, 180)
Neutrons = np.linspace(1,180)
AValues = np.array([BindingEnergy(Protons[i] + Neutrons[j], Protons[i]) for i in range(len(Protons)) for j in range(len(Neutrons))])
plt.plot(AValues, '.')
plt.xlabel('#Neutrons')
plt.ylabel("#Protons")
plt.show()
