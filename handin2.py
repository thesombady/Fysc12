import math


def Expression1(r):
    k2 = math.sqrt(938.9 * 2.2/(197.3**2))
    powerfunction = math.exp(-2 * k2 * r)
    return 1/(-2 * k2) * powerfunction

def Expression2(r):
    k1 = math.sqrt(2 * 938.9 * (-2.2 + 35)/(197.3**2))
    sinusfunction = (r / 2 - 1/(4 * k1) * math.sin(k1 * 2 * r))
    return sinusfunction

def COverA():
    k1 = math.sqrt(938.9 * (-2.2 + 35)/(197.3**2))
    k2 = math.sqrt(938.9 * 2.2/(197.3**2))
    return (math.sin(k1 * 2.1)/math.exp(-k2 * 2.1))**(-2)



Numerator = -Expression1(2.1)
Denominartor = Numerator + Expression2(2.1) * COverA()
print(Numerator/ Denominartor)
