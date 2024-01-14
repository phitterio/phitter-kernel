import numpy

def adinf(z):
    if z < 2:
        return (z ** -0.5) * numpy.exp(-1.2337141 / z) * (2.00012 + (0.247105 - (0.0649821 - (0.0347962 - (0.011672-0.00168691 * z) * z) * z) * z) * z)
    return numpy.exp(-numpy.exp(1.0776 - (2.30695 - (0.43424 - (0.082433 - (0.008056-0.0003146 * z) * z) * z) * z) * z))
                                              
def errfix(n, x):
    if x > 0.8:
        return (-130.2137 + (745.2337 - (1705.091 - (1950.646 - (1116.360 - 255.7844 * x) * x) * x) * x) * x) / n
    
    c = 0.01265 + 0.1757 / n
    if x < c:
        t= x / c
        t= numpy.sqrt(t) * (1 - t) * (49 * t - 102)
        return t * (0.0037 / (n ** 2) + 0.00078 / n + 0.00006) / n
    
    t = (x - c) / (0.8 - c)
    t = -0.00022633 + (6.54034 - (14.6538 - (14.458 - (8.259 - 1.91864 * t) * t) * t) * t) * t
    return t * (0.04213 + 0.01365 / n) / n


def AD(n, z):
    
    x = adinf(z)
    if x > 0.8:
        v = (-130.2137 + (745.2337 - (1705.091 - (1950.646 - (1116.360 - 255.7844 * x) * x) * x) * x) * x) / n
        return x + v
    
    c = 0.01265 + 0.1757 / n
    if x < c:
        v= x / c
        v= numpy.sqrt(v) * (1 - v) * (49 * v - 102)
        return x + v * (0.0037 / (n ** 2) + 0.00078 / n + 0.00006) / n

    v = (x - c) / (0.8 - c)
    v = -0.00022633 + (6.54034 - (14.6538 - (14.458 - (8.259 - 1.91864 * v) * v) * v) * v) * v
    
    return x + v * (.04213 + .01365 / n) / n

def main():
    while True:
        n = int(input("Enter n: "))
        z = float(input("Enter z: "))
        x = adinf(z)
        print("adinf(z)=", x)
        print("errfix(n,x)=",errfix(n,x))
        print("Prob(A_{} < {})={}".format(n,z,AD(n,z)))
        
if __name__ == "__main__":
    main()