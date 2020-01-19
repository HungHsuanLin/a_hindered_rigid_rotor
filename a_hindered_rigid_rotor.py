import numpy as np
from scipy import special
import matplotlib.pyplot as plt
import quadpy
import math

#using a basis of l spherical harmonics
def matlab_legendre(n,X):
    res = []
    for m in range(n+1):
        res.append(np.array(special.lpmv(m,n,X)))
    return np.array(res)

#using a basis of l spherical harmonics
#value of l to use
degree=7

### parameters
#Rotational constant for a hydrogen molecule in wavenumber
B=61.20979
# B=1.0
#conversion factor from wavenumber to meV for JZL
w2meV=0.1239842

### not using at the moment
#h/2pi
hbar = 1.05457173E-34
#the mass of a hydorgen atom
Hmass = 1.6737236E-27
#the reduced mass of a hydrogen molecule
mu = Hmass*Hmass/(Hmass+Hmass)
#the bond length of a hydrogen molecule
R = 0.741*10E-10
I = mu*R**2
BB = hbar**2 / (2*I) * (6.24150913*10E18)*1000;
### not using at the moment

# to caclculate the total number of basis functions
NBas = 0
for i in range(0,degree+1):
    numfun = 2*i + 1
    NBas = NBas + numfun

# to setup the initial matrix
v = np.zeros(phi.size)
H = np.zeros([NBas, NBas],dtype=complex)
J = np.array([])
for i in range(0,degree+1):
    for j in range(0,2*i+1):
        J = np.append(J,[int(i)])

# to build the Lededev Sphere using 146 points
leb = quadpy.sphere.lebedev_019()
phi = np.arctan2(leb.points[:,1],leb.points[:,0])
theta = np.arccos(leb.points[:,2])

# to construct the Spherical Harmonics basis over theta and phi
k = (degree + 1)**2
Y = np.zeros([leb.points[:,0].size, k],dtype=complex)
for j in range(0,degree+1):
    Pm = np.matrix.transpose(matlab_legendre(j,leb.points[:,2]))
    lconstant = ((2*j +1)/(4*np.pi))**(0.5)
    #calculate where to put the vector
    center = (j+1)**2 -j
    # calculate the Yj0
    Y[:,center-1] = lconstant*Pm[:,0]
    # calculate the order Ylm of the set (if any)
    for m in range(1,j+1):
        precoeff = lconstant * (math.factorial(j-m)/math.factorial(j+m))**(0.5)
        mod = m%2
        if mod == 1:
            Y[:,center + m-1] =  precoeff*Pm[:,m]*np.exp(1j*m*phi)
            Y[:,center - m-1] = -precoeff*Pm[:,m]*np.exp(-1j*m*phi)
        else:
            Y[:,center + m-1] =  precoeff*Pm[:,m]*np.exp(1j*m*phi)
            Y[:,center - m-1] =  precoeff*Pm[:,m]*np.exp(-1j*m*phi)

# to calculate the potential matrix
## we first build the diagonal terms
H_diag = np.zeros([NBas, NBas],dtype=complex)
for n in range(0,NBas):
    H_diag[n,n]=B*J[n]*(J[n]+1)+H[n,n]
rawdata=np.array([])
## the strength of rotational barrier
b=0.0
f=open('rawdata.csv','a')
# the strength of rotational barrier
for a in np.arange(0,100,1):
    # Generte rotational barrier
    # v = a*B*(np.sin(theta)**2)+(0.5*b)*(np.cos(2*phi))*(np.sin(theta)**2) # 2-D rotational barrier
    # v = a*B*(np.sin(theta)**2) # 1-D rotational barrier
    # v = a*(np.sin(theta)**2)
    H = H_diag + np.matmul(np.matrix.getH(Y),v.reshape(-1,1)*(leb.weights*4*np.pi).reshape(-1,1)*Y)
    eigenvalues, eigenvectors= np.linalg.eig(H)
    newline=np.append(np.array(a),np.sort(eigenvalues.real)*w2meV)
    np.savetxt(f, newline, fmt='%1.3f', newline=", ")
    f.write("\n")
    #plot rotational transition
    for i in range(0,9):
        plt.plot(a*w2meV,w2meV*np.sort(eigenvalues.real)[i],'bo')

f.close()
plt.show()
