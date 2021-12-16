import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

# Se ajustan los datos del catalogo a la ley de Guttenberg-Richter.


# Creación del vector de frecuencias y magnitudes a partir del catálogo.

def frecmag(catalogo):
    
    magnitudes = catalogo[:, 5]         # Se guardan las magnitudes en un vector.
    # Se crea un array de magnitudes para clasificar el catálogo por frecuencias.
    x_mag = np.arange(min(magnitudes), max(magnitudes), 0.1)
    # Se crea un array que contendrá los valores de las cuentas.
    y_frec = np.zeros(len(x_mag), int)
    contador = Counter(magnitudes)      # Se crea el contador de magnitudes.
    for i, m in enumerate(x_mag):
        y_frec[i] = contador[m]
    
    return x_mag, y_frec


cat = np.loadtxt("Catálogo_españa_matlab.txt", delimiter=",", skiprows = 1, dtype = float)
mag, frec = frecmag(cat)

# Representación de resultados.

############################# Acumulativo #########################

ac_frec = np.cumsum(frec[::-1])[::-1]

fig, ax = plt.subplots(figsize = (8,8))
plt.title("Frecuencia-Magnitud")
plt.xlabel("Magnitud")
plt.ylabel("Frecuencia")
ax.set_yscale("log");
ax.grid(b=True, which='major', color='b', linestyle='-')
ax.grid(b=True, which='minor', color='r', linestyle='--')
ax.set_xlim(0, 8)

ax.scatter(mag, ac_frec, label = "Acumulativo")

###############################  Normal ###########################
          
ax.scatter(mag, frec,label="Frecuencias")
ax.scatter(1.8, 70000, c= "b", label="Mc") 
ax.plot(mag[18:],10**(6.028801 -  0.594731*mag[18:]), c = "purple", label="G-L") 
plt.legend(loc="best")
plt.show()
        

