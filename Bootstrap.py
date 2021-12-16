import pandas as pd
import numpy as np
import math

# Bootstrapping para el cálculo de Mc y b, con sus respectivos errores.

##############################################################################
#################### Magnitud de Completitud del Catálogo ####################
##############################################################################

# Método de máxima curvatura.

# Requisitos:
# Catálogo (cat): magnitudes de eventos.
# Resolución de magnitud (res).

# Opcional:
# Corrección a la magnitud(corr).

def magcomp(cat, res, corr):
    if res == 0:
        res = 0.1                       # El binning por defecto de los catálogos.
    
    maglist = sorted(cat)               # Se ordena la lista de magnitudes.
     
    cMc = 0                             # Candidato a magnitud de completitud.
    
    # Se calculan el mínimo y el máximo del catálogo.
    minMag = min(maglist)
    maxMag = max(maglist)
    
    # Creación del histograma de magnitudes.
    magrange = np.arange(minMag, maxMag, 0.1)                   # Dominio del histograma.
    hist, bins_ed = np.histogram(cat, bins = magrange)          # Histograma.
    maxcount = max(hist)                                        # Se toma el máximo de cada bin.
    # Se recorre el histograma para obtener el índice del máximo, nos quedaremos
    # con la última magnitud que tenga el máximo de cuentas. 
    for j in range(len(hist)):
        if hist[j] == maxcount:       
            cMc = bins_ed[j]
            # Si se quiere mantener el primer valor de magnitud en el caso
            # de que coincidan, descomentar el break que hay debajo.
            # break;
    return cMc


#############################################################################
################################# Parámetro b ###############################
#############################################################################

# Método de máxima verosimilitud (maximum likelihood).
# Aki, K. (1965), Maximum likelihood estimate of b in the formula log N = a - bm and its confidence limits.
# Bull. Earthquake Res. Inst., Tokyo Univ. 43, 237-238. 

# Requisitos:
# Catálogo (cat): magnitudes de eventos.
# Valor de la magnitud de completitud (mediante la función anterior o pasada
# como parámetro de simulación).

def b_param(cat, mc):
    
    binning = 0.1               # Binning por defecto de los catálogos.
    minMag = min(cat)           # Cálculo de la magnitud mínima del catálogo.
    meanMag = np.mean(cat)      # Cálculo de la magnitud promedio.
    n = len(cat)                # Tamaño del catálogo.
    
    # Cálculo de b usando la expresión del método: 
    b = (np.log10(np.exp(1)))/(meanMag + binning*0.5 - minMag)
    # En caso de que se quiera un valor de b constante, descomentar linea siguiente:
    # b = 1.0
    
    # Cálculo de la desviación estándar.
    magminprom = cat-np.ones(len(cat))*meanMag      # Catálogo menos promedio de magnitud.
    bVar = np.var(magminprom)/n                     # Varianza de b/n.
    bstdev = 2.30*np.sqrt(bVar)*b** 2              # Desviación estándar.
    a = np.log10(n) + b*minMag                      # a usando la ley de G-R.
    
    return  a, b, bstdev


#############################################################################
################################# Bootstraping ##############################
#############################################################################

# Requisitos:
# Catálogo (cat): magnitudes de eventos.
# Número de simulaciones a realizar (nsim) (normalmente entre 20 y 30).
# Número mínimo de eventos con M > Mc para computar el valor de b (neve)
# (tras bootstrapping).
# Método para calcular la magnitud de completitud (máxima curvatura de momento).
# Resolución de magnitud (res).

def Bootstrap(cat, nsim, neve, met, res, corr):
    
    nmc = len(cat)-1                        # Número de eventos en el catálogo.
    mC_list= []                             # Lista de valores de magnitud de completitud.
    b_list = []                             # Lista de valores de b.
    a_list = []                             # Lista de valores de a.
    b_std_list = []                         # Lista de desv. std. de b.
    for i in range(nsim):
        sintcat = cat[:,5].copy()           # Lista de catálogos sintéticos.
        # Crea una lista con los índices de la lista del catálogo para
        # más  tarde crear un subconjunto aleatorio del mismo (con repeticiones).
        vRand = np.ceil(np.random.rand(nmc)*nmc)
        sintcatb = [sintcat[int(i)] for i in vRand]    # Catálogo aleatorio.
        # Solución de: https://stackoverflow.com/questions/25431850/passing-a-list-of-indices-to-another-list-in-python-correct-syntax
        # Para reducir el catálogo se calcula la magnitud de completidud del mismo
        Mcomp = magcomp(sintcatb, res, corr)
        mC_list.append(Mcomp)
        # A continuación se filtra el catálogo quitando los eventos con M < Mc.
        sintcatb[:] = [j for j in sintcatb if j >= Mcomp]
        # Filtrado de lista obtenido de stackoverflow:
        # https://stackoverflow.com/questions/1352885/remove-elements-as-you-traverse-a-list-in-python
        if len(sintcatb) >= neve:
            aval, bval, bstd = b_param(sintcatb, Mcomp)  # Cálculo de b si hay suficientes eventos.
        else:
            print("No se ha podido calcuar b por falta de eventos > Mc.")
            bval = 0
        b_list.append(bval)
        a_list.append(aval)
        b_std_list.append(bstd)

    # Cálculo de los valores promedio de Mc y b tras los bucles.
    bprom, bstdev = np.mean(b_list), np.std(b_std_list)
    mcprom, mcstdev = np.mean(mC_list), np.std(mC_list)
    aprom, astd = np.mean(a_list), np.std(a_list)
    
    return bprom, bstdev, aprom, astd, mcprom, mcstdev, b_list, mC_list

################################################################################
################################################################################
################################################################################

cat = np.loadtxt("Catálogo_españa_matlab.txt", delimiter=",", skiprows = 1, dtype = float)

b, berr, a, aerr, mc, mcerr, b_l, mC_l = Bootstrap(cat, 20, 10000, 0, 0.1, 0)

print("La ley de Gutenberg-Richter para el catálogo es: log(N) = {:f} + {:f}M".format(a, b))
print("La magnitud de completidud es :", mc)
