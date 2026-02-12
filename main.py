import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from utils import nan_replace, to_dataframe, corelograma, plot_corelatii, plot_componente

NUME_FISIER = "Date_Proiect_PCA_DoarTari.xlsx"

try:
    t = pd.read_excel(NUME_FISIER, index_col=0)
except FileNotFoundError:
    print(f"EROARE: Nu gasesc fisierul '{NUME_FISIER}'.")
    exit()

nan_replace(t)

variabile_observate = list(t.columns)
print("Variabile analizate:", variabile_observate)

x_orig = t[variabile_observate]
x = (x_orig - np.mean(x_orig, axis=0)) / np.std(x_orig, axis=0)

n, m = x.shape
print(f"Dimensiuni: {n} tari x {m} indicatori")

model_acp = PCA()
model_acp.fit(x)

alpha = model_acp.explained_variance_
print("Valori proprii (Alpha):", alpha)

a = model_acp.components_
c = model_acp.transform(x)

labels = ["C" + str(i+1) for i in range(len(alpha))]
componente_df = to_dataframe(c, t.index, labels, "componente.csv")

# figura 1
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(alpha) + 1), alpha, 'bo-', linewidth=2, markersize=8)
plt.title('Figura 1: Scree Plot (Valorile Proprii)', fontsize=16, color='b')
plt.xlabel('Componente Principale', fontsize=12)
plt.ylabel('Valoare Proprie (Eigenvalue)', fontsize=12)
plt.axhline(1, color='r', linestyle='--', label='Criteriul Kaiser (1)')
plt.legend()
plt.grid(True)
plt.show()
print("S-a generat Figura 1 (Scree Plot)")

# Kaiser
conditie = np.where(alpha > 1)
nr_comp_s_kaiser = len(conditie[0])
print(f"Componente semnificative (Kaiser > 1): {nr_comp_s_kaiser}")

# Cattell
eps = alpha[0 : (m-1)] - alpha[1 : m]
sigma = eps[0: (m-2)] - eps[1: len(eps)]
indici_negativi = (sigma < 0)
if any(indici_negativi):
    nr_comp_s_cattel = np.where(indici_negativi)[0][0] + 1
else:
    nr_comp_s_cattel = "Nedeterminat"
print(f"Componente semnificative (Cattell): {nr_comp_s_cattel}")

# Procent de acoperire
ponderi = np.cumsum(alpha / sum(alpha))
conditie_proc = np.where(ponderi > 0.8)
nr_comp_s_procent = conditie_proc[0][0] + 1 if len(conditie_proc[0]) > 0 else m
print(f"Componente semnificative (Acoperire > 80%): {nr_comp_s_procent}")

# figura 3
plot_componente(componente_df, "C1", "C2", titlu="Figura 3: Harta Țărilor (Score Plot)", aspect=1)
print("S-a generat Figura 3 (Harta Tărilor)")

corr = np.corrcoef(x, c, rowvar=False)
r_x_c = corr[:m, m:]
r_x_c_df = to_dataframe(r_x_c, variabile_observate, labels, "corelatii_factoriale.csv")

# figura 2
plot_corelatii(r_x_c_df, "C1", "C2", titlu="Figura 2: Cercul Corelațiilor")
print("S-a generat Figura 2 (Cercul Corelațiilor)")

# figura 4
r_patrat = r_x_c * r_x_c
comunalitati = np.cumsum(r_patrat, axis=1)
comunalitati_df = to_dataframe(comunalitati, variabile_observate, labels, "comunalitati.csv")

corelograma(comunalitati_df, vmin=0, vmax=1, titlu="Figura 4: Calitatea Reprezentării (Comunalități)")
print("S-a generat Figura 4 (Heatmap Comunalități)")

# cosinusuri
c_patrat = c ** 2
sume = c_patrat.sum(axis=1, keepdims=True)
cosin = c_patrat / sume
cosin_df = to_dataframe(cosin, t.index, labels, "cosinusuri.csv")

