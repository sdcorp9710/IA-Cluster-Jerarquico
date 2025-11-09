
"""
Doctorado en Tecnologías para la Transformación Digital
Materia:      Inteligencia Artificial
Unidad:       2
Práctica      2.2. Máquina de Aprendizaje No Supervisado
                   Clúster Jerárquico (Ward) con soporte visual (PCA)
Autor:        Luis Alejandro Santana Valadez

Requisitos:
    pip install pandas numpy matplotlib scipy scikit-learn

Entrada:
    - CSV "alumnos_notas.csv" con columnas:
      Alumno, Matemáticas, Ciencias, Español, Historia, EdFísica

Salida (gráficas):
    - dendrograma_ward.png
    - mapa_calor_distancias.png
    - pca_individuos_clusters.png
    - pca_circulo_correlaciones.png
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from matplotlib.patches import Ellipse

# Paleta de colores para aplicación en gráficas
COLORS = ["#005F73", "#0A9396", "#94D2BD", "#EE9B00", "#BB3E03", "#CA6702", "#001219"]

def plot_cluster_ellipse(ax, X2, edge="#222222", face="#94D2BD",
                         n_std=2.0, lw=1.6, alpha=0.14,
                         min_axis_frac=0.15, eps=1e-3):
    """
    Dibuja una elipse de covarianza para los puntos X2 (N x 2).
    - Regulariza la covarianza (eps) para evitar aplanamiento 
      (para el caso de 2 puntos o colinealidad).
    - Asegura un eje menor mínimo (min_axis_frac del eje mayor).
    """
    N = X2.shape[0]
    if N < 2:
        return

    # Covarianza regularizada
    cov = np.cov(X2, rowvar=False)
    cov = cov + eps * np.eye(2)

    # Autovalores/vectores
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]

    # Ángulo y semiejes (n_std desviaciones)
    theta = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    major = 2 * n_std * np.sqrt(max(vals[0], eps))
    minor = 2 * n_std * np.sqrt(max(vals[1], eps))

    # Si el eje menor es demasiado pequeño (2 puntos o colinealidad), forzarlo
    minor = max(minor, major * min_axis_frac)

    mean = X2.mean(axis=0)
    ell = Ellipse(xy=mean, width=major, height=minor, angle=theta,
                  edgecolor=edge, facecolor=face, lw=lw, alpha=alpha, zorder=1)
    ax.add_patch(ell)

# ------------------
# 1) Cargar datos
df = pd.read_csv("alumnos_notas.csv")
X = df[["Matemáticas","Ciencias","Español","Historia","EdFísica"]].values
nombres = df["Alumno"].values

# ------------------
# 2) Estandarizar (recomendado para aplicar correctamente Ward)
scaler = StandardScaler()
Z = scaler.fit_transform(X)

# ------------------
# 3) Enlace Ward aplicado para el dendrograma
D = pdist(Z, metric="euclidean")
L = linkage(D, method="ward")

plt.figure(figsize=(9, 6))
dendrogram(L, labels=nombres, leaf_rotation=0, leaf_font_size=10, color_threshold=None)
plt.title("Dendrograma – Enlace Ward")
plt.tight_layout()
plt.savefig("dendrograma_ward.png", dpi=200)
plt.close()

# ------------------
# 4) Mapa de calor de distancias
dist_matrix = squareform(D)
plt.figure(figsize=(6, 5))
im = plt.imshow(dist_matrix, cmap="viridis")
plt.colorbar(im, fraction=0.046, pad=0.04, label="Distancia euclidiana (Z-score)")
plt.xticks(range(len(nombres)), nombres, rotation=90)
plt.yticks(range(len(nombres)), nombres)
plt.title("Mapa de calor de distancias entre alumnos")
plt.tight_layout()
plt.savefig("mapa_calor_distancias.png", dpi=200)
plt.close()

# ------------------
# 5) Etiquetas de clúster (Ward) con AgglomerativeClustering
model = AgglomerativeClustering(n_clusters=3, metric="euclidean", linkage="ward")

etiquetas = model.fit_predict(Z)

# ------------------
# 6) PCA para plano de individuos (visual)
pca = PCA(n_components=2, random_state=42)
Z2 = pca.fit_transform(Z)
var = pca.explained_variance_ratio_ * 100

plt.figure(figsize=(7, 6))
for k in np.unique(etiquetas):
    idx = etiquetas == k
    plt.scatter(Z2[idx,0], Z2[idx,1], s=80, edgecolor="#222222", linewidth=0.8,
                c=COLORS[k % len(COLORS)], label=f"Cluster {k+1}")
    for i in np.where(idx)[0]:
        plt.text(Z2[i,0]+0.05, Z2[i,1]+0.05, nombres[i], fontsize=9)

# Generación del clústers con elipses de covarianza
fig = plt.gcf()
ax = plt.gca()
for k in np.unique(etiquetas):
    idx = etiquetas == k
    # Elipse de covarianza para el clúster
    plot_cluster_ellipse(ax, Z2[idx], face=COLORS[k % len(COLORS)], alpha=0.14, lw=1.6)

plt.xlabel(f"PC1 ({var[0]:.1f}%)")
plt.ylabel(f"PC2 ({var[1]:.1f}%)")
plt.legend(frameon=False)
plt.title("Plano de individuos (PCA) apoyo a clúster jerárquico (Ward)")
plt.tight_layout()
plt.savefig("pca_individuos_clusters.png", dpi=200)
plt.close()

# ------------------
# 7) Círculo de correlaciones (variables de materias)
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
vars_names = ["Matemáticas","Ciencias","Español","Historia","EdFísica"]

fig, ax = plt.subplots(figsize=(6,6))
ang = np.linspace(0, 2*np.pi, 200)
ax.plot(np.cos(ang), np.sin(ang), linewidth=1.0, color="#888888")
for i, v in enumerate(vars_names):
    ax.arrow(0, 0, loadings[i,0], loadings[i,1], head_width=0.04, head_length=0.04,
             fc=COLORS[i % len(COLORS)], ec=COLORS[i % len(COLORS)], length_includes_head=True)
    ax.text(loadings[i,0]*1.08, loadings[i,1]*1.08, v, color=COLORS[i % len(COLORS)], fontsize=10)
ax.set_xlabel(f"PC1 ({var[0]:.1f}%)")
ax.set_ylabel(f"PC2 ({var[1]:.1f}%)")
ax.set_title("Círculo de correlaciones (PCA) – apoyo a clúster jerárquico (Ward)")
ax.set_aspect('equal', adjustable='box')
ax.axhline(0, color="#BBBBBB", linewidth=0.8)
ax.axvline(0, color="#BBBBBB", linewidth=0.8)
plt.tight_layout()
plt.savefig("pca_circulo_correlaciones.png", dpi=200)
plt.close()

print("OK, Figuras generadas en el directorio actual.")
