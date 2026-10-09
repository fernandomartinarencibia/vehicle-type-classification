"""Analiza las resoluciones de las imágenes del dataset y sugiere un tamaño de entrada para la red.

Uso: python analyze_resolutions.py   (espera el dataset en data/archive, con una carpeta por clase)
"""
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from PIL import Image
from tqdm import tqdm

PATH_DATASET = 'data/archive'
VALID_FORMATS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')


def analizar_resoluciones(ruta_dataset):
    """Devuelve un DataFrame con la clase, el ancho, el alto y la relación de aspecto de cada imagen."""
    datos = []
    print(f"Explorando: {ruta_dataset}...")

    for clase in os.listdir(ruta_dataset):
        ruta_clase = os.path.join(ruta_dataset, clase)
        if not os.path.isdir(ruta_clase):
            continue
        for archivo in tqdm(os.listdir(ruta_clase), desc=f"Clase: {clase}"):
            if not archivo.lower().endswith(VALID_FORMATS):
                continue
            try:
                # Solo se lee la cabecera de la imagen para obtener sus dimensiones
                with Image.open(os.path.join(ruta_clase, archivo)) as img:
                    ancho, alto = img.size
                datos.append({'clase': clase, 'ancho': ancho, 'alto': alto,
                              'relacion_aspecto': ancho / alto})
            except Exception as e:
                print(f"Error con {archivo}: {e}")

    return pd.DataFrame(datos)


def main():
    df = analizar_resoluciones(PATH_DATASET)

    # Dispersión de resoluciones por clase, con sus histogramas marginales
    sns.set_theme(style="whitegrid")
    g = sns.jointplot(data=df, x='ancho', y='alto', hue='clase',
                      alpha=0.5, palette='viridis', kind='scatter')
    g.fig.suptitle('Distribución de resoluciones por clase', y=1.02)
    g.set_axis_labels('Ancho (píxeles)', 'Alto (píxeles)')
    plt.show()

    print("\nEstadísticas de las resoluciones:")
    print(df[['ancho', 'alto']].describe())

    # La mediana es más robusta que la media frente a valores extremos
    print(f"Resolución mediana: {df['ancho'].median()}x{df['alto'].median()}")

    # Tamaño de entrada sugerido a partir del percentil 75 del ancho
    p75_ancho = df['ancho'].quantile(0.75)
    if p75_ancho <= 80:
        target = 64
    elif p75_ancho <= 160:
        target = 128
    else:
        target = 224
    print(f"El 75 % de las imágenes tiene un ancho menor que {p75_ancho:.2f} px")
    print(f"Tamaño de entrada sugerido: {target}x{target}")


if __name__ == '__main__':
    main()
