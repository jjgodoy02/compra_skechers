# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 15:23:19 2025

@author: Jose Godoy
"""

import numpy as np
from PIL import Image, ImageDraw
import math
from sklearn.cluster import KMeans
import colorsys

def DominantColor(img, n_clusters=3):
    img = Image.open(img)
    arr = np.array(img).reshape(-1, 3)
    
    mask = ~np.all(arr > [250, 250, 250], axis=1)
    arr_filtered = arr[mask]
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(arr_filtered)
    
    counts = np.bincount(kmeans.labels_)
    sorted_indices = np.argsort(counts)[::-1]   
    
    dominant_color = kmeans.cluster_centers_[sorted_indices[0]].astype(int)
    
    return dominant_color

def ColorMap(colores, sku_list, tamaño_imagen=500):
    valid_colores = {k: v for k, v in colores.items() if k in sku_list}
    skus = sorted(
    valid_colores.keys(),
    key=lambda k: colorsys.rgb_to_hsv(*[x / 255 for x in valid_colores[k]])
    )
    n = len(skus)

    # Calcular columnas y filas
    columnas = math.ceil(math.sqrt(n))
    filas = math.ceil(n / columnas)

    # Calcular tamaño del cuadrito dinámicamente
    cuadrito = tamaño_imagen // max(columnas, filas)

    # Recalcular tamaño real de la imagen (puede ser menor a 500x500 para evitar bordes extra)
    ancho = columnas * cuadrito
    alto = filas * cuadrito

    img = Image.new("RGB", (ancho, alto), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    for idx, sku in enumerate(skus):
        color = tuple(valid_colores[sku])
        x = (idx % columnas) * cuadrito
        y = (idx // columnas) * cuadrito
        draw.rectangle([x, y, x + cuadrito, y + cuadrito], fill=color)

    return img