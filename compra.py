# -*- coding: utf-8 -*-
"""
Created on Tue Jul  8 10:22:56 2025

@author: Jose Godoy
"""

# ─────────────────────────────────────────────────────────────────────────────
# 1. Librerías
# ─────────────────────────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import os
from PIL import Image, ImageOps
from io import BytesIO
import glob
import time
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from DominantColor import DominantColor, ColorMap
import io

# ─────────────────────────────────────────────────────────────────────────────
# 2. Funciones de carga de datos
# ─────────────────────────────────────────────────────────────────────────────
def importar_imagenes(df, better_img_folder):
    """
    Carga imágenes desde una carpeta local, y le asigna la ruta de la imagen correspondiente a cada producto dentro del Dataframe df.
    
    Parameters
    ----------
    df : DATAFRAME
        Dataframe con información de los productos.
    better_img_folder : str
        Ruta a la carpeta con las imágenes de alta calidad

    Returns
    -------
    image_buffers : dict
        Diccionario donde las keys son los sku del producto (de la forma 'STYLE_COLOR') y el value es un buffer de la imagen.

    """
    image_buffers = {} # Diccionario a retornar. key = style_color, value = buffer de imgaen
    total = len(df)
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, row in enumerate(df.itertuples()): # Itera sobre filas del df para ir llenado el diccionario image_buffers
        style = getattr(row, 'STYLE')
        color = getattr(row, 'COLOR')
        pattern = os.path.join(better_img_folder, f"{style}_{color}_*.jpg")
        matches = glob.glob(pattern)

        buffer = None
        if matches:
            try:
                img = Image.open(matches[0]).convert("RGB") # Extrae la imagen de la ruta especificada
                img = ImageOps.contain(img, (1000, 1000)) # Redimensiona la imagen
                buffer = BytesIO()
                img.save(buffer, format="JPEG", quality=75, optimize=True)
                buffer.seek(0)
            except:
                buffer = None
        else:
            img = Image.open(row.IMAGE).convert("RGB")
            buffer = BytesIO()
            img.save(buffer, format="JPEG", quality=75, optimize=True)
            buffer.seek(0)
        
        image_buffers[row.SKU] = buffer # Asocia el SKU con la imagen en buffer en el diccionario

        progress_bar.progress((i + 1) / total)
        status_text.text(f"Cargando imágenes: {i + 1} de {total}")

    status_text.text("Carga de imágenes completada.")
    time.sleep(0.5)
    progress_bar.empty()
    status_text.empty()
    return image_buffers

def procesar_imagenes(imgs):
    """
    A partir de un diccionario de imágenes, obtiene el color dominante de cada una por medio de kMeans.

    Parameters
    ----------
    imgs : dict
        Diccionario donde las keys son los sku del producto (de la forma 'STYLE_COLOR') y el value es un buffer de la imagen.

    Returns
    -------
    images_colors : dict
        Diccionario donde las keys son los sku del producto (de la forma 'STYLE_COLOR') y el value es un numpy array del color dominante ([R,G,B]).

    """
    total = len(imgs)
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    images_colors = {}

    for i, (sku, img) in enumerate(imgs.items()):
        if img is None:
            continue  
        try:
            dominant_color = DominantColor(img)
            images_colors[sku] = dominant_color
        except Exception as e:
            print(f"Error procesando {sku}: {e}")
            continue  
        
        progress_bar.progress((i + 1) / total)
        status_text.text(f"Procesando imágenes: {i + 1} de {total}")
    status_text.text("Procesamiento de imágenes completado.")
    time.sleep(0.5)
    progress_bar.empty()
    status_text.empty()
    return images_colors

@st.cache_data
def cargar_catalogo():
    """
    Carga el Dataframe con información de los productos, y crea una columna con la ruta de la imagen correspondiente

    Returns
    -------
    df : DATAFRAME
        Dataframe original, habiendole agregado una columna ['IMAGE'] con la ruta de la imagen correspondiente.
    img_folder : STR
        Ruta al folder con las imágenes originales (extraídas del excel).
    better_img_folder : STR
        Ruta al folder con la imágenes de mejor calidad (extraídas de Its the S).

    """
    df = pd.read_excel(r"C:\Users\ACCJGODOY\OneDrive - Grupo Disresa\Documentos\Skechers\misc\imagenes\compra.xlsx")
    img_folder = r"C:\Users\ACCJGODOY\OneDrive - Grupo Disresa\Documentos\Skechers\misc\imagenes\img"
    better_img_folder = r"C:\Users\ACCJGODOY\OneDrive - Grupo Disresa\Documentos\Skechers\misc\imagenes\its-styles_hd"

    df['STYLE'] = df['STYLE'].astype(str)
    df['COLOR'] = df['COLOR'].astype(str)
    df['SKU'] = df['SKU'].astype(str)
    df['IMAGE'] = (df['SKU'] + ".jpg").apply(lambda x: os.path.join(img_folder, x))

    return df, img_folder, better_img_folder

@st.cache_data
def cargar_ventas(fecha_exacta="2025-06-26"):
    """
    Carga todos los archivos que tienen la fecha EXACTA en el nombre (ventas acumuladas hasta esa fecha),
    para todos los países disponibles.
    """
    sales_pattern = rf"C:\Users\ACCJGODOY\OneDrive - Grupo Disresa\Documentos\Python\sales\sales_*_{fecha_exacta}.csv"
    sales_files = glob.glob(sales_pattern)
    ventas_dfs = []

    for file in sales_files:
        try:
            filename = os.path.basename(file)
            pais = filename.split('_')[1].split('-')[0]
            pais = "ES" if pais == "SV" else pais

            df_sales = pd.read_csv(file)
            df_sales['Pais'] = pais
            df_sales['Fecha'] = pd.to_datetime(
                df_sales[['Anio', 'Mes', 'Dia']].rename(columns={'Anio': 'year', 'Mes': 'month', 'Dia': 'day'})
            )
            df_sales['DivisionGenero'] = df_sales['U_Division'].astype(str) + " " + df_sales['U_Genero'].astype(str)

            ventas_dfs.append(df_sales)

        except Exception as e:
            print(f"Error procesando {file}: {e}")

    if ventas_dfs:
        df_all = pd.concat(ventas_dfs, ignore_index=True)
        df_all['YearMonth'] = df_all['Fecha'].dt.to_period('M').astype(str)
        ventas_mensuales = (
            df_all.groupby(['YearMonth', 'Pais', 'DivisionGenero'], as_index=False)['Cantidad']
            .sum()
            .rename(columns={'Cantidad': 'Ventas'})
        )
        return ventas_mensuales
    else:
        return pd.DataFrame(columns=['YearMonth', 'Pais', 'DivisionGenero', 'Ventas'])

@st.cache_data
def cargar_presupuesto():
    df = pd.read_excel(r"C:\Users\ACCJGODOY\OneDrive - Grupo Disresa\Documentos\Skechers\misc\imagenes\presupuesto.xlsx")
    num_cols = df.select_dtypes(include='number').columns
    df[num_cols] = (np.ceil(df[num_cols] / 12) * 12).astype(int)
    return df

# ─────────────────────────────────────────────────────────────────────────────
# 3. Helper functions
# ─────────────────────────────────────────────────────────────────────────────
# Función para checkear que se haya introducido un texto válido (Número entre 0-10, o vacío)
def validar_entrada(texto):
    if texto == "":
        return True, None
    try:
        val = float(texto)
        if 0 <= val <= 10:
            return True, val
        else:
            return False, None
    except:
        return False, None
    
def greedy_rounder(relative_scores, budget):
    budget = budget.iloc[0]
    relative_scores = relative_scores.fillna(0)
    raw_alloc = relative_scores * budget
    dozens_raw = raw_alloc / 12
    dozens_floored = np.floor(dozens_raw)

    final_dozens = pd.Series(dozens_floored, index=relative_scores.index)

    budget_dozens = int(budget) // 12
    current_sum = int(final_dozens.sum())
    diff = budget_dozens - current_sum

    frac_parts = dozens_raw - dozens_floored
    indices = np.argsort(-frac_parts.to_numpy())  # posiciones, no etiquetas

    if diff > 0:
        for i in indices[:diff]:
            final_dozens.iloc[i] += 1
    elif diff < 0:
        for i in indices[:abs(diff)]:
            if final_dozens.iloc[i] > 0:
                final_dozens.iloc[i] -= 1

    final_alloc = (final_dozens * 12).astype(int)
    return final_alloc

def random_number(std_dev=2, lower=0, upper=10):
    mean = np.random.uniform(3, 7)
    x = np.random.normal(loc=mean, scale=std_dev)
    x= np.clip(x, lower, upper)
    return x

def random_boolean(prob_1=0.5):  
    return np.random.choice([0, 1], p=[1 - prob_1, prob_1])

def apply_ratings_to_df():
    for rating in st.session_state.ratings.values():
        idx = rating.get('df_index')
        if idx is not None:
            st.session_state.df.at[idx, 'GT score'] = rating.get('gt_score')
            st.session_state.df.at[idx, 'ES score'] = rating.get('es_score')
            st.session_state.df.at[idx, 'HN TGU score'] = rating.get('tgu_score')
            st.session_state.df.at[idx, 'HN SPS score'] = rating.get('sps_score')

# ─────────────────────────────────────────────────────────────────────────────
# 4. Configuración inicial
# ─────────────────────────────────────────────────────────────────────────────
cmap = plt.cm.magma
new_cmap = mcolors.LinearSegmentedColormap.from_list('magma_trunc', cmap(np.linspace(0, 0.3, 256)))

st.set_page_config(layout="wide")

if 'df' not in st.session_state:
    df_loaded, img_folder, better_img_folder = cargar_catalogo()
    # Añadir columnas nuevas para calificaciones si no existen
    new_cols = ['GT','ES','RD','HN TGU','HN SPS']
    for col in new_cols:
        if col not in df_loaded.columns:
            df_loaded[col] = 0
    new_score_cols = ['GT score','ES score','RD score','HN TGU score','HN SPS score']
    for col in new_score_cols:
        if col not in df_loaded.columns:
            df_loaded[col] = None
            
    st.session_state.df = df_loaded.copy()
    st.session_state.img_folder = img_folder
    st.session_state.better_img_folder = better_img_folder
else:
    df_loaded = st.session_state.df
    img_folder = st.session_state.img_folder
    better_img_folder = st.session_state.better_img_folder

# Añade el diccionario de 'image_buffers' al session_state, para evitar que las imágenes se procesen cada vez
if 'image_buffers' not in st.session_state:
    st.session_state.image_buffers = importar_imagenes(df_loaded, better_img_folder)

if 'images_colors' not in st.session_state:
    st.session_state.images_colors = procesar_imagenes(st.session_state.image_buffers)
    
ventas_historicas = cargar_ventas()
budget = cargar_presupuesto()

# Crea un diccionario en session_sate donde guardar las calificaciones por SKU
if 'ratings' not in st.session_state:
    st.session_state.ratings = {}
    for idx, row in st.session_state.df.iterrows():
        sku = row['SKU']
        product_key = f"{row['DIVISION NAME']}_{sku}"
        scoreGT = random_number()
        scoreES = random_number()
        scoreTGU = random_number()
        scoreSPS = random_number()
        scoreRD = random_number()
        
        validGT, scoreGT_val = validar_entrada(scoreGT)
        validES, scoreES_val = validar_entrada(scoreES)
        validTGU, scoreTGU_val = validar_entrada(scoreTGU)
        validSPS, scoreSPS_val = validar_entrada(scoreSPS)
        validRD, scoreRD_val = validar_entrada(scoreRD)

        st.session_state.ratings[product_key] = {
            'valid': all([validGT, validES, validTGU, validSPS, validRD]),
            'gt_score': scoreGT_val,
            'es_score': scoreES_val,
            'tgu_score': scoreTGU_val,
            'sps_score': scoreSPS_val,
            'rd_score': scoreRD_val,
            'df_index': idx
        }
# ─────────────────────────────────────────────────────────────────────────────
# 5. Selección de división o grupo
# ─────────────────────────────────────────────────────────────────────────────
display_choice = st.sidebar.radio(
    "Seleccione una opción para mostrar los productos:",
    ["Por división", "Por orden de catálogo"])

st.sidebar.write("")

if display_choice == "Por división":
    divisiones = sorted(df_loaded['DIVISION NAME'].unique())
    div_options =  ['Todas las divisiones'] + divisiones
    division_seleccionada = st.sidebar.selectbox("Selecciona una División", div_options)
    
    if division_seleccionada == "Todas las divisiones":
        df_div = st.session_state.df.copy()
    else:
        df_div = st.session_state.df[st.session_state.df['DIVISION NAME'] == division_seleccionada].copy()
    
    grouped = df_div.groupby('STYLE')
    
    st.title(f"PREVENTA SS26 - {division_seleccionada}")

if display_choice == "Por orden de catálogo":
    groups = df_loaded['GrupoCatalogo'].unique()
    group_options =  groups
    grupo_seleccionado = st.sidebar.selectbox("Selecciona un Grupo", group_options)
    st.sidebar.write("")
    
    if grupo_seleccionado == "Todos los grupos":
        df_div = st.session_state.df.copy()
    else:
        df_div = st.session_state.df[st.session_state.df['GrupoCatalogo'] == grupo_seleccionado].copy()
    
    grouped = df_div.groupby('STYLE')
    
    st.title(f"PREVENTA SS26 - {grupo_seleccionado}")
    

# ─────────────────────────────────────────────────────────────────────────────
# 6. Presupuesto de la división seleccionada
# ─────────────────────────────────────────────────────────────────────────────
if display_choice == 'Por división':
    if division_seleccionada == 'Todas las divisiones': 
        budget_div = budget.iloc[:-1].copy()   
        budget_div['Total'] = budget_div.drop(columns=('DIVISION NAME')).sum(axis=1)
        subset_for_gradient = budget_div.columns.difference(['DIVISION NAME'])
        subset_for_format = budget_div.columns.difference(['DIVISION NAME'])
        cmap = plt.cm.magma
        new_cmap = mcolors.LinearSegmentedColormap.from_list(
            'magma_trunc', cmap(np.linspace(0, 0.3, 256))
        )
        gradient_args = {
        'cmap': new_cmap,
        'subset': subset_for_gradient,
        'axis': 0
        }
        
        fmt_dict = {col: "{:.0f}" for col in subset_for_format}
        styled_budget_div = (budget_div.sort_values('Total',ascending=False)
                             .style
                             .background_gradient(**gradient_args)
                             .format(fmt_dict)
        )
        st.markdown("#### Presupuesto por país")
        st.dataframe(styled_budget_div, hide_index=True)
        
    else:
        budget_div = budget[budget['DIVISION NAME'] == division_seleccionada].copy()  
        budget_div['Total'] = budget_div.drop(columns=('DIVISION NAME')).sum(axis=1)
        subset_for_gradient = budget_div.columns.difference(['Total', 'DIVISION NAME'])
        subset_for_format = budget_div.columns.difference(['DIVISION NAME'])
        gradient_args = {
        'cmap': new_cmap,
        'subset': subset_for_gradient,
        'axis': 1
        }
        
        fmt_dict = {col: "{:.0f}" for col in subset_for_format}
        styled_budget_div = (budget_div.sort_values('Total',ascending=False)
                             .style
                             .background_gradient(**gradient_args)
                             .format(fmt_dict)
        )
        st.markdown("#### Presupuesto por país")
        st.dataframe(styled_budget_div, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# 7. Estado de la compra de la división seleccionada
# ─────────────────────────────────────────────────────────────────────────────
if display_choice == 'Por división':
    st.markdown("#### Estado de la compra")
    if division_seleccionada == 'Todas las divisiones':
        subdf = st.session_state.df.copy()

        country_cols = ['GT','ES','RD','HN TGU','HN SPS']

        suggested_matrix = subdf[['SKU', *country_cols]].copy()

        suggested_matrix['Total'] = suggested_matrix[country_cols].sum(axis=1)

        suggested_matrix = suggested_matrix.merge(
            subdf[['SKU', 'DIVISION NAME']],
            on='SKU',
            how='left'
        )

        # Agrupar totales por división
        totals = suggested_matrix.groupby('DIVISION NAME')[list(country_cols) + ['Total']].sum().reset_index()

        # Estilos para mostrar
        subset_for_gradient = country_cols + ['Total']
        gradient_args = {
            'cmap': new_cmap,
            'subset': subset_for_gradient,
            'axis': 0
        }

        styled_totals = (totals.sort_values('Total', ascending=False)
                             .style
                             .background_gradient(**gradient_args)
                             .format(fmt_dict)
        )

        styled_suggested = (suggested_matrix.style.background_gradient(
            cmap=new_cmap,
            subset=subset_for_gradient,
            axis=0
        ))

        st.dataframe(styled_totals, hide_index=True)

    else:
        for key, val in st.session_state.ratings.items():
            if key.startswith(division_seleccionada):
                i = val['df_index']
                st.session_state.df.at[i, 'GT score'] = val['gt_score']
                st.session_state.df.at[i, 'ES score'] = val['es_score']
                st.session_state.df.at[i, 'HN TGU score'] = val['tgu_score']
                st.session_state.df.at[i, 'HN SPS score'] = val['sps_score']
                st.session_state.df.at[i, 'RD score'] = val['rd_score']
                st.session_state.df.at[i, 'valid'] = val['valid']

        subdf = st.session_state.df[st.session_state.df['DIVISION NAME'] == division_seleccionada].copy()
        score_cols = ['GT score','ES score','HN TGU score','HN SPS score','RD score']
        
        suggested_matrix = pd.DataFrame(
        0,
        index=subdf.index,
        columns=budget_div.drop(columns='DIVISION NAME').columns
        )

        for col in score_cols:
            country = col.split('score')[0].strip()
            total_scores = subdf[col].sum()
            if total_scores == 0:
                continue

            relative_col = f'relative {col}'
            subdf.loc[:, relative_col] = (subdf[col] / total_scores).fillna(0)
            budget = budget_div[country]

            suggested_col = greedy_rounder(subdf[relative_col], budget)

            # Poner las sugerencias en la columna correspondiente
            suggested_matrix[country] = suggested_col

        # Agregar columna SKU para luego hacer merge/update
        suggested_matrix['SKU'] = subdf['SKU'].values


        # Indexar por SKU para actualizar
        st.session_state.df.set_index('SKU', inplace=True)
        suggested_matrix.set_index('SKU', inplace=True)

        # Actualizar valores solo de las columnas relevantes (países)
        st.session_state.df.update(suggested_matrix)

        st.session_state.df.reset_index(inplace=True)
        
        suggested_matrix['Total'] = suggested_matrix.sum(axis=1)
        suggested_matrix.reset_index(inplace=True)
        gradient_args = {'cmap': new_cmap, 'subset': subset_for_gradient, 'axis': 0}
        styled_suggested = (suggested_matrix.style.background_gradient(**gradient_args))
        
        
        totals = pd.DataFrame([suggested_matrix.sum()])
        totals['DIVISION NAME'] = division_seleccionada
        totals = totals[['DIVISION NAME','GT','ES','RD','HN TGU','HN SPS','Total']]
        
        subset_for_gradient = budget_div.columns.difference(['Total', 'DIVISION NAME'])
        subset_for_format = budget_div.columns.difference(['DIVISION NAME'])
        gradient_args = {
        'cmap': new_cmap,
        'subset': subset_for_gradient,
        'axis': 1
        }
        styled_totals = (totals.sort_values('Total',ascending=False)
                             .style
                             .background_gradient(**gradient_args)
                             .format(fmt_dict)
        )
        st.dataframe(styled_totals, hide_index=True)
        
        st.dataframe(styled_suggested, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# 8. Información general de división seleccionada
# ─────────────────────────────────────────────────────────────────────────────
if display_choice == 'Por división':
    styles_in_division = sorted(df_div['STYLE'].unique())
    style_options = ["Todos los estilos"] + styles_in_division
    selected_style = st.sidebar.selectbox("Selecciona un Style", style_options)

    for key, val in st.session_state.ratings.items():
        if key.startswith(division_seleccionada):
            i = val['df_index']
            st.session_state.df.at[i, 'GT score'] = val['gt_score']
            st.session_state.df.at[i, 'ES score'] = val['es_score']
            st.session_state.df.at[i, 'HN TGU score'] = val['tgu_score']
            st.session_state.df.at[i, 'HN SPS score'] = val['sps_score']
            st.session_state.df.at[i, 'RD score'] = val['rd_score']
            st.session_state.df.at[i, 'valid'] = val['valid']

    st.session_state.df.set_index('SKU', inplace=True)
    suggested_matrix.set_index('SKU', inplace=True)
    st.session_state.df.update(suggested_matrix)
    st.session_state.df.reset_index(inplace=True)
    suggested_matrix.reset_index(inplace=True)

    if division_seleccionada == 'Todas las divisiones':
        df_division = st.session_state.df.copy()
    else:
        df_division = st.session_state.df[st.session_state.df['DIVISION NAME'] == division_seleccionada].copy()

    # Información de estilos en catálogo
    sku_unicos = df_division['SKU'].nunique()
    style_unicos = len(styles_in_division)
    gt_ChosenSKUS = df_division[df_division['GT']>0]['SKU'].nunique()
    gt_ChosenStyles = len(df_division[df_division['GT']>0]['STYLE'].unique())
    es_ChosenSKUS = df_division[df_division['ES']>0]['SKU'].nunique()
    es_ChosenStyles = len(df_division[df_division['ES']>0]['STYLE'].unique())
    rd_ChosenSKUS = df_division[df_division['RD']>0]['SKU'].nunique()
    rd_ChosenStyles = len(df_division[df_division['RD']>0]['STYLE'].unique())
    tgu_ChosenSKUS = df_division[df_division['HN TGU']>0]['SKU'].nunique()
    tgu_ChosenStyles = len(df_division[df_division['HN TGU']>0]['STYLE'].unique())
    sps_ChosenSKUS = df_division[df_division['HN SPS']>0]['SKU'].nunique()
    sps_ChosenStyles = len(df_division[df_division['HN SPS']>0]['STYLE'].unique())

    #Información de estilos/SKUs calificados
    gt_GradedSKUS = len(df_division[~df_division['GT score'].isna()]['SKU'].unique())
    es_GradedSKUS = len(df_division[~df_division['ES score'].isna()]['SKU'].unique())
    rd_GradedSKUS = len(df_division[~df_division['RD score'].isna()]['SKU'].unique())
    tgu_GradedSKUS = len(df_division[~df_division['HN TGU score'].isna()]['SKU'].unique())
    sps_GradedSKUS = len(df_division[~df_division['HN SPS score'].isna()]['SKU'].unique())
    gt_GradedStyles = len(df_division[~df_division['GT score'].isna()]['STYLE'].unique())
    es_GradedStyles = len(df_division[~df_division['ES score'].isna()]['STYLE'].unique())
    rd_GradedStyles = len(df_division[~df_division['RD score'].isna()]['STYLE'].unique())
    tgu_GradedStyles = len(df_division[~df_division['HN TGU score'].isna()]['STYLE'].unique())
    sps_GradedStyles = len(df_division[~df_division['HN SPS score'].isna()]['STYLE'].unique())

    col1, col2, col3 = st.columns([7,7,5])
    with col1:
        st.markdown("#### Productos seleccionados")
        Chosen = pd.DataFrame(columns=['Estilos','SKUs'],index=['Disponible','GT','ES','RD','HN TGU','HN SPS'])
        Chosen.at['Disponible','Estilos'] = style_unicos
        Chosen.at['GT','Estilos'] = gt_ChosenStyles
        Chosen.at['ES','Estilos'] = es_ChosenStyles
        Chosen.at['RD','Estilos'] = rd_ChosenStyles
        Chosen.at['HN TGU','Estilos'] = tgu_ChosenStyles
        Chosen.at['HN SPS','Estilos'] = sps_ChosenStyles
        Chosen.at['Disponible','SKUs'] = sku_unicos
        Chosen.at['GT','SKUs'] = gt_ChosenSKUS
        Chosen.at['ES','SKUs'] = es_ChosenSKUS
        Chosen.at['RD','SKUs'] = rd_ChosenSKUS
        Chosen.at['HN TGU','SKUs'] = tgu_ChosenSKUS
        Chosen.at['HN SPS','SKUs'] = sps_ChosenSKUS
        st.dataframe(Chosen)

    with col2:
        st.markdown("#### Productos calificados")
        Graded = pd.DataFrame(columns=['Estilos','SKUs'],index=['Disponible','GT','ES','RD','HN TGU','HN SPS'])
        Graded.at['Disponible','Estilos'] = style_unicos
        Graded.at['GT','Estilos'] = f"{gt_GradedStyles} calificados, faltan {style_unicos-gt_GradedStyles}"
        Graded.at['ES','Estilos'] = f"{es_GradedStyles} calificados, faltan {style_unicos-es_GradedStyles}"
        Graded.at['RD','Estilos'] = f"{rd_GradedStyles} calificados, faltan {style_unicos-rd_GradedStyles}"
        Graded.at['HN TGU','Estilos'] = f"{tgu_GradedStyles} calificados, faltan {style_unicos-tgu_GradedStyles}"
        Graded.at['HN SPS','Estilos'] = f"{sps_GradedStyles} calificados, faltan {style_unicos-sps_GradedStyles}"
        Graded.at['Disponible','SKUs'] = sku_unicos
        Graded.at['GT','SKUs'] = f"{gt_GradedSKUS} calificados, faltan {sku_unicos-gt_GradedSKUS}"
        Graded.at['ES','SKUs'] = f"{es_GradedSKUS} calificados, faltan {sku_unicos-es_GradedSKUS}"
        Graded.at['RD','SKUs'] = f"{rd_GradedSKUS} calificados, faltan {sku_unicos-rd_GradedSKUS}"
        Graded.at['HN TGU','SKUs'] = f"{tgu_GradedSKUS} calificados, faltan {sku_unicos-tgu_GradedSKUS}"
        Graded.at['HN SPS','SKUs'] = f"{sps_GradedSKUS} calificados, faltan {sku_unicos-sps_GradedSKUS}"
        st.dataframe(Graded)
        
    with col3:
        try:
            st.markdown("#### Colores disponibles")
            color_map = ColorMap(st.session_state.images_colors,df_division['SKU'].unique())
            st.image(color_map, width=250)
        except:
            st.write("No hay producto seleccionado")
        
# ─────────────────────────────────────────────────────────────────────────────
# 9. Ventas históricas de la división seleccionada
# ─────────────────────────────────────────────────────────────────────────────
if display_choice == "Por división":
    if division_seleccionada != "Todas las divisiones":
        col1, col2 = st.columns([1,1])
        with col1:
            ventas_div_hist = ventas_historicas[
                ventas_historicas['DivisionGenero'].str.startswith(division_seleccionada)
            ]

            if not ventas_div_hist.empty:
                st.markdown("#### Ventas mensuales por país")

                pivot_ventas = ventas_div_hist.pivot_table(
                    index='YearMonth',
                    columns='Pais',
                    values='Ventas',
                    aggfunc='sum'
                ).fillna(0)

                fig, ax = plt.subplots(figsize=(10, 5))
                pivot_ventas.plot(ax=ax, marker='o')
                ax.set_title(f"Ventas mensuales - {division_seleccionada}")
                ax.set_xlabel("YearMonth")
                ax.set_ylabel("Ventas")
                ax.grid(True)
                ax.legend(title='País', bbox_to_anchor=(1.05, 1), loc='upper left')
                st.pyplot(fig)
            
            else:
                st.info(f"No hay datos históricos para la división {division_seleccionada}.")
            

            
# ─────────────────────────────────────────────────────────────────────────────
# 10. Formularios por estilo
# ─────────────────────────────────────────────────────────────────────────────
if display_choice == 'Por división':
    if division_seleccionada != 'Todas las divisiones' or selected_style != 'Todos los estilos':
        for style, group in grouped:
            if selected_style != "Todos los estilos" and style != selected_style:
                continue

            st.markdown("---")
            st.markdown(f"### Style: `{style}`")

            row0 = group.iloc[0]
            col_meta1, col_meta2, col_meta3 = st.columns([1, 3, 3])
            with col_meta1:
                st.markdown(f"**LTA:** ${row0['LTA']}")
                st.markdown(f"**OUTSOLE:** {row0['OUTSOLE']}")
                st.markdown(f"**keyItem:** {row0['keyItem']}")
            with col_meta2:
                st.markdown(f"**FEATURES:** {row0['FEATURES']}")
                st.markdown(f"**DESCRIPTION:** {row0['DESCRIPTION']}")

            with st.form(key=f"form_{style}"):
                Ncols = 6
                cols = st.columns(Ncols)

                for idx_row, (idx, row) in enumerate(group.iterrows()):
                    col = cols[idx_row % Ncols]
                    sku = row['SKU']
                    product_key = f"{row['DIVISION NAME']}_{sku}"

                    # Imagen
                    img_buffer = st.session_state.image_buffers.get(sku)
                    if img_buffer:
                        col.image(img_buffer, caption=row['COLOR'], use_container_width=True)
                    else:
                        try:
                            img = Image.open(row['IMAGE']).convert("RGB")
                            col.image(img, caption=row['COLOR'], use_container_width=True)
                        except:
                            col.warning("⚠️ Imagen no encontrada")


                    # Inicializar valores desde session_state
                    rating = st.session_state.ratings.get(product_key)
                    initial_gt_score = rating.get("gt_score")
                    initial_es_score = rating.get("es_score")
                    initial_rd_score = rating.get("rd_score")
                    initial_tgu_score = rating.get("tgu_score")
                    initial_sps_score = rating.get("sps_score")

                    # Inputs
                    gt_score = col.number_input("GT puntuación (0-10)", min_value=0, max_value=10, value=int(initial_gt_score), step=1, key=f"gt_score_{product_key}")
                    es_score = col.number_input("ES puntuación (0-10)", min_value=0, max_value=10, value=int(initial_es_score), step=1, key=f"es_score_{product_key}")
                    rd_score = col.number_input("RD puntuación (0-10)", min_value=0, max_value=10, value=int(initial_rd_score), step=1, key=f"rd_score_{product_key}")
                    tgu_score = col.number_input("HN TGU puntuación (0-10)", min_value=0, max_value=10, value=int(initial_tgu_score), step=1, key=f"tgu_score_{product_key}")
                    sps_score = col.number_input("HN SPS puntuación (0-10)", min_value=0, max_value=10, value=int(initial_sps_score), step=1, key=f"sps_score_{product_key}")

                    # Validaciones (opcional mostrar mensajes)
                    valid_gt_score, gt_score_val = validar_entrada(gt_score)
                    valid_es_score, es_score_val = validar_entrada(es_score)
                    valid_rd_score, rd_score_val = validar_entrada(rd_score)
                    valid_tgu_score, tgu_score_val = validar_entrada(tgu_score)
                    valid_sps_score, sps_score_val = validar_entrada(sps_score)

                    if not valid_gt_score:
                        col.warning("GT debe ser un número entre 0 y 10")
                    if not valid_es_score:
                        col.warning("ES debe ser un número entre 0 y 10")
                    if not valid_rd_score:
                        col.warning("RD debe ser un número entre 0 y 10")
                    if not valid_tgu_score:
                        col.warning("HN TGU debe ser un número entre 0 y 10")
                    if not valid_sps_score:
                        col.warning("HN SPS debe ser un número entre 0 y 10")

                # Botón para guardar el estilo actual
                col_guardar = st.columns([1])[0]
                submitted = col_guardar.form_submit_button("Actualizar")

                if submitted:
                    st.session_state[f'submitted_success_{style}'] = True

                    for idx_row, (idx, row) in enumerate(group.iterrows()):
                        sku = row['SKU']
                        product_key = f"{row['DIVISION NAME']}_{sku}"

                        gt_score = st.session_state.get(f"gt_score_{product_key}", "")
                        es_score = st.session_state.get(f"es_score_{product_key}", "")
                        rd_score = st.session_state.get(f"rd_score_{product_key}", "")
                        tgu_score = st.session_state.get(f"tgu_score_{product_key}", "")
                        sps_score = st.session_state.get(f"sps_score_{product_key}", "")

                        valid_gt_score, gt_score_val = validar_entrada(gt_score)
                        valid_es_score, es_score_val = validar_entrada(es_score)
                        valid_rd_score, rd_score_val = validar_entrada(rd_score)
                        valid_tgu_score, tgu_score_val = validar_entrada(tgu_score)
                        valid_sps_score, sps_score_val = validar_entrada(sps_score)

                        st.session_state.ratings[product_key] = {
                            'gt_score': gt_score_val,
                            'es_score': es_score_val,
                            'rd_score': rd_score_val,
                            'tgu_score': tgu_score_val,
                            'sps_score': sps_score_val,
                            'valid': all([valid_gt_score,valid_es_score,valid_rd_score,valid_tgu_score,valid_sps_score]),
                            'df_index': idx
                        }

                    st.rerun()

            # Aplicar cambios si se guardó ese estilo
            if st.session_state.get(f'submitted_success_{style}', False):
                st.session_state[f'submitted_success_{style}'] = False
                apply_ratings_to_df()
                
                st.success("Calificaciones actualizadas correctamente.")
                
    if division_seleccionada == 'Todas las divisiones':
        excel_buffer = io.BytesIO()
        st.session_state.df.to_excel(excel_buffer, index=False, engine='openpyxl')
        excel_buffer.seek(0)

        st.sidebar.download_button(
            label="⬇️ Descargar sugerencias",
            data=excel_buffer,
            file_name="sugerencias.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

if display_choice == 'Por orden de catálogo':
    if grupo_seleccionado != 'Todos los grupos':
        for style, group in grouped:
            st.markdown("---")
            st.markdown(f"### Style: `{style}`")

            row0 = group.iloc[0]
            col_meta1, col_meta2, col_meta3 = st.columns([1, 3, 3])
            with col_meta1:
                st.markdown(f"**LTA:** ${row0['LTA']}")
                st.markdown(f"**OUTSOLE:** {row0['OUTSOLE']}")
                st.markdown(f"**keyItem:** {row0['keyItem']}")
            with col_meta2:
                st.markdown(f"**FEATURES:** {row0['FEATURES']}")
                st.markdown(f"**DESCRIPTION:** {row0['DESCRIPTION']}")

            with st.form(key=f"form_{style}"):
                Ncols = 6
                cols = st.columns(Ncols)

                for idx_row, (idx, row) in enumerate(group.iterrows()):
                    col = cols[idx_row % Ncols]
                    sku = row['SKU']
                    product_key = f"{row['DIVISION NAME']}_{sku}"

                    # Imagen
                    img_buffer = st.session_state.image_buffers.get(sku)
                    if img_buffer:
                        col.image(img_buffer, caption=row['COLOR'], use_container_width=True)
                    else:
                        try:
                            img = Image.open(row['IMAGE']).convert("RGB")
                            col.image(img, caption=row['COLOR'], use_container_width=True)
                        except:
                            col.warning("⚠️ Imagen no encontrada")


                    # Inicializar valores desde session_state
                    rating = st.session_state.ratings.get(product_key)
                    initial_gt_score = rating.get("gt_score")
                    initial_es_score = rating.get("es_score")
                    initial_rd_score = rating.get("rd_score")
                    initial_tgu_score = rating.get("tgu_score")
                    initial_sps_score = rating.get("sps_score")

                    # Inputs
                    gt_score = col.number_input("GT puntuación (0-10)", min_value=0, max_value=10, value=int(initial_gt_score), step=1, key=f"gt_score_{product_key}")
                    es_score = col.number_input("ES puntuación (0-10)", min_value=0, max_value=10, value=int(initial_es_score), step=1, key=f"es_score_{product_key}")
                    rd_score = col.number_input("RD puntuación (0-10)", min_value=0, max_value=10, value=int(initial_rd_score), step=1, key=f"rd_score_{product_key}")
                    tgu_score = col.number_input("HN TGU puntuación (0-10)", min_value=0, max_value=10, value=int(initial_tgu_score), step=1, key=f"tgu_score_{product_key}")
                    sps_score = col.number_input("HN SPS puntuación (0-10)", min_value=0, max_value=10, value=int(initial_sps_score), step=1, key=f"sps_score_{product_key}")

                    # Validaciones (opcional mostrar mensajes)
                    valid_gt_score, gt_score_val = validar_entrada(gt_score)
                    valid_es_score, es_score_val = validar_entrada(es_score)
                    valid_rd_score, rd_score_val = validar_entrada(rd_score)
                    valid_tgu_score, tgu_score_val = validar_entrada(tgu_score)
                    valid_sps_score, sps_score_val = validar_entrada(sps_score)

                    if not valid_gt_score:
                        col.warning("GT debe ser un número entre 0 y 10")
                    if not valid_es_score:
                        col.warning("ES debe ser un número entre 0 y 10")
                    if not valid_rd_score:
                        col.warning("RD debe ser un número entre 0 y 10")
                    if not valid_tgu_score:
                        col.warning("HN TGU debe ser un número entre 0 y 10")
                    if not valid_sps_score:
                        col.warning("HN SPS debe ser un número entre 0 y 10")

                # Botón para guardar el estilo actual
                col_guardar = st.columns([1])[0]
                submitted = col_guardar.form_submit_button("Actualizar")

                if submitted:
                    st.session_state[f'submitted_success_{style}'] = True

                    for idx_row, (idx, row) in enumerate(group.iterrows()):
                        sku = row['SKU']
                        product_key = f"{row['DIVISION NAME']}_{sku}"

                        gt_score = st.session_state.get(f"gt_score_{product_key}", "")
                        es_score = st.session_state.get(f"es_score_{product_key}", "")
                        rd_score = st.session_state.get(f"rd_score_{product_key}", "")
                        tgu_score = st.session_state.get(f"tgu_score_{product_key}", "")
                        sps_score = st.session_state.get(f"sps_score_{product_key}", "")

                        valid_gt_score, gt_score_val = validar_entrada(gt_score)
                        valid_es_score, es_score_val = validar_entrada(es_score)
                        valid_rd_score, rd_score_val = validar_entrada(rd_score)
                        valid_tgu_score, tgu_score_val = validar_entrada(tgu_score)
                        valid_sps_score, sps_score_val = validar_entrada(sps_score)

                        st.session_state.ratings[product_key] = {
                            'gt_score': gt_score_val,
                            'es_score': es_score_val,
                            'rd_score': rd_score_val,
                            'tgu_score': tgu_score_val,
                            'sps_score': sps_score_val,
                            'valid': all([valid_gt_score,valid_es_score,valid_rd_score,valid_tgu_score,valid_sps_score]),
                            'df_index': idx
                        }

                    st.rerun()

            # Aplicar cambios si se guardó ese estilo
            if st.session_state.get(f'submitted_success_{style}', False):
                st.session_state[f'submitted_success_{style}'] = False
                apply_ratings_to_df()
                
                st.success("Calificaciones actualizadas correctamente.")
                
    excel_buffer = io.BytesIO()
    st.session_state.df.to_excel(excel_buffer, index=False, engine='openpyxl')
    excel_buffer.seek(0)

    st.sidebar.download_button(
        label="⬇️ Descargar sugerencias",
        data=excel_buffer,
        file_name="sugerencias.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


