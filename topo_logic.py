import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap, BoundaryNorm
import os

# Fix for Streamlit Cloud (Headless)
plt.switch_backend('Agg')
import traceback

# ==========================================
# CONSTANTES Y CONFIGURACIÓN
# ==========================================
AREA_MINIMA_M2 = 9.0
GRID_SIZE = 1.0

# Colores Base (Originales 8 bandas para Estadísticas)
COLORES_FINALES_BASE = [
    '#C00000', # < -3x
    '#FF0000', # -3x a -2x
    '#FFC000', # -2x a -1x
    '#92D050', # -1x a 0
    '#92D050', # 0 a 1x 
    '#00B0F0', # 1x a 2x
    '#0070C0', # 2x a 3x
    '#002060'  # > 3x
]

# Colores Base Simplificados (Para Mapas y Zonas Defectuosas)
COLORES_SEMAFORO = [
    '#FF0000', # Rojo (Bajo Tolerancia)
    '#FFC000', # Amarillo (En Tolerancia pero bajo 0)
    '#00B050'  # Verde (Sobre 0)
]

def get_dynamic_ranges(step):
    """Genera límites y etiquetas basados en el paso de tolerancia (Visual)."""
    limits = [-np.inf, -3*step, -2*step, -1*step, 0, 1*step, 2*step, 3*step, np.inf]
    labels = [
        f'< -{3*step}', f'-{3*step} a -{2*step}', f'-{2*step} a -{step}', 
        f'-{step} a 0', f'0 a {step}', f'{step} a {2*step}', 
        f'{2*step} a {3*step}', f'> {3*step}'
    ]
    return limits, labels

def calcular_rangos(df, rasante=None, step=4.0, dynamic_tol=None):
    """
    Calcula estadísticas de distribución (Tabla Resumen).
    MANTENIENDO LÓGICA V1 (8 Bandas) según solicitud del usuario.
    """
    if df.empty: return pd.DataFrame(), df
    df = df.copy()
    
    # Asegurar existencia de Desv_cm
    if 'desviacion' in df.columns:
        df['Desv_cm'] = df['desviacion']
    elif 'Desv_cm' not in df.columns:
        if rasante is not None:
            df['Desv_cm'] = (df['Cota_Calc'] - rasante) * 100
        else:
            df['Desv_cm'] = 0.0
            
    # Usamos 'step' para la distribución estadística (Visual), no 'dynamic_tol'
    # dynamic_tol se usará SOLO para detectar zonas defectuosas después.
    
    limits, labels_txt = get_dynamic_ranges(step)
            
    rangos = [
        (labels_txt[7], lambda x: x > limits[7], 'Corte Crítico'),
        (labels_txt[6], lambda x: (x > limits[6]) & (x <= limits[7]), 'Corte Alto'),
        (labels_txt[5], lambda x: (x > limits[5]) & (x <= limits[6]), 'Corte Alerta'),
        (labels_txt[4], lambda x: (x >= 0) & (x <= limits[5]), 'OK (Corte)'),
        (labels_txt[3], lambda x: (x >= limits[3]) & (x < 0), 'OK (Relleno)'),
        (labels_txt[2], lambda x: (x >= limits[2]) & (x < limits[3]), 'Relleno Alerta'),
        (labels_txt[1], lambda x: (x >= limits[1]) & (x < limits[2]), 'Relleno Bajo'),
        (labels_txt[0], lambda x: x < limits[1], 'Relleno Crítico')
    ]
    
    res, total = [], len(df)
    for i, (lbl, cond, grp) in enumerate(rangos):
        c = len(df[cond(df['Desv_cm'])])
        # Mapeo de color V1 (8 colores)
        color_idx = 7 - i
        color_hex = COLORES_FINALES_BASE[color_idx]
        
        res.append({
            'Tipo': grp, 'Rango': lbl, 'Puntos': c, 
            'Porcentaje': (c/total)*100 if total > 0 else 0,
            'Color': color_hex
        })
    return pd.DataFrame(res), df

def calculate_dynamic_tolerance(cover_cm):
    """
    Calcula la tolerancia dinámica basada en el espesor (Cover).
    
    Reglas:
    - Espesor > 45 cm: 50% del espesor
    - Espesor 30 a 44.9 cm: 30% del espesor
    - Espesor 20 a 29.9 cm: 10% del espesor
    - Espesor < 20 cm: 0.5 cm (Estricto)
    """
    if pd.isna(cover_cm) or cover_cm <= 0:
        return 0.5 # Valor por defecto seguro (Estricto)
        
    if cover_cm > 45:
        return cover_cm * 0.50
    elif cover_cm >= 30:
        return cover_cm * 0.30
    elif cover_cm >= 20:
        return cover_cm * 0.10
    else:
        return 0.5 # Menor a 20cm, tolerancia cero (usamos 0.5 para estabilidad numérica)

def flood_fill_matrix(matrix):
    """Identifica zonas conectadas en una matriz binaria."""
    rows, cols = matrix.shape
    labeled_matrix = np.zeros_like(matrix, dtype=int)
    current_label = 1
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    
    for r in range(rows):
        for c in range(cols):
            if matrix[r, c] == 1 and labeled_matrix[r, c] == 0:
                stack = [(r, c)]
                labeled_matrix[r, c] = current_label
                while stack:
                    curr_r, curr_c = stack.pop()
                    for nr, nc in neighbors:
                        next_r, next_c = curr_r + nr, curr_c + nc
                        if (0 <= next_r < rows and 0 <= next_c < cols and matrix[next_r, next_c] == 1 and labeled_matrix[next_r, next_c] == 0):
                            labeled_matrix[next_r, next_c] = current_label
                            stack.append((next_r, next_c))
                current_label += 1
    return labeled_matrix, current_label - 1

def detectar_zonas(df, col_n, col_e, col_z, tol):
    """Detecta zonas contiguas que exceden la tolerancia. Estrategia: Filtro Puntos -> Grilla."""
    if df.empty: return pd.DataFrame(), 0
    
    # 1. FILTRAR PRIMERO: Solo puntos que son defecto
    # Esto asegura que no se "promedien" defectos con puntos buenos en la misma celda de 1m2
    df_defects = df[df['Desv_cm'] < -tol].copy()
    
    if df_defects.empty:
        return pd.DataFrame(), 0
        
    # Crear grilla solo con puntos defectuosos
    df_defects['GN'] = (df_defects[col_n]//GRID_SIZE)*GRID_SIZE
    df_defects['GE'] = (df_defects[col_e]//GRID_SIZE)*GRID_SIZE
    
    # Agrupar: Z promedio y Desviación promedio (de los malos)
    grid = df_defects.groupby(['GN','GE'])[['Desv_cm', col_z]].mean().reset_index()
    atot = len(grid)*(GRID_SIZE**2) # Area estimada bruta
    
    n_min, e_min = grid['GN'].min(), grid['GE'].min()
    rows = int(grid['GN'].max() - n_min) + 5
    cols = int(grid['GE'].max() - e_min) + 5
    
    if rows > 8000 or cols > 8000: 
        return pd.DataFrame(), atot
    
    mat = np.zeros((rows, cols))
    
    # Map index to grid row and fill matrix
    grid['r_idx'] = ((grid['GN'] - n_min)).astype(int)
    grid['c_idx'] = ((grid['GE'] - e_min)).astype(int)
    
    for _, r in grid.iterrows():
        # Ya filtramos por tolerancia, así que todas las celdas aquí son defecto
        mat[int(r['r_idx']), int(r['c_idx'])] = 1
            
    lbl, num = flood_fill_matrix(mat)
    zonas = []
    
    # Asignar Label a Grid
    grid['Label'] = 0
    for idx, r in grid.iterrows():
        r_i, c_i = int(r['r_idx']), int(r['c_idx'])
        if 0 <= r_i < rows and 0 <= c_i < cols:
            grid.at[idx, 'Label'] = lbl[r_i, c_i]
            
    # Filter only labeled cells (Clusters)
    defect_grid = grid[grid['Label'] > 0]
    
    if not defect_grid.empty:
        zone_grps = defect_grid.groupby('Label')
        
        for label_id, grp in zone_grps:
            area = len(grp) * (GRID_SIZE**2)
            if area >= AREA_MINIMA_M2:
                # Find worst point
                worst_idx = grp['Desv_cm'].idxmin()
                worst_row = grp.loc[worst_idx]
                
                zonas.append({
                    'ID': int(label_id),
                    'Area_Efectiva_m2': area,
                    'Norte': worst_row['GN'],
                    'Este': worst_row['GE'],
                    'Elev_Min': worst_row[col_z],
                    'Desv_Min (cm)': worst_row['Desv_cm']
                })

    return pd.DataFrame(zonas), atot

# ... (Previous imports) ...

def generar_mapa_interactivo(df, zonas_df, col_n, col_e, titulo, tol):
    """Genera un mapa INTERACTIVO (Plotly) de CALOR (Estilo Técnico XY - Match Excel)."""
    try:
        # Limpieza
        df[col_n] = pd.to_numeric(df[col_n], errors='coerce')
        df[col_e] = pd.to_numeric(df[col_e], errors='coerce')
        df_clean = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[col_n, col_e, 'Desv_cm'])
        
        if df_clean.empty: return f"Error: Sin datos válidos."
        
        # Color Semáforo (Rojo/Amarillo/Verde)
        def get_color_semaforo(x):
            if x < -tol: return '#FF0000' # Rojo
            elif x < 0: return '#FFC000'  # Amarillo
            else: return '#00B050'        # Verde

        point_colors = df_clean['Desv_cm'].apply(get_color_semaforo).tolist()

        fig = go.Figure()
        
        # 1. PUNTOS DEL MAPA DE CALOR (Scatter Simple)
        # Usamos Scatter en lugar de Scattergl para asegurar estabilidad visual si el dataset no es masivo.
        fig.add_trace(go.Scatter(
            x=df_clean[col_e],
            y=df_clean[col_n],
            mode='markers',
            marker=dict(
                size=5,
                color=point_colors,
                opacity=0.9,
                line=dict(width=0) # Sin borde para limpieza
            ),
            text=df_clean['Desv_cm'].apply(lambda x: f"Desv: {x:.1f}cm"),
            hoverinfo='text',
            name='Puntos'
        ))
        
        # 2. MARCADORES DE ZONAS DEFECTUOSAS (Negros)
        if not zonas_df.empty:
             if 'Norte' in zonas_df.columns and 'Este' in zonas_df.columns:
                z_n = zonas_df['Norte'].values
                z_e = zonas_df['Este'].values
                z_ids = zonas_df['ID'].values
                z_areas = zonas_df['Area_Efectiva_m2'].values
                z_desv = zonas_df['Desv_Min (cm)'].values if 'Desv_Min (cm)' in zonas_df.columns else [0]*len(z_ids)
                z_elev = zonas_df['Elev_Min'].values if 'Elev_Min' in zonas_df.columns else [0]*len(z_ids)
                
                fig.add_trace(go.Scatter(
                    x=z_e, y=z_n,
                    mode='markers+text',
                    marker=dict(size=12, color='black', symbol='circle', line=dict(color='white', width=1)),
                    text=[str(i) for i in z_ids],
                    textposition='top center',
                    name='Zonas ID',
                    textfont=dict(size=14, color='black', family="Arial Black"),
                    hoverinfo='text',
                    hovertext=[f"ID: {i}<br>Area: {a:.0f} m2<br>Desv Min: {d:.1f} cm<br>Elev Min: {el:.3f} m<br>N: {n:.0f}<br>E: {e:.0f}" 
                               for i, a, d, el, n, e in zip(z_ids, z_areas, z_desv, z_elev, z_n, z_e)]
                ))

        # Configurar Layout Cartesian (Imitando Matplotlib)
        # Fix: Ensure axes are formatted as integers (d) and aspect ratio is 1.
        fig.update_layout(
            # title removed per user request (overlap issues)
            plot_bgcolor='#EBEBEB', # Gris claro de fondo (Estilo Matplotlib)
            xaxis=dict(
                title="Este (X)",
                showgrid=True, gridcolor='white', gridwidth=1,
                zeroline=False,
                scaleanchor="y", scaleratio=1, # Aspect Ratio 1:1
                tickformat="d" # Enteros estrictos
            ),
            yaxis=dict(
                title="Norte (Y)",
                showgrid=True, gridcolor='white', gridwidth=1,
                zeroline=False,
                tickformat="d" # Enteros estrictos (sin comas ni puntos)
            ),
            margin={"r":20,"t":40,"l":20,"b":20},
            height=700, # Un poco más alto para ver mejor
            showlegend=False,
            dragmode='zoom', # Default to zoom
            hovermode='closest'
        )
        return fig
            
    except Exception as e:
        traceback.print_exc()
        return f"Error General Mapa: {str(e)}"

def procesar_turno(df, rasante, tolerancia, col_z, col_n, col_e, step=4.0, cover_cm=0.0):
    """Procesa un turno completo y retorna resultados."""
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), 0.0

    # 1. Tolerancia Dinámica (Si hay Cover)
    tol_detect = tolerancia
    # NOTE: topo_dashboard passes calculated tolerance in 'tolerancia' arg already if logic updated there.
    # But if not, we re-calc here? 
    # Current topo_dashboard calls with 'tolerancia=tol_calculated'. So we trust input.
    pass 
    # Logic note: 'calculate_dynamic_tolerance' usage inside here might be redundant if caller handles it.
    # Let's trust the caller provided 'tolerancia' is the correct cut-off.
        
    # 2. Calcular Estadísticas (VISUAL / DISTRIBUCIÓN)
    tbl_rangos, df_cal = calcular_rangos(df, rasante, step=step)
    
    # 3. Detectar Zonas Defectuosas (CRITERIO TÉCNICO / TRAFICO)
    # UPDATED call with col_z
    zonas, area_defectos_bruta = detectar_zonas(df_cal, col_n, col_e, col_z, tol_detect)
    
    # NEW KPI LOGIC (User Request):
    # Incidencia = (Area Zona Defecto / Area TOTAL Trabajada del Turno) * 100
    
    # Calcular Área Total Trabajada (Todo lo levantado/pintado) estimando por grilla
    # Usamos df_cal que ya tiene puntos validos.
    # Grid Size es constante global.
    if not df_cal.empty:
        # Repetimos logica grilla rapida
        gn_all = (df_cal[col_n]//GRID_SIZE).astype(int)
        ge_all = (df_cal[col_e]//GRID_SIZE).astype(int)
        # Unique cells
        unique_cells = len(df_cal.groupby([gn_all, ge_all]).size())
        area_turno_total = unique_cells * (GRID_SIZE**2)
    else:
        area_turno_total = 1.0 # Evitar div/0
        
    # 3. Calcular KPI (Incidencia)
    if not zonas.empty and area_turno_total > 0:
        zonas['KPI Incidencia'] = (zonas['Area_Efectiva_m2'] / area_turno_total)
    elif not zonas.empty:
        zonas['KPI Incidencia'] = 0.0
        
    return tbl_rangos, df_cal, zonas, area_defectos_bruta
    
    # Return tolerance used so dashboard can show it? 
    # Current signature doesn't support returning it, but implementation is enough for now.
    return tbl_rangos, df_cal, zonas, area_tot

def generar_texto_analisis(stats_df, zonas_df, atot, poza):
    """Genera el texto de análisis técnico para el reporte."""
    if stats_df.empty: return "Sin datos."
    
    # Encontrar rango predominante
    pred = stats_df.loc[stats_df['Puntos'].idxmax()]
    
    cant_zonas = len(zonas_df) if not zonas_df.empty else 0
    area_mala = zonas_df['Area_Efectiva_m2'].sum() if not zonas_df.empty else 0
    
    return (f"ANÁLISIS TÉCNICO - {poza}\n\n1. SITUACIÓN GENERAL:\n"
            f"   El rango predominante es '{pred['Rango']}', con un {pred['Porcentaje']:.1f}% de la superficie.\n\n"
            f"2. ÁREAS DEFECTUOSAS:\n   Se detectaron {cant_zonas} zonas críticas (>{AREA_MINIMA_M2}m²). "
            f"La superficie total afectada es de {int(area_mala)} m² sobre un total de {int(atot)} m².\n\n"
            f"3. RECOMENDACIÓN:\n   Se sugiere priorizar las zonas identificadas para trabajos de renivelación.")

# ==========================================
# NUEVA LÓGICA: MAPAS SATELITALES (MATPLOTLIB + CONTEXTILY)
# ==========================================
import contextily as ctx
try:
    import pyproj
except ImportError:
    pass

def generar_mapa_satelital_interactivo(df, zonas_df, col_n, col_e, titulo, tol):
    """Genera un mapa INTERACTIVO (Plotly) con fondo Satelital (Esri)."""
    try:
        # Limpieza y Conversión Numérica
        df[col_n] = pd.to_numeric(df[col_n], errors='coerce')
        df[col_e] = pd.to_numeric(df[col_e], errors='coerce')
        
        df_clean = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[col_n, col_e, 'Desv_cm'])
        if df_clean.empty: 
            return f"Error: Sin datos válidos. Revise columnas {col_n}/{col_e} o valores vacíos."
        
        # CONVERSIÓN DE COORDENADAS (UTM 19S -> WGS84 Lat/Lon)
        # Requerido para mapas web (Plotly Mapbox)
        transformer = None
        try:
            transformer = pyproj.Transformer.from_crs("EPSG:32719", "EPSG:4326", always_xy=True)
            # Transform expects (x, y) -> returns (lon, lat)
            lons, lats = transformer.transform(df_clean[col_e].values, df_clean[col_n].values)
            df_clean['lon'] = lons
            df_clean['lat'] = lats
        except Exception as e:
            return f"Error Proyección: {str(e)}"

        # Configurar colores y rangos (igual que antes)
        limits, labels = get_dynamic_ranges(tol)
        
        fig = go.Figure()
        
        # --- ZONAS DEFECTUOSAS MARCADORES ---
        if not zonas_df.empty:
             if 'Norte' in zonas_df.columns and 'Este' in zonas_df.columns:
                z_n = zonas_df['Norte'].values
                z_e = zonas_df['Este'].values
                z_ids = zonas_df['ID'].values
                
                if transformer:
                    lat_z, lon_z = transformer.transform(z_e, z_n)
                else:
                    lat_z, lon_z = z_n, z_e
                    
                fig.add_trace(go.Scattermapbox(
                    lat=lat_z, lon=lon_z,
                    mode='markers+text',
                    marker=dict(size=14, color='black', symbol='circle'),
                    text=[str(i) for i in z_ids],
                    textposition='top center',
                    name='Zonas ID',
                    textfont=dict(size=14, color='black', family="Arial Black"),
                    hoverinfo='text',
                    hovertext=[f"ID: {i}<br>Area: {a:.0f}m2<br>Desv Min: {d:.1f}cm" 
                               for i, a, d in zip(z_ids, zonas_df['Area_Efectiva_m2'], zonas_df['Desv_Min (cm)'])]
                ))


        # Añadir Puntos
        fig.add_trace(go.Scattermapbox(
            lat=df_clean['lat'],
            lon=df_clean['lon'],
            mode='markers',
            marker=go.scattermapbox.Marker(
                size=8, # Tamaño pixelado visible
                color=df_clean['Desv_cm'],
                colorscale=[
                    [0.0, '#C00000'],    # < -3x
                    [0.125, '#FF0000'],  # -3x a -2x
                    [0.25, '#FFC000'],   # -2x a -1x
                    [0.375, '#92D050'],  # -1x a 0
                    [0.5, '#92D050'],    # 0 a 1x 
                    [0.625, '#00B0F0'],  # 1x a 2x
                    [0.75, '#0070C0'],   # 2x a 3x
                    [1.0, '#002060']     # > 3x
                ],
                cmin=-3*tol, cmax=3*tol, # Fijar min/max para estabilidad visual
                opacity=0.8,
            ),
            text=df_clean['Desv_cm'].apply(lambda x: f"Desv: {x:.1f}cm"),
            hoverinfo='text'
        ))

        # Configurar Layout Maps (Esri Satellite)
        fig.update_layout(
            title=dict(text=f"Satelital Interactivo - {titulo}", y=0.98),
            mapbox=dict(
                style="white-bg", # Estilo base vacío para poner capas encima
                layers=[
                    {
                        "below": 'traces',
                        "sourcetype": "raster",
                        "sourceattribution": "Esri World Imagery",
                        "source": [
                            "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
                        ]
                    }
                ],
                center=dict(lat=df_clean['lat'].mean(), lon=df_clean['lon'].mean()),
                zoom=16 # Zoom inicial cercano
            ),
            margin={"r":0,"t":40,"l":0,"b":0},
            height=600
        )
        return fig
            
    except Exception as e:
        traceback.print_exc()
        return f"Error General Satélite: {str(e)}"

# ==========================================
# LÓGICA DE VISUALIZACIÓN (ANTIGUA)
# ==========================================

def generar_mapa_matplotlib(df, zonas, col_n, col_e, titulo, tol):
    """
    Genera un objeto Figura de Matplotlib con el mapa de calor.
    Retorna (fig, ax) para ser usado en Streamlit.
    """
    try:
        # Limpieza robusta y Coerción Numérica
        df[col_n] = pd.to_numeric(df[col_n], errors='coerce')
        df[col_e] = pd.to_numeric(df[col_e], errors='coerce')

        df_clean = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[col_n, col_e, 'Desv_cm'])
        if df_clean.empty:
            raise ValueError(f"Sin datos válidos para graficar. Revisar columnas {col_n}, {col_e} o Desv_cm.")

        x_min, x_max = df_clean[col_e].min(), df_clean[col_e].max()
        y_min, y_max = df_clean[col_n].min(), df_clean[col_n].max()
        
        # Validación de rangos
        if pd.isna(x_min) or pd.isna(x_max) or pd.isna(y_min) or pd.isna(y_max):
             raise ValueError(f"Rango de coordenadas inválido (NaN). Min/Max E: {x_min}/{x_max}, N: {y_min}/{y_max}")

        # Margen dinámico (5% del rango o min 5m)
        dx, dy = x_max - x_min, y_max - y_min
        margin = max(5, max(dx, dy) * 0.05)

        # Configurar Figura con estilo "ggplot" like (Gris de fondo)
        plt.style.use('ggplot')
        fig, ax = plt.subplots(figsize=(12, 10), dpi=120)
        ax.set_facecolor('#EBEBEB') # Gris claro de fondo
        
        # Obtener rangos dinámicos
        # limits, _ = get_dynamic_ranges(tol) # Old logic
        
        # New Logic: 3 Colors (Red, Yellow, Green) matching the Interactive Map
        # Boundaries: [-inf, -tol, 0, inf]
        # But for BoundaryNorm we need finite values. We can use min/max of data.
        
        # Colormap Simplificado
        # Rojo: < -tol
        # Amarillo: [-tol, 0)
        # Verde: >= 0
        
        # Definir Colores y Limites
        cmap_custom = ListedColormap(['#FF0000', '#FFC000', '#00B050']) # Rojo, Amarillo, Verde
        
        # Limites para la normalización: 
        # [VeryLow, -tol, 0, VeryHigh]
        # Usamos valores extremos seguros
        bounds = [df_clean['Desv_cm'].min() - 1, -tol, 0, df_clean['Desv_cm'].max() + 1]
        
        # Ajuste si min > -tol (no hay rojos) o similares, pero BoundaryNorm maneja bins.
        # Necesitamos asegurar que -tol y 0 esten en orden y dentro del rango si es posible, o forzarlos.
        # Mejor: Usar 'Desv' values to map colors manually implies no Colorbar gradient, straightforward mapping.
        # But colorbar is nice.
        
        # Let's try explicit coloring like Plotly to be 100% exact.
        colors_mapped = []
        for x in df_clean['Desv_cm']:
            if x < -tol: colors_mapped.append('#FF0000')
            elif x < 0:  colors_mapped.append('#FFC000')
            else:        colors_mapped.append('#00B050')
            
        # Scatter Plot con colores directos
        sc = ax.scatter(
            df_clean[col_e], df_clean[col_n],
            c=colors_mapped,
            s=15, marker='o', alpha=0.9, edgecolors='none', zorder=10
        )

        # Colorbar - Custom legend instead? 
        # Since we use direct colors, a standard colorbar won't work automatically attached to 'sc'.
        # We create a dummy mappable or just a Legend. 
        # Legend is better for discrete categories.
        legend_elements = [
            patches.Patch(facecolor='#FF0000', edgecolor='none', label=f'< -{tol} cm'),
            patches.Patch(facecolor='#FFC000', edgecolor='none', label=f'-{tol} a 0 cm'),
            patches.Patch(facecolor='#00B050', edgecolor='none', label='> 0 cm')
        ]
        ax.legend(handles=legend_elements, loc='upper right', title="Desviación")


        if not zonas.empty:
            # --- MARCADORES DE PUNTOS CRÍTICOS (ID) ---
            # Check if we have point coordinates (New Logic)
            if 'Norte' in zonas.columns and 'Este' in zonas.columns:
                 # Plot markers (Black dots)
                 ax.scatter(
                     zonas['Este'], zonas['Norte'],
                     c='black', s=40, marker='o', edgecolors='white', linewidth=1, zorder=25, label='Zona Defectuosa'
                 )
                 # Add ID labels
                 for _, z in zonas.iterrows():
                     ax.text(
                         z['Este'], z['Norte'], str(int(z['ID'])),
                         color='black', fontsize=9, fontweight='bold', ha='center', va='bottom', zorder=30
                     )
            # Fallback for old data (Rectangles)
            elif 'E_Min' in zonas.columns:
                for _, z in zonas.iterrows():
                    if pd.isna(z['E_Min']) or pd.isna(z['N_Min']): continue
                    width = z['E_Max'] - z['E_Min']
                    height = z['N_Max'] - z['N_Min']
                    rect = patches.Rectangle(
                        (z['E_Min'], z['N_Min']), width, height,
                        linewidth=2.0, edgecolor='#FF0000', facecolor='none', zorder=20
                    )
                    ax.add_patch(rect)

        # Titulo y Ejes
        ax.set_title(f"{titulo}", fontsize=14, fontweight='bold', pad=15)
        ax.set_aspect('equal')
        ax.set_xlim(x_min - margin, x_max + margin)
        ax.set_ylim(y_min - margin, y_max + margin)
        
        # Grid blanco fino
        ax.grid(True, color='white', linestyle='-', linewidth=0.5, alpha=0.8)
        
        # Etiquetas de ejes
        ax.set_xlabel('Este (X)', fontsize=10)
        ax.set_ylabel('Norte (Y)', fontsize=10)
        
        return fig
        
    except Exception:
        traceback.print_exc()
        return None

# ==========================================
# NUEVA LÓGICA: MAPAS INTERACTIVOS (PLOTLY)
# ==========================================
import plotly.graph_objects as go

# Legacy function removed. Now standardized on generar_mapa_interactivo (formerly satelital).

# Legacy function removed. Now standardized on generar_mapa_interactivo (formerly satelital).
