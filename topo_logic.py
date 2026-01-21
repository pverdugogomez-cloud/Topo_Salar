import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap, BoundaryNorm
import traceback

# ==========================================
# CONSTANTES Y CONFIGURACIÓN
# ==========================================
AREA_MINIMA_M2 = 9.0
GRID_SIZE = 1.0

# Colores Base (Solo definimos colores, los límites ahora son dinámicos)
COLORES_FINALES_BASE = [
    '#C00000', # < -3x      (Crítico Bajo)
    '#FF0000', # -3x a -2x  (Bajo)
    '#FFC000', # -2x a -1x  (Alerta Bajo)
    '#92D050', # -1x a 0    (OK - Verde Claro)
    '#92D050', # 0 a 1x     (OK - Verde Claro)
    '#00B0F0', # 1x a 2x    (Alerta Alto)
    '#0070C0', # 2x a 3x    (Alto)
    '#002060'  # > 3x       (Crítico Alto)
]

def get_dynamic_ranges(step):
    """Genera límites y etiquetas basados en el paso de tolerancia."""
    # Limits: [-inf, -3s, -2s, -s, 0, s, 2s, 3s, inf]
    limits = [-np.inf, -3*step, -2*step, -1*step, 0, 1*step, 2*step, 3*step, np.inf]
    
    # Labels for display
    labels = [
        f'< -{3*step}', 
        f'-{3*step} a -{2*step}', 
        f'-{2*step} a -{step}', 
        f'-{step} a 0', 
        f'0 a {step}', 
        f'{step} a {2*step}', 
        f'{2*step} a {3*step}', 
        f'> {3*step}'
    ]
    return limits, labels

# ==========================================
# LÓGICA DE CÁLCULO
# ==========================================

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

def calcular_rangos(df, rasante=None, step=4.0):
    """Calcula estadísticas de desviación respecto a la rasante con rangos dinámicos."""
    if df.empty: return pd.DataFrame(), df
    df = df.copy()
    
    # Asegurar existencia de Desv_cm (si ya viene calculada, usarla)
    if 'desviacion' in df.columns:
        # Compatibilidad si ya se calculó fuera
        df['Desv_cm'] = df['desviacion']
    elif 'Desv_cm' not in df.columns:
        if rasante is not None:
            df['Desv_cm'] = (df['Cota_Calc'] - rasante) * 100
        else:
            df['Desv_cm'] = 0.0
    
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
        # rangos está ordenado de Mayor (>12) a Menor (<-12)
        # COLORES_FINALES_BASE está ordenado de Menor (<-12) a Mayor (>12)
        # Por tanto, el índice de color correspondiente es (7 - i)
        color_idx = 7 - i
        color_hex = COLORES_FINALES_BASE[color_idx]
        
        res.append({
            'Tipo': grp, 
            'Rango': lbl, 
            'Puntos': c, 
            'Porcentaje': (c/total)*100 if total > 0 else 0,
            'Color': color_hex
        })
    return pd.DataFrame(res), df

def detectar_zonas(df, col_n, col_e, tol):
    """Detecta zonas contiguas que exceden la tolerancia."""
    if df.empty: return pd.DataFrame(), 0
    
    # Crear grilla
    df['GN'] = (df[col_n]//GRID_SIZE)*GRID_SIZE
    df['GE'] = (df[col_e]//GRID_SIZE)*GRID_SIZE
    
    grid = df.groupby(['GN','GE'])['Desv_cm'].mean().reset_index()
    atot = len(grid)*(GRID_SIZE**2)
    
    n_min, e_min = grid['GN'].min(), grid['GE'].min()
    rows = int(grid['GN'].max() - n_min) + 5
    cols = int(grid['GE'].max() - e_min) + 5
    
    if rows > 8000 or cols > 8000: 
        return pd.DataFrame(), atot # Evitar crash por memoria
    
    mat = np.zeros((rows, cols))
    for _, r in grid.iterrows():
        if abs(r['Desv_cm']) > tol: 
            mat[int(r['GN']-n_min), int(r['GE']-e_min)] = 1
            
    lbl, num = flood_fill_matrix(mat)
    zonas = []
    for i in range(1, num+1):
        inds = np.where(lbl == i)
        area = len(inds[0]) * (GRID_SIZE**2)
        if area >= AREA_MINIMA_M2:
            rmin, rmax = np.min(inds[0]), np.max(inds[0])
            cmin, cmax = np.min(inds[1]), np.max(inds[1])
            zonas.append({
                'ID': i, 'Area_Efectiva_m2': area, 
                'N_Min': n_min + rmin, 'N_Max': n_min + rmax + 1, 
                'E_Min': e_min + cmin, 'E_Max': e_min + cmax + 1
            })
    return pd.DataFrame(zonas), atot

def procesar_turno(df, rasante, tolerancia, col_z, col_n, col_e, step=4.0):
    """Procesa un turno completo y retorna resultados."""
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), 0.0

    # 1. Calcular Desviaciones (Si ya no vienen)
    # Si viene 'desviacion' from dashboard usamos esa, calcular_rangos lo maneja.
    tbl_rangos, df_cal = calcular_rangos(df, rasante, step=step)
    
    # 2. Detectar Zonas Defectuosas
    zonas, area_tot = detectar_zonas(df_cal, col_n, col_e, tolerancia)
    
    # 3. Calcular KPI (Incidencia)
    if not zonas.empty and area_tot > 0:
        zonas['Incidencia (%)'] = (zonas['Area_Efectiva_m2'] / area_tot) * 100
    elif not zonas.empty:
        zonas['Incidencia (%)'] = 0.0
    
    return tbl_rangos, df_cal, zonas, area_tot

def generar_texto_analisis(stats_df, zonas_df, atot, poza):
    """Genera el texto de análisis técnico para el reporte."""
    if stats_df.empty: return "Sin datos."
    
    # Encontrar rango predominante
    pred = stats_df.loc[stats_df['Puntos'].idxmax()]
    
    cant_zonas = len(zonas_df) if not zonas_df.empty else 0
    area_mala = zonas_df['Area_Efectiva_m2'].sum() if not zonas_df.empty else 0
    
    return (f"ANÁLISIS TÉCNICO - {poza}\n\n1. SITUACIÓN GENERAL:\n"
            f"   El rango predominante es '{pred['Rango']}', con un {pred['Porcentaje']:.1%} de la superficie.\n\n"
            f"2. ÁREAS DEFECTUOSAS:\n   Se detectaron {cant_zonas} zonas críticas (>{AREA_MINIMA_M2}m²). "
            f"La superficie total afectada es de {int(area_mala)} m² sobre un total de {int(atot)} m².\n\n"
            f"3. RECOMENDACIÓN:\n   Se sugiere priorizar las zonas identificadas para trabajos de renivelación.")

# ==========================================
# LÓGICA DE VISUALIZACIÓN
# ==========================================

def generar_mapa_matplotlib(df, zonas, col_n, col_e, titulo, tol):
    """
    Genera un objeto Figura de Matplotlib con el mapa de calor.
    Retorna (fig, ax) para ser usado en Streamlit.
    """
    try:
        # Limpieza robusta
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
        
        # Colormap
        cmap_custom = ListedColormap(COLORES_FINALES)
        norm_custom = BoundaryNorm(LIMITES_FINALES, cmap_custom.N)

        # Scatter Plot - Puntos de fondo más grandes para visibilidad
        sc = ax.scatter(
            df_clean[col_e], df_clean[col_n],
            c=df_clean['Desv_cm'],
            cmap=cmap_custom,
            norm=norm_custom,
            s=15, marker='o', alpha=0.9, edgecolors='none', zorder=10
        )

        # Colorbar
        ticks_bar = [-12, -8, -4, 0, 4, 8, 12]
        cbar = plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.04, ticks=ticks_bar)
        cbar.ax.set_yticklabels([str(x) for x in ticks_bar])
        cbar.set_label('Desviación (cm)')

        # Zonas Defectuosas
        if not zonas.empty:
            for _, z in zonas.iterrows():
                if pd.isna(z['E_Min']) or pd.isna(z['N_Min']): continue
                width = z['E_Max'] - z['E_Min']
                height = z['N_Max'] - z['N_Min']
                
                # Rectángulo rojo brillante
                rect = patches.Rectangle(
                    (z['E_Min'], z['N_Min']), width, height,
                    linewidth=2.0, edgecolor='#FF0000', facecolor='none', zorder=20
                )
                ax.add_patch(rect)
                
                # Etiqueta estilo "Tag" (Rojo con texto blanco)
                label_txt = f"ID:{int(z['ID'])}\n{int(z['Area_Efectiva_m2'])}m²"
                ax.text(
                    z['E_Min'], z['N_Max'], label_txt,
                    color='white', fontsize=8, fontweight='bold',
                    bbox=dict(facecolor='#D32F2F', alpha=0.9, edgecolor='white', boxstyle='round,pad=0.3'),
                    verticalalignment='bottom', horizontalalignment='left', zorder=21
                )

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

def generar_mapa_interactivo(df, zonas, col_n, col_e, titulo="Mapa Calor", step=4.0):
    """
    Genera un mapa interactivo con Plotly graph_objects.
    """
    try:
        # Limpieza robusta
        df_clean = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[col_n, col_e, 'Desv_cm'])
        if df_clean.empty:
            return None

        limits, labels = get_dynamic_ranges(step)
        # Limits: [-inf, -3s, -2s, -s, 0, s, 2s, 3s, inf]
        # Colors: C00000 (-inf..-3s), FF0000, FFC000, 92D050 (-s..0), 92D050 (0..s), 00B0F0, 0070C0, 002060 (>3s)
        
        # Color mapping logic
        def get_color(v):
            if v < limits[1]: return COLORES_FINALES_BASE[0] # < -3s
            elif v < limits[2]: return COLORES_FINALES_BASE[1] # -3s to -2s
            elif v < limits[3]: return COLORES_FINALES_BASE[2] # -2s to -s
            elif v < limits[4]: return COLORES_FINALES_BASE[3] # -s to 0 (OK)
            elif v < limits[5]: return COLORES_FINALES_BASE[4] # 0 to s (OK)
            elif v < limits[6]: return COLORES_FINALES_BASE[5] # s to 2s
            elif v < limits[7]: return COLORES_FINALES_BASE[6] # 2s to 3s
            else: return COLORES_FINALES_BASE[7] # > 3s

        # Asignar colores a cada punto
        colors = df_clean['Desv_cm'].apply(get_color)

        fig = go.Figure()

        # 1. Capa de Puntos (Scattergl para mejor performance con muchos puntos)
        fig.add_trace(go.Scattergl(
            x=df_clean[col_e],
            y=df_clean[col_n],
            mode='markers',
            marker=dict(
                size=5,
                color=colors,
                opacity=0.8
            ),
            text=[f"Desv: {d:.1f}cm<br>N: {n:.1f}<br>E: {e:.1f}" 
                  for d, n, e in zip(df_clean['Desv_cm'], df_clean[col_n], df_clean[col_e])],
            name='Puntos'
        ))

        # 2. Capa de Rectángulos (Zonas)
        if not zonas.empty:
            for _, row in zonas.iterrows():
                fig.add_shape(
                    type="rect",
                    x0=row['E_Min'], y0=row['N_Min'],
                    x1=row['E_Max'], y1=row['N_Max'],
                    line=dict(color="Black", width=2),
                    fillcolor="rgba(0,0,0,0)",
                )
                fig.add_annotation(
                    x=(row['E_Min']+row['E_Max'])/2,
                    y=(row['N_Max']),
                    text=f"Zona {row['ID']}<br>{row['Area_Efectiva_m2']:.1f}m²",
                    showarrow=False,
                    yshift=10,
                    bgcolor="white",
                    opacity=0.7
                )
        
        # Layout
        fig.update_layout(
            title=titulo,
            xaxis_title="Este (X)",
            yaxis_title="Norte (Y)",
            yaxis=dict(
                scaleanchor="x",
                scaleratio=1,
            ),
            hovermode='closest',
            dragmode='pan',
            height=600,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        return fig

    except Exception as e:
        traceback.print_exc()
        return None

        
    except Exception as e:
        traceback.print_exc()
        return None
