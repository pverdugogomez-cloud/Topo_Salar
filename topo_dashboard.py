import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import io
from datetime import datetime
import topo_logic_v5 as topo_logic
import traceback
import db_manager # Feature: History DB

# --- HISTORY MANAGEMENT DIALOG ---
@st.dialog("Gestor de Historial de Calidad", width="large")
def show_history_modal():
     st.info("Aquí podrá visualizar y editar el historial de puntos bajos detectados.")
     
     hist_df = db_manager.load_history()
     
     if hist_df.empty:
         st.warning("No hay historial disponible aún.")
     else:
         # Filters for View
         col_f1, col_f2 = st.columns(2)
         with col_f1:
             f_poza = st.selectbox("Filtrar por Poza:", ["TODOS"] + sorted(hist_df['Poza'].unique().tolist()), key="hist_filter_poza")
         with col_f2:
             f_estado = st.selectbox("Filtrar por Estado:", ["TODOS"] + sorted(hist_df['Estado'].unique().tolist()), key="hist_filter_estado")
         
         # Apply Filters
         df_view = hist_df.copy()
         if f_poza != "TODOS":
             df_view = df_view[df_view['Poza'] == f_poza]
         if f_estado != "TODOS":
             df_view = df_view[df_view['Estado'] == f_estado]
             
         # Editable Dataframe
         edited_hist = st.data_editor(
             df_view,
             column_order=("ID", "Fecha_Reporte", "Turno", "Poza", "Cota_Teorica", "Norte", "Este", "Cota_GPS", "Desv_GPS", "Cota_Real_Terreno", "Desv_Real", "Observacion", "Estado"),
             column_config={
                 "ID": st.column_config.NumberColumn("ID", disabled=True, width="small"),
                 "ID_Unico": None, # Hide Hash
                 "Fecha_Reporte": st.column_config.TextColumn("Fecha", disabled=True),
                 "Turno": st.column_config.TextColumn("Turno", disabled=True, width="small"),
                 "Poza": st.column_config.TextColumn("Poza", disabled=True),
                 "Cota_Teorica": st.column_config.NumberColumn("Rasante (m)", disabled=True, format="%.3f"),
                 "Norte": st.column_config.NumberColumn("Norte", disabled=True, format="%.0f"),
                 "Este": st.column_config.NumberColumn("Este", disabled=True, format="%.0f"),
                 "Cota_GPS": st.column_config.NumberColumn("Cota Sistema (m)", disabled=True, format="%.3f"),
                 "Desv_GPS": st.column_config.NumberColumn("Desv. Sistema (cm)", disabled=True, format="%.1f"),
                 "Cota_Real_Terreno": st.column_config.NumberColumn("Cota Real (m)", required=False, format="%.3f"),
                 "Desv_Real": st.column_config.NumberColumn("Desv Real (cm)", disabled=True, format="%.1f"),
                 "Observacion": st.column_config.TextColumn("Observación", width="medium"),
                 "Estado": st.column_config.SelectboxColumn("Estado", options=["Pendiente", "Revisado", "Corregido", "Descartado"])
             },
             hide_index=True,
             use_container_width=True,
             key="hist_editor_main",
             num_rows="dynamic"
         )
         
         if st.button("Guardar Cambios Historial"):
             # --- UPDATE LOGIC ---
             for idx, row in edited_hist.iterrows():
                 # Auto-calc deviation if real z changed
                 if pd.notnull(row['Cota_Real_Terreno']) and pd.notnull(row['Cota_Teorica']):
                     try:
                         val_real = float(row['Cota_Real_Terreno'])
                         val_teor = float(row['Cota_Teorica'])
                         row['Desv_Real'] = (val_real - val_teor) * 100.0
                     except: pass
                 
                 mask_id = hist_df['ID_Unico'] == row['ID_Unico']
                 if mask_id.any():
                     hist_df.loc[mask_id, 'Cota_Real_Terreno'] = row['Cota_Real_Terreno']
                     hist_df.loc[mask_id, 'Desv_Real'] = row['Desv_Real']
                     hist_df.loc[mask_id, 'Observacion'] = row['Observacion']
                     hist_df.loc[mask_id, 'Estado'] = row['Estado']
             
             if db_manager.save_history(hist_df):
                 st.success("Historial Actualizado.")
                 st.rerun()
             else:
                 st.error("Error guardando historial.")
         
         # Export Button
         excel_bytes = db_manager.export_to_excel(hist_df) 
         st.download_button("📥 Descargar Excel Historial", excel_bytes, f"Historial_Calidad_{datetime.now().strftime('%Y%m%d')}.xlsx")

from topo_logic_v5 import (
    procesar_excel, 
    generar_mapa_interactivo, 
    filtrar_datos,
    load_settings,
    save_settings,
    generar_pptx_report
)
import time
import math

try:
    from streamlit_plotly_events import plotly_events
except ImportError:
    st.error("Librería 'streamlit-plotly-events' no instalada.")

import google.generativeai as genai
from google.api_core import exceptions as google_exceptions

def generar_comentario_ia(contexto_texto, api_key, model_name='gemini-2.0-flash', custom_instruction=None):
    """
    Genera un comentario técnico usando Gemini.
    Retorna tupla: (texto_respuesta, info_uso)
    """
    try:
        genai.configure(api_key=api_key)
        # Fix: Strip 'models/' prefix if present to avoid 404 errors
        clean_model_name = model_name.replace("models/", "")
        model = genai.GenerativeModel(clean_model_name)
        
        base_instruction = """
        ERES UN MOTOR DE ANÁLISIS TÉCNICO AUTOMATIZADO. 
        TU SALIDA DEBE SER ESTRICTAMENTE EL REPORTE TÉCNICO.
        PROHIBIDO: Saludos, introducciones, despedidas.
        PROHIBIDO: Repetir el nombre de la Poza o Títulos como "Informe Técnico".
        SOLO DATOS Y RECOMENDACIONES.
        """
        
        user_criteria = custom_instruction if custom_instruction else """
        Actúa como un Ingeniero Geomensor experto en control de calidad.
        Genera un comentario técnico breve (máximo 3 líneas) con recomendaciones operativas.
        """

        prompt = f"""
        {base_instruction}
        
        CRITERIOS DEL USUARIO:
        {user_criteria}
        
        Datos del Análisis:
        {contexto_texto}
        """
        
        response = model.generate_content(prompt)
        
        # Token usage extraction (if available in response object)
        usage_info = "Información de tokens no disponible"
        if hasattr(response, 'usage_metadata'):
            u = response.usage_metadata
            usage_info = f"Tokens: {u.prompt_token_count} (Entrada) + {u.candidates_token_count} (Salida) = {u.total_token_count} Total"
            
        return response.text, usage_info

    except google_exceptions.ResourceExhausted:
        return "⏳ **Límite de Cuota Alcanzado (Error 429):** Por favor espera unos 30 segundos antes de intentar de nuevo.", None
    except Exception as e:
        return f"❌ Error generando comentario ({type(e).__name__}): {str(e)}", None

# ... (Metrics function unchanged) ...

# ... (Main processing loop context) ...



def calculate_metrics_from_points(points):
    """Calcula distancia (2 ptos) o área/perímetro (>2 ptos)."""
    if not points or len(points) < 2: return None
    coords = [(p['x'], p['y']) for p in points]
    results = {}
    
    # 1. Distancia Total (Perímetro)
    perimeter = 0.0
    for i in range(len(coords)-1):
        x1, y1 = coords[i]
        x2, y2 = coords[i+1]
        perimeter += math.sqrt((x2-x1)**2 + (y2-y1)**2)
    results['Longitud'] = perimeter
    
    # 2. Área (Solo si >= 3 puntos)
    if len(coords) >= 3:
        x_pts = [c[0] for c in coords]
        y_pts = [c[1] for c in coords]
        # Shoelace formula (assuming closed loop last->first for area)
        area = 0.5 * abs(sum(x_pts[i]*y_pts[(i+1)%len(coords)] - x_pts[(i+1)%len(coords)]*y_pts[i]
                             for i in range(len(coords))))
        results['Area'] = area
        
    return results

import plotly.graph_objects as go

# ==========================================
# CONFIGURACIÓN PAGINA
# ==========================================
st.set_page_config(page_title="Topo Dashboard V27", layout="wide", page_icon="Logo_TS.ico")

# --- 0. INITIAL SETUP & CONFIG LOAD (FORCE AI INIT) ---
if 'app_ai_settings' not in st.session_state:
    st.session_state['app_ai_settings'] = load_settings()

# Initialize API Key from Secrets if available (Automatic Load)
if 'api_key_to_use' not in st.session_state:
    secret_key = st.secrets.get("gemini_api_key", None)
    if secret_key:
        st.session_state['api_key_to_use'] = secret_key

# ==========================================
# GESTIÓN DE BASES DE DATOS (JSON)
# ==========================================
@st.cache_data
def load_db():
    if not os.path.exists("base_datos_pozas.json"):
        return {}
    with open("base_datos_pozas.json", "r") as f:
        return json.load(f)

def save_db(db_dict):
    """Guarda la base de datos de pozas en JSON."""
    with open("base_datos_pozas.json", "w") as f:
        json.dump(db_dict, f, indent=4)

if 'db_pozas' not in st.session_state:
    st.session_state.db_pozas = load_db()

# Helper for Metrics
def calculate_metrics_from_points(points, col_x='Este', col_y='Norte'):
    """Calcula distancia (2 ptos) o área/perímetro (>2 ptos)."""
    if not points or len(points) < 2: return None
    
    # Extract coordinates robustly
    try:
        coords = [(float(p[col_x]), float(p[col_y])) for p in points]
    except (KeyError, ValueError):
        return None
        
    results = {}
    
    # 1. Distancia Total (Perímetro)
    perimeter = 0.0
    for i in range(len(coords)-1):
        x1, y1 = coords[i]
        x2, y2 = coords[i+1]
        perimeter += math.sqrt((x2-x1)**2 + (y2-y1)**2)
    results['Longitud'] = perimeter
    
    # 2. Área (Solo si >= 3 puntos)
    if len(coords) >= 3:
        x_pts = [c[0] for c in coords]
        y_pts = [c[1] for c in coords]
        # Shoelace formula (assuming closed loop last->first for area)
        area = 0.5 * abs(sum(x_pts[i]*y_pts[(i+1)%len(coords)] - x_pts[(i+1)%len(coords)]*y_pts[i]
                             for i in range(len(coords))))
        results['Area'] = area
        
    return results

# ==========================================
# GESTIÓN DE BASES DE DATOS (JSON)
# ==========================================
DB_FILE = "base_datos_pozas.json"

def load_db():
    if os.path.exists(DB_FILE):
        try:
            with open(DB_FILE, "r") as f: return json.load(f)
        except: return {}
    return {}

def save_db(data):
    with open(DB_FILE, "w") as f: json.dump(data, f)

if 'db_pozas' not in st.session_state:
    st.session_state.db_pozas = load_db()

# ==========================================
# FUNCIONES AUXILIARES UI
# ==========================================
def get_automatic_tolerance(cover):
    if cover >= 44: return cover * 0.50
    elif cover >= 40: return cover * 0.30
    elif cover >= 30: return cover * 0.30
    elif cover >= 20: return cover * 0.10
    return 0.0

def generate_excel_report(global_res_dict, col_map=None):
    """
    global_res_dict structure: ...
    col_map: dict with keys 'Z', 'N', 'E' mapped to actual column names.
    """
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        wb = writer.book
        f_tit = wb.add_format({'bold':True,'size':14,'color':'#003366'})
        f_b = wb.add_format({'border':1})
        f_p = wb.add_format({'num_format':'0.000000','border':1, 'align':'center'}) # Pure Ratio
        f_custom_pct = wb.add_format({'num_format':'0.000000', 'border':1, 'align': 'center'})
        
        # CENTERED HEADERS
        f_hy = wb.add_format({'bold':True,'bg_color':'#FFC000','border':1,'align':'center', 'valign':'vcenter'})
        f_w = wb.add_format({'text_wrap':True,'border':0,'valign':'top'})
        f_center = wb.add_format({'align':'center', 'border':1})
        f_subtit = wb.add_format({'bold':True,'size':12,'color':'#000000', 'underline':True})

        # 1. Resumen Ejecutivo (All Pozas)
        ws1 = wb.add_worksheet('Resumen Ejecutivo')
        # Enhanced Header
        top_header_format = wb.add_format({'bold':True,'size':16,'color':'#003366','align':'left'})
        sub_header_format = wb.add_format({'bold':True,'size':11,'color':'#404040','align':'left'})
        
        ws1.write('B2', "REPORTE CONSOLIDADO DE CALIDAD DE NIVELACIÓN", top_header_format)
        ws1.write('B3', f"Fecha de Generación: {datetime.now().strftime('%d/%m/%Y %H:%M')}", sub_header_format)
        ws1.write('B4', "Generado por: Topo Dashboard V27 (AI Powered)", sub_header_format)
        
        curr_row = 6
        
        for poza_id, poza_data in global_res_dict.items():
            conf = poza_data.get('Config', {})
            ras, cov, tol = conf.get('Rasante',0), conf.get('Cover',0), conf.get('Tol',0)
            src = conf.get('Source', 'N/A')
            
            # Poza Header
            ws1.write(curr_row, 1, f"📍 POZA: {poza_id}", f_tit)
            ws1.write(curr_row+1, 1, f"Rasante: {ras:.3f}m | Cover: {cov:.1f}cm ({src}) | Tol: {tol:.1f}cm")
            curr_row += 3

            for t in ['A', 'B', 'General']:
                if t not in poza_data or poza_data[t]['vacio']: continue
                data = poza_data[t]
                
                ws1.merge_range(curr_row, 1, curr_row, 4, f"TURNO {t}", f_hy)
                curr_row += 1
                
                # --- 1. ANALYSIS TEXT (First) ---
                # Retrieve Analysis Text
                if 'texto_analisis' in data and data['texto_analisis']:
                    texto = data['texto_analisis']
                else:
                    texto = topo_logic.generar_texto_analisis(data['tbl'], data['zonas'], data['atot'], poza_id)
                
                ws1.write(curr_row, 1, "Análisis Técnico:", f_subtit) # Subtitle style
                ws1.merge_range(curr_row+1, 1, curr_row+4, 12, texto, f_w) # Text block
                
                curr_row += 6 # Space after text
                
                # --- 2. TABLE (Left) ---
                table_start_row = curr_row
                ws1.write_row(table_start_row, 1, ['Tipo','Rango','Cant','%'], f_hy) # Header
                for i, r in data['tbl'].iterrows():
                    ws1.write(table_start_row+1+i,1,r['Tipo'],f_b)
                    ws1.write(table_start_row+1+i,2,r['Rango'],f_center)
                    ws1.write(table_start_row+1+i,3,r['Puntos'],f_center)
                    ws1.write(table_start_row+1+i,4,r['Porcentaje'],f_custom_pct)
                
                # --- 3. CHART (Right) ---
                ch = wb.add_chart({'type':'column'})
                points_list = []
                for _, r in data['tbl'].iterrows():
                    points_list.append({'fill': {'color': r['Color'] if 'Color' in r else '#203764'}})
                
                ch.add_series({
                    'name': f'{poza_id} - Turno {t}',
                    'categories': ['Resumen Ejecutivo', table_start_row+1, 2, table_start_row+1+len(data['tbl'])-1, 2],
                    'values':     ['Resumen Ejecutivo', table_start_row+1, 3, table_start_row+1+len(data['tbl'])-1, 3],
                    'points': points_list,
                    'data_labels': {'value': True, 'position': 'outside_end'}
                })
                ch.set_title({'name': f"Distribución Turno {t}"})
                ch.set_y_axis({'name': 'Cantidad'})
                ch.set_x_axis({'name': 'Rango'})
                ch.set_legend({'position': 'none'}) 
                
                # Insert Chart to the right of table (approx Column F/G)
                ws1.insert_chart(f'G{table_start_row}', ch, {'x_scale':1.5, 'y_scale':1.0})
                
                curr_row += max(16, len(data['tbl']) + 2) + 2 # Move down past chart/table
            
            curr_row += 2 # Spacer between pozas

        # 2. Zonas Defectuosas
        ws2 = wb.add_worksheet('Zonas_Defectuosas')
        curr = 0
        for poza_id, poza_data in global_res_dict.items():
            ws2.write(curr, 0, f"POZA: {poza_id}", f_tit)
            curr += 2
            
            for t in ['A', 'B', 'General']:
                if t not in poza_data or poza_data[t]['vacio']: continue
                data = poza_data[t]
                
                # Analysis Text (CRITICAL ONLY)
                if 'texto_analisis_critico' in data and data['texto_analisis_critico']:
                     texto = data['texto_analisis_critico']
                elif 'texto_analisis' in data and data['texto_analisis']:
                     texto = data['texto_analisis']
                else:
                     texto = "Sin análisis crítico disponible."
                
                ws2.merge_range(curr, 0, curr+6, 11, texto, f_w)
                
                # KPI Calculation
                area_mala = data['zonas']['Area_Efectiva_m2'].sum() if not data['zonas'].empty else 0
                kpi_val = (area_mala / data['atot']) * 100 if data['atot'] > 0 else 0
                ws2.write(curr+7, 0, f"KPI Incidencia (Area Defectuosa / Total): {kpi_val:.2f}%", f_tit)

                # INSERT MAPS (Heatmap Only)
                import matplotlib.pyplot as plt
                
                # Heatmap (Bottom) - "Abajo"
                row_heat = curr + 0 # Now directly below tables
                try:
                    ws2.write(row_heat, 14, "Mapa de Calor", f_tit)
                    fig_map = topo_logic.generar_mapa_matplotlib(
                         data['df'], data['zonas'], 
                         col_n=col_map['N'], col_e=col_map['E'],
                         titulo=f"Mapa Calor {poza_id}-{t}",
                         tol=poza_data['Config']['Tol']
                    )
                    if fig_map:
                        img_heat = io.BytesIO()
                        fig_map.savefig(img_heat, format='png', bbox_inches='tight', dpi=100)
                        img_heat.seek(0)
                        ws2.insert_image(row_heat+1, 14, 'map_heat.png', {'image_data': img_heat, 'x_scale': 0.6, 'y_scale': 0.6})
                        plt.close(fig_map)
                        map_inserted = True
                except Exception as e:
                    ws2.write(row_heat+1, 14, f"Error Heatmap: {e}")
                
                curr += 9

                
                # Table of Zones
                if not data['zonas'].empty:
                    # Merge Title across all columns
                    cols_count = len(data['zonas'].columns)
                    ws2.merge_range(curr, 0, curr, cols_count - 1, f"Detalle Zonas Turno {t}", f_hy)
                    
                    # Headers
                    ws2.write_row(curr+1, 0, data['zonas'].columns, f_hy)
                    for i, row in data['zonas'].iterrows():
                         ws2.write_row(curr+2+i, 0, row.values, f_center)
                         # Explicit format for Incidencia column (last one?)
                         # Assuming 'Incidencia (%)' is last
                         if 'KPI Incidencia' in data['zonas'].columns:
                             idx_inc = data['zonas'].columns.get_loc('KPI Incidencia')
                             val = row['KPI Incidencia']
                             ws2.write(curr+2+i, idx_inc, val, f_custom_pct)

                    # Move cursor down based on table size OR map size
                    rows_table = len(data['zonas']) + 4
                    rows_map = 20 # approx for map height
                    curr += rows_table + 2 
                else:
                    ws2.write(curr, 0, f"Turno {t}: No se detectaron zonas críticas.")
                    curr += 5

            curr += 2

        # 3. Datos Mapas (Optimized)
        export_list = []
        for poza_id, poza_data in global_res_dict.items():
            for t in ['A', 'B', 'General']:
                if t not in poza_data or poza_data[t]['vacio']: continue
                df_pts = poza_data[t]['df'].copy()
                if df_pts.empty: continue
                
                df_pts['Poza'] = poza_id
                df_pts['Turno_Rep'] = t
                
                # Map coordinates
                if col_map:
                    if col_map.get('N') in df_pts.columns: df_pts['Norte'] = df_pts[col_map['N']]
                    if col_map.get('E') in df_pts.columns: df_pts['Este'] = df_pts[col_map['E']]
                    if col_map.get('Z') in df_pts.columns: df_pts['Elev'] = df_pts[col_map['Z']]
                        
                for std, actual in [('Norte', 'CellN_m'), ('Este', 'CellE_m'), ('Elev', 'Elevation_m')]:
                    if std not in df_pts.columns and actual in df_pts.columns:
                        df_pts[std] = df_pts[actual]
                
                target_cols = ['Poza', 'Turno_Rep', 'Norte', 'Este', 'Elev', 'desviacion', 'Rango', 'Tipo']
                for c in target_cols:
                    if c not in df_pts.columns: df_pts[c] = np.nan
                export_list.append(df_pts[target_cols])

        if export_list:
            df_full_export = pd.concat(export_list, ignore_index=True)
            df_full_export.to_excel(writer, sheet_name='Datos_Mapas', index=False)
        else:
            wb.add_worksheet('Datos_Mapas').write('A1', "No hay datos para exportar.")

    return output.getvalue()

# ==========================================
# MAIN APP INIT
# ==========================================
# Title with Logo
# Title with Logo
col_h1, col_h2 = st.columns([0.5, 4])
with col_h1:
    if os.path.exists("Logo_TS.ico"):
        st.image("Logo_TS.ico", width=60)
with col_h2:
    st.title("Dashboard Topo V27")

# Global Status Container for top-level alerts
status_container = st.container()

# Initial Vars
df = None
unique_pozas_all = []
unique_maqs_all = []
date_options_all = []
turn_options_all = []
c_design, c_maq, c_time, cz, cn, ce = None, None, None, None, None, None
cols_needed_map = {}

# 1. SIDEBAR CONFIG (BD & INPUTS)
# ==========================================
# 1. SIDEBAR STRUCTURE (Top to Bottom)
# ==========================================
# ==========================================
# 1. SIDEBAR STRUCTURE (Top to Bottom)
# ==========================================
with st.sidebar:
    # A. BRANDING (Top)
    if os.path.exists("logo.png"):
        st.image("logo.png", width=300)
    
    st.divider()
    
    # B. DATOS DE ENTRADA (Top Priority)
    st.subheader("Datos de Entrada")
    uploaded_files = st.file_uploader("Cargar Archivos (CSV/Excel)", type=["csv", "xlsx"], accept_multiple_files=True, key="main_file_uploader")
    st.divider()

    # C. PREPARACIÓN (Placeholders removed here, rendered below)


# ==========================================
# 2. LOAD & PREPARE DATA (Logic Only)
# ==========================================
df = None
if uploaded_files:
    try:
        df_list = []
        for u_file in uploaded_files:
            if u_file.name.endswith('.csv'):
                temp_df = pd.read_csv(u_file, low_memory=False)
            else:
                temp_df = pd.read_excel(u_file)
            temp_df.columns = temp_df.columns.str.strip()
            df_list.append(temp_df)
        
        if df_list:
            df = pd.concat(df_list, ignore_index=True)

            # Detect columns (Logic applies to consolidated DF)
            cols_needed = {
                'Z': ['Elevation_m','Z','Cota'],
                'N': ['CellN_m','Norte','Y'],
                'E': ['CellE_m','Este','X']
            }
            found_cols = {}
            for k, candidates in cols_needed.items():
                found = next((c for c in df.columns if c in candidates), None)
                found_cols[k] = found
            cz, cn, ce = found_cols['Z'], found_cols['N'], found_cols['E']
            
            c_design = next((c for c in df.columns if c in ['DesignName','Design','Diseño']), None)
            c_maq = next((c for c in df.columns if c in ['MachineName','Machine','Maquina','Máquina','Equipo','Excavadora']), None)
            c_time = next((c for c in df.columns if c in ['Time','Fecha','Hora', 'Date']), None)

            # Pre-process
            if c_design:
                df['PozaID'] = df[c_design].apply(lambda x: str(x).split('_')[0].strip().upper())
                unique_pozas_all = sorted(df['PozaID'].unique())
            else:
                df['PozaID'] = 'GENERAL'
                unique_pozas_all = ['GENERAL']

            if c_time:
                # Robust parsing for Trimble format (e.g., 2026/Jan/01)
                # Map English months to numbers to avoid locale issues
                month_map = {
                    'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04', 'May': '05', 'Jun': '06',
                    'Jul': '07', 'Aug': '08', 'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'
                }
                
                # Create a temporary series for parsing
                date_series = df[c_time].astype(str)
                for eng, num in month_map.items():
                    date_series = date_series.str.replace(eng, num, regex=False)
                
                # Normalize slashes just in case and parse
                # Values become 2026/01/01 ...
                df['DT'] = pd.to_datetime(date_series, errors='coerce')
                
                df['Fecha'] = df['DT'].dt.date
                df['Turno'] = df['DT'].apply(lambda x: 'A' if pd.notnull(x) and 7 <= x.hour < 19 else ('B' if pd.notnull(x) else 'Desconocido'))
            else:
                df['Turno'] = 'General'
                df['Fecha'] = 'General'
            
    except Exception as e:
        st.error(f"Error cargando archivos: {e}")
        traceback.print_exc()
        df = None


# ==========================================
# 3. SIDEBAR CONFIG & FILTERS (PERSISTENT UI)
# ==========================================
apply_filters = False 
import re

# Always render Sidebar elements
with st.sidebar:
    # --- CONFIGURATION EXPANDER (Menu 2 - TOP) ---
    with st.expander("Configuración", expanded=False):
        # 1. DB Management
        st.subheader("1. Base de Datos (Pozas)")
        db_upload = st.file_uploader("Subir archivo BD (.xlsx)", type=["xlsx"], key="db_loader_main")
        
        if db_upload:
            try:
                df_new = pd.read_excel(db_upload)
                col_poza = next((c for c in df_new.columns if "poza" in c.lower()), None)
                col_cover = next((c for c in df_new.columns if "cover" in c.lower()), None)
                if col_poza and col_cover:
                    if st.button("📥 Importar/Fusionar Datos", key="btn_imp_db"):
                        for _, row in df_new.iterrows():
                            p_val = str(row[col_poza]).strip().upper()
                            c_val = pd.to_numeric(row[col_cover], errors='coerce')
                            if pd.notna(c_val) and p_val != "NAN":
                                st.session_state.db_pozas[p_val] = float(c_val)
                        save_db(st.session_state.db_pozas)
                        st.success("Registros importados.")
                        st.rerun()
                else:
                    st.error("Cols 'Poza'/'Cover' no encontradas.")
            except Exception as e:
                st.error(str(e))
        
        # Line removed
        


        # Editor de Base de Datos
        if st.checkbox("Gestionar Base de Datos Manualmente", key="chk_db_edit"):
            st.info("Puede agregar, editar o eliminar filas. Pulse 'Guardar' al finalizar.")
            
            # Convert Dict to DataFrame for Editor
            current_data = [{"Poza": k, "Cover": v} for k, v in st.session_state.db_pozas.items()]
            df_db_edit = pd.DataFrame(current_data)
            
            edited_df = st.data_editor(
                df_db_edit,
                num_rows="dynamic",
                column_config={
                    "Poza": st.column_config.TextColumn("Poza ID", required=True),
                    "Cover": st.column_config.NumberColumn("Cover (cm)", required=True, min_value=0.0)
                },
                use_container_width=True,
                key="db_editor_widget"
            )
            
            if st.button("💾 Guardar Cambios en BD", key="btn_save_db"):
                # Reconstruct Dictionary
                new_db = {}
                for idx, row in edited_df.iterrows():
                    p_id = str(row['Poza']).strip().upper()
                    if p_id and p_id != "NAN" and p_id != "NONE":
                        new_db[p_id] = float(row['Cover'])
                
                st.session_state.db_pozas = new_db
                save_db(new_db)
                st.success(f"Base de datos actualizada: {len(new_db)} registros.")
                st.rerun()

        st.divider()

        active_pozas = unique_pozas_all if df is not None else []
        tol_step_val = 4.0
        
        # STATEFUL COVER EDITOR logic
        # Check if we need to rebuild the state (file changed or first run)
        current_pozas_set = set(active_pozas)
        
        # Helper to compare sets safely dealing with potentially unhashable types if any (though strings are safe)
        cached_pozas = st.session_state.get('last_pozas_set', set())
        
        cover_rows = []
        if 'df_covers_state' not in st.session_state or cached_pozas != current_pozas_set:
            # Rebuild state from DB and persistence


            for pid in active_pozas:
                if pid == "GENERAL": continue
                c_db = st.session_state.db_pozas.get(pid, 0.0)
                
                # Default manual is 0.0
                c_man = 0.0
                
                # Try to preserve manual value from previous state if available for this PID
                if 'df_covers_state' in st.session_state and not st.session_state.df_covers_state.empty:
                     old_df = st.session_state.df_covers_state
                     if 'PozaID' in old_df.columns:
                         match = old_df[old_df['PozaID'] == pid]
                         if not match.empty:
                             try:
                                c_man = float(match.iloc[0]['Cover Manual'])
                             except: pass
                
                cover_rows.append({
                    "PozaID": pid,
                    "Cover BD": float(c_db),
                    "Cover Manual": float(c_man)
                })
            st.session_state.df_covers_state = pd.DataFrame(cover_rows)
            st.session_state.last_pozas_set = current_pozas_set

        # Render Editor using Session State DF
        if 'df_covers_state' in st.session_state and not st.session_state.df_covers_state.empty:
            edited_covers = st.data_editor(
                st.session_state.df_covers_state,
                column_config={
                    "PozaID": st.column_config.TextColumn("Poza", disabled=True),
                    "Cover BD": st.column_config.NumberColumn("BD (cm)", disabled=True, format="%.1f"),
                    "Cover Manual": st.column_config.NumberColumn("Manual (cm)", required=True, min_value=0.0, format="%.1f")
                },
                disabled=["PozaID", "Cover BD"],
                hide_index=True,
                key="cover_editor_main",
                use_container_width=True
            )
            # Update state with edits immediately
            st.session_state.df_covers_state = edited_covers
            
            # --- PROACTIVE WARNING IN MAIN AREA ---
            # Check for missing covers to warn user immediately under Title
            missing_pozas_warn = []
            for _, row in edited_covers.iterrows():
                c_d = float(row['Cover BD'])
                c_m = float(row['Cover Manual'])
                if c_d <= 0 and c_m <= 0:
                    missing_pozas_warn.append(str(row['PozaID']))
            
            if missing_pozas_warn:
                status_container.error(
                    f"⚠️ ALERTA: No se encontraron los covers para las pozas {missing_pozas_warn} en la Base de Datos. "
                    "Debe ingresarse el valor en la columna 'Manual' de la tabla de Configuración (Barra Lateral)."
                )

        else:
            edited_covers = pd.DataFrame(columns=["PozaID", "Cover BD", "Cover Manual"])
            if df is None: st.info("⚠️ Cargue archivo para configurar.")

        # --- 3. RASANTES POR POZA ---
        st.subheader("3. Rasantes por Poza")
        
        # Calculate Config Rows based on EDITED Covers
        config_rows = []
        if not edited_covers.empty:
            for _, row in edited_covers.iterrows():
                pid = row['PozaID']
                c_db = row['Cover BD']
                c_man = row['Cover Manual']
                
                # Logic: Priority DB > Manual
                cov_eff = c_db if c_db > 0 else c_man
                source = "BD" if c_db > 0 else ("Manual" if c_man > 0 else None)
                
                ras_auto = 0.0
                if cov_eff > 0:
                    ras_auto = 2300.0 + (cov_eff/100.0)
                else:
                    # Fallback logic if needed, but primary is now cover
                    # Try to parse from file data if available? 
                    # Re-using legacy logic might be complex here as iterate over covers df, not full df.
                    # We can lookup in df if strictly needed, but let's rely on cover.
                    pass
                
                status_emoji = "✅" if ras_auto > 0 else "⚠️"
                status_text = f"{status_emoji} {source}" if source else f"{status_emoji} Falta Cover"
                config_rows.append({"PozaID": pid, "Rasante": ras_auto, "Info": status_text})

        if config_rows:
            df_config = pd.DataFrame(config_rows).set_index("PozaID")
            edited_config = st.data_editor(
                df_config,
                column_config={
                    "Rasante": st.column_config.NumberColumn("Rasante (m)", format="%.3f", required=True),
                    "Info": st.column_config.TextColumn("Estado", disabled=True),
                },
                disabled=["PozaID", "Info"],
                key="ras_editor_main",
                use_container_width=True
            )
        else:
            edited_config = pd.DataFrame(columns=["Rasante"])
            if edited_covers.empty and df is not None: st.info("No hay pozas activas.")

        # --- 4. CRITERIO DE EVALUACIÓN (GLOBAL) ---
        st.divider()
        st.subheader("4. Criterio Técnicos")
        c_options = ["Criterio SQM", "Criterio Excon"]
        # Use session state to persist choice if needed, but simple selectbox works for now if top-down
        criterio_eval = st.selectbox("Criterio de Evaluación", c_options, index=1, key="criterio_selector_sidebar") 

        # Legend Table
        if criterio_eval == "Criterio Excon":
            st.markdown("""
            **Leyenda Criterio Excon:**
            | Color | Clasificación | Rango (cm) |
            | :---: | :--- | :--- |
            | <span style='color:#FF0000; font-size:1.5em;'>●</span> | **Crítico Bajo** | $\le$ Tol (-15 o -10) |
            | <span style='color:#FFC000; font-size:1.5em;'>●</span> | **Bajo Tolerable** | > Tol y $\le$ -4 |
            | <span style='color:#00B050; font-size:1.5em;'>●</span> | **Conforme** | > -4 y $\le$ 4 |
            | <span style='color:#00B0F0; font-size:1.5em;'>●</span> | **Sobrelevación Leve** | > 4 y $\le$ 10 |
            | <span style='color:#002060; font-size:1.5em;'>●</span> | **Sobrelevación Crítica** | > 10 |
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            **Leyenda Criterio SQM:**
            | Color | Clasificación | Rango |
            | :---: | :--- | :--- |
            | <span style='color:#FF0000; font-size:1.5em;'>●</span> | **Corte Crítico** | > 3x Tol |
            | <span style='color:#FF0000; font-size:1.5em;'>●</span> | **Corte Alto** | 2x a 3x Tol |
            | <span style='color:#FF8C00; font-size:1.5em;'>●</span> | **Corte Alerta** | 1x a 2x Tol |
            | <span style='color:#00B050; font-size:1.5em;'>●</span> | **OK (Corte)** | 0 a 1x Tol |
            | <span style='color:#00B050; font-size:1.5em;'>●</span> | **OK (Relleno)** | -1x Tol a 0 |
            | <span style='color:#FFC000; font-size:1.5em;'>●</span> | **Relleno Alerta** | -2x a -1x Tol |
            | <span style='color:#FFC000; font-size:1.5em;'>●</span> | **Relleno Bajo** | -3x a -2x Tol |
            | <span style='color:#FF0000; font-size:1.5em;'>●</span> | **Relleno Crítico** | < -3x Tol |
            """, unsafe_allow_html=True)

 


    # 4. Filtros de Visualización
    with st.expander("Filtros de Visualización", expanded=True):
        if df is not None:
            # 1. Poza
            poza_opts = ["TODOS"] + (unique_pozas_all if unique_pozas_all else [])
            sel_poza = st.selectbox("1. Poza:", poza_opts, key="f_poza")
            
            # Filter Step 1
            # Optimized Cascading Filter (Avoids Deep Copies)
            # 1. Poza
            mask_poza = pd.Series(True, index=df.index)
            if sel_poza != "TODOS":
                mask_poza = df['PozaID'] == sel_poza
            
            # 2. Machine
            # Get valid options relative to Current Mask
            df_m1 = df.loc[mask_poza]
            maqs_avail = sorted(df_m1[c_maq].dropna().unique()) if (c_maq and not df_m1.empty) else []
            sel_maq = st.selectbox("2. Máquina:", ["TODOS"] + maqs_avail, key="f_maq")

            mask_maq = pd.Series(True, index=df.index)
            if sel_maq != "TODOS" and c_maq:
                mask_maq = df[c_maq] == sel_maq

            # 3. Date
            mask_current = mask_poza & mask_maq
            df_m2 = df.loc[mask_current]
            
            fechas_avail = sorted(df_m2['Fecha'].dropna().unique()) if (c_time and not df_m2.empty) else []
            dates_opts = ["TODOS"] + [str(d) for d in fechas_avail]
            sel_date = st.selectbox("3. Fecha:", dates_opts, key="f_date")
            
            mask_date = pd.Series(True, index=df.index)
            if sel_date != "TODOS" and c_time:
                target_date = datetime.strptime(sel_date, '%Y-%m-%d').date()
                mask_date = df['Fecha'] == target_date

            # 4. Turn
            mask_current = mask_poza & mask_maq & mask_date
            df_m3 = df.loc[mask_current]
            
            turnos_avail = sorted(df_m3['Turno'].dropna().unique()) if ('Turno' in df_m3 and not df_m3.empty) else []
            turns_opts = ["TODOS"] + turnos_avail
            sel_turn = st.selectbox("4. Turno:", turns_opts, key="f_turn")
        else:
            # Empty User Interface when no file is loaded
            st.selectbox("1. Poza:", ["(Cargar Archivo)"], disabled=True, key="f_poza_dummy")
            st.selectbox("2. Máquina:", ["(Cargar Archivo)"], disabled=True, key="f_maq_dummy")
            st.selectbox("3. Fecha:", ["(Cargar Archivo)"], disabled=True, key="f_date_dummy")
            st.selectbox("4. Turno:", ["(Cargar Archivo)"], disabled=True, key="f_turn_dummy")
            # Initialize dummy vars for logic below
            sel_poza, sel_maq, sel_date, sel_turn = "TODOS", "TODOS", "TODOS", "TODOS"
            df_f3 = pd.DataFrame() # Empty DF

    # --- ACTION BUTTON ---
    # Gestionar Estado de Procesamiento
    if 'processing_active' not in st.session_state:
        st.session_state['processing_active'] = False
        
    # Detectar cambios en archivos para resetear
    current_file_names = [f.name for f in uploaded_files] if uploaded_files else []
    if 'last_uploaded_files' not in st.session_state:
        st.session_state['last_uploaded_files'] = current_file_names
    
    if st.session_state['last_uploaded_files'] != current_file_names:
        st.session_state['processing_active'] = False
        st.session_state['last_uploaded_files'] = current_file_names

    if st.button("PROCESAR RESULTADOS", type="primary", use_container_width=True, key="btn_process"):
        st.session_state['processing_active'] = True
        # Clear AI Cache on explicit new process
        keys_to_clear = [k for k in st.session_state.keys() if k.startswith("ai_res_")]
        for k in keys_to_clear:
            del st.session_state[k]
        st.rerun()
        
    st.divider()
    if st.button("GESTIONAR PUNTOS BAJOS (HISTORIAL)", type="secondary", use_container_width=True):
        show_history_modal()

    # --- ADMIN PANEL (BOTTOM) ---
    st.divider()
    st.caption("Configuración Avanzada v4.1")
    with st.expander("🔐 Panel de Administrador (IA)", expanded=False):
        # Load current settings from file
        current_settings = load_settings()
        
        # Simple Session State Login
        if 'admin_logged_in' not in st.session_state:
            st.session_state['admin_logged_in'] = False
        
        if not st.session_state['admin_logged_in']:
            pwd = st.text_input("Contraseña de Admin:", type="password", key="admin_pwd_input")
            if st.button("Ingresar Panel"):
                if pwd == current_settings.get("admin_password", "excon"):
                    st.session_state['admin_logged_in'] = True
                    st.rerun()
                else:
                    st.error("Contraseña incorrecta")
        else:
            st.success("🔓 Acceso Concedido")
            if st.button("Cerrar Sesión"):
                st.session_state['admin_logged_in'] = False
                st.rerun()
            
            st.subheader("Configuración General")
            
            # 1. AI Toggle
            ai_enabled = st.toggle("Activar Generación IA", value=current_settings.get("ai_enabled", True))
            
            # 2. API Key Management
            st.markdown("**Gestión API Key:**")
            api_key_system = st.secrets.get("gemini_api_key", None)
            if api_key_system:
                    st.info(f"🔑 Clave cargada desde archivo (secrets.toml).")
                    st.session_state['api_key_to_use'] = api_key_system
            else:
                    st.warning("No hay clave en secrets.toml")
                    user_key = st.text_input("Ingresar Key Manualmente:", type="password", value=st.session_state.get('api_key_to_use', ''))
                    if user_key:
                        st.session_state['api_key_to_use'] = user_key
            
            # 3. Model Selector (Dynamic)
            st.markdown("**Modelo de IA:**")
            if st.session_state.get('api_key_to_use'):
                try:
                    genai.configure(api_key=st.session_state['api_key_to_use'])
                    if 'gemini_models_list' not in st.session_state:
                            models_iter = genai.list_models()
                            st.session_state['gemini_models_list'] = [
                                m.name.replace("models/", "") 
                                for m in models_iter 
                                if 'generateContent' in m.supported_generation_methods
                            ]
                    
                    default_ix = 0
                    saved_model = st.session_state.get('selected_ai_model', 'gemini-2.0-flash')
                    if saved_model in st.session_state['gemini_models_list']:
                            default_ix = st.session_state['gemini_models_list'].index(saved_model)
                    
                    selected_model = st.selectbox("Seleccionar Modelo:", st.session_state['gemini_models_list'], index=default_ix)
                    st.session_state['selected_ai_model'] = selected_model
                except:
                    st.error("Error cargando modelos (Revisar Key)")
            
            # 4. Custom Prompt Editor
            st.subheader("🧠 Cerebro de la IA (Prompt)")
            new_prompt = st.text_area(
                "Instrucciones para la IA (Definir rol y enfoque):", 
                value=current_settings.get("system_prompt", ""),
                height=200
            )
            
            # Save Button
            if st.button("💾 Guardar Configuración"):
                new_settings = {
                    "ai_enabled": ai_enabled,
                    "system_prompt": new_prompt,
                    "admin_password": current_settings.get("admin_password", "excon") # Keep same pwd for now
                }
                if save_settings(new_settings):
                    st.success("Configuración guardada exitosamente.")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("Error guardando configuración.")

    # Load clean settings for usage in app (Read-only if not logged in)
    app_settings = load_settings()
    st.session_state['app_ai_settings'] = app_settings

# ==========================================
# PROCESSING & RESULTS
# ==========================================
apply_filters = st.session_state['processing_active']
# ==========================================
# PROCESSING & RESULTS
# ==========================================
if apply_filters and df is not None:
    
    # 1. Apply Filters
    # 1. Apply Filters (Optimized)
    # Validate masks exist (Safety Check)
    if 'mask_date' not in locals():
        # This can happen if sidebar execution failed or scope issue. Fallback to all true.
        mask_final = pd.Series(True, index=df.index)
    else:
        mask_final = mask_date 

    if sel_turn != "TODOS":
        mask_final = mask_final & (df['Turno'] == sel_turn)
    
    df_final = df.loc[mask_final].copy() # Single Copy for processing
        
    if df_final.empty:
        st.warning("⚠️ No hay datos visibles con los filtros actuales.")
        st.stop()
        
    # --- VALIDATION STEP: CHECK CONFIGURATION BEFORE CALCULATION ---
    # Normalize PozaID first
    df_final['PozaID'] = df_final['PozaID'].astype(str).str.strip().str.upper()
    active_pozas_process = df_final['PozaID'].unique()
    missing_config_pozas = []
    
    # Build Cover Map from Sidebar Editor
    cover_val_map = {}
    cover_src_map = {}
    
    # Ensure edited_covers exists (it should from sidebar)
    if 'edited_covers' in locals() and not edited_covers.empty:
        for _, row in edited_covers.iterrows():
            p_key = str(row['PozaID']).strip().upper()
            c_db = float(row['Cover BD'])
            c_man = float(row['Cover Manual'])
            
            c_eff = c_db if c_db > 0 else c_man
            src = "BD" if c_db > 0 else ("Manual" if c_man > 0 else "Faltante")
            
            cover_val_map[p_key] = c_eff
            cover_src_map[p_key] = src

    # Check for missing covers
    for pid in active_pozas_process:
        if pid == "GENERAL": continue
        c_check = cover_val_map.get(pid, 0.0)
        if c_check <= 0:
            missing_config_pozas.append(pid)
            
    if missing_config_pozas:
        status_container.error(f"⛔ Faltan datos de Espesor (Cover) para las pozas: {', '.join(missing_config_pozas)}")
        status_container.info("💡 Por favor, diríjase a la sección 'Configuración' en la barra lateral e ingrese manualmente los valores en la columna 'Manual' para continuar.")
        st.stop()
        
    # 2. Vectorized Deviation Calculation
    rasante_map = {str(k).strip(): v for k, v in edited_config['Rasante'].to_dict().items()}
    df_final['Rasante_Teorica'] = df_final['PozaID'].map(rasante_map).fillna(0.0)
    df_final['Cota_Calc'] = df_final[cz] 
    df_final['desviacion'] = (df_final['Cota_Calc'] - df_final['Rasante_Teorica']) * 100.0
    
    # 3. GLOBAL CALCULATION PHASE
    # Criterio taken from Sidebar (criterio_eval)

    global_results = {}
    groups = df_final.groupby('PozaID')
    
    # Prepare result container for all pozas first
    ai_enabled = st.session_state.get('app_ai_settings', {}).get("ai_enabled", True)
    active_key = st.session_state.get('api_key_to_use')
    
    # --- HISTORY: INIT BUFFER ---
    hist_df_current = db_manager.load_history()
    new_findings_buffer = []
    
    # FORCE 2.0 FLASH if session has old model
    curr_model = st.session_state.get('selected_ai_model', 'gemini-2.0-flash')
    if "1.5-flash" in curr_model: 
        curr_model = "gemini-2.0-flash"
        st.session_state['selected_ai_model'] = curr_model
        
    model_to_use = curr_model
    custom_prompt = st.session_state.get('app_ai_settings', {}).get("system_prompt", None)

    # Initialize progress bar for AI processing
    ai_progress_bar = None
    ai_task_count = 0 
    
    # WRAP WHOLE PROCESSING IN SPINNER
    with st.spinner("🤖 Generando Análisis Técnico con Inteligencia Artificial para todas las pozas... (Esto puede tardar unos segundos)"):
        for pid, df_grp in groups:
            poza_res = {}
            try:
                # Config used
                cov_used = cover_val_map.get(pid, 0.0)
                source_cov = cover_src_map.get(pid, "Desconocido")
            
                # --- NEW TIERED TOLERANCE CALCULATION ---
                if criterio_eval == "Criterio Excon":
                    tol_calculated = 15.0 if cov_used > 50 else 10.0
                else:
                    if cov_used > 45: tol_calculated = cov_used * 0.50
                    elif cov_used >= 30: tol_calculated = cov_used * 0.30
                    elif cov_used >= 20: tol_calculated = cov_used * 0.10
                    elif cov_used > 0: tol_calculated = 0.0 
                    else: tol_calculated = 4.0

                if pid in edited_config.index:
                    ras_used = edited_config.loc[pid, 'Rasante']
                else:
                    ras_used = 0.0
                
                poza_res = {'Config': {'Rasante': ras_used, 'Tol': tol_calculated, 'Cover': cov_used, 'Source': source_cov}}
                turns_present = df_grp['Turno'].unique()
                
                # INNER LOOP (MOVED INSIDE TRY BLOCK)
                for t in turns_present:
                    df_t = df_grp[df_grp['Turno'] == t]
                    
                    try:
                        # Logic calculation
                        res_tbl, df_proc, zonas, atot, df_visual = topo_logic.procesar_turno(
                            df_t, ras_used, 
                            tolerancia=tol_calculated, 
                            col_z=cz, # Passing Elevation Column
                            col_n=cn, col_e=ce,
                            step=tol_step_val, 
                            cover_cm=cov_used,
                            criterio=criterio_eval
                        )

                        # --- CAPTURE NEW FINDINGS (FEATURE V6) ---
                        if not zonas.empty:
                            rep_date = df_t['Fecha'].iloc[0] if 'Fecha' in df_t.columns and not df_t.empty else datetime.now().date()
                            for _, z_row in zonas.iterrows():
                                new_rec = {
                                    'Poza': pid,
                                    'Turno': t,
                                    'Fecha_Reporte': str(rep_date),
                                    'Norte': z_row.get('Norte', 0),
                                    'Este': z_row.get('Este', 0),
                                    'Cota_Teorica': ras_used,
                                    'Cota_GPS': z_row.get('Elev_Min', 0),
                                    'Desv_GPS': z_row.get('Desv_Min (cm)', 0)
                                }
                                new_findings_buffer.append(new_rec)
                        
                        # STANDARDIZE COLUMNS FOR DOWNSTREAM (PPTX/Excel/Maps)
                        # This ensures 'Norte' and 'Este' exist for map generation
                        if cn in df_proc.columns: df_proc['Norte'] = df_proc[cn]
                        if ce in df_proc.columns: df_proc['Este'] = df_proc[ce]
                        if cz in df_proc.columns: df_proc['Elev'] = df_proc[cz]
                        
                        # Prepare Context (Classic Text) needed for AI or as Standalone
                        txt_context_classic = topo_logic.generar_texto_analisis(res_tbl, zonas, atot, f"{pid}_{t}")
                        
                        # --- DECISION LOGIC: AI vs CLASSIC ---
                        if ai_enabled and active_key:
                            ai_task_count += 1
                            cache_key = f"ai_res_{pid}_{t}"
                            cached_text = st.session_state.get(cache_key)
                            
                            if cached_text:
                                 # Split Cache if possible
                                 if "|||" in cached_text:
                                     parts = cached_text.split("|||")
                                     txt_analysis = parts[0].strip()
                                     txt_analysis_crit = parts[1].strip()
                                 else:
                                     txt_analysis = cached_text
                                     txt_analysis_crit = "Análisis Crítico no disponible en caché (Regenerar)."
                            else:
                                txt_analysis = "🔄 Generando Análisis Técnico con IA... (Procesando)"
                                try:
                                    # 1. General Analysis
                                    txt_ai_gen, _ = generar_comentario_ia(
                                        txt_context_classic, 
                                        active_key, 
                                        model_name=model_to_use, 
                                        custom_instruction=custom_prompt
                                    )
                                    
                                    # 2. Critical Analysis (New)
                                    txt_ai_crit = "No hay zonas críticas para analizar."
                                    if not zonas.empty:
                                         context_crit = f"Tabla de Zonas Críticas (Puntos Bajos):\n{zonas.to_string()}\nTotal Puntos Bajos: {len(zonas)}\nÁrea Total Defectos: {atot:.1f} m2"
                                         prompt_crit = "Analiza BREVEMENTE estas zonas críticas (puntos bajos). Menciona cuántas son, el área afectada y la gravedad. Foco en corrección."
                                         txt_ai_crit, _ = generar_comentario_ia(
                                            context_crit, 
                                            active_key, 
                                            model_name=model_to_use, 
                                            custom_instruction=prompt_crit
                                         )

                                    if "Error" not in txt_ai_gen:
                                        txt_analysis = txt_ai_gen
                                        txt_analysis_crit = txt_ai_crit
                                        combined_cache = f"{txt_analysis} ||| {txt_analysis_crit}"
                                        st.session_state[cache_key] = combined_cache
                                    else:
                                        txt_analysis = f"❌ Error GenAI: {txt_ai_gen}"
                                        txt_analysis_crit = "Error."
                                except Exception as e:
                                     st.warning(f"Error GenAI Poza {pid}: {e}")
                                     txt_analysis = f"❌ Error inesperado: {e}"
                                     txt_analysis_crit = f"Error: {e}"
                        else:
                            txt_analysis = txt_context_classic
                            txt_analysis_crit = "Análisis IA Desactivado."
                        
                        poza_res[t] = {
                            'tbl': res_tbl, 'df': df_proc, 'df_visual': df_visual, 'zonas': zonas, 'atot': atot, 'vacio': False,
                            'texto_analisis': txt_analysis,
                            'texto_analisis_critico': txt_analysis_crit,
                            'metrics': calculate_metrics_from_points(zonas.to_dict('records') if not zonas.empty else [], col_x=ce, col_y=cn)
                        }

                    except Exception as e:
                        import traceback
                        print(f"Error CRITICO procesando Poza {pid} Turno {t}: {e}")
                        traceback.print_exc()
                        st.error(f"Error procesando {pid} Turno {t}: {e}")
                        
                        poza_res[t] = {
                            'vacio': True, 
                            'texto_analisis': f"Error de Cálculo: {e}",
                            'tbl': pd.DataFrame({'Rango':[], 'Puntos':[]}),
                            'cols': [],
                            'df': df_t,
                            'zonas': pd.DataFrame(),
                            'zonas': pd.DataFrame(),
                            'zonas': pd.DataFrame(),
                            'atot': 0,
                            'df_visual': df_t # Fallback to input
                        }

            except Exception as eOuter:
                 st.error(f"Error procesando configuración Poza {pid}: {eOuter}")
                 poza_res = {'Config': {'Rasante': 0, 'Tol': 0, 'Cover': 0, 'Source': 'Error'}}
            
            global_results[pid] = poza_res



    # --- HISTORY: MERGE & SAVE NEW FINDINGS ---
    if new_findings_buffer:
        df_new_findings = pd.DataFrame(new_findings_buffer)
        hist_df_updated, count_new = db_manager.merge_new_findings(hist_df_current, df_new_findings)
        if count_new > 0:
            db_manager.save_history(hist_df_updated)
            st.toast(f"✅ Se agregaron {count_new} nuevos puntos al historial.", icon="💾")

    # 4. REPORT GENERATION PHASE (ONCE)
    col_mapping = {'N': cn, 'E': ce, 'Z': cz}
    excel_data = generate_excel_report(global_results, col_map=col_mapping)
    
    # 4.1 PPTX GENERATION
    try:
        pptx_data = topo_logic.generar_pptx_report(global_results)
    except Exception as e:
        pptx_data = None
        st.error(f"Error generando PPTX: {e}")

    # 5. RENDERING PHASE
    st.header("Resultados por Poza")
    tabs = st.tabs([f"Poza {pid}" for pid in groups.groups.keys()])
    
    for (pid, df_grp), tab in zip(groups, tabs):
        with tab:
            poza_data = global_results[pid]
            ras_used = poza_data['Config']['Rasante']
            cov_used = poza_data['Config']['Cover']
            src_cov = poza_data['Config']['Source']
            
            # Calculate what tolerance was likely used for display
            # Calculate what tolerance was likely used for display
            if cov_used > 0:
                if criterio_eval == "Criterio Excon":
                     dyn_tol = 15.0 if cov_used > 50 else 10.0
                     tol_label = f"{dyn_tol:.1f}cm (Excon)"
                else:
                    dyn_tol = topo_logic.calculate_dynamic_tolerance(cov_used)
                    tol_label = f"{dyn_tol:.1f}cm (Dinámica)"
            else:
                dyn_tol = 4.0 # Default fallback
                tol_label = "PENDIENTE (Falta Espesor)"
            
            # Display Config Info
            if cov_used <= 0:
                 st.error("⚠️ **ALERTA: Falta Espesor (Cover).** No se encontró en la BD ni se ingresó valor manual.")
            else:
                cov_msg = f"{cov_used:.1f}cm ({src_cov})"
                st.info(f"📐 **Rasante:** {ras_used:.3f}m | **Espesor (Cover):** {cov_msg} | **Tolerancia Detección:** {tol_label}")
            
            t_res, t_map, t_down = st.tabs(["Estadísticas", "Mapas", "Descargas"])
            
            turns_present = df_grp['Turno'].unique()
            
            with t_res:
                col_charts = st.columns(len(turns_present)) if len(turns_present) > 0 else [st.container()]
                
                for idx, t in enumerate(turns_present):
                    data = poza_data.get(t)
                    if not data:
                        st.warning(f"⚠️ Datos no disponibles para Turno {t}")
                        continue
                        
                    with col_charts[idx]:
                        st.subheader(f"Turno {t}")
                        
                        # ANALYSIS TEXT (Calculated in Batch)
                        # ANALYSIS TEXT (Calculated in Batch)
                        txt = data.get('texto_analisis', "Error cargando texto.")
                        
                        # Full Expansion, Solid Color (Info Box Style)
                        st.markdown("**Análisis Técnico:**")
                        st.info(txt, icon="📝")
                        
                        # Manual AI Refresh (Optional, kept for "Re-run")
                        settings = st.session_state.get('app_ai_settings', {})
                        ai_enabled = settings.get("ai_enabled", True)
                        
                        

                        
                        # TABLE (Restore 'Tipo', hide 'Color')
                        df_show = data['tbl'].drop(columns=['Color'], errors='ignore')
                        # Explicit integer formatting for coordinates and area
                        format_dict = {
                            'Porcentaje': "{:.1f}%",
                            'Area_Efectiva_m2': "{:.0f}",
                            'Norte': "{:.0f}", 'Este': "{:.0f}",
                            'Elev_Min': "{:.3f}"
                        }
                        st.dataframe(df_show.style.format(format_dict, na_rep=""), use_container_width=True)
                        
                        # CHART (Vertical Bars with Colors)
                        fig = go.Figure()
                        # Use colors from logic if available
                        colors_mapped = [row['Color'] for _, row in data['tbl'].iterrows()] if 'Color' in data['tbl'].columns else None
                        
                        fig.add_trace(go.Bar(
                            x=data['tbl']['Rango'], # X Axis = Labels (Ranges)
                            y=data['tbl']['Puntos'], # Y Axis = Count (Height)
                            text=data['tbl']['Puntos'], 
                            textposition='auto',
                            marker_color=colors_mapped # Specific colors per bar
                        ))
                        
                        fig.update_layout(
                            title=f"Distribución Turno {t}", 
                            height=300, 
                            margin=dict(l=20,r=20,t=40,b=20),
                            xaxis_title="Rango",
                            yaxis_title="Cantidad de Puntos"
                        )
                        st.plotly_chart(fig, use_container_width=True)

            with t_map:
                # 1. Combined Satellite Map (Full Width) - HIDDEN TEMPORARILY
                # ... (Hidden code) ...

                st.divider()

                # 2. Individual Heatmaps
                col_maps = st.columns(len(turns_present)) if len(turns_present) > 0 else [st.container()]
                for idx, t in enumerate(turns_present):
                     data = poza_data.get(t)
                     if not data:
                         st.warning(f"⚠️ Datos no disponibles para Mapa - Turno {t}")
                         continue

                     with col_maps[idx]:
                         st.subheader(f"Mapa - Turno {t}")
                         # Removed per-turn satellite map from here

                         # 2. Heatmap (Plotly)
                         st.caption(f"Mapa de Calor Interactivo (Tol: {dyn_tol:.1f}cm)")
                         
                         fig_map = topo_logic.generar_mapa_interactivo(
                             data.get('df_visual', data['df']), data['zonas'], 
                             col_n=cn, col_e=ce,
                             titulo=f"Mapa {pid} - Turno {t}",
                             tol=dyn_tol,
                             criterio=criterio_eval,
                             cover_cm=cov_used
                         )
                         
                         if isinstance(fig_map, str):
                             st.error(fig_map)
                         elif fig_map is None:
                             st.warning("No se pudo generar el mapa.")
                         else:
                             # Restore standard interactive map (No clicking measurement)
                             st.plotly_chart(fig_map, use_container_width=True)

                         # TABLE & KPI BELOW MAP
                         if not data['zonas'].empty:
                             with st.expander(f"Detalle Puntos Bajos - {t}", expanded=True):
                                 # Analysis Text for Critical Zones
                                 st.markdown("**Análisis de Defectos:**")
                                 crit_txt = data.get('texto_analisis_critico', "Sin comentarios adicionales.")
                                 st.info(crit_txt, icon="⚠️")

                                 # Format KPI column as percentage
                                 # Formato solicitado: Norte/Este sin decimales, Elev_Min con 3.
                                 fmt_zonas = {
                                     'KPI Incidencia': "{:.6f}", # Pure Ratio
                                     'Area_Efectiva_m2': "{:.0f}",
                                     'Norte': "{:.0f}", 'Este': "{:.0f}",
                                     'Elev_Min': "{:.3f}",
                                     'Desv_Min (cm)': "{:.1f}"
                                 }
                                 st.dataframe(
                                     data['zonas'].style.format(fmt_zonas, na_rep=""), 
                                     use_container_width=True
                                 )
                         else:
                             st.success("No hay zonas defectuosas.")

            with t_down:
                 st.success("Reporte listo para descargar.")
                 st.download_button(
                    label="Descargar Reporte Consolidado (Excel)",
                    data=excel_data,
                    file_name=f"Reporte_Topo_Consolidado_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    type="primary",
                    key=f"btn_down_{pid}"
                 )
                 
                 if pptx_data:
                     # Dynamic filename matching user request format
                     pptx_name = f"Regis_GPS_Control_{pid}_{datetime.now().strftime('%d_%B_%H%M')}.pptx"
                     st.download_button(
                        label="Descargar Reporte PowerPoint (.pptx)",
                        data=pptx_data,
                        file_name=pptx_name,
                        mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                        key=f"btn_down_ppt_{pid}"
                     )

# Branding Bottom Sidebar
with st.sidebar:
    st.markdown("---")
    st.markdown("<div style='text-align: right; font-style: italic;'>Desarrollado por Departamento de Innovación Excon.</div>", unsafe_allow_html=True)



