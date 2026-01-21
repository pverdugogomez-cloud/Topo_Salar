import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import io
from datetime import datetime
import topo_logic
import traceback
import plotly.graph_objects as go

# ==========================================
# CONFIGURACIÓN PAGINA
# ==========================================
st.set_page_config(page_title="Topo Dashboard V27", layout="wide", page_icon="Logo_TS.ico")

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
        f_p = wb.add_format({'num_format':'0.0%','border':1}) # Excel expects ratio for %, logic gives 0-100?
        # Logic gives 15.4 (0-100). So format should be just '0.0' with suffix? 
        # Actually user wants "15.4%". If logic gives 15.4, format '0.0"%"' works.
        f_custom_pct = wb.add_format({'num_format':'0.0"%"', 'border':1, 'align': 'center'})
        
        # CENTERED HEADERS
        f_hy = wb.add_format({'bold':True,'bg_color':'#FFC000','border':1,'align':'center', 'valign':'vcenter'})
        f_w = wb.add_format({'text_wrap':True,'border':0,'valign':'top'})
        f_center = wb.add_format({'align':'center', 'border':1})

        # 1. Resumen Ejecutivo (All Pozas)
        ws1 = wb.add_worksheet('Resumen Ejecutivo')
        ws1.write('B2', "REPORTE CONSOLIDADO DE CALIDAD DE NIVELACIÓN", f_tit)
        ws1.write('B3', f"Fecha Gen: {datetime.now().strftime('%d/%m/%Y')}")
        
        curr_row = 5
        
        for poza_id, poza_data in global_res_dict.items():
            conf = poza_data.get('Config', {})
            ras, cov, tol = conf.get('Rasante',0), conf.get('Cover',0), conf.get('Tol',0)
            
            ws1.write(curr_row, 1, f"POZA: {poza_id}", f_tit)
            ws1.write(curr_row+1, 1, f"Rasante: {ras:.3f} | Tol: {tol:.1f}cm")
            curr_row += 3

            for t in ['A', 'B', 'General']:
                if t not in poza_data or poza_data[t]['vacio']: continue
                data = poza_data[t]
                
                ws1.merge_range(curr_row, 1, curr_row, 4, f"TURNO {t}", f_hy)
                ws1.write_row(curr_row+1, 1, ['Tipo','Rango','Cant','%'], f_hy) # Centered Header
                for i, r in data['tbl'].iterrows():
                    ws1.write(curr_row+2+i,1,r['Tipo'],f_b)
                    ws1.write(curr_row+2+i,2,r['Rango'],f_center)
                    ws1.write(curr_row+2+i,3,r['Puntos'],f_center)
                    # Logic gives 0-100 value.
                    ws1.write(curr_row+2+i,4,r['Porcentaje'],f_custom_pct)
                
                # Chart
                ch = wb.add_chart({'type':'bar'})
                ch.add_series({
                    'name': f'{poza_id} - Turno {t}',
                    'categories': ['Resumen Ejecutivo', curr_row+2, 2, curr_row+9, 2],
                    'values':     ['Resumen Ejecutivo', curr_row+2, 4, curr_row+9, 4],
                    'fill':       {'color': '#203764'} # Uniform Color
                })
                ws1.insert_chart(f'G{curr_row}', ch, {'x_scale':0.8, 'y_scale':0.8})
                curr_row += 16
            
            curr_row += 2 # Espacio entre pozas

        # 2. Zonas Defectuosas
        ws2 = wb.add_worksheet('Zonas_Defectuosas')
        curr = 0
        for poza_id, poza_data in global_res_dict.items():
            ws2.write(curr, 0, f"POZA: {poza_id}", f_tit)
            curr += 2
            
            for t in ['A', 'B', 'General']:
                if t not in poza_data or poza_data[t]['vacio']: continue
                data = poza_data[t]
                
                # Analysis Text
                texto = topo_logic.generar_texto_analisis(data['tbl'], data['zonas'], data['atot'], poza_id)
                ws2.merge_range(curr, 0, curr+6, 11, texto, f_w)
                
                # KPI Calculation
                area_mala = data['zonas']['Area_Efectiva_m2'].sum() if not data['zonas'].empty else 0
                kpi_val = (area_mala / data['atot']) * 100 if data['atot'] > 0 else 0
                ws2.write(curr+7, 0, f"KPI Incidencia (Area Defectuosa / Total): {kpi_val:.2f}%", f_tit)

                # Insert Map Image NEXT to table (approx col 13?)
                # We need to generate it here
                map_inserted = False
                try:
                    fig_map = topo_logic.generar_mapa_interactivo(
                         data['df'], data['zonas'], 
                         col_n=col_mapping['N'], col_e=col_mapping['E'],
                         titulo=f"Mapa {poza_id} - Turno {t}",
                         step=poza_data['Config']['Tol']
                    )
                    if fig_map:
                        # Ensure kaleido is present or handle
                        img_data = io.BytesIO(fig_map.to_image(format="png", width=600, height=500, scale=1.5))
                        # Insert at 'curr' row, col 14 (N)
                        ws2.insert_image(curr, 14, 'map.png', {'image_data': img_data, 'x_scale': 0.8, 'y_scale': 0.8})
                        map_inserted = True
                except Exception as e:
                    print(f"Map gen failed: {e}")
                    ws2.write(curr, 14, "No se pudo generar imagen (Kaleido missing?)")

                curr += 9
                
                # Table of Zones
                if not data['zonas'].empty:
                    ws2.write(curr, 0, f"Detalle Zonas Turno {t}", f_hy)
                    # Headers centered
                    ws2.write_row(curr+1, 0, data['zonas'].columns, f_hy)
                    for i, row in data['zonas'].iterrows():
                         ws2.write_row(curr+2+i, 0, row.values, f_center)
                         # Explicit format for Incidencia column (last one?)
                         # Assuming 'Incidencia (%)' is last
                         if 'Incidencia (%)' in data['zonas'].columns:
                             idx_inc = data['zonas'].columns.get_loc('Incidencia (%)')
                             val = row['Incidencia (%)']
                             ws2.write(curr+2+i, idx_inc, val, f_custom_pct)

                    # Move cursor down based on table size OR map size
                    rows_table = len(data['zonas']) + 4
                    rows_map = 20 # approx for map height
                    curr += max(rows_table, rows_map) + 2
                else:
                    ws2.write(curr, 0, f"Turno {t}: No se detectaron zonas críticas.")
                    curr += 20 if map_inserted else 5

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
                temp_df = pd.read_csv(u_file)
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
                df['DT'] = pd.to_datetime(df[c_time], errors='coerce')
                df['Fecha'] = df['DT'].dt.date
                df['Turno'] = df['DT'].apply(lambda x: 'A' if 7<=x.hour<19 else 'B')
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
    # --- CONFIGURATION EXPANDER (Menu 2) ---
    # User requested removing the "Gear" header and just having the menu.
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
        
        # 2. Rasantes Configuration
        st.subheader("2. Rasantes por Poza")
        
        # Calculate Config Rows (Handle if df is None)
        active_pozas = unique_pozas_all if df is not None else []
        config_rows = []
        
        if df is not None:
            for pid in active_pozas:
                if pid == "GENERAL": continue
                cov = st.session_state.db_pozas.get(pid, 0.0)
                source = "BD" if cov > 0 else None
                ras_auto = 0.0
                if cov > 0:
                    ras_auto = 2300.0 + (cov/100.0)
                else:
                    mask = df['PozaID'] == pid
                    if mask.any():
                        rep_name = str(df.loc[mask, c_design].iloc[0])
                        match = re.search(r'(23\d{2}[.,]?\d*)', rep_name)
                        if match:
                            try: ras_auto = float(match.group(1).replace(',', '.')); source = "Nom"
                            except: pass
                
                status_emoji = "✅" if ras_auto > 0 else "⚠️"
                status_text = f"{status_emoji} {source}" if source else f"{status_emoji} Falta"
                config_rows.append({"PozaID": pid, "Rasante": ras_auto, "Info": status_text})
        
        if config_rows:
            df_config = pd.DataFrame(config_rows).set_index("PozaID")
            edited_config = st.data_editor(
                df_config,
                column_config={
                    "Rasante": st.column_config.NumberColumn("Rasante (m)", format="%.3f", required=True),
                    "Info": st.column_config.TextColumn("Fuente", disabled=True),
                },
                disabled=["PozaID", "Info"],
                key="ras_editor_main",
                use_container_width=True
            )
        else:
            if df is None:
                st.info("⚠️ Cargue un archivo CSV/Excel para configurar Rasantes.")
            else:
                st.info("No se detectaron pozas para configurar.")
            edited_config = pd.DataFrame(columns=["Rasante"])

        st.divider()
        
        # 3. Tolerance
        st.subheader("3. Tolerancia Visual")
        tol_step_val = st.number_input("Paso de Tolerancia (cm)", value=4.0, step=0.5, key="tol_step_main")

    # --- FILTERS EXPANDER (Menu 3) ---
    st.divider()
    with st.expander("Filtros", expanded=True):
        if df is not None:
            # 1. Poza
            poza_opts = ["TODOS"] + (unique_pozas_all if unique_pozas_all else [])
            sel_poza = st.selectbox("1. Poza:", poza_opts, key="f_poza")
            
            # Filter Step 1
            df_f1 = df.copy()
            if sel_poza != "TODOS":
                df_f1 = df_f1[df_f1['PozaID'] == sel_poza]
            
            # 2. Machine
            maqs_avail = sorted(df_f1[c_maq].dropna().unique()) if (c_maq and not df_f1.empty) else []
            sel_maq = st.selectbox("2. Máquina:", ["TODOS"] + maqs_avail, key="f_maq")

            # Filter Step 2
            df_f2 = df_f1.copy()
            if sel_maq != "TODOS" and c_maq:
                df_f2 = df_f2[df_f2[c_maq] == sel_maq]

            # 3. Date
            fechas_avail = sorted(df_f2['Fecha'].dropna().unique()) if (c_time and not df_f2.empty) else []
            dates_opts = ["TODOS"] + [str(d) for d in fechas_avail]
            sel_date = st.selectbox("3. Fecha:", dates_opts, key="f_date")

            # Filter Step 3
            df_f3 = df_f2.copy()
            if sel_date != "TODOS" and c_time:
                df_f3 = df_f3[df_f3['Fecha'] == datetime.strptime(sel_date, '%Y-%m-%d').date()]
            
            # 4. Turn
            turnos_avail = sorted(df_f3['Turno'].dropna().unique()) if ('Turno' in df_f3 and not df_f3.empty) else []
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
    st.divider()
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
        st.rerun()

# ==========================================
# PROCESSING & RESULTS
# ==========================================
apply_filters = st.session_state['processing_active']
# ==========================================
# PROCESSING & RESULTS
# ==========================================
if apply_filters and df is not None:
    
    # 1. Apply Filters
    df_final = df_f3.copy()
    if sel_turn != "TODOS":
        df_final = df_final[df_final['Turno'] == sel_turn]
        
    if df_final.empty:
        st.warning("No hay datos tras filtrar.")
        st.stop()
        
    # 2. Vectorized Deviation Calculation
    df_final['PozaID'] = df_final['PozaID'].astype(str).str.strip()
    rasante_map = {str(k).strip(): v for k, v in edited_config['Rasante'].to_dict().items()}
    df_final['Rasante_Teorica'] = df_final['PozaID'].map(rasante_map).fillna(0.0)
    df_final['Cota_Calc'] = df_final[cz] 
    df_final['desviacion'] = (df_final['Cota_Calc'] - df_final['Rasante_Teorica']) * 100.0
    
    # 3. GLOBAL CALCULATION PHASE
    global_results = {}
    groups = df_final.groupby('PozaID')
    
    # Prepare result container for all pozas first
    for pid, df_grp in groups:
        # Config used
        if pid in edited_config.index:
            ras_used = edited_config.loc[pid, 'Rasante']
        else:
            ras_used = 0.0
        
        poza_res = {'Config': {'Rasante': ras_used, 'Tol': tol_step_val, 'Cover': 0}}
        turns_present = df_grp['Turno'].unique()
        
        for t in turns_present:
            df_t = df_grp[df_grp['Turno'] == t]
            
            # Logic calculation
            res_tbl, df_proc, zonas, atot = topo_logic.procesar_turno(
                df_t, ras_used, 
                tolerancia=tol_step_val, 
                col_z=cz, col_n=cn, col_e=ce,
                step=tol_step_val
            )
            
            poza_res[t] = {
                'tbl': res_tbl, 'df': df_proc, 'zonas': zonas, 'atot': atot, 'vacio': False
            }
        global_results[pid] = poza_res

    # 4. REPORT GENERATION PHASE (ONCE)
    col_mapping = {'N': cn, 'E': ce, 'Z': cz}
    excel_data = generate_excel_report(global_results, col_map=col_mapping)

    # 5. RENDERING PHASE
    # 5. RENDERING PHASE
    st.header("Resultados por Poza")
    tabs = st.tabs([f"Poza {pid}" for pid in groups.groups.keys()])
    
    for (pid, df_grp), tab in zip(groups, tabs):
        with tab:
            poza_data = global_results[pid]
            ras_used = poza_data['Config']['Rasante']
            
            st.caption(f"Rasante: **{ras_used:.3f}** | Paso Tolerancia: **{tol_step_val}cm**")
            
            t_res, t_map, t_down = st.tabs(["Estadísticas", "Mapas", "Descargas"])
            
            turns_present = df_grp['Turno'].unique()
            
            with t_res:
                col_charts = st.columns(len(turns_present)) if len(turns_present) > 0 else [st.container()]
                
                for idx, t in enumerate(turns_present):
                    data = poza_data[t]
                    with col_charts[idx]:
                        st.subheader(f"Turno {t}")
                        
                        # ANALYSIS TEXT FIRST (Expanded)
                        txt = topo_logic.generar_texto_analisis(data['tbl'], data['zonas'], data['atot'], f"{pid}_{t}")
                        st.info(txt)
                        
                        # TABLE (Hide Color column)
                        df_show = data['tbl'].drop(columns=['Color'], errors='ignore')
                        st.dataframe(df_show.style.format({'Porcentaje': "{:.1f}%"}), use_container_width=True)
                        
                        # CHART (Uniform Color)
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                            x=data['tbl']['Rango'], y=data['tbl']['Puntos'],
                            text=data['tbl']['Puntos'], textposition='auto'
                        ))
                        # Default Plotly blue is fine, or specify a single color like 'marker_color="#203764"' if desired. 
                        # User said "all same color", default is same color.
                        fig.update_layout(title=f"Distribución Turno {t}", height=300, margin=dict(l=20,r=20,t=40,b=20))
                        st.plotly_chart(fig, use_container_width=True)

            with t_map:
                col_maps = st.columns(len(turns_present)) if len(turns_present) > 0 else [st.container()]
                for idx, t in enumerate(turns_present):
                     data = poza_data[t]
                     with col_maps[idx]:
                         st.subheader(f"Mapas - Turno {t}")
                         fig_map = topo_logic.generar_mapa_interactivo(
                             data['df'], data['zonas'], 
                             col_n=cn, col_e=ce,
                             titulo=f"Mapa {pid} - Turno {t}",
                             step=tol_step_val
                         )
                         st.plotly_chart(fig_map, use_container_width=True)

                         # TABLE & KPI BELOW MAP
                         if not data['zonas'].empty:
                             with st.expander(f"Detalle Puntos Bajos - {t}", expanded=True):
                                 # Format KPI column as percentage
                                 st.dataframe(
                                     data['zonas'].style.format({'Incidencia (%)': "{:.2f}%", 'Area_Efectiva_m2': "{:.1f}"}), 
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

# Branding Bottom Sidebar
with st.sidebar:
    st.markdown("---")
    st.markdown("<div style='text-align: right; font-style: italic;'>Desarrollado por Departamento de Innovación Excon.</div>", unsafe_allow_html=True)



