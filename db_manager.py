import json
import os
import pandas as pd
import hashlib
import io
from datetime import datetime

# Path to local DB file
DB_FILE = "db_historial_calidad.json"

def get_db_path():
    return os.path.abspath(DB_FILE)

def load_history():
    """Load the history database from JSON file."""
    if not os.path.exists(DB_FILE):
        return pd.DataFrame(columns=[
            'ID', 'ID_Unico', 'Fecha_Reporte', 'Poza', 'Turno', 
            'Norte', 'Este', 'Cota_Teorica', 'Cota_GPS', 'Desv_GPS',
            'Cota_Real_Terreno', 'Desv_Real', 'Observacion', 'Estado', 'Color_Estado'
        ])
    
    try:
        with open(DB_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        df_hist = pd.DataFrame(data)
        
        # Enforce Schema
        required_cols = [
            'ID', 'ID_Unico', 'Fecha_Reporte', 'Poza', 'Turno', 
            'Norte', 'Este', 'Cota_Teorica', 'Cota_GPS', 'Desv_GPS',
            'Cota_Real_Terreno', 'Desv_Real', 'Observacion', 'Estado', 'Color_Estado'
        ]
        for col in required_cols:
            if col not in df_hist.columns:
                df_hist[col] = None # Or appropriate default
                
        return df_hist
    except Exception as e:
        print(f"Error loading DB: {e}")
        return pd.DataFrame(columns=[
            'ID', 'ID_Unico', 'Fecha_Reporte', 'Poza', 'Turno', 
            'Norte', 'Este', 'Cota_Teorica', 'Cota_GPS', 'Desv_GPS',
            'Cota_Real_Terreno', 'Desv_Real', 'Observacion', 'Estado', 'Color_Estado'
        ])

def save_history(df):
    """Save the history dataframe to JSON file."""
    try:
        # Convert date objects to strings for JSON serialization
        df_save = df.copy()
        if 'Fecha_Reporte' in df_save.columns:
            df_save['Fecha_Reporte'] = df_save['Fecha_Reporte'].astype(str)
            
        data = df_save.to_dict(orient='records')
        with open(DB_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Error saving DB: {e}")
        return False

def generate_unique_id(row):
    """Generate a unique hash ID based on fixed attributes."""
    # Components: Poza, Date, Turn, N (int), E (int) -> Enough uniqueness
    raw_str = f"{row['Poza']}_{row['Fecha_Reporte']}_{row['Turno']}_{int(row['Norte'])}_{int(row['Este'])}"
    return hashlib.md5(raw_str.encode()).hexdigest()[:12]

def merge_new_findings(history_df, new_findings_df):
    """
    Merge new findings into history, avoiding duplicates.
    new_findings_df expects columns: ['Poza', 'Turno', 'Fecha_Reporte', 'Norte', 'Este', 'Cota_Teorica', 'Cota_GPS', 'Desv_GPS']
    """
    if new_findings_df.empty:
        return history_df, 0
    
    # Ensure ID exists in new findings
    new_findings_df['ID_Unico'] = new_findings_df.apply(generate_unique_id, axis=1)
    
    # Prepare standard columns
    new_findings_df['Cota_Real_Terreno'] = None
    new_findings_df['Desv_Real'] = None
    new_findings_df['Observacion'] = ""
    new_findings_df['Estado'] = "Pendiente"
    new_findings_df['Color_Estado'] = "#FF0000" # Red for new/pending
    
    # Filter out existing IDs
    existing_ids = set(history_df['ID_Unico'].values)
    
    unique_new = new_findings_df[~new_findings_df['ID_Unico'].isin(existing_ids)].copy()
    
    if unique_new.empty:
        return history_df, 0
    
    # Assign Sequential IDs
    max_id = 0
    if 'ID' in history_df.columns and not history_df.empty:
        try:
            max_id = history_df['ID'].astype(int).max()
            if pd.isna(max_id): max_id = 0
        except: max_id = 0
    
    unique_new['ID'] = range(max_id + 1, max_id + 1 + len(unique_new))
        
    # Concatenate
    updated_history = pd.concat([history_df, unique_new], ignore_index=True)
    
    return updated_history, len(unique_new)

def export_to_excel(df):
    """Generate Excel bytes for download."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Historial_Calidad')
        # Auto-adjust columns width (basic)
        worksheet = writer.sheets['Historial_Calidad']
        for i, col in enumerate(df.columns):
            worksheet.set_column(i, i, 20)
    return output.getvalue()
