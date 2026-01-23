# Release Notes - Topo Dashboard V27

## 🚀 Nuevas Funcionalidades

### 1. Gestión Avanzada de Covers (Espesores)
- **Editor Manual por Poza**: Nueva tabla interactiva en la barra lateral que permitir definir el *Cover* de forma individual para cada poza detectada.
- **Persistencia**: Los valores manuales ingresados se guardan durante la sesión y no se pierden al interactuar con otros elementos.
- **Prioridad**: El sistema prioriza automáticamente el valor de la Base de Datos; si no existe, usa el valor Manual.

### 2. Sistema de Validación Proactiva
- **Alertas Tempranas**: Ahora el sistema verifica automáticamente si faltan configuraciones de *Cover* apenas se carga un archivo.
- **Visibilidad Mejorada**: Las alertas aparecen en un contenedor rojo visible en la parte superior del Dashboard (bajo el título principal), indicando exactamente qué pozas requieren atención antes de procesar.
- **Bloqueo Seguro**: El botón "PROCESAR RESULTADOS" se bloquea lógicamente si faltan datos críticos, evitando cálculos erróneos.

### 3. Mejoras en Compatibilidad de Archivos
- **Soporte de Fechas Robustas**: Se añadió un pre-procesador para manejar fechas con meses en inglés (ej. "2026/Jan/01") correctamente, incluso en sistemas con configuración regional en español.

## 🎨 Mejoras de Interfaz (UI/UX)
- **Limpieza Visual**: Se eliminaron divisores redundantes (`---`) y líneas excesivas en la barra lateral para una apariencia más limpia y profesional.
- **Iconos Actualizados**: Se simplificaron los títulos de los expansores (eliminado icono de carpeta en Filtros).

## 🔧 Correcciones Técnicas
- **KPI Incidencia**: Ajustado el formato numérico para mostrar el ratio exacto (4 decimales) en lugar de porcentaje.
- **Corrección de Bugs**: Solucionado un error crítico donde se sobrescribían los datos filtrados en ciertos flujos de ejecución.

---
**Archivos Actualizados:**
- `topo_dashboard_v2.py`
- `topo_logic_v2.py`
