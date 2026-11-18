import gradio as gr
import numpy as np
import cv2

# CSS personalizado para remarcar las casillas de subida y darles un aspecto limpio
custom_css = """
.sat-box {
    border: 2px dashed #94a3b8 !important;
    border-radius: 10px !important;
    background-color: #f8fafc !important;
    transition: all 0.2s ease-in-out !important;
}
.sat-box:hover {
    border-color: #f97316 !important;
    background-color: #fff7ed !important;
}
.panel-card {
    border: 1px solid #e2e8f0 !important;
    border-radius: 10px !important;
    padding: 12px !important;
    background-color: #ffffff !important;
}
"""

def hacer_mapa_calor(matriz_64x64):
    """
    Convierte una matriz de probabilidades (64x64, valores 0..1) en una imagen 
    con mapa de calor (COLORMAP_INFERNO) reescalada a 320x320 sin difuminar los píxeles.
    """
    matriz_uint8 = np.clip(matriz_64x64 * 255, 0, 255).astype(np.uint8)
    mapa_bgr = cv2.applyColorMap(matriz_uint8, cv2.COLORMAP_INFERNO)
    mapa_rgb = cv2.cvtColor(mapa_bgr, cv2.COLOR_BGR2RGB)
    mapa_grande = cv2.resize(mapa_rgb, (320, 320), interpolation=cv2.INTER_NEAREST)
    return mapa_grande

def predecir_incendio(olci, dem, wc, h):
    if not olci or not dem or not wc:
        return None, None, "Pendiente: Por favor, carga los 3 archivos satelitales (OLCI, DEM y WC) antes de continuar."
    
    if h is None or h <= 0:
        return None, None, "Error: El horizonte de predicción (H) debe ser mayor a 0 días."

    # =========================================================================
    # ZONA DE INTEGRACIÓN DEL MODELO (Para Antonio, Sebastián y Eduardo)
    # =========================================================================
    # 1. Cargar tensores desde: olci.name, dem.name, wc.name
    # 2. Normalizar:
    #    - OLCI: (7, 5, 64, 64) float32 -> dividir por 1000.0
    #    - DEM:  (1, 64, 64) float32    -> dividir por 1000.0 (pasa a km)
    #    - WC:   (1, 64, 64) float32    -> dividir por 100.0 (0 = sin dato)
    # 3. Generar Entrada 4 (SCAR) con modelo congelado TMB-S3 (OLCI + WC en frame f)
    # 4. Ejecutar modelo final: prob_quema_nueva = sigmoid(model(OLCI, DEM, WC, SCAR, H))
    # =========================================================================

    # --- SIMULACIÓN VISUAL (Mientras se conecta el modelo PyTorch) ---
    y, x = np.ogrid[:64, :64]
    scar_simulado = np.exp(-((x - 28)**2 + (y - 32)**2) / 35.0)
    desplazamiento = min(int(h * 2), 18)
    quema_nueva_simulada = np.exp(-((x - (28 + desplazamiento))**2 + (y - 30)**2) / (50.0 + h * 8))
    quema_nueva_simulada = np.clip(quema_nueva_simulada + np.random.uniform(0, 0.08, (64, 64)), 0, 1)

    img_scar = hacer_mapa_calor(scar_simulado)
    img_pred = hacer_mapa_calor(quema_nueva_simulada)

    # Cálculo de superficie: 1 píxel de 300x300 m = 9 hectáreas
    pixeles_criticos = int(np.sum(quema_nueva_simulada > 0.5))
    hectareas_riesgo = pixeles_criticos * 9

    reporte = (
        f"RESUMEN DE LA PREDICCIÓN (Zona de 19.2 x 19.2 km | 64x64 px)\n"
        f"------------------------------------------------------------\n"
        f"• Horizonte proyectado (H): {h} días\n"
        f"• Píxeles con alta prob. de quema (>50%): {pixeles_criticos} px\n"
        f"• Superficie estimada de expansión: ~{hectareas_riesgo} hectáreas\n"
        f"• Estado del pipeline: Entradas validadas (Modo Demo Activo)"
    )

    return img_scar, img_pred, reporte


# --- DISEÑO DE INTERFAZ ---
with gr.Blocks(theme=gr.themes.Default(), css=custom_css) as demo:
    gr.Markdown(
        """
        # Sistema de Predicción de Propagación de Incendios (Sen2Fire)
        **Demo de inferencia satelital** sobre cuadrículas locales de **19.2 × 19.2 km** (64×64 píxeles a 300 m de resolución).
        """
    )

    with gr.Row(equal_height=False):
        # COLUMNA IZQUIERDA: Entradas con marcos definidos y sin solapamiento
        with gr.Column(scale=5):
            gr.Markdown("### 1. Datos Satelitales de Entrada")
            
            file_olci = gr.File(
                label="Secuencias Sentinel-3 OLCI (7 frames, 5 bandas)",
                height=145,
                elem_classes=["sat-box"]
            )
            
            with gr.Row():
                file_dem = gr.File(
                    label="Mapa de Elevación (DEM GLO30)",
                    height=145,
                    elem_classes=["sat-box"]
                )
                file_wc = gr.File(
                    label="Uso de Suelo (WorldCover v200)",
                    height=145,
                    elem_classes=["sat-box"]
                )

            gr.Markdown("### 2. Parámetros de Proyección")
            num_h = gr.Number(
                label="Horizonte de predicción H (en días)",
                info="Ej: 0.5 (12 horas), 2 (dos días), 7 (una semana). La cicatriz actual (SCAR) se calcula internamente con TMB-S3.",
                value=2.0,
                minimum=0.1,
                step=0.5,
                elem_classes=["panel-card"]
            )

            with gr.Row():
                btn_limpiar = gr.Button("Limpiar datos", variant="secondary")
                btn_predecir = gr.Button("Calcular Predicción", variant="primary")

        # COLUMNA DERECHA: Salidas visuales y panel de métricas
        with gr.Column(scale=6):
            gr.Markdown("### 3. Mapas de Probabilidad y Métricas")
            
            with gr.Row():
                out_scar = gr.Image(
                    label="Cicatriz Actual (SCAR - Modelo TMB-S3)",
                    height=265,
                    elem_classes=["panel-card"]
                )
                out_pred = gr.Image(
                    label="Predicción de Quema Nueva (Salida Sigmoid)",
                    height=265,
                    elem_classes=["panel-card"]
                )
            
            out_status = gr.Textbox(
                label="Reporte de Superficie y Estado",
                lines=6,
                interactive=False,
                elem_classes=["panel-card"]
            )

    btn_predecir.click(
        fn=predecir_incendio,
        inputs=[file_olci, file_dem, file_wc, num_h],
        outputs=[out_scar, out_pred, out_status]
    )

    btn_limpiar.click(
        fn=lambda: [None, None, None, 2.0, None, None, ""],
        inputs=[],
        outputs=[file_olci, file_dem, file_wc, num_h, out_scar, out_pred, out_status]
    )

if __name__ == "__main__":
    demo.launch()