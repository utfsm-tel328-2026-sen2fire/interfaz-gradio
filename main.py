import gradio as gr
import numpy as np
import cv2
import torch
from burn_scar.models.propagation import BurnScarPropagationModel

# --- 1. CARGA ÚNICA DEL MODELO ENTRENADO ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_PATH = "weights/best_fold_0.pt"

ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
MODEL_CONFIG = ckpt["model_config"]
BEST_THRESHOLD = float(ckpt.get("metrics", {}).get("best_threshold", 0.2))

model = BurnScarPropagationModel(**MODEL_CONFIG).to(DEVICE)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# CSS para mantener la estética limpia con bordes definidos
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
    """Convierte matriz (64, 64) en [0, 1] a mapa de calor RGB reescalado a 320x320."""
    matriz_uint8 = np.clip(matriz_64x64 * 255, 0, 255).astype(np.uint8)
    mapa_bgr = cv2.applyColorMap(matriz_uint8, cv2.COLORMAP_INFERNO)
    mapa_rgb = cv2.cvtColor(mapa_bgr, cv2.COLOR_BGR2RGB)
    return cv2.resize(mapa_rgb, (320, 320), interpolation=cv2.INTER_NEAREST)

def preprocesar_olci(olci_raw: np.ndarray) -> np.ndarray:
    """Rellena NaNs con la media de su banda en ese frame y normaliza radiancia / 1000."""
    olci = olci_raw.astype(np.float32).copy()
    if olci.shape[0] < 7:
        faltantes = 7 - olci.shape[0]
        pad = np.repeat(olci[0:1], faltantes, axis=0)
        olci = np.concatenate([pad, olci], axis=0)
    for t in range(olci.shape[0]):
        for b in range(olci.shape[1]):
            banda = olci[t, b]
            if np.isnan(banda).any():
                media = np.nanmean(banda)
                banda[np.isnan(banda)] = 0.0 if np.isnan(media) else media
    return olci / 1000.0

def predecir_incendio(archivo_npz, h_usuario):
    if archivo_npz is None:
        return None, None, "Pendiente: Carga un archivo satelital (.npz) para ejecutar la inferencia."

    try:
        datos = np.load(archivo_npz.name)
    except Exception as e:
        return None, None, f"Error al leer el archivo .npz: {e}"

    requeridas = ["x_olci", "x_dem", "x_worldcover", "x_scar"]
    faltantes = [k for k in requeridas if k not in datos]
    if faltantes:
        return None, None, f"Error: El archivo .npz no contiene las claves esperadas: {faltantes}"

    # 1. Preprocesamiento según contrato del modelo
    olci = preprocesar_olci(datos["x_olci"])                      # (7, 5, 64, 64)
    dem = datos["x_dem"].astype(np.float32) / 1000.0              # metros -> km
    wc_raw = datos["x_worldcover"].astype(np.float32)
    valid_fuel = wc_raw > 0                                       # clase 0 = sin dato/combustible
    wc = np.where(valid_fuel, wc_raw / 100.0, 0.0)
    scar = datos["x_scar"].astype(np.float32)

    # Asegurar forma (1, 64, 64) en capas 2D
    if dem.ndim == 2:
        dem = dem[None]
    if wc.ndim == 2:
        wc = wc[None]
    if scar.ndim == 2:
        scar = scar[None]

    # Usar días ingresados por el usuario o el horizonte propio del archivo si es <= 0
    if h_usuario is not None and h_usuario > 0:
        h_final = float(h_usuario)
    elif "horizon_days" in datos:
        h_final = float(datos["horizon_days"])
    else:
        h_final = 2.0

    # 2. Inferencia con PyTorch
    with torch.no_grad():
        t_olci = torch.from_numpy(olci[None]).to(DEVICE)
        t_dem = torch.from_numpy(dem[None]).to(DEVICE)
        t_wc = torch.from_numpy(wc[None]).to(DEVICE)
        t_scar = torch.from_numpy(scar[None]).to(DEVICE)
        t_h = torch.tensor([h_final], dtype=torch.float32, device=DEVICE)

        logits = model(t_olci, t_dem, t_wc, t_scar, t_h)
        prob_quema = torch.sigmoid(logits).squeeze().cpu().numpy()

    # Descartar zonas sin dato de combustible (clase WC == 0)
    prob_quema = np.where(valid_fuel, prob_quema, 0.0)

    # 3. Visualización y cálculo de hectáreas (1 píxel 300x300m = 9 ha)
    img_scar = hacer_mapa_calor(scar.squeeze())
    img_pred = hacer_mapa_calor(prob_quema)

    pixeles_actuales = int(np.sum(scar.squeeze() > 0.5))
    pixeles_nuevos = int(np.sum(prob_quema > BEST_THRESHOLD))
    ha_actuales = pixeles_actuales * 9
    ha_nuevas = pixeles_nuevos * 9

    nota_target = ""
    if "target" in datos:
        ha_reales = int(np.sum(datos["target"] > 0.5)) * 9
        nota_target = f"• Expansión real registrada (ground truth): ~{ha_reales} ha\n"

    reporte = (
        f"REPORTE DE INFERENCIA REAL (Zona 19.2 × 19.2 km | 64×64 px)\n"
        f"-------------------------------------------------------------\n"
        f"• Horizonte evaluado (H): {h_final:.2f} días | Umbral: {BEST_THRESHOLD}\n"
        f"• Cicatriz actual detectada (SCAR): {pixeles_actuales} px (~{ha_actuales} ha)\n"
        f"• Quema nueva proyectada: {pixeles_nuevos} px (~{ha_nuevas} ha)\n"
        f"{nota_target}"
        f"• Probabilidad máxima en escena: {prob_quema.max()*100:.1f}%"
    )

    return img_scar, img_pred, reporte


# --- INTERFAZ GRADIO ---
with gr.Blocks() as demo:
    gr.Markdown(
        """
        # Sistema de Predicción de Propagación de Incendios (Sen2Fire)
        **Inferencia en vivo (ConvLSTM + MiniSegFormer)** sobre cuadrículas locales de **19.2 × 19.2 km** (64×64 píxeles a 300 m).
        """
    )

    with gr.Row(equal_height=False):
        with gr.Column(scale=5):
            gr.Markdown("### 1. Carga de Escena Satelital")
            file_npz = gr.File(
                label="Paquete Satelital del Evento (.npz: OLCI + DEM + WorldCover + SCAR)",
                file_types=[".npz"],
                height=170,
                elem_classes=["sat-box"]
            )

            gr.Markdown("### 2. Horizonte Temporal")
            num_h = gr.Number(
                label="Días a proyectar (H)",
                info="Ingresa un valor mayor a 0 (ej. 0.5, 2, 5). Si pones 0, usa el horizonte original del archivo.",
                value=2.0,
                minimum=0.0,
                step=0.5,
                elem_classes=["panel-card"]
            )

            with gr.Row():
                btn_limpiar = gr.Button("Limpiar", variant="secondary")
                btn_predecir = gr.Button("Calcular Predicción", variant="primary")

        with gr.Column(scale=6):
            gr.Markdown("### 3. Resultados del Modelo")
            with gr.Row():
                out_scar = gr.Image(
                    label="Cicatriz Actual (SCAR - Entrada f)",
                    height=265,
                    elem_classes=["panel-card"]
                )
                out_pred = gr.Image(
                    label="Probabilidad de Quema Nueva (f + H)",
                    height=265,
                    elem_classes=["panel-card"]
                )

            out_status = gr.Textbox(
                label="Métricas de Superficie y Validación",
                lines=6,
                interactive=False,
                elem_classes=["panel-card"]
            )

    btn_predecir.click(
        fn=predecir_incendio,
        inputs=[file_npz, num_h],
        outputs=[out_scar, out_pred, out_status]
    )

    btn_limpiar.click(
        fn=lambda: [None, 2.0, None, None, ""],
        inputs=[],
        outputs=[file_npz, num_h, out_scar, out_pred, out_status]
    )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Default(), css=custom_css)