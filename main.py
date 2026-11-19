import gradio as gr
import numpy as np
import cv2
import torch
from burn_scar.models.propagation import BurnScarPropagationModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_PATH = "weights/best_fold_0.pt"

ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
MODEL_CONFIG = ckpt["model_config"]
BEST_THRESHOLD = float(ckpt.get("metrics", {}).get("best_threshold", 0.2))

model = BurnScarPropagationModel(**MODEL_CONFIG).to(DEVICE)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

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


def crear_fondo_terreno(olci_norm: np.ndarray, dem_norm: np.ndarray) -> np.ndarray:
    banda_sat = olci_norm[-1, 0]
    p2, p98 = np.percentile(banda_sat, (2, 98))
    sat_gris = np.clip((banda_sat - p2) / (p98 - p2 + 1e-6), 0.0, 1.0)

    dem_2d = dem_norm.squeeze()
    d_min, d_max = dem_2d.min(), dem_2d.max()
    relieve = (dem_2d - d_min) / (d_max - d_min + 1e-6)

    base = np.clip(0.75 * sat_gris + 0.25 * relieve, 0.15, 0.88)
    return np.stack([base, base, base], axis=-1)


def renderizar_scar(fondo: np.ndarray, scar_2d: np.ndarray) -> np.ndarray:
    canvas = fondo.copy()
    mask = scar_2d > 0.5
    color_scar = np.array([0.98, 0.25, 0.09], dtype=np.float32)
    canvas[mask] = 0.25 * canvas[mask] + 0.75 * color_scar
    img_uint8 = np.clip(canvas * 255, 0, 255).astype(np.uint8)
    return cv2.resize(img_uint8, (350, 350), interpolation=cv2.INTER_NEAREST)


def renderizar_prediccion(
    fondo: np.ndarray,
    scar_2d: np.ndarray,
    prob_2d: np.ndarray,
    umbral: float,
    target_2d: np.ndarray | None = None,
) -> np.ndarray:
    canvas = fondo.copy()

    mask_scar = scar_2d > 0.5
    color_base = np.array([0.72, 0.11, 0.11], dtype=np.float32)
    canvas[mask_scar] = 0.30 * canvas[mask_scar] + 0.70 * color_base

    mask_pred = prob_2d > umbral
    if np.any(mask_pred):
        intensidad = np.clip((prob_2d - umbral) / (1.0 - umbral + 1e-6), 0.35, 1.0)
        r = np.ones_like(prob_2d)
        g = 0.95 - 0.35 * intensidad
        b = 0.05 * np.ones_like(prob_2d)
        color_pred = np.stack([r, g, b], axis=-1)
        canvas[mask_pred] = 0.15 * canvas[mask_pred] + 0.85 * color_pred[mask_pred]

    img_uint8 = np.clip(canvas * 255, 0, 255).astype(np.uint8)
    img_big = cv2.resize(img_uint8, (350, 350), interpolation=cv2.INTER_NEAREST)

    if target_2d is not None and np.any(target_2d > 0.5):
        gt_big = cv2.resize((target_2d > 0.5).astype(np.uint8), (350, 350), interpolation=cv2.INTER_NEAREST)
        contornos, _ = cv2.findContours(gt_big, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img_big, contornos, -1, (0, 255, 255), 2)

    return img_big


def preprocesar_olci(olci_raw: np.ndarray) -> np.ndarray:
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

    if h_usuario is None or float(h_usuario) <= 0:
        return None, None, "Error: Ingresa un horizonte temporal (H) mayor a 0 días (ej. 0.5, 1.0, 2.0, 5.0)."

    try:
        datos = np.load(archivo_npz.name)
    except Exception as e:
        return None, None, f"Error al leer el archivo .npz: {e}"

    requeridas = ["x_olci", "x_dem", "x_worldcover", "x_scar"]
    faltantes = [k for k in requeridas if k not in datos]
    if faltantes:
        return None, None, f"Error: El archivo .npz no contiene las claves esperadas: {faltantes}"

    olci = preprocesar_olci(datos["x_olci"])
    dem = datos["x_dem"].astype(np.float32) / 1000.0
    wc_raw = datos["x_worldcover"].astype(np.float32)
    valid_fuel = wc_raw > 0
    wc = np.where(valid_fuel, wc_raw / 100.0, 0.0)
    scar = datos["x_scar"].astype(np.float32)

    if dem.ndim == 2:
        dem = dem[None]
    if wc.ndim == 2:
        wc = wc[None]
    if scar.ndim == 2:
        scar = scar[None]

    h_final = float(h_usuario)
    h_archivo = float(datos["horizon_days"]) if "horizon_days" in datos else None

    with torch.no_grad():
        t_olci = torch.from_numpy(olci[None]).to(DEVICE)
        t_dem = torch.from_numpy(dem[None]).to(DEVICE)
        t_wc = torch.from_numpy(wc[None]).to(DEVICE)
        t_scar = torch.from_numpy(scar[None]).to(DEVICE)
        t_h = torch.tensor([h_final], dtype=torch.float32, device=DEVICE)

        logits = model(t_olci, t_dem, t_wc, t_scar, t_h)
        prob_quema = torch.sigmoid(logits).squeeze().cpu().numpy()

    prob_quema = np.where(valid_fuel, prob_quema, 0.0)
    target_2d = datos["target"].squeeze() if "target" in datos else None

    fondo = crear_fondo_terreno(olci, dem)
    img_scar = renderizar_scar(fondo, scar.squeeze())
    img_pred = renderizar_prediccion(fondo, scar.squeeze(), prob_quema, BEST_THRESHOLD, target_2d)

    pixeles_actuales = int(np.sum(scar.squeeze() > 0.5))
    pixeles_nuevos = int(np.sum(prob_quema > BEST_THRESHOLD))
    ha_actuales = pixeles_actuales * 9
    ha_nuevas = pixeles_nuevos * 9

    ref_h = f" (captura real del archivo: {h_archivo:.2f} d)" if h_archivo is not None else ""
    nota_target = ""
    if target_2d is not None:
        ha_reales = int(np.sum(target_2d > 0.5)) * 9
        nota_target = f"• Expansión real registrada (contorno cian): ~{ha_reales} ha\n"

    reporte = (
        f"REPORTE DE INFERENCIA REAL (Zona 19.2 × 19.2 km | 64×64 px)\n"
        f"-------------------------------------------------------------\n"
        f"• Horizonte evaluado (H): {h_final:.2f} días{ref_h}\n"
        f"• Umbral de decisión: {BEST_THRESHOLD}\n"
        f"• Cicatriz previa (rojo): {pixeles_actuales} px (~{ha_actuales} ha)\n"
        f"• Propagación predicha (amarillo): {pixeles_nuevos} px (~{ha_nuevas} ha)\n"
        f"{nota_target}"
        f"• Probabilidad máxima en escena: {prob_quema.max()*100:.1f}%"
    )

    return img_scar, img_pred, reporte


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
                info="Ingresa los días a futuro que quieres proyectar (ej. 0.5, 1, 2, 5).",
                value=1.0,
                minimum=0.25,
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
                    label="Cicatriz Actual (Rojo: Área quemada en t=0)",
                    height=300,
                    elem_classes=["panel-card"]
                )
                out_pred = gr.Image(
                    label="Propagación (Rojo: Base | Amarillo: Predicción | Cian: Real)",
                    height=300,
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
        fn=lambda: [None, 1.0, None, None, ""],
        inputs=[],
        outputs=[file_npz, num_h, out_scar, out_pred, out_status]
    )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Default(), css=custom_css)
