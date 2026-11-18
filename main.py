import gradio as gr
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

# ==============================================================================
# 1. ARQUITECTURA MODELO 1: DETECCIÓN ESTÁTICA (U-Net)
# ==============================================================================
class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, mid_ch: int = None) -> None:
        super().__init__()
        if not mid_ch: mid_ch = out_ch
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch), nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x): return self.conv(self.pool(x))

class Up(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, skip_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv(skip_ch * 2, out_ch, skip_ch)
    def forward(self, x, skip):
        x = self.up(x)
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        return self.conv(torch.cat([skip, x], dim=1))

class OutConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)
    def forward(self, x): return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels: int = 12, n_classes: int = 2) -> None:
        super().__init__()
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.bot = DoubleConv(512, 1024)
        self.up1 = Up(1024, 512, 256)
        self.up2 = Up(256, 256, 128)
        self.up3 = Up(128, 128, 64)
        self.up4 = Up(64, 64, 64)
        self.outc = OutConv(64, n_classes)
    def forward(self, x):
        s1 = self.inc(x)
        s2 = self.down1(s1)
        s3 = self.down2(s2)
        s4 = self.down3(s3)
        b = self.bot(s4)
        x = self.up1(b, s4)
        x = self.up2(x, s3)
        x = self.up3(x, s2)
        x = self.up4(x, s1)
        return self.outc(x)

# ==============================================================================
# 2. CARGA DE PESOS (MODELO 1)
# ==============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modelo_estatico = UNet(in_channels=12, n_classes=2)
ruta_ckpt = "pesos/pesos-sen2fire-21-10-2026.ckpt"

try:
    checkpoint = torch.load(ruta_ckpt, map_location=device, weights_only=False)
    state = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
    esperados = modelo_estatico.state_dict()
    primera_clave = next(iter(esperados))
    prefijos = [clave[:-len(primera_clave)] for clave in state if clave.endswith(primera_clave)]

    for prefijo in prefijos:
        if all(prefijo + k in state and state[prefijo + k].shape == valor.shape for k, valor in esperados.items()):
            pesos_unet = {k: state[prefijo + k] for k in esperados}
            modelo_estatico.load_state_dict(pesos_unet, strict=True)
            modelo_estatico.to(device).eval()
            print("✅ EXCELENTE: Pesos U-Net cargados correctamente.")
            break
except FileNotFoundError:
    print("❌ ERROR: No encuentro el archivo 'pesos/pesos-sen2fire-21-10-2026.ckpt'.")

# ==============================================================================
# 3. FUNCIONES DE PROCESAMIENTO
# ==============================================================================
transformacion = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((512, 512), antialias=True),
])

def calcular_hectareas(mascara_predicha, resolucion_pixel_m=10):
    pixeles_fuego = np.sum(mascara_predicha == 1) 
    return pixeles_fuego * ((resolucion_pixel_m ** 2) / 10000.0)

def procesar_estatico(imagen):
    """Lógica del Modelo 1 (U-Net). Funciona con el smoke test RGB adaptado a 12 canales."""
    if imagen is None: return None, "Sube una imagen."
    alto_orig, ancho_orig = imagen.shape[:2]
    imagen_tensor = transformacion(imagen).to(device)
    imagen_tensor_12 = imagen_tensor.repeat(4, 1, 1).unsqueeze(0)
    
    with torch.inference_mode():
        logits = modelo_estatico(imagen_tensor_12)
        prob_fuego = torch.softmax(logits, dim=1)[0, 1].cpu().numpy()
        mascara_512 = (prob_fuego >= 0.5).astype(np.uint8)
    
    mascara = cv2.resize(mascara_512, (ancho_orig, alto_orig), interpolation=cv2.INTER_NEAREST)
    imagen_coloreada = imagen.copy()
    capa_roja = np.zeros_like(imagen_coloreada, dtype=np.uint8)
    capa_roja[mascara == 1] = [255, 0, 0]
    imagen_salida = cv2.addWeighted(imagen_coloreada, 1.0, capa_roja, 0.65, 0)
    
    ha_fuego = calcular_hectareas(mascara)
    reporte = f"📊 Fuego Activo detectado: {ha_fuego:.2f} hectáreas.\n"
    return imagen_salida, reporte

def procesar_predictivo(archivo_npz, dias_h):
    """Lógica del Modelo 2 (Simulador). En construcción."""
    if archivo_npz is None: return "Sube el paquete de datos (.npz)"
    
    # Aquí irá la lógica de extracción de OLCI, DEM, WC y generación del tensor SCAR
    # Aquí irá la inferencia del nuevo modelo con la entrada (5)
    
    mensaje_temporal = (
        f"🚧 EN CONSTRUCCIÓN 🚧\n"
        f"Recibido horizonte predictivo: {dias_h} días.\n"
        f"Falta inyectar arquitecturas TMB-S3 y Predictiva."
    )
    return mensaje_temporal

# ==============================================================================
# 4. DASHBOARD INTERFAZ GRÁFICA (Gradio Blocks)
# ==============================================================================
with gr.Blocks(title="Sistema Integral de Incendios", theme=gr.themes.Soft()) as plataforma:
    gr.Markdown("# 🛰️ Plataforma de Monitoreo y Predicción de Incendios")
    gr.Markdown("Selecciona la herramienta de análisis en las pestañas inferiores.")
    
    with gr.Tabs():
        # --- PESTAÑA 1: DETECCIÓN ESTÁTICA ---
        with gr.TabItem("Detección Inmediata (Actual)"):
            with gr.Row():
                with gr.Column():
                    entrada_estatica = gr.Image(label="Subir captura visible / infrarroja")
                    boton_detectar = gr.Button("Analizar Fuego Actual", variant="primary")
                with gr.Column():
                    salida_imagen = gr.Image(label="Zonas de fuego detectadas")
                    salida_texto = gr.Textbox(label="Panel de Métricas")
            
            boton_detectar.click(
                fn=procesar_estatico, 
                inputs=entrada_estatica, 
                outputs=[salida_imagen, salida_texto]
            )
            
        # --- PESTAÑA 2: SIMULADOR PREDICTIVO ---
        with gr.TabItem("Simulación Predictiva (Futuro)"):
            with gr.Row():
                with gr.Column():
                    entrada_datos = gr.File(label="Paquete de Escenario (OLCI, DEM, WC)")
                    entrada_dias = gr.Slider(minimum=0.5, maximum=14.0, step=0.5, value=1.0, label="Días a predecir (H)")
                    boton_simular = gr.Button("Proyectar Incendio", variant="primary")
                with gr.Column():
                    salida_prediccion = gr.Textbox(label="Resultados de Predicción", lines=5)
            
            boton_simular.click(
                fn=procesar_predictivo, 
                inputs=[entrada_datos, entrada_dias], 
                outputs=salida_prediccion
            )

if __name__ == "__main__":
    plataforma.launch()
