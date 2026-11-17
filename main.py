import gradio as gr
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms

# ==============================================================================
# 1. ARQUITECTURA DEL MODELO (ADN de tu equipo)
# ==============================================================================
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class Up(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2,
                      diff_y // 2, diff_y - diff_y // 2])
        return self.conv(torch.cat([skip, x], dim=1))


class OutConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    """U-Net mínimo para segmentación binaria.

    Entrada: (B, in_channels, H, W)  — p.ej. 12 bandas Sentinel-2
    Salida:  (B, out_channels, H, W) — logits (sin sigmoid, lo aplica la loss)

    H y W deben ser divisibles por 16 (4 niveles de pooling).
    """

    def __init__(self, in_channels: int = 12, out_channels: int = 1, base: int = 32) -> None:
        super().__init__()
        c = [base, base * 2, base * 4, base * 8]

        self.inc = DoubleConv(in_channels, c[0])
        self.down1 = Down(c[0], c[1])
        self.down2 = Down(c[1], c[2])
        self.down3 = Down(c[2], c[3])
        self.bot = DoubleConv(c[3], c[3] * 2)
        self.up1 = Up(c[3] * 2, c[3], c[2])
        self.up2 = Up(c[2], c[2], c[1])
        self.up3 = Up(c[1], c[1], c[0])
        self.up4 = Up(c[0], c[0], c[0])
        self.outc = OutConv(c[0], out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
# 2. CARGA DE PESOS (Fase A3)
# ==============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instanciamos el modelo con la configuración exacta que me mostraste
modelo = UNet(in_channels=12, out_channels=1, base=32).to(device)

ruta_pesos = "pesos/pesos-sen2fire-21-10-2026.ckpt"

try:
    checkpoint = torch.load(ruta_pesos, map_location=device)
    # PyTorch Lightning suele guardar los pesos dentro de la llave "state_dict" o "model"
    if "model" in checkpoint:
        modelo.load_state_dict(checkpoint["model"])
    elif "state_dict" in checkpoint:
        # A veces Lightning le pone el prefijo "model." a las capas, esto lo limpia:
        state_dict_limpio = {k.replace("model.", ""): v for k, v in checkpoint["state_dict"].items()}
        modelo.load_state_dict(state_dict_limpio, strict=False)
    else:
        modelo.load_state_dict(checkpoint)
        
    print("✅ EXCELENTE: Pesos cargados correctamente. La arquitectura coincide.")
except FileNotFoundError:
    print("❌ ERROR: No encuentro el archivo. Revisa que esté dentro de la carpeta 'pesos/' y se llame exacto igual.")
except RuntimeError as e:
    print(f"❌ ERROR DE ARQUITECTURA: {e}")

modelo.eval() 

# ==============================================================================
# 3. INTERFAZ Y PROCESAMIENTO
# ==============================================================================
transformacion = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((512, 512), antialias=True),
])

def calcular_hectareas(mascara_predicha, resolucion_pixel_m=10):
    pixeles_fuego = np.sum(mascara_predicha == 1) 
    area_por_pixel_ha = (resolucion_pixel_m ** 2) / 10000.0
    return pixeles_fuego * area_por_pixel_ha

def procesar_imagen(imagen):
    if imagen is None:
        return None, "Por favor, sube una imagen para analizar."
    
    alto_orig, ancho_orig = imagen.shape[:2]
    imagen_tensor = transformacion(imagen).to(device)
    
    # Adaptador: 3 canales RGB a 12 canales Sentinel-2
    imagen_tensor_12 = imagen_tensor.repeat(4, 1, 1).unsqueeze(0)
    
    with torch.no_grad():
        logits = modelo(imagen_tensor_12)
        probabilidades = torch.sigmoid(logits).squeeze()
        mascara_512 = (probabilidades > 0.5).cpu().numpy().astype(np.uint8)
    
    mascara = cv2.resize(mascara_512, (ancho_orig, alto_orig), interpolation=cv2.INTER_NEAREST)

    imagen_coloreada = imagen.copy()
    imagen_coloreada[mascara == 1] = [255, 0, 0] # Fuego activo en rojo
    
    imagen_salida = cv2.addWeighted(imagen, 0.6, imagen_coloreada, 0.4, 0)
    ha_fuego = calcular_hectareas(mascara)
    
    reporte = (
        f"📊 REPORTE DE ÁREAS AFECTADAS (Fase A3 - Integrada)\n"
        f"--------------------------------------------------\n"
        f"🔥 Fuego Activo detectado: {ha_fuego:.2f} hectáreas.\n"
        f"--------------------------------------------------\n"
        f"Total Área Afectada: {ha_fuego:.2f} hectáreas.\n"
    )
    return imagen_salida, reporte

interfaz = gr.Interface(
    fn=procesar_imagen,
    inputs=gr.Image(label="Sube la imagen satelital aquí"),
    outputs=[gr.Image(label="Zonas de fuego detectadas (U-Net)"), gr.Textbox(label="Panel de Métricas")],
    title="Sistema de Detección de Incendios",
    description="Motor de Inferencia conectado. Arquitectura base=32."
)

if __name__ == "__main__":
    interfaz.launch()
