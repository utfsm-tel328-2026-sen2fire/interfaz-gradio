import gradio as gr
import cv2
import numpy as np

def simular_deteccion(imagen):
    # Función dummy: Aplica un mapa de calor para simular áreas detectadas
    if imagen is None:
        return None
    imagen_calor = cv2.applyColorMap(imagen, cv2.COLORMAP_HOT)
    return imagen_calor

interfaz = gr.Interface(
    fn=simular_deteccion,
    inputs=gr.Image(label="Sube una imagen satelital o de dron"),
    outputs=gr.Image(label="Zonas de fuego/daño detectadas"),
    title="Detección de Incendios Forestales (Prototipo)",
    description="Sube una imagen para probar la interfaz. (Actualmente usa una función simulada)."
)

if __name__ == "__main__":
    interfaz.launch()