import gradio as gr

# =========================================================
# import cv2          # Se usará para aplicar mapas de color y procesar la salida visual del modelo.
# import numpy as np  # Se usará para operaciones de matrices y el conteo de píxeles por hectárea.
# =========================================================


# --- BLOQUE 1: CÁLCULOS MATEMÁTICOS (En espera) ---
# Esta función está lista en estructura. 
# Recibirá la máscara real cuando el modelo U-Net esté integrado.
def calcular_hectareas_quemadas(mascara_predicha, resolucion_pixel_m=10):
    # TODO: Implementar conteo de píxeles usando numpy (ej: np.sum) 
    # una vez que se reciba la matriz del modelo.
    pass 


# --- BLOQUE 2: PROCESAMIENTO E INTEGRACIÓN (Plantilla) ---
# Recibe la imagen y muestra la estructura de cómo se entregarán los resultados.
def procesar_imagen(imagen):
    if imagen is None:
        return None, "Por favor, sube una imagen para analizar."
    
    # Placeholder: Como aún no hay IA, devolvemos la misma imagen en la salida
    # para demostrar que el flujo de la interfaz funciona.
    imagen_salida = imagen 
    
    # Estructura definitiva del reporte. 
    # Los valores están en espera de los datos del Sub-equipo 1.
    reporte = (
        f" REPORTE DE ÁREAS AFECTADAS (Plantilla Base)\n"
        f"--------------------------------------------------\n"
        f" Fuego Activo detectado: [Esperando modelo U-Net] hectáreas.\n"
        f" Cicatriz de quemadura:  [Esperando modelo U-Net] hectáreas.\n"
        f"--------------------------------------------------\n"
        f"Total Área Afectada: [Esperando modelo U-Net] hectáreas.\n\n"
        f"*Nota: Interfaz gráfica y estructura matemática finalizadas.\n"
        f"A la espera de integración con la red neuronal (Actividad A3)."
    )
    
    return imagen_salida, reporte


# --- BLOQUE 3: INTERFAZ DE USUARIO ---
# Configura y lanza la aplicación web estableciendo las zonas de entrada y salida.
interfaz = gr.Interface(
    fn=procesar_imagen,
    inputs=gr.Image(label="Sube la imagen a analizar"),
    outputs=[
        gr.Image(label="Zonas de fuego detectadas (Espacio reservado)"),
        gr.Textbox(label="Panel de Métricas y Análisis", lines=8)
    ],
    title="Sistema de Detección de Incendios - Interfaz Base",
    description="Estructura de la interfaz. Preparada para conectarse con los resultados del modelo U-Net."
)

if __name__ == "__main__":
    interfaz.launch()
