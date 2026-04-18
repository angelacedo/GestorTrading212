import os
import logging
from datetime import datetime
import markdown

# Configuración del logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ReportBuilder")

# Constante de versión del bot
BOT_VERSION = "v1.0.0"

class ReportBuilder:
    """
    Clase encargada de estructurar el análisis crudo en un reporte formal,
    guardarlo como archivo y prepararlo para su distribución por email.
    """

    def __init__(self):
        # Directorio raíz del proyecto (calculado de forma relativa a este módulo)
        # ../reports/
        self.base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'reports'))

    def build_report(self, analysis_markdown: str) -> dict:
        """
        1. Compone el archivo final y lo parsea.
           a. Añade la cabecera (Fecha, Hora, y "TRADE REPUBLIC")
           b. Inyecta el reporte de la IA
           c. Añade el pie de página con versión.
        3. Retorna un diccionario con la versión puramente Markdown y la versión renderizada a HTML.
        """
        logger.info("Construyendo el formato final del reporte y parseando a HTML...")
        
        ahora = datetime.now()
        fecha_str = ahora.strftime("%Y-%m-%d")
        hora_str = ahora.strftime("%H:%M:%S")

        # a. Cabecera fija
        # Nota: Has indicado TRADE REPUBLIC en tus instrucciones a pesar de que el resto
        # del código y el .env decían Trading 212. He acatado literalmente tu petición 📝.
        cabecera = f"# 📋 REPORTE DIARIO DE INVERSIÓN - TRADE REPUBLIC\n"
        cabecera += f"**Fecha:** {fecha_str} | **Hora de Análisis:** {hora_str}\n\n"
        cabecera += "---\n\n"

        # c. Pie de página fijo con metadatos del bot
        pie_pagina = f"\n\n---\n*Generado automáticamente de forma local por Bot Analista ({BOT_VERSION}) - Proceso finalizado a las {hora_str}*"

        # b. Ensamblaje: Cabecera + Cuerpo IA + Pie
        md_content = cabecera + analysis_markdown + pie_pagina

        # 3. Conversión a HTML
        # Utilizamos la librería 'markdown' e incluimos la extensión 'tables' (útil cuando el LLM las usa)
        html_content = markdown.markdown(md_content, extensions=['tables', 'fenced_code'])

        return {
            "markdown": md_content,
            "html": html_content,
            "fecha_objeto": ahora # Útil para que la función save_report no recalcule la hora
        }

    def save_report(self, md_content: str, date: datetime = None) -> str:
        """
        2. Guarda el contenido del reporte en la estructura de directorios:
           /reports/YYYY-MM/reporte-YYYY-MM-DD.md
        """
        if date is None:
            date = datetime.now()
            
        yyyy_mm = date.strftime("%Y-%m")
        yyyy_mm_dd = date.strftime("%Y-%m-%d")
        
        # Generar ruta del directorio y asegurar que exista
        target_dir = os.path.join(self.base_dir, yyyy_mm)
        os.makedirs(target_dir, exist_ok=True)
        
        # Formato de archivo especificado
        file_path = os.path.join(target_dir, f"reporte-{yyyy_mm_dd}.md")
        
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(md_content)
            logger.info(f"Reporte archivado localmente con éxito en: {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"Fallo al intentar escribir el archivo MD en disco: {str(e)}")
            return ""

# Código para testear este módulo de forma independiente
if __name__ == "__main__":
    try:
        builder = ReportBuilder()
        
        # Fake input del motor de LLM
        mock_analysis = "## 📈 Predicciones a Corto Plazo (7 días)\nTodo apunta a un alza en el sector tecnológico debido a las métricas del Q3.\n\n## ⚠️ Alertas y Riesgos\nLa macroeconomía actual sigue presentando niveles notables de volatilidad."
        
        # Test construir el reporte
        resultados = builder.build_report(mock_analysis)
        
        print("--- RESULTADO MARKDOWN ---")
        print(resultados["markdown"])
        
        print("\n--- RESULTADO HTML (Para correo email) ---")
        print(resultados["html"])
        
        # Test de guardado
        print("\n--- ARCHIVADO ---")
        ruta = builder.save_report(resultados["markdown"], resultados["fecha_objeto"])
        
    except Exception as e:
        logger.error(f"Fallo en prueba de creación de reporte: {str(e)}")
