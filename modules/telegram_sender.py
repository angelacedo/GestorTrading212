import os
import io
import re
import asyncio
import logging
from datetime import datetime
from typing import List
from telegram import Bot
from telegram.constants import ParseMode
from telegram.error import TelegramError
from dotenv import load_dotenv

# Cargar variables de entorno del archivo .env
load_dotenv()

# Configuración del logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TelegramSender")

class TelegramSender:
    """
    Módulo para enviar las alertas y reportes completos directamente a un chat/canal 
    de Telegram a través de un Bot automatizado.
    """

    def __init__(self):
        """
        1. Conectarse a la API de Telegram usando el token guardado en .env
        """
        self.token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")

        if not self.token or not self.chat_id:
            logger.error("TELEGRAM_BOT_TOKEN o TELEGRAM_CHAT_ID no configurado en el archivo .env.")
            raise ValueError("Credenciales de Telegram incompletas.")

    def _split_message(self, text: str, max_length: int = 4000) -> List[str]:
        """
        2b. Dividir el mensaje si supera el límite de Telegram manteniendo la coherencia.
        Telegram limita cada mensaje a aproximadamente 4096 caracteres.
        """
        partes = []
        while len(text) > 0:
            if len(text) <= max_length:
                partes.append(text)
                break
                
            # Tratamos de cortar limpiamente por doble salto de línea (separación de párrafos)
            split_at = text.rfind("\n\n", 0, max_length)
            
            # Si no hay doble salto, probamos buscar un salto normal
            if split_at == -1:
                split_at = text.rfind("\n", 0, max_length)
                
                # En último recurso, lo cortamos por fuerza bruta
                if split_at == -1:
                    split_at = max_length
                    
            partes.append(text[:split_at])
            
            # Recortamos la parte ya guardada para iterar por lo restante
            text = text[split_at:].lstrip()
            
        return partes

    def _escape_markdown_v2(self, text: str) -> str:
        """
        Aplica un escape estricto de los caracteres reservados por Telegram MarkdownV2 
        según lo estipulado en su documentación oficial.
        """
        # Reemplazamos constructores clásicos de MD no soportados directamente por formato válido en TG
        text = re.sub(r'^(#+)\s*(.*?)$', r'*\2*', text, flags=re.MULTILINE)
        
        # Escapado estricto: Todo carácter reservado (salvo * y ` que queremos mantener de parte del LLM).
        # Lista de reservados a purgar con \\: _ [ ] ( ) ~ > # + - = | { } . !
        caracteres_reservados = ['_', '[', ']', '(', ')', '~', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']
        for c in caracteres_reservados:
            text = text.replace(c, f"\\{c}")
            
        return text

    async def send_report(self, markdown_content: str, date: datetime = None) -> bool:
        """
        2. Enviar el reporte formateado y adjunto al Chat_ID.
        Utilizamos 'async' porque python-telegram-bot v20+ es por defecto asíncrono.
        """
        if date is None:
            date = datetime.now()
            
        fecha_str = date.strftime("%Y-%m-%d")
        
        # 2a. Envío del reporte dividiendo si es muy largo
        partes_reporte = self._split_message(markdown_content)
        
        async with Bot(token=self.token) as bot:
            try:
                logger.info(f"Enviando reporte a Telegram a través de {len(partes_reporte)} mensaje(s)...")
                
                for idx, parte in enumerate(partes_reporte):
                    # 2c. Uso del formato MarkdownV2 de Telegram
                    texto_formateado = self._escape_markdown_v2(parte)
                    
                    try:
                        await bot.send_message(
                            chat_id=self.chat_id,
                            text=texto_formateado,
                            parse_mode=ParseMode.MARKDOWN_V2
                        )
                    except TelegramError as e:
                        logger.warning(f"Telegram rechazó el parseo MarkdownV2. Aplicando fallback a texto plano... ({e})")
                        # 4. Fallback de seguridad vital: Si el LLM inventó una sintaxis rara y el escape falla,
                        # se envía sin formateo para asegurar la entrega antes que la estética.
                        await bot.send_message(
                            chat_id=self.chat_id,
                            text=parte
                        )
                
                # 2d. Guardar como archivo temporal (en memoria) para enviarlo como doc adjunto
                # Así se cumple la entrega en chat y en archivo descargable
                archivo_memoria = io.BytesIO(markdown_content.encode('utf-8'))
                archivo_memoria.name = f"reporte-{fecha_str}.md"
                
                await bot.send_document(
                    chat_id=self.chat_id,
                    document=archivo_memoria,
                    caption=f"📎 Reporte completo - {fecha_str}",
                    parse_mode=None
                )
                
                logger.info("Reporte enviado con éxito a Telegram (Mensajes y Documento).")
                return True
                
            except Exception as e:
                logger.error(f"Error masivo al intentar contactar con Telegram: {str(e)}")
                return False

    async def send_error_alert(self, error_message: str) -> bool:
        """
        3. Enviar mensaje de alerta rápida si algún paso del main.py falla.
        """
        fecha_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        alerta = f"⚠️ *Alerta Crítica del Bot Financiero* ⚠️\n\n"
        alerta += f"Ha ocurrido un fallo irrecuperable en la ejecución de hoy \\({fecha_str}\\)\\.\n\n"
        alerta += f"*Detalle técnico:*\n`{self._escape_markdown_v2(error_message)}`"
        
        async with Bot(token=self.token) as bot:
            try:
                # Se envía como aviso urgente al canal
                await bot.send_message(
                    chat_id=self.chat_id,
                    text=alerta,
                    parse_mode=ParseMode.MARKDOWN_V2
                )
                logger.info("Alerta de error enviada por Telegram exitosamente.")
                return True
            except Exception as e:
                logger.error(f"Imposible enviar siquiera el mensaje de error por Telegram: {str(e)}")
                return False

# Pruebas asíncronas independientes del módulo
if __name__ == "__main__":
    import asyncio
    
    async def correr_pruebas():
        emisor = TelegramSender()
        
        print("--- PRUEBA DE ERROR ---")
        exito_error = await emisor.send_error_alert("Este es un error crítico de simulador. Testing API!")
        print(f"Estado de entrega de error: {exito_error}\n")
        
        print("--- PRUEBA DE REPORTE NORMAL E INYECCIÓN ADJUNTA ---")
        mock_reporte = "## 📊 Resumen del Mercado\nHoy es un día de tremendo impacto.\n\n## ⚠️ Alertas\nEl oro sube - Ten mucho cuidado!"
        exito_reporte = await emisor.send_report(mock_reporte)
        print(f"Estado de entrega final: {exito_reporte}")
        
    try:
        # Corremos el loop asíncrono en caso de ser ejecutado el fichero localmente
        asyncio.run(correr_pruebas())
    except Exception as e:
        logger.error(f"No se pudieron ejecutar las pruebas locales: {str(e)}")
