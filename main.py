import os
import sys
import asyncio
import logging
import traceback
from datetime import datetime
from dotenv import load_dotenv

# 2. Cargar las variables de entorno con python-dotenv al inicio
load_dotenv()

# Configuración del registro de terminal (consola) con timestamps explícitos
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("Orquestador")

# 1. Importar TODOS los módulos creados
try:
    from modules.t212_client import T212Client
    from modules.finnhub_client import FinnhubClient
    from modules.news_client import NewsClient
    from modules.llm_analyzer import LLMAnalyzer
    from modules.report_builder import ReportBuilder
    from modules.telegram_sender import TelegramSender
    from modules.symbol_resolver import SymbolResolver
except ImportError as imp_err:
    logger.critical(f"Error fatal: No se pudieron importar los submódulos locales. ¿Aseguraste ejecutar el script desde el root? ({imp_err})")
    sys.exit(1)

async def main():
    """
    3. Función principal orquestadora. Toma el control lineal del proceso de inicio a fin.
    Se declara asíncrona porque la interacción principal con Telegram requiere I/O asíncrona.
    """
    logger.info("=" * 60)
    logger.info("🚀 INICIANDO EJECUCIÓN DEL ORQUESTADOR FINANCIERO")
    logger.info("=" * 60)

    # 4. Manejo global de tracking de excepciones ("silenciosas") de cada paso
    errores_ocurridos = []

    def log_error(contexto: str, error: Exception):
        """Función auxiliar que encapsula el error, lo añade a la pila para no matar el flujo."""
        msg = f"Excepción en '{contexto}': {str(error)}"
        logger.error(msg)
        errores_ocurridos.append(msg)

    # Declaración previa de telemetría de fallos
    telegram_bot = None

    try:
        logger.info("[PASO 0] Autentificando e instanciando clientes API de la arquitectura...")
        telegram_bot = TelegramSender()
        t212_client = T212Client()
        market_client = FinnhubClient() # Basado en yfinance como parche de seguridad a Finnhub
        news_client = NewsClient()
        llm_analyzer = LLMAnalyzer()
        builder = ReportBuilder()
        resolver = SymbolResolver()

        # ---------------------------------------------------------
        # a. Obtener cartera y resumen de cuenta de Trading 212
        # ---------------------------------------------------------
        logger.info("[PASO 1] Conectando a Trading 212: Descarga de estado de la cuenta...")
        try:
            account_summary = t212_client.get_account_summary()
            portfolio = t212_client.get_portfolio()

            # Encapsulamos ambos para que el LLM reciba una visión estática completa
            portfolio_data = {
                "resumen_general_cuenta": account_summary,
                "posiciones_abiertas": portfolio
            }
        except Exception as e:
            raise RuntimeError(f"Fallo vital irrecuperable con el Broker Trading 212: {str(e)}")

        if not portfolio:
            logger.warning("No hay posiciones abiertas registradas en el broker actual.")

        logger.info("[PASO 1.5] Truducción cruzada inteligente de Símbolos Trading 212 vía LLM...")
        lista_simbolos_raw = [p.get("simbolo") for p in portfolio]
        dicc_traducciones = resolver.resolve_symbols_batch(lista_simbolos_raw)

        # ---------------------------------------------------------
        # b. Datos históricos y  c. Noticias específicas + generales
        # ---------------------------------------------------------
        logger.info("[PASO 2] Web Scraping & Finanzas: Recopilando inteligencia de mercado...")

        indicators_data = {}
        news_data = []
        analyst_ratings_data = {}

        # Primero capturamos el entorno Macro de la economía pura mundial
        try:
            macro_news = news_client.get_market_news(max_articles=6)
            if macro_news:
                news_data.extend(macro_news)
        except Exception as e:
            log_error("Módulo de Noticias Macro del Mercado", e)

        # 3b y 3c: Ciclo (For Each) por posición accionaria
        logger.info("         -> Analizando individualmente tus activos de la cartera...")

        # Limite opcional sugerido en la iteración por si la cartera es inmensa
        # Aquí tomamos toda la cartera pero podrías acotarla: portfolio[:15]
        for posicion in portfolio:
            simbolo = posicion.get("simbolo")
            if not simbolo or simbolo == "Desconocido" or simbolo == "Cash":
                continue

            # Traducción dinámica de símbolos propietario a global vía caché LLM
            simbolo_traducido = dicc_traducciones.get(simbolo, simbolo.replace("_EQ", "").split("_")[0].upper())
            # Base del símbolo para noticias (extrae el ticker raíz sin extensión de mercado)
            simbolo_base = simbolo_traducido.split(".")[0].split("_")[0]

            logger.info(f"            - Procesando telemetría para: {simbolo}")

            # Obtener indicadores diarios para análisis métrico (b)
            try:
                indicadores = market_client.get_market_indicators(simbolo_traducido)
                if indicadores:
                    indicators_data[simbolo] = indicadores
            except Exception as e:
                log_error(f"Indicadores de {simbolo}", e)

            # Extraer titulares dedicados de esa compañía particular (c)
            try:
                simbolo_news = news_client.get_news_for_symbol(simbolo_base, max_articles=2)
                if simbolo_news:
                    news_data.extend(simbolo_news)
            except Exception as e:
                log_error(f"Noticias dedicadas de {simbolo}", e)

            # Obtener dictamen de firmas analistas (Mejora 1)
            try:
                ratings = news_client.get_analyst_ratings(simbolo_traducido)
                if ratings:
                    analyst_ratings_data[simbolo] = ratings
            except Exception as e:
                log_error(f"Ratings de analistas de {simbolo}", e)

            # Identificación heurística de ETFs de Raw Materials para noticias de sector amplio (Mejora 3)
            # Ej: CS1l_EQ (Crude Oil), PHAG_EQ (Silver), GLD/IGLN/SGLN (Gold)
            if simbolo.endswith("_EQ") and any(sub in simbolo.upper() for sub in ["GLD", "PHAG", "CS1", "IGLN", "SGLN", "OIL", "CMD"]):
                try:
                    sector_news = news_client.get_sector_news("commodities OR precious metals", max_articles=2)
                    if sector_news:
                        news_data.extend(sector_news)
                except Exception as e:
                    log_error(f"Noticias de sector vital para el ETF {simbolo}", e)

        # ---------------------------------------------------------
        # d. Enviar todo al LLM de Claude para el análisis integral
        # ---------------------------------------------------------
        logger.info("[PASO 3] Pensamiento Artificial O1: Lanzando inyección al LLM de OpenRouter...")
        try:
            raw_analysis = llm_analyzer.analyze(portfolio_data, indicators_data, news_data, analyst_ratings_data)
        except Exception as e:
            raise RuntimeError(f"El LLM falló por completo y el reporte inteligente no pudo fluir: {str(e)}")

        # ---------------------------------------------------------
        # e. Construir el reporte con ReportBuilder
        # f. Guardar el reporte físicamente en /reports/YYYY-MM/
        # ---------------------------------------------------------
        logger.info("[PASO 4] Construcción Táctica: Ensamblando archivo final y asegurando Write/IO...")
        try:
            estructuras = builder.build_report(raw_analysis)
            ruta_reporte = builder.save_report(estructuras["markdown"], date=estructuras["fecha_objeto"])
            if not ruta_reporte:
                log_error("Escritura local", Exception("La ruta o permisos fallaron, se emitió vacío."))
        except Exception as e:
            log_error("Ensamblaje y Formateo ReportBuilder", e)
            # Salvavidas para no sacrificar el envío si el builder falla
            estructuras = {"markdown": raw_analysis}

        # ---------------------------------------------------------
        # g. Despachar el reporte finalmente al Telegram Personal Vía API del Bot
        # ---------------------------------------------------------
        logger.info("[PASO 5] Operaciones de Despliegue: Contactando con la central de Telegram...")
        try:
            await telegram_bot.send_report(estructuras.get("markdown", raw_analysis))
        except Exception as e:
            # Falla terminal, Telegram es el canal de entrega, de nada sirvió construir todo
            raise RuntimeError(f"Fallo terminal en emisor Telegram: {str(e)}")

        # ---------------------------------------------------------
        # 5. Volcado final del resumen de la ejecución al humano / consola
        # ---------------------------------------------------------
        logger.info("=" * 60)
        if errores_ocurridos:
            logger.warning("🏁 EJECUCIÓN DEL CRON FINALIZADA. Hubo ADVERTENCIAS secundarias:")
            for e in errores_ocurridos:
                logger.warning(f"     -> {e}")
        else:
            logger.info("🏁 EJECUCIÓN FINALIZADA. ÉXITO ABSOLUTO AL 100% (0 colisiones).")
        logger.info("=" * 60)

    # 4. Captura del colapso maestro de la aplicación entera y ejecución de salvamento
    except Exception as fatal_err:
        logger.critical(f"🛑 CRASH TOTAL EN EL MOTOR: {str(fatal_err)}")
        logger.critical("Iniciando secuencia SOS automatizada hacia tu panel de Telegram...")

        # Recuperamos la traza (traceback) para imprimir dónde falló exactamente nuestro código de python
        traza = traceback.format_exc()

        if telegram_bot:
            try:
                # Limitamos los caracteres a los primeros 800 de la trace para evitar un saturamiento de API
                alerta_msj = f"{str(fatal_err)}\n\nTraceback:\n{traza[:800]}..."
                await telegram_bot.send_error_alert(alerta_msj)
                logger.info("Alerta enviada para advertir al administrador remotamente.")
            except:
                logger.critical("El bot no cuenta con internet o conectividad para Telegram. Fallo silencioso final.")

        logger.error("EJECUCIÓN DEL ESCRIPT ABORTADA INCONCLUSAMENTE.")

# Punto principal de arranque
if __name__ == "__main__":
    # Necesario arrancar con run() para gestionar todas las coroutinas generadas
    # por python-telegram-bot dentro de nuestro algoritmo lineal
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.warning("\n[INTERRUPCIÓN MANUAL] - Se ha cancelado la señal de proceso.")
