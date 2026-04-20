import os
import json
import logging
from typing import Dict, Any, List
import openai
from openai import OpenAI
from dotenv import load_dotenv

# Cargar variables de entorno desde el archivo .env
load_dotenv()

# Configuración del logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("LLMAnalyzer")

class LLMAnalyzer:
    """
    Motor central de inteligencia del proyecto.
    Utiliza un LLM hospedado en OpenRouter (por defecto Claude) para interpretar
    los datos cuantitativos (mercado/cartera) y cualitativos (noticias) y generar el reporte diario.
    """

    def __init__(self):
        """
        1. Conectarse a OpenRouter con la API key del .env.
        Instancia el cliente oficial de OpenAI adaptando la base_url hacia los servidores de OpenRouter.
        """
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        # El modelo puede configurarse vía .env, por defecto usamos claude-3-opus
        self.model = os.getenv("OPENROUTER_MODEL", "anthropic/claude-3-opus")
        
        if not self.api_key:
            logger.error("OPENROUTER_API_KEY no se encontró en las variables de entorno.")
            raise ValueError("Credenciales de OpenRouter insuficientes.")
            
        # Configuramos la base_url hacia OpenRouter
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
        )

    def analyze(self, portfolio_data: List[Dict[str, Any]], indicators_data: Dict[str, Any], news_data: List[Dict[str, str]], analyst_ratings_data: Dict[str, Any] = None, watchlist_data: List[str] = None) -> str:
        """
        2. Analiza los datos dict/list inyectados usando las directrices (System Prompt) 
        y devuelve una salida Markdown altamente formateada.
        """
        logger.info(f"Generando el análisis financiero vía IA usando el modelo: {self.model} ...")

        # Convertimos los diccionarios y listas a strings JSON formateados 
        # para que el LLM lo delimite y parseé inequívocamente en el Prompt
        portfolio_str = json.dumps(portfolio_data, indent=2, ensure_ascii=False)
        indicators_str = json.dumps(indicators_data, indent=2, ensure_ascii=False)
        news_str = json.dumps(news_data, indent=2, ensure_ascii=False)
        ratings_str = json.dumps(analyst_ratings_data or {}, indent=2, ensure_ascii=False)
        
        if watchlist_data is None:
            watchlist_str = "No definida. Infiere oportunidades relevantes basándote en el contexto macro de las noticias y en los sectores ya presentes en la cartera del usuario."
        else:
            watchlist_str = json.dumps(watchlist_data, indent=2, ensure_ascii=False)

        # a. Construcción del Prompt de Sistema definiendo el rol (Senior, objetivo, conservador)
        # c. Imposición estricta del formato Markdown solicitado en la salida
        # d. Inclusión estricta del disclaimer legal obligatorio por IA
        system_prompt = """Eres un Analista Financiero Senior de alto prestigio internacional. Tu perfil se caracteriza por ser extremadamente objetivo, analítico y conservador respecto al riesgo de volatilidad en los mercados.
Tu tarea es ingerir datos numéricos diarios, métricas técnicas que te pasaré, y el contexto macro/micro que te dan las noticias, cruzarlas con el estado real de la cartera del usuario, y redactar un informe de alto nivel.
Debes mantener siempre un tono profesional, sin preámbulos robóticos ni "Hola, soy tu IA de hoy". Entra directo al análisis y basa tus conclusiones exclusivamente en la data que proceses.

INSTRUCCIONES ANALÍTICAS ADICIONALES Y OBLIGATORIAS:
- Para cada posición de la cartera, debes mencionar obligatoriamente al menos un rating o precio objetivo de analista externo si está disponible en los datos proporcionados.
- Al analizar ETFs de materias primas (oro, plata, petróleo u otros), dedica un análisis propio y detallado, no los agrupes con renta variable. Incluye contexto macro de oferta/demanda y tendencia estructural del activo subyacente.
- Al analizar noticias, distingue entre eventos ocurridos hoy y contexto acumulado de los últimos 30 días. Indica explícitamente si una noticia es reciente (menos de 48h) o de contexto previo.
- El tono del análisis debe ser siempre objetivo y basado en datos. Evita afirmaciones especulativas sin respaldo en los datos proporcionados.

REQUISITO INQUEBRANTABLE FORMATO MD: Tu respuesta final DEBE utilizar EXACTAMENTE este esqueleto Markdown estructurado y sus respectivas cabeceras, sin omitir ni añadir ninguna otra sección de Título (##). Escribe el contenido dentro de cada bloque.

## 📊 Resumen del Mercado
## 💼 Estado de la Cartera
## 📰 Impacto de Noticias
## 📈 Predicciones a Corto Plazo (7 días)
## ⚠️ Alertas y Riesgos
## ✅ Recomendaciones del Día
## 🔭 Oportunidades de Mercado Externas

REGLAS DE FORMATO VISUAL OBLIGATORIAS (para máxima legibilidad):

1. ## 📊 Resumen del Mercado
   - Usa 3-4 bullets cortos (máx. 2 líneas cada uno) agrupados por tema: Macro, Geopolítica, Sectores, Materias Primas.
   - NO escribas bloques de texto seguido de más de 3 líneas sin salto.

2. ## 💼 Estado de la Cartera
   - Empieza SIEMPRE con este bloque de resumen rápido (rellena los valores reales):
     > 💰 **Valor total:** $X | 📈 **Beneficio acumulado:** +$X | 💵 **Liquidez:** $X
   - A continuación presenta TODAS las posiciones en una ÚNICA tabla Markdown con estas columnas EXACTAS:
     | Símbolo | P/L | RSI | Tendencia | Var. 30d | Rating | Señal |
     donde:
       - "Tendencia" = ↗ Alcista / ↘ Bajista / → Lateral (según SMA7 vs SMA30)
       - "Var. 30d" = el porcentaje de variación a 30 días con signo (ej. +12.6%)
       - "Rating" = rating de analista si existe, si no: —
       - "Señal" = 🟢 si P/L>0 y RSI<70 | 🟡 si RSI≥70 o posición neutral | 🔴 si P/L<0
   - Después de la tabla añade UNA subsección ### 🔍 Análisis Destacado con 2-4 bullets sobre las posiciones más relevantes (mayores P/L positivos/negativos, alertas RSI severas, anomalías). NO repitas cada posición individualmente.

3. ## 📰 Impacto de Noticias
   - Lista cada noticia relevante con: **[RECIENTE <48h]** o **[CONTEXTO]** + título en negrita + 1 línea de impacto en cartera.

4. ## 📈 Predicciones a Corto Plazo (7 días)
   - Usa una tabla con columnas: | Activo | Perspectiva | Catalizador principal |
   - Solo incluye posiciones con perspectiva diferenciada. No repitas todas las posiciones si su perspectiva es neutral.

5. ## ⚠️ Alertas y Riesgos
   - Lista con emojis de severidad: 🔴 Crítico | 🟠 Moderado | 🟡 Informativo
   - Máximo 6 alertas, ordenadas de mayor a menor severidad.

6. ## ✅ Recomendaciones del Día
   - Máximo 5 recomendaciones accionables, numeradas, en formato: **Acción** — justificación breve.

7. ## 🔭 Oportunidades de Mercado Externas
   - Tabla con columnas: | Señal | Activo / Ticker | Justificación |
   - donde Señal = 🟢 COMPRAR / 🟡 OBSERVAR / 🔴 EVITAR
   - Mínimo 3 filas, máximo 6.

***
*Disclaimer Legal: Este reporte ha sido generado automatizadamente por Inteligencia Artificial y posee un propósito exclusivamente informativo. De ninguna manera constituye un asesoramiento financiero fiduciario regulado, ni una recomendación oficial o determinante de inversión.*

INSTRUCCIÓN CRÍTICA DE OUTPUT: No incluyas ningún proceso de razonamiento,
planificación, borradores, ni metacomentarios sobre cómo vas a estructurar
la respuesta. Empieza a escribir directamente el contenido de la primera
sección ## 📊 Resumen del Mercado sin ningún preámbulo. El output debe ser
EXCLUSIVAMENTE el reporte final en Markdown, nada más."""

        # b. Inyección de todos los datos recibidos (Estructurados JSON vs string plano) en el User Prompt
        user_prompt = f"""Aquí tienes tu compilación del dataset diario para evaluación:

[ESTADO ACTUAL DE LA CARTERA TRADING 212]
{portfolio_str}

[MÉTRICAS E INDICADORES TÉCNICOS POR ACTIVO]
{indicators_str}

[RATINGS Y OBJETIVOS DE PRECIO DE ANALISTAS]
{ratings_str}

[NOTICIAS GLOBALES Y PUNTUALES DE LOS ÚLTIMOS 30 DÍAS]
{news_str}

[WATCHLIST EXTERNA - ACTIVOS FUERA DE CARTERA A EVALUAR]
{watchlist_str}

Basado única y exclusivamente en esto (junto a tu extenso conocimiento del entorno macroeconómico general consolidado hasta tu fecha de corte), redacta el informe íntegramente respetando las secciones exigidas. Inicia directamente desde la primera cabecera Markdown (## 📊 Resumen del Mercado)."""

        try:
            # 3. Llamada segura al provider manejando límites, con tokens acotados
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=6000, # Límite alto para dejar que fluya el reporte sin cortar
                temperature=0.3,  # Baja "creatividad". Favorece lógica analítica, predictibilidad y estilo conservador.
                extra_body={"reasoning": {"effort": "none"}}
            )
            
            report = response.choices[0].message.content
            logger.info("El reporte IA se ha redactado exitosamente.")
            return report

        # 3. Manejo explícito de Excepciones del Wrapper OpenAI (API Limit, Errores Red, etc)
        except openai.RateLimitError as e:
            logger.error("Límite de Cota/Tokens de OpenRouter excedido (RateLimitError).")
            return self._generar_error_markdown("Límite de peticiones o balance en OpenRouter excedido. Por favor, revisa tus créditos/suscripción.")
        
        except openai.APIError as e:
            logger.error(f"Fallo del servidor o endpoint de OpenRouter: {str(e)}")
            return self._generar_error_markdown(f"La API de OpenRouter respondió con un error de servicio: {e}")
            
        except openai.APIConnectionError as e:
            logger.error("No se pudo conectar con el endpoint de OpenRouter.")
            return self._generar_error_markdown("Fallo en la conexión de red limitando el contacto con los servidores de IA.")
            
        except Exception as e:
            logger.error(f"Fallo crítico e inesperado durante el análisis IA: {str(e)}")
            return self._generar_error_markdown(f"Error inesperado procesando la predicción: {str(e)}")

    def _generar_error_markdown(self, causa: str) -> str:
        """
        En vez de crashear el email entero, si el LLM falla, el email recibirá esta única alerta Markdown 
        para notificar al usuario final limpiamente que hubo un error con la Inteligencia artificial.
        """
        return f"## ⚠️ Error de Generación del Reporte (IA Fuera de Servicio)\n\nLamentablemente el análisis inteligente no pudo completarse el día de hoy.\n\n**Motivo devuelto por el sistema:**\n{causa}\n\n*Por favor, contacta con tu administrador o revisa los logs del servidor de OpenRouter.*"

# Entorno de pruebas
if __name__ == "__main__":
    try:
        ia_analista = LLMAnalyzer()
        
        # Simulación de datos parseados
        mock_portfolio = [{"simbolo": "AAPL", "cantidad": 5.0, "valor_actual": 850.0, "ganancia_perdida": 23.4}]
        mock_indicators = {"AAPL": {"sma_7": 172.5, "sma_30": 170.1, "rsi_14": 45.3, "variacion_30d_pct": 1.2}}
        mock_news = [{"titular": "Apple lanzará nuevo procesador centrado en su propio modelo de IA integrador", "resumen": "Reportes indican una renovación total del chip m4.", "fuente": "Bloomberg", "fecha": "Hace 4 horas"}]
        
        print("--- GENERANDO ANÁLISIS DE PRUEBA (Puede tardar entre 5 y 20 segundos) ---")
        reporte = ia_analista.analyze(mock_portfolio, mock_indicators, mock_news)
        print("\n" + "="*50 + "\nRESULTADO:\n" + "="*50)
        print(reporte)
        
    except Exception as e:
        logger.error(f"Error fatal probando módulo LLM: {str(e)}")
