import os
import requests
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any
from dotenv import load_dotenv

# Cargar variables de entorno desde el archivo .env
load_dotenv()

# Configuración del logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("NewsClient")

class NewsClient:
    """
    Cliente para interactuar con la API de NewsData.io y extraer información 
    relevante del mercado de forma resumida para el LLM.
    """

    def __init__(self):
        """
        1. Conectarse a NewsData.io usando la API key del .env.
        Inicializamos identificando nuestros dominios de prensa oficial preferidos.
        """
        self.api_key = os.getenv("NEWSDATA_API_KEY")
        self.finnhub_api_key = os.getenv("FINNHUB_API_KEY")
        if not self.api_key:
            logger.error("NEWSDATA_API_KEY no se encontró en las variables de entorno.")
            raise ValueError("Credenciales de Newsdata incompletas.")
            
        self.base_url = "https://newsdata.io/api/1/news"
        # Lista de dominios recomendados (fuentes financieras oficiales)
        self.fuentes_financieras = "reuters,bloomberg,ft,cnbc,marketwatch,wsj,forbes,yahoofinance"

    def _clean_ticker(self, ticker: str) -> str:
        """
        Normaliza símbolos generados por brokers para la búsqueda limpiando los sufijos.
        Ej: AAPL_US_EQ -> AAPL
        """
        return ticker.split('_')[0]

    def _procesar_y_filtrar_noticias(self, results: List[Dict], max_articles: int) -> List[Dict[str, str]]:
        """
        5. Filtrar noticias duplicadas o sin resumen.
        También procesa de forma estricta que la noticia no tenga más de 24h
        para no depender de atributos "premium" o restricciones de la API.
        """
        noticias_procesadas = []
        titulares_vistos = set()
        
        # Determinamos el timestamp de los últimos 30 días
        limite_30d = datetime.now() - timedelta(days=30)

        for article in results:
            titulo = article.get("title", "")
            if not titulo:
                continue
            titulo = titulo.strip()
            
            # Cogemos la descripción y como fallback el 'content'
            resumen = article.get("description") or article.get("content", "")
            resumen = str(resumen).strip()
            
            # Descartamos si no hay resumen real, o el titular ya fue capturado (duplicado)
            if not resumen or titulo in titulares_vistos or resumen.lower() == "none" or len(resumen) < 10:
                continue
                
            # Filtro manual de fechas: asegurándonos que son de las últimas 24h
            pub_date_str = article.get("pubDate", "")
            if pub_date_str:
                try:
                    # El formato común es "YYYY-MM-DD HH:MM:SS"
                    pub_date = datetime.strptime(pub_date_str, "%Y-%m-%d %H:%M:%S")
                    if pub_date < limite_30d:
                        continue  # La noticia es antigua, saltamos a la siguiente
                except ValueError:
                    # En caso de formatos irregulares en NewsData, lo dejamos pasar
                    pass 

            # Registramos como procesado
            titulares_vistos.add(titulo)
            
            # 3. Truncar el resumen a un máximo de 200 caracteres
            if len(resumen) > 200:
                resumen = resumen[:197] + "..."
                
            # 3. Construimos el diccionario de salida
            noticias_procesadas.append({
                "titular": titulo,
                "resumen": resumen,
                "fuente": article.get("source_id", "Medio Financiero"),
                "fecha": pub_date_str
            })
            
            # Finalizamos si hemos alcanzado nuestro máximo solicitado
            if len(noticias_procesadas) >= max_articles:
                break
                
        return noticias_procesadas

    def get_news_for_symbol(self, symbol: str, max_articles: int = 5) -> List[Dict[str, str]]:
        """
        2. Busca noticias recientes para un símbolo bursátil específico.
        """
        clean_symbol = self._clean_ticker(symbol)
        logger.info(f"Buscando noticias recientes (últimos 30 días) para {clean_symbol}...")
        
        # Filtramos por el símbolo (q), en la categoría de negocios (business) en In/Es
        params = {
            "apikey": self.api_key,
            "q": clean_symbol,
            "language": "en,es",
            "category": "business"
            # Omitimos 'domain' aquí porque a nivel de empresas específicas
            # Reuters o CNBC pueden no cubrirlas todas en las últimas 24 horas,
            # pero la categoría asegura que son portales financieros.
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data.get("status") == "error":
                logger.error(f"Error devuelto por NewsData: {data.get('results', {}).get('message', 'Desconocido')}")
                return []

            results = data.get("results", [])
            return self._procesar_y_filtrar_noticias(results, max_articles)
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error de red al buscar noticias de {clean_symbol}: {str(e)}")
            return []

    def get_market_news(self, max_articles: int = 10) -> List[Dict[str, str]]:
        """
        4. Obtiene las noticias generales más relevantes del mercado global actual.
        """
        logger.info("Buscando las noticias globales más relevantes del mercado...")
        
        # Aquí sí aplicamos el filtro por nuestras fuentes de élite para el contexto del mercado
        params = {
            "apikey": self.api_key,
            "language": "en,es",
            "category": "business,top",
            "domain": self.fuentes_financieras
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data.get("status") == "error":
                logger.error(f"Error devuelto por NewsData: {data.get('results', {}).get('message', 'Desconocido')}")
                return []

            results = data.get("results", [])
            return self._procesar_y_filtrar_noticias(results, max_articles)
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error de red al buscar noticias globales: {str(e)}")
            return []

    def get_analyst_ratings(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Consulta la API de Finnhub para obtener ratings de analistas externos y objetivos de precio.
        Endpoints: /stock/recommendation y /stock/price-target
        """
        clean_symbol = self._clean_ticker(symbol)
        if not getattr(self, 'finnhub_api_key', None):
            logger.warning("FINNHUB_API_KEY ausente. Omitiendo ratings de analistas.")
            return []
            
        logger.info(f"Obteniendo ratings de analistas y precio para {clean_symbol}...")
        try:
            params = {"symbol": clean_symbol, "token": self.finnhub_api_key}
            
            # Obtener recomendación grupal
            res_rec = requests.get("https://finnhub.io/api/v1/stock/recommendation", params=params, timeout=5)
            data_rec = res_rec.json() if res_rec.ok else []
            
            # Obtener precio objetivo
            res_tgt = requests.get("https://finnhub.io/api/v1/stock/price-target", params=params, timeout=5)
            data_tgt = res_tgt.json() if res_tgt.ok else {}
            
            ratings = []
            if isinstance(data_rec, list) and len(data_rec) > 0:
                latest = data_rec[0]
                
                votos = {
                    "strongBuy": latest.get("strongBuy", 0),
                    "buy": latest.get("buy", 0),
                    "hold": latest.get("hold", 0),
                    "sell": latest.get("sell", 0),
                    "strongSell": latest.get("strongSell", 0)
                }
                rating_mayoritario = max(votos, key=votos.get)
                
                precio_objetivo = data_tgt.get("targetMean", "No disponible")
                
                ratings.append({
                    "firma": "Consenso Bloomberg/Finnhub", # Simulamos una firma externa basándonos en el consenso agregado
                    "rating": rating_mayoritario.upper(),
                    "objetivo_precio": precio_objetivo,
                    "fecha": latest.get("period", "Desconocida")
                })
            return ratings
            
        except Exception as e:
            logger.warning(f"Error consultando ratings de Finnhub para {clean_symbol}: {str(e)}")
            return []

    def get_sector_news(self, sector: str, max_articles: int = 5) -> List[Dict[str, str]]:
        """
        Busca noticias de contexto agregadas por sector, especialmente útil para activos sin ticker particular como Materias Primas.
        """
        logger.info(f"Buscando contexto macro para el sector: {sector}...")
        params = {
            "apikey": self.api_key,
            "q": sector,
            "language": "en,es",
            "category": "business"
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data.get("status") == "error":
                logger.error(f"Error de NewsData en sector: {data.get('results', {}).get('message', 'Desconocido')}")
                return []

            results = data.get("results", [])
            return self._procesar_y_filtrar_noticias(results, max_articles)
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Falla de red buscando el sector {sector}: {str(e)}")
            return []

# Código de ejecución para testing individual
if __name__ == "__main__":
    try:
        cliente_noticias = NewsClient()
        
        # 1. Macro mercado
        print("\n--- NOTICIAS DEL MERCADO GLOBAL ---")
        macro_news = cliente_noticias.get_market_news(max_articles=3)
        if not macro_news:
            print("No se encontraron noticias globales recientes.")
        for i, n in enumerate(macro_news):
            print(f"[{i+1}] {n['titular']}")
            print(f"    Fuente: {n['fuente']} | {n['fecha']}")
            print(f"    Resumen: {n['resumen']}\n")

        # 2. Especifico
        ticker = "TSLA"
        print(f"\n--- NOTICIAS PARA {ticker} ---")
        symbol_news = cliente_noticias.get_news_for_symbol(ticker, max_articles=2)
        if not symbol_news:
            print("No se encontraron noticias recientes válidas para este símbolo.")
        for i, n in enumerate(symbol_news):
            print(f"[{i+1}] {n['titular']}")
            print(f"    Fuente: {n['fuente']} | {n['fecha']}")
            print(f"    Resumen: {n['resumen']}\n")

    except Exception as e:
        logger.error(f"Fallo en la prueba de NewsData: {str(e)}")
