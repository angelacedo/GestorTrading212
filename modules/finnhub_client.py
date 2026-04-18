import os
import logging
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Cargar variables de entorno desde el archivo .env
load_dotenv()

# Configuración del logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MarketDataClient")

class FinnhubClient:
    """
    Nota: Aunque el módulo se llame FinnhubClient por consistencia con el plan inicial,
    utilizaremos yfinance para los datos históricos debido a las restricciones (403) 
    de la capa gratuita de Finnhub para velas (candles).
    """

    def __init__(self):
        """
        Inicializa el cliente. Mantenemos la estructura por si en el futuro
        añadimos funciones de Finnhub que sí funcionen (como perfiles de empresa).
        """
        self.api_key = os.getenv("FINNHUB_API_KEY")

    def _clean_ticker(self, ticker: str) -> str:
        """
        Limpia los tickers de Trading 212 (ej: AAPL_US_EQ -> AAPL)
        para que funcionen en yfinance/Finnhub.
        """
        return ticker.split('_')[0]

    def get_historical_data(self, symbol: str, days: int = 30) -> Optional[pd.DataFrame]:
        """
        Obtiene datos históricos usando yfinance (mucho más fiable para histórico gratuito).
        Mantiene el formato de salida: fecha, apertura, cierre, máximo, mínimo, volumen.
        """
        clean_symbol = self._clean_ticker(symbol)
        logger.info(f"Obteniendo histórico para {clean_symbol} via yfinance (últimos {days} días)...")
        
        try:
            # yfinance permite descargar el histórico directamente
            # Usamos '1d' como intervalo y calculamos el periodo
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days + 10) # Añadimos margen por fines de semana
            
            df_yf = yf.download(clean_symbol, start=start_date, end=end_date, progress=False)
            
            if df_yf.empty:
                logger.warning(f"No hay datos históricos disponibles para el símbolo: {clean_symbol}")
                return None
                
            # Formateamos para que coincida con nuestra estructura previa
            df = df_yf.reset_index()
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns] # Fix para multi-index de yfinance reciente
            
            df = df.rename(columns={
                'Date': 'fecha',
                'Open': 'apertura',
                'Close': 'cierre',
                'High': 'máximo',
                'Low': 'mínimo',
                'Volume': 'volumen'
            })
            
            return df[['fecha', 'apertura', 'cierre', 'máximo', 'mínimo', 'volumen']]

        except Exception as e:
            logger.error(f"Error al obtener datos de yfinance para {clean_symbol}: {str(e)}")
            return None

    def get_market_indicators(self, symbol: str) -> Optional[Dict[str, float]]:
        """
        Calcula indicadores técnicos (SMA, RSI, Variación) usando los datos de yfinance.
        """
        # Pedimos 90 días para asegurar el cálculo del RSI y SMA30
        df = self.get_historical_data(symbol, days=90)
        
        if df is None or len(df) < 30:
            logger.warning(f"No hay datos suficientes para calcular indicadores de {symbol}.")
            return None
            
        try:
            # a) Variación porcentual 30 días
            cierre_actual = float(df['cierre'].iloc[-1])
            cierre_previo = float(df['cierre'].iloc[-30])
            var_pct_30d = ((cierre_actual - cierre_previo) / cierre_previo) * 100

            # b) Medias Móviles
            df['sma_7'] = df['cierre'].rolling(window=7).mean()
            df['sma_30'] = df['cierre'].rolling(window=30).mean()
            
            # c) RSI 14
            delta = df['cierre'].diff()
            gain = delta.where(delta > 0, 0.0)
            loss = -delta.where(delta < 0, 0.0)
            avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
            avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
            rs = avg_gain / avg_loss
            df['rsi_14'] = 100 - (100 / (1 + rs))

            return {
                "sma_7": round(float(df['sma_7'].iloc[-1]), 2),
                "sma_30": round(float(df['sma_30'].iloc[-1]), 2),
                "rsi_14": round(float(df['rsi_14'].iloc[-1]), 2),
                "variacion_30d_pct": round(float(var_pct_30d), 2)
            }
            
        except Exception as e:
            logger.error(f"Error en indicadores técnicos para {symbol}: {str(e)}")
            return None

# Prueba
if __name__ == "__main__":
    cliente = FinnhubClient()
    for s in ["AAPL", "TTWO_US_EQ"]:
        print(f"\n--- PROBANDO CON {s} ---")
        indicadores = cliente.get_market_indicators(s)
        if indicadores:
            print(f"Indicadores para {s}: {indicadores}")
