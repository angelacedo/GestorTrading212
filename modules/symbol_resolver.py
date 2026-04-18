import os
import json
import logging
import threading
from typing import Dict, List
from openai import OpenAI
import openai
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Configuración del logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SymbolResolver")

class SymbolResolver:
    """
    Módulo inteligente que mapea de forma dinámica los tokens/símbolos propietarios
    de los brókers a su ticker internacional estándar para posibilitar el scraping.
    """

    def __init__(self):
        # 1c. Configuración del modelo gratuito vía OpenRouter
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.model = os.getenv("OPENROUTER_MODEL_MAP_SYMBOL")

        if not self.api_key:
            logger.error("OPENROUTER_API_KEY no encontrada.")
            raise ValueError("Credenciales de OpenRouter insuficientes.")

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
        )

        # 3. Configuramos la persistencia en disco del caché
        self.cache_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'symbol_cache.json'))
        self.cache_lock = threading.Lock()
        self.cache = self._load_cache()

    def _load_cache(self) -> Dict[str, str]:
        """Carga el historial de resoluciones previas guardadas desde el disco."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error al leer el caché {self.cache_file}: {e}")
                return {}
        return {}

    def _save_cache(self) -> None:
        """Graba la sesión de resoluciones al archivo JSON de caché."""
        with self.cache_lock:
            try:
                with open(self.cache_file, 'w', encoding='utf-8') as f:
                    json.dump(self.cache, f, indent=4, ensure_ascii=False)
            except Exception as e:
                logger.warning(f"Error escribiendo el caché de símbolos: {e}")

    def resolve_symbol(self, t212_symbol: str) -> str:
        """
        1. Resuelve un único símbolo a su estándar de YFinance.
        """
        # 3a. Consultando el caché para optimizar tiempo y cuotas
        with self.cache_lock:
            if t212_symbol in self.cache:
                return self.cache[t212_symbol]

        # 1c/1d. Diseño de los Prompts exigidos
        system_prompt = "You are a financial data assistant. Your only task is to convert a Trading 212 ticker symbol to its exact equivalent ticker symbol valid for use with the yfinance Python library. Reply with ONLY the ticker symbol, no explanations, no punctuation, nothing else. If you are not sure, make your best guess based on the symbol structure."
        user_prompt = f"Convert this Trading 212 symbol to its yfinance equivalent: {t212_symbol}"

        try:
            # 1b. Interrogación al LLM
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=20,
                temperature=0.0
            )

            # 1e. Limpieza exhaustiva
            respuesta_cruda = response.choices[0].message.content or ""
            resultado = respuesta_cruda.strip().replace('`', '').replace('"', '').replace("'", "")

            if not resultado:
                raise ValueError("Respuesta LLM vacía")

            # 3c. Almacenamos el descubrimiento de la Inteligencia Artificial
            with self.cache_lock:
                self.cache[t212_symbol] = resultado
            self._save_cache()

            return resultado

        except Exception as e:
            # 1f. Fallback de rescate heurístico (ej: AAPL_US_EQ -> AAPL)
            fallback = t212_symbol.replace("_EQ", "").split('_')[0].upper()
            logger.warning(f"Error resolviendo '{t212_symbol}' por LLM ({e}). Se ha inyectado el Fallback: {fallback}")
            return fallback

    def resolve_symbols_batch(self, t212_symbols: List[str]) -> Dict[str, str]:
        """
        2. Función que orquesta la traducción masiva pero secuencial.
        """
        resultados = {}
        for sym in t212_symbols:
            if not sym or sym in ["Desconocido", "Cash"]:
                continue

            res_yf = self.resolve_symbol(sym)

            # 2d. Logging exhaustivo
            logger.info(f"[SYMBOL RESOLVER] {sym} → {res_yf}")
            resultados[sym] = res_yf

        return resultados
