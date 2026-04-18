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
        # Mantiene el fallback local explícito de la instrucción anterior por si en .env falla o cambia
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

    def resolve_symbols_batch(self, t212_symbols: List[str]) -> Dict[str, str]:
        """
        Resuelve una lista completa de símbolos en UNA sola llamada al LLM,
        optimizando drásticamente el uso del caché y evitando rate-limits iterativos.
        """
        resultados = {}
        símbolos_a_consultar = []

        # 1. Filtrar de la lista de entrada los que ya estén en caché
        with self.cache_lock:
            for sym in t212_symbols:
                if not sym or sym in ["Desconocido", "Cash"]:
                    continue

                if sym in self.cache:
                    resultados[sym] = self.cache[sym]
                else:
                    if sym not in símbolos_a_consultar:
                        símbolos_a_consultar.append(sym)

        # 2. Si todos los símbolos ya están en caché, abortar petición LLM
        if not símbolos_a_consultar:
            return resultados

        logger.info(f"Consultando traducción LLM en bloque para {len(símbolos_a_consultar)} símbolos nuevos...")

        # 3. Construir mensaje único de usuario sin system prompt
        lista_str = "\n".join(símbolos_a_consultar)
        user_prompt = (
            "You are a financial data assistant. Convert each Trading 212 ticker symbol "
            "to its exact yfinance equivalent. Reply ONLY with a JSON object where keys "
            "are the original symbols and values are the yfinance tickers. No explanations, "
            "no markdown, no code blocks, just the raw JSON.\n\n"
            "Symbols to convert:\n"
            f"{lista_str}"
        )

        nuevos_resueltos = {}
        try:
            # 8. Petición singular usando el enfoque de LLama 1-prompt-only
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0
            )

            # Extracción del cuerpo textual
            respuesta_cruda = response.choices[0].message.content or ""
            respuesta_limpia = respuesta_cruda.strip()

            # Defensiva ligera en caso de que el LLM devuelva formato markdown de bloque
            if respuesta_limpia.startswith("```json"):
                respuesta_limpia = respuesta_limpia[7:]
            if respuesta_limpia.startswith("```"):
                respuesta_limpia = respuesta_limpia[3:]
            if respuesta_limpia.endswith("```"):
                respuesta_limpia = respuesta_limpia[:-3]

            respuesta_limpia = respuesta_limpia.strip()

            # 4. Parseamos la respuesta directamente como diccionario JSON JSON JSON JSON
            nuevos_resueltos = json.loads(respuesta_limpia)

        except Exception as e:
            logger.warning(f"Error procesando el Batch de JSON del LLM: {str(e)}. Aplicando iteración de fallback...")

        # 4(Cont.) Aplicamos validación y Fallbacks
        with self.cache_lock:
            for sym in símbolos_a_consultar:
                if sym in nuevos_resueltos and isinstance(nuevos_resueltos[sym], str) and nuevos_resueltos[sym]:
                    traducido = nuevos_resueltos[sym]
                else:
                    # Fallback individual robusto estipulado en tu requerimiento (strip + upper)
                    traducido = sym.replace("_EQ", "").split('_')[0].upper()
                    logger.warning(f"Usando fallback semántico interno para {sym} -> {traducido}")

                # Asignamos al diccionario en caliente y a la caché en memoria simultáneamente
                self.cache[sym] = traducido
                resultados[sym] = traducido

                logger.info(f"[SYMBOL RESOLVER] {sym} → {traducido}")

        # 5. Guardar la caché final persistiendo todos de golpe
        self._save_cache()

        # 6. Devolver el diccionario combinado
        return resultados
