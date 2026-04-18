import os
import requests
import base64
import logging
from typing import List, Dict, Any
from dotenv import load_dotenv

# Cargar variables de entorno desde el archivo .env
load_dotenv()

# Configurar un logger básico para este módulo
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("T212Client")

class T212Client:
    """
    Cliente para interactuar con la API oficial de Trading 212.
    """

    def __init__(self):
        """
        1. Conectarse a la API de Trading 212 usando la API key y Secret del .env.
        5. Usar el entorno LIVE explícitamente (https://live.trading212.com).
        """
        self.api_key = os.getenv("T212_API_KEY")
        self.api_secret = os.getenv("T212_API_SECRET")

        if not self.api_key or not self.api_secret:
            logger.error("T212_API_KEY o T212_API_SECRET no encontrada. Asegúrate de configurarlas en el archivo .env.")
            raise ValueError("Credenciales de Trading 212 incompletas.")

        # URL base del entorno LIVE oficial de Trading 212
        self.base_url = "https://live.trading212.com/api/v0"

        # Trading 212 requiere Basic Authentication (API_KEY:API_SECRET en base64)
        auth_str = f"{self.api_key}:{self.api_secret}"
        encoded_auth = base64.b64encode(auth_str.encode('utf-8')).decode('utf-8')

        self.headers = {
            "Authorization": f"Basic {encoded_auth}"
        }

    def _manejar_peticion(self, endpoint: str) -> Any:
        """
        4. Manejar errores de conexión con mensajes claros.
        Método interno para aislar la lógica HTTP y controlar los errores de manera centralizada.
        """
        url = f"{self.base_url}/{endpoint}"
        try:
            # Hacemos la petición con un tiempo límite para evitar que el script se cuelgue
            response = requests.get(url, headers=self.headers, timeout=10)

            # Si la respuesta es de error (4xx o 5xx), la captura y lanza una excepción HTTP
            response.raise_for_status()

            # Devolvemos los datos parseados en JSON
            return response.json()

        except requests.exceptions.HTTPError:
            logger.error(f"Error de Servidor/Auth ({response.status_code}): {response.text}")
        except requests.exceptions.ConnectionError:
            logger.error("Error de conexión: No se pudo contactar con los servidores de Trading 212.")
        except requests.exceptions.Timeout:
            logger.error("Error de tiempo de espera: La API de Trading 212 tardó demasiado en responder.")
        except requests.exceptions.RequestException as e:
            logger.error(f"Error inesperado de red al contactar Trading 212: {str(e)}")

        return None

    def get_portfolio(self) -> List[Dict[str, Any]]:
        """
        2. Función que devuelve la lista de posiciones actuales.
        Extrae: símbolo, cantidad, valor actual, ganancia/pérdida.
        """
        logger.info("Solicitando posiciones actuales (Portfolio) a Trading 212 LIVE...")
        data = self._manejar_peticion("equity/portfolio")

        # Si hubo un error en la conexión, data será None
        if not data:
            logger.warning("No se devolvieron datos de la cartera.")
            return []

        posiciones = []
        # La API de T212 devuelve una lista de diccionarios, iteramos para extraer lo necesario
        for posicion in data:
            # Construimos la estructura consolidada
            # T212 usa 'ppl' para Profit and Loss (ganancias y pérdidas)
            # Calculamos el valor_actual como cantidad * precio actual
            cantidad = posicion.get("quantity", 0.0)
            precio_actual = posicion.get("currentPrice", 0.0)

            posiciones.append({
                "simbolo": posicion.get("ticker", "Desconocido"),
                "cantidad": cantidad,
                "valor_actual": cantidad * precio_actual,
                "ganancia_perdida": posicion.get("ppl", 0.0)
            })

        return posiciones

    def get_account_summary(self) -> Dict[str, Any]:
        """
        3. Función que devuelve el valor total de la cuenta, cash disponible y resultado total (P/L).
        """
        logger.info("Solicitando el resumen de la cuenta a Trading 212 LIVE...")
        data = self._manejar_peticion("equity/account/cash")

        if not data:
            logger.warning("No se pudo obtener el estado de la cuenta por un error de conexión.")
            return {
                "valor_total": 0.0,
                "efectivo_disponible": 0.0,
                "resultado_total": 0.0
            }

        # Estructuramos el resumen según los requerimientos
        resumen = {
            "valor_total": data.get("total", 0.0),
            "efectivo_disponible": data.get("free", 0.0),
            "resultado_total": data.get("ppl", 0.0)
        }

        return resumen

# Código de prueba rápida (solo se ejecuta si corres este archivo directamente)
if __name__ == "__main__":
    try:
        cliente_t212 = T212Client()
        print("--- CONECTANDO A TRADING 212 ---")

        resumen = cliente_t212.get_account_summary()
        print(f"Resumen de cuenta: {resumen}")

        posiciones = cliente_t212.get_portfolio()
        print(f"Posiciones abiertas detectadas: {len(posiciones)}")
        for i, pos in enumerate(posiciones):
            print(f"[{i+1}] {pos['simbolo']}: Qty {pos['cantidad']} | Valor: {pos['valor_actual']} | P/L: {pos['ganancia_perdida']}")

    except Exception as e:
        logger.error(f"Fallo en la prueba de cliente: {str(e)}")
