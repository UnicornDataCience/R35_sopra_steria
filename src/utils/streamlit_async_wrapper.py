"""
Wrapper síncrono para evitar problemas de event loop en Streamlit
"""

import asyncio
import threading
import concurrent.futures
import functools
import logging
import time
import datetime
from typing import Any, Callable, Dict

logger = logging.getLogger(__name__)

class StreamlitAsyncWrapper:
    """Wrapper que ejecuta funciones async de manera segura en Streamlit"""
    
    def __init__(self):
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
    
    def run_async(self, coro_func: Callable, *args, timeout: int = 60, **kwargs) -> Any:
        """
        Ejecuta una función async de manera síncrona y segura para Streamlit
        
        Args:
            coro_func: Función async a ejecutar
            *args: Argumentos posicionales
            timeout: Timeout en segundos (default: 60)
            **kwargs: Argumentos con nombre
            
        Returns:
            Resultado de la función async
        """
        start_time = time.time()
        print(f"🔄 [{datetime.datetime.now().strftime('%H:%M:%S')}] StreamlitAsyncWrapper iniciando con timeout {timeout}s...")
        print(f"🔄 Función: {coro_func.__name__ if hasattr(coro_func, '__name__') else str(coro_func)}")
        print(f"🔄 Args: {len(args)}, Kwargs: {len(kwargs)}")
        
        def run_in_thread():
            thread_start = time.time()
            print(f"🧵 [{datetime.datetime.now().strftime('%H:%M:%S')}] Thread iniciado...")
            
            # Crear un nuevo event loop en este hilo
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                print(f"🧵 [{datetime.datetime.now().strftime('%H:%M:%S')}] Event loop creado, ejecutando función...")
                
                # Ejecutar la corrutina
                if asyncio.iscoroutinefunction(coro_func):
                    result = loop.run_until_complete(coro_func(*args, **kwargs))
                else:
                    # Si no es async, ejecutar directamente
                    result = coro_func(*args, **kwargs)
                
                thread_end = time.time()
                print(f"✅ [{datetime.datetime.now().strftime('%H:%M:%S')}] Thread completado en {thread_end - thread_start:.2f}s")
                return result
            except Exception as e:
                thread_end = time.time()
                print(f"❌ [{datetime.datetime.now().strftime('%H:%M:%S')}] Error en thread después de {thread_end - thread_start:.2f}s: {e}")
                logger.error(f"Error en run_in_thread: {e}")
                raise
            finally:
                loop.close()
        
        try:
            print(f"🔄 [{datetime.datetime.now().strftime('%H:%M:%S')}] Enviando a ThreadPoolExecutor...")
            # Ejecutar en un hilo separado con timeout
            future = self._executor.submit(run_in_thread)
            result = future.result(timeout=timeout)
            
            total_time = time.time() - start_time
            print(f"🎉 [{datetime.datetime.now().strftime('%H:%M:%S')}] StreamlitAsyncWrapper completado exitosamente en {total_time:.2f}s")
            return result
        except concurrent.futures.TimeoutError:
            total_time = time.time() - start_time
            print(f"⏰ [{datetime.datetime.now().strftime('%H:%M:%S')}] TIMEOUT después de {timeout}s (total: {total_time:.2f}s)")
            logger.error(f"Timeout después de {timeout}s")
            raise TimeoutError(f"Operación timeout después de {timeout} segundos")
        except Exception as e:
            total_time = time.time() - start_time
            print(f"❌ [{datetime.datetime.now().strftime('%H:%M:%S')}] Error después de {total_time:.2f}s: {e}")
            logger.error(f"Error en run_async: {e}")
            raise

# Instancia global del wrapper
_async_wrapper = StreamlitAsyncWrapper()

def sync_run(timeout: int = 120):
    """Decorador para convertir funciones async en síncronas"""
    def decorator(async_func):
        @functools.wraps(async_func)
        def wrapper(*args, **kwargs):
            return _async_wrapper.run_async(async_func, *args, timeout=timeout, **kwargs)
        return wrapper
    return decorator

def run_async_safe(coro_func: Callable, *args, timeout: int = 120, **kwargs) -> Any:
    """Función helper para ejecutar async de manera segura"""
    return _async_wrapper.run_async(coro_func, *args, timeout=timeout, **kwargs)
