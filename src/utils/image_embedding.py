"""
Utilidades para embeber imágenes en markdown y respuestas de chat.
"""

import base64
from pathlib import Path
from typing import Optional, Dict, Any
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def image_to_base64(image_path: str) -> Optional[str]:
    """
    Convierte una imagen a base64 para embeber en HTML/Markdown.
    
    Args:
        image_path: Ruta al archivo de imagen
        
    Returns:
        String base64 de la imagen o None si falla
    """
    try:
        path = Path(image_path)
        
        if not path.exists():
            logger.warning(f"Imagen no encontrada: {image_path}")
            return None
        
        with open(path, 'rb') as image_file:
            encoded = base64.b64encode(image_file.read()).decode('utf-8')
            
        # Detectar tipo MIME basado en extensión
        mime_type = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif',
            '.svg': 'image/svg+xml'
        }.get(path.suffix.lower(), 'image/png')
        
        return f"data:{mime_type};base64,{encoded}"
    
    except Exception as e:
        logger.error(f"Error convirtiendo imagen a base64: {e}")
        return None


def create_markdown_image(image_path: str, alt_text: str = "Visualización", width: Optional[int] = None) -> str:
    """
    Crea un markdown con imagen embebida en base64.
    
    Args:
        image_path: Ruta a la imagen
        alt_text: Texto alternativo
        width: Ancho opcional en píxeles
        
    Returns:
        String con markdown HTML embebido
    """
    base64_data = image_to_base64(image_path)
    
    if not base64_data:
        # Fallback a ruta relativa
        return f"![{alt_text}]({image_path})"
    
    # Usar HTML para tener control sobre tamaño
    width_attr = f' width="{width}"' if width else ''
    return f'<img src="{base64_data}" alt="{alt_text}"{width_attr} style="max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);" />'


def create_visualization_section(
    timeline_path: Optional[str] = None,
    heatmap_path: Optional[str] = None,
    embed_images: bool = True
) -> str:
    """
    Crea una sección de markdown con visualizaciones.
    
    Args:
        timeline_path: Ruta al gráfico de timeline
        heatmap_path: Ruta al heatmap
        embed_images: Si True, embebe imágenes en base64; si False, usa rutas
        
    Returns:
        String con sección de markdown formateada
    """
    sections = []
    
    if timeline_path:
        sections.append("## 📈 Evolución Temporal de Pacientes\n")
        
        if embed_images:
            sections.append(create_markdown_image(timeline_path, "Timeline de Evolución", width=800))
        else:
            sections.append(f"![Timeline de Evolución]({timeline_path})")
        
        sections.append("\nEste gráfico muestra la evolución de los parámetros clínicos clave a lo largo del tiempo para cada paciente simulado.\n")
    
    if heatmap_path:
        sections.append("## 🔥 Mapa de Calor de Transiciones\n")
        
        if embed_images:
            sections.append(create_markdown_image(heatmap_path, "Heatmap de Evolución", width=800))
        else:
            sections.append(f"![Heatmap de Evolución]({heatmap_path})")
        
        sections.append("\nEl mapa de calor visualiza los patrones de cambio en los parámetros clínicos, permitiendo identificar tendencias de mejoría o deterioro.\n")
    
    return "\n".join(sections)


def enhance_response_with_images(
    response: Dict[str, Any],
    embed_images: bool = True
) -> Dict[str, Any]:
    """
    Mejora una respuesta del agente añadiendo visualizaciones embebidas.
    
    Args:
        response: Diccionario de respuesta del agente
        embed_images: Si True, embebe imágenes; si False, solo añade rutas
        
    Returns:
        Respuesta mejorada con visualizaciones
    """
    timeline_path = response.get('timeline_path')
    heatmap_path = response.get('heatmap_path')
    
    if not timeline_path and not heatmap_path:
        return response
    
    try:
        # Obtener el mensaje existente
        message = response.get('message', '')
        
        # Crear sección de visualizaciones
        viz_section = create_visualization_section(
            timeline_path=timeline_path,
            heatmap_path=heatmap_path,
            embed_images=embed_images
        )
        
        # Insertar visualizaciones antes de las conclusiones o al final
        if "## Conclusiones" in message or "## Recomendaciones" in message:
            # Insertar antes de conclusiones
            parts = message.split("## Conclusiones", 1) if "## Conclusiones" in message else message.split("## Recomendaciones", 1)
            enhanced_message = f"{parts[0]}\n\n{viz_section}\n\n## {'Conclusiones' if '## Conclusiones' in message else 'Recomendaciones'}{parts[1]}"
        else:
            # Añadir al final
            enhanced_message = f"{message}\n\n---\n\n{viz_section}"
        
        response['message'] = enhanced_message
        response['images_embedded'] = embed_images
        
        logger.info(f"✅ Respuesta mejorada con visualizaciones (embed={embed_images})")
        
    except Exception as e:
        logger.error(f"Error mejorando respuesta con imágenes: {e}")
    
    return response


def save_response_as_html(response: Dict[str, Any], output_path: str = None) -> str:
    """
    Guarda la respuesta como archivo HTML para visualización en navegador.
    
    Args:
        response: Respuesta del agente con markdown
        output_path: Ruta donde guardar el HTML (opcional)
        
    Returns:
        Ruta del archivo HTML generado
    """
    try:
        import markdown
        from datetime import datetime
        
        message = response.get('message', '')
        
        # Convertir markdown a HTML
        html_content = markdown.markdown(
            message,
            extensions=['extra', 'codehilite', 'tables', 'toc']
        )
        
        # Plantilla HTML con estilos
        html_template = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Informe de Simulación - Patient-IA</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .container {{
            background: white;
            padding: 40px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1, h2, h3 {{
            color: #2c3e50;
        }}
        h1 {{
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            margin-top: 30px;
            border-left: 4px solid #3498db;
            padding-left: 15px;
        }}
        img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            margin: 20px 0;
        }}
        ul, ol {{
            padding-left: 25px;
        }}
        code {{
            background: #f8f9fa;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
        }}
        .timestamp {{
            text-align: right;
            color: #7f8c8d;
            font-size: 0.9em;
            margin-top: 40px;
            border-top: 1px solid #ecf0f1;
            padding-top: 15px;
        }}
    </style>
</head>
<body>
    <div class="container">
        {html_content}
        <div class="timestamp">
            Generado el {datetime.now().strftime('%d/%m/%Y a las %H:%M:%S')} por Patient-IA
        </div>
    </div>
</body>
</html>"""
        
        # Determinar ruta de salida
        if not output_path:
            from pathlib import Path
            output_dir = Path("temp_generations/reports")
            output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = output_dir / f"simulation_report_{timestamp}.html"
        
        # Guardar archivo
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_template)
        
        logger.info(f"📄 Informe HTML guardado en: {output_path}")
        return str(output_path)
        
    except ImportError:
        logger.warning("⚠️ Módulo 'markdown' no disponible. Instalar con: pip install markdown")
        return None
    except Exception as e:
        logger.error(f"Error guardando HTML: {e}")
        return None
