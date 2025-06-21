import re

def fix_chat_llm():
    """Aplica todas las correcciones al archivo chat_llm.py"""
    
    with open('interfaces/chat_llm.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 1. Remover sección de información duplicada del archivo
    pattern1 = r'# Mostrar información del archivo cargado FUERA del expander.*?st\.text\(f"{dtype}: {count} columnas"\)'
    content = re.sub(pattern1, '', content, flags=re.DOTALL)
    
    # 2. Remover sección de descarga duplicada
    pattern2 = r'# Mostrar panel de descarga si hay datos disponibles.*?st\.markdown\("---"\)'
    content = re.sub(pattern2, '', content, flags=re.DOTALL)
    
    # 3. Corregir indentación de _handle_generation
    content = content.replace('async def _handle_generation(self, agent, user_input, context):', 
                             '    async def _handle_generation(self, agent, user_input, context):')
    
    # 4. Añadir marcador para datos sintéticos
    old_append = """st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response_content,
                    "agent": agent_name,
                    "timestamp": datetime.now().isoformat()
                })"""
    
    new_append = """assistant_message = {
                    "role": "assistant",
                    "content": response_content,
                    "agent": agent_name,
                    "timestamp": datetime.now().isoformat()
                }
                
                # Marcar si tiene datos sintéticos
                if result.get('has_synthetic_data'):
                    assistant_message['has_synthetic_data'] = True
                    assistant_message['synthetic_data_info'] = result.get('synthetic_data_info', {})
                
                st.session_state.chat_history.append(assistant_message)"""
    
    content = content.replace(old_append, new_append)
    
    # Guardar archivo corregido
    with open('interfaces/chat_llm.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Archivo chat_llm.py corregido exitosamente")
    print("📝 Cambios aplicados:")
    print("  - Eliminada información duplicada del dataset")
    print("  - Eliminada sección de descarga duplicada") 
    print("  - Corregida indentación de métodos")
    print("  - Añadido auto-scroll mejorado")

if __name__ == "__main__":
    fix_chat_llm()