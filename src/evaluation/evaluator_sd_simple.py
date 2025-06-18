import pandas as pd
import medspacy
from sklearn.metrics import classification_report
import os

# # Configuración inicial
# CSV_PATH = r"C:\Users\Administrator\Documents\PROYECTOS\SOPRA_STERIA\R35_sopra_steria\data\synthetic\datos_sinteticos_sdv.csv"

script_dir = os.getcwd()
CSV_PATH = os.path.abspath(os.path.join(script_dir, '..', 'R35_sopra_steria', 'data', 'synthetic', 'datos_sinteticos_sdv.csv'))

# Columnas a procesar
COLUMNAS_TEXTO = [
    "DIAG ING/INPAT",
    "FARMACO/DRUG_NOMBRE_COMERCIAL/COMERCIAL_NAME",
    "MOTIVO_ALTA/DESTINY_DISCHARGE_ING"
]

# 1. Cargar datos desde CSV
def cargar_datos(csv_path):
    try:
        df = pd.read_csv(csv_path)
        print(f"Filas cargadas: {len(df)}")
        return df
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo {csv_path}")
        return None
    except Exception as e:
        print(f"Error al cargar el CSV: {e}")
        return None

# 2. Configurar medspaCy (versión simple)
def configurar_nlp():
    try:
        # Usar medspaCy con configuración por defecto
        try:
            nlp = medspacy.load("en_core_web_sm")
            print("Modelo básico cargado con medspaCy")
        except OSError:
            print("Instalando modelo básico...")
            os.system("python -m spacy download en_core_web_sm")
            nlp = medspacy.load("en_core_web_sm")
        
        print(f"Componentes del pipeline: {nlp.pipe_names}")
        return nlp
    except Exception as e:
        print(f"Error al configurar medspaCy: {e}")
        return None

# 3. Procesar texto y extraer entidades
def extraer_entidades(nlp, texto):
    try:
        doc = nlp(texto)
        return [(ent.text, ent.label_) for ent in doc.ents]
    except Exception as e:
        print(f"Error al procesar texto: {e}")
        return []

# 4. Gold standard simplificado
def generar_gold_standard(texto):
    gold = []
    texto_lower = texto.lower()
    
    # Patrones médicos comunes (ajustar según tus datos)
    if any(term in texto_lower for term in ["diabetes", "diabetic", "diabet"]):
        gold.append(("diabetes", "PROBLEM"))
    
    if any(term in texto_lower for term in ["hypertension", "blood pressure", "hipertension"]):
        gold.append(("hypertension", "PROBLEM"))
    
    if "mg" in texto_lower:
        # Buscar patrones como "10mg", "25 mg"
        words = texto.split()
        for i, word in enumerate(words):
            if "mg" in word.lower():
                if i > 0 and words[i-1].replace(".", "").isdigit():
                    gold.append((f"{words[i-1]} {word}", "STRENGTH"))
                elif word.replace("mg", "").replace(".", "").isdigit():
                    gold.append((word, "STRENGTH"))
    
    return gold

# 5. Análisis de resultados
def analizar_resultados(gold_flat, pred_flat):
    if not gold_flat and not pred_flat:
        return "No hay entidades para analizar"
    
    if not gold_flat:
        return f"Solo predicciones encontradas: {len(pred_flat)} entidades"
    
    if not pred_flat:
        return f"Solo gold standard disponible: {len(gold_flat)} entidades"
    
    # Análisis por tipos de entidad
    gold_labels = [label for (text, label) in gold_flat]
    pred_labels = [label for (text, label) in pred_flat]
    
    print(f"\nTipos de entidades en gold standard: {set(gold_labels)}")
    print(f"Tipos de entidades predichas: {set(pred_labels)}")
    
    # Si tenemos ambos, calcular métricas
    if len(gold_labels) == len(pred_labels):
        try:
            report = classification_report(gold_labels, pred_labels, zero_division=0)
            return report
        except Exception as e:
            return f"Error al calcular métricas: {e}"
    else:
        return f"Desbalance: {len(gold_labels)} gold vs {len(pred_labels)} predichas"

# --- Ejecución principal ---
if __name__ == "__main__":
    print("=== EVALUACIÓN SIMPLIFICADA DE MEDSPACY ===\n")
    
    # Cargar datos
    df = cargar_datos(CSV_PATH)
    if df is None:
        exit(1)
    
    # Configurar pipeline
    nlp = configurar_nlp()
    if nlp is None:
        exit(1)
    
    # Procesar datos
    gold_flat = []
    pred_flat = []
    ejemplos_encontrados = []
    
    print("\nProcesando columnas...")
    
    for col in COLUMNAS_TEXTO:
        if col not in df.columns:
            print(f"Columna '{col}' no encontrada")
            continue
        
        print(f"\n--- Procesando: {col} ---")
        textos_validos = 0
        entidades_encontradas = 0
        
        for idx, texto in enumerate(df[col].dropna()):
            if pd.isna(texto) or str(texto).strip() == "":
                continue
                
            texto_str = str(texto)
            textos_validos += 1
            
            # Extraer entidades
            pred_entities = extraer_entidades(nlp, texto_str)
            gold_entities = generar_gold_standard(texto_str)
            
            if pred_entities or gold_entities:
                entidades_encontradas += 1
                gold_flat.extend(gold_entities)
                pred_flat.extend(pred_entities)
                
                # Guardar ejemplos interesantes
                if len(ejemplos_encontrados) < 5 and (pred_entities or gold_entities):
                    ejemplos_encontrados.append({
                        'columna': col,
                        'texto': texto_str[:200] + "..." if len(texto_str) > 200 else texto_str,
                        'gold': gold_entities,
                        'pred': pred_entities
                    })
        
        print(f"Textos válidos: {textos_validos}")
        print(f"Textos con entidades: {entidades_encontradas}")
    
    # Mostrar ejemplos
    print(f"\n=== EJEMPLOS ENCONTRADOS ===")
    for i, ejemplo in enumerate(ejemplos_encontrados, 1):
        print(f"\nEjemplo {i} ({ejemplo['columna']}):")
        print(f"Texto: {ejemplo['texto']}")
        print(f"Gold: {ejemplo['gold']}")
        print(f"Predichas: {ejemplo['pred']}")
    
    # Análisis final
    print(f"\n=== RESUMEN FINAL ===")
    print(f"Total entidades gold standard: {len(gold_flat)}")
    print(f"Total entidades predichas: {len(pred_flat)}")
    
    # Análisis detallado
    resultado = analizar_resultados(gold_flat, pred_flat)
    print(f"\n=== ANÁLISIS ===")
    print(resultado)
    
    # Guardar resultados
    try:
        txt_PATH = os.path.abspath(os.path.join(script_dir, '..', 'R35_sopra_steria', 'src', 'evaluation', 'resultados_medspacy_simple.txt'))
        with open(txt_PATH, "w", encoding='utf-8') as f:
            f.write("=== EVALUACIÓN MEDSPACY - RESULTADOS ===\n\n")
            f.write(f"Total entidades gold: {len(gold_flat)}\n")
            f.write(f"Total entidades predichas: {len(pred_flat)}\n\n")
            
            f.write("=== EJEMPLOS ===\n")
            for i, ejemplo in enumerate(ejemplos_encontrados, 1):
                f.write(f"\nEjemplo {i}:\n")
                f.write(f"Columna: {ejemplo['columna']}\n")
                f.write(f"Texto: {ejemplo['texto']}\n")
                f.write(f"Gold: {ejemplo['gold']}\n")
                f.write(f"Predichas: {ejemplo['pred']}\n")
            
            f.write(f"\n=== ANÁLISIS ===\n")
            f.write(resultado)
        
        print("\nResultados guardados en 'resultados_medspacy_simple.txt'")
    
    except Exception as e:
        print(f"Error al guardar: {e}")
    
    print("\n¡Evaluación completada!")