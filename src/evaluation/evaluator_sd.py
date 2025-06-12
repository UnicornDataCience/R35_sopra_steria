import pandas as pd
import medspacy
from medspacy.ner import TargetRule
from sklearn.metrics import classification_report
from spacy import displacy
import os

# Importación opcional de QuickUMLS
try:
    from medspacy.umls_lookup import QuickUMLS
    QUICKUMLS_AVAILABLE = True
except ImportError:
    print("Warning: QuickUMLS no está disponible. Continuando sin UMLS...")
    QUICKUMLS_AVAILABLE = False

# Configuración inicial
# SOLUCIÓN 1: Raw string (r"") - RECOMENDADA
CSV_PATH = r"C:\Users\Administrator\Documents\PROYECTOS\SOPRA_STERIA\R35_sopra_steria\data\synthetic\datos_sinteticos_sdv.csv"

# SOLUCIÓN 2: Barras normales (funciona en Windows también)
# CSV_PATH = "C:/Users/Administrator/Documents/PROYECTOS/SOPRA_STERIA/R35_sopra_steria/data/synthetic/datos_sinteticos_sdv.csv"

# SOLUCIÓN 3: Escapar las barras invertidas
# CSV_PATH = "C:\\Users\\Administrator\\Documents\\PROYECTOS\\SOPRA_STERIA\\R35_sopra_steria\\data\\synthetic\\datos_sinteticos_sdv.csv"

# SOLUCIÓN 4: Usar os.path.join (más portable)
# CSV_PATH = os.path.join("C:", "Users", "Administrator", "Documents", "PROYECTOS", "SOPRA_STERIA", "R35_sopra_steria", "data", "synthetic", "datos_sinteticos_sdv.csv")

UMLS_DATA_PATH = r"ruta/a/umls/data"  # Descargar de https://www.nlm.nih.gov/research/umls/

# Columnas a procesar (según tu lista)
COLUMNAS_TEXTO = [
    "DIAG ING/INPAT",  # Diagnósticos
    "FARMACO/DRUG_NOMBRE_COMERCIAL/COMERCIAL_NAME",  # Medicamentos
    "MOTIVO_ALTA/DESTINY_DISCHARGE_ING"  # Texto libre
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

# 2. Configurar medspaCy con reglas y UMLS
def configurar_nlp():
    try:
        # Intentar cargar el modelo médico, si no existe usar el básico de spaCy
        try:
            nlp = medspacy.load("en_core_med7_lg")  # Modelo médico
            print("Modelo médico en_core_med7_lg cargado")
        except OSError:
            print("Modelo médico no encontrado. Intentando con modelo básico...")
            try:
                nlp = medspacy.load("en_core_web_sm")  # Modelo básico
                print("Modelo básico en_core_web_sm cargado")
            except OSError:
                print("No se encontró ningún modelo. Instalando modelo básico...")
                os.system("python -m spacy download en_core_web_sm")
                nlp = medspacy.load("en_core_web_sm")
        
        print(f"Pipeline components: {nlp.pipe_names}")
        
        # Verificar si medspacy_target_matcher ya existe antes de añadirlo
        target_matcher_name = "medspacy_target_matcher"
        if target_matcher_name not in nlp.pipe_names:
            # Añadir reglas para validación
            rules = [
                TargetRule("mg", "STRENGTH", pattern=[{"IS_DIGIT": True}, {"LOWER": "mg"}]),  # Dosis
                TargetRule("alergi", "ALLERGY", pattern=[{"LOWER": {"REGEX": "alergi[aá]"}}]),  # Alergias
            ]
            nlp.add_pipe("medspacy_target_matcher", config={"rules": rules})
            print("medspacy_target_matcher añadido con reglas personalizadas")
        else:
            # Si ya existe, obtener el componente y añadir reglas
            target_matcher = nlp.get_pipe("medspacy_target_matcher")
            rules = [
                TargetRule("mg", "STRENGTH", pattern=[{"IS_DIGIT": True}, {"LOWER": "mg"}]),  # Dosis
                TargetRule("alergi", "ALLERGY", pattern=[{"LOWER": {"REGEX": "alergi[aá]"}}]),  # Alergias
            ]
            # Añadir reglas al matcher existente
            for rule in rules:
                target_matcher.add([rule])
            print("Reglas personalizadas añadidas al medspacy_target_matcher existente")
        
        # Añadir UMLS solo si está disponible y los datos existen
        if QUICKUMLS_AVAILABLE and os.path.exists(UMLS_DATA_PATH):
            try:
                if "quickumls" not in nlp.pipe_names:
                    nlp.add_pipe("quickumls", config={"quickumls_fp": UMLS_DATA_PATH, "threshold": 0.7})
                    print("QuickUMLS añadido exitosamente")
            except Exception as e:
                print(f"Error al añadir QuickUMLS: {e}")
        else:
            print("QuickUMLS no disponible o datos UMLS no encontrados. Continuando sin UMLS...")
        
        print(f"Pipeline final: {nlp.pipe_names}")
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

# 4. Validar contra gold standard (ejemplo simplificado)
def generar_gold_standard(texto):
    # Mock: Simular anotaciones manuales (ajustar según tus datos reales)
    gold = []
    if "diabet" in texto.lower():
        gold.append(("diabetes", "PROBLEM"))
    if "mg" in texto:
        # Mejorar la extracción de dosis
        words = texto.split()
        for i, word in enumerate(words):
            if "mg" in word.lower():
                if i > 0:
                    gold.append((words[i-1] + " " + word, "STRENGTH"))
                else:
                    gold.append((word, "STRENGTH"))
    return gold

# 5. Métricas y reporte
def evaluar_metricas(gold_flat, pred_flat):
    if not gold_flat or not pred_flat:
        print("Warning: No hay datos suficientes para evaluar métricas")
        return "No hay datos suficientes para la evaluación"
    
    try:
        report = classification_report(
            [label for (text, label) in gold_flat],
            [label for (text, label) in pred_flat],
            zero_division=0
        )
        return report
    except Exception as e:
        print(f"Error al generar métricas: {e}")
        return f"Error en evaluación: {e}"

# --- Ejecución principal ---
if __name__ == "__main__":
    print("Iniciando evaluación de medspaCy...")
    
    # Cargar datos
    df = cargar_datos(CSV_PATH)
    if df is None:
        exit(1)
    
    # Inicializar pipeline
    nlp = configurar_nlp()
    if nlp is None:
        exit(1)
    
    # Procesar columnas de texto
    gold_flat = []
    pred_flat = []
    
    print("\nProcesando columnas de texto...")
    for col in COLUMNAS_TEXTO:
        if col in df.columns:
            print(f"Procesando columna: {col}")
            textos_procesados = 0
            
            for texto in df[col].dropna():
                # Extraer entidades
                pred_entities = extraer_entidades(nlp, str(texto))
                gold_entities = generar_gold_standard(str(texto))
                
                # Acumular para métricas
                gold_flat.extend(gold_entities)
                pred_flat.extend(pred_entities)
                
                textos_procesados += 1
                
                # Opcional: Visualizar un ejemplo (solo los primeros 3)
                if len(pred_entities) > 0 and textos_procesados <= 3:
                    print(f"\nEjemplo {textos_procesados}:")
                    print(f"Texto: {texto[:100]}...")
                    print(f"Entidades encontradas: {pred_entities}")
                    
                    # Comentar la siguiente línea si no quieres visualización HTML
                    # doc = nlp(str(texto))
                    # displacy.render(doc, style="ent", jupyter=False)
            
            print(f"Textos procesados en {col}: {textos_procesados}")
        else:
            print(f"Warning: Columna '{col}' no encontrada en el CSV")
    
    # Generar reporte
    print(f"\nTotal entidades gold: {len(gold_flat)}")
    print(f"Total entidades predichas: {len(pred_flat)}")
    
    if gold_flat and pred_flat:
        report = evaluar_metricas(gold_flat, pred_flat)
        print("\n--- Reporte de Métricas ---")
        print(report)
        
        # Guardar resultados
        try:
            with open("resultados_evaluacion.txt", "w", encoding='utf-8') as f:
                f.write("=== REPORTE DE EVALUACIÓN DE MEDSPACY ===\n\n")
                f.write(f"Total entidades gold standard: {len(gold_flat)}\n")
                f.write(f"Total entidades predichas: {len(pred_flat)}\n\n")
                f.write("--- Métricas de Clasificación ---\n")
                f.write(report)
            print("\nResultados guardados en 'resultados_evaluacion.txt'")
        except Exception as e:
            print(f"Error al guardar resultados: {e}")
    else:
        print("No se encontraron entidades para evaluar. Revisa tu dataset y las reglas de extracción.")
    
    print("\n¡Evaluación completada!")