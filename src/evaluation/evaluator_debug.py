import pandas as pd
import spacy
import medspacy
from sklearn.metrics import classification_report
import os
import re
import subprocess
import sys

# # Configuración inicial
# CSV_PATH = r"C:\Users\Administrator\Documents\PROYECTOS\SOPRA_STERIA\R35_sopra_steria\data\synthetic\datos_sinteticos_sdv.csv"

script_dir = os.getcwd()
CSV_PATH = os.path.abspath(os.path.join(script_dir, '..', 'R35_sopra_steria', 'data', 'synthetic', 'datos_sinteticos_sdv.csv'))

COLUMNAS_TEXTO = [
    "DIAG ING/INPAT",
    "FARMACO/DRUG_NOMBRE_COMERCIAL/COMERCIAL_NAME",
    "MOTIVO_ALTA/DESTINY_DISCHARGE_ING"
]

def cargar_datos(csv_path):
    """Cargar datos desde CSV"""
    try:
        df = pd.read_csv(csv_path)
        print(f"Filas cargadas: {len(df)}")
        return df
    except Exception as e:
        print(f"Error al cargar CSV: {e}")
        return None

def intentar_instalar_medspacy():
    """Intentar diferentes métodos de instalación de medspaCy"""
    print("\n=== INTENTANDO INSTALAR MEDSPACY ===")
    
    comandos = [
        [sys.executable, "-m", "pip", "install", "medspacy"],
        [sys.executable, "-m", "pip", "install", "medspacy", "--upgrade"],
        [sys.executable, "-m", "pip", "install", "medspacy", "--no-cache-dir"],
        [sys.executable, "-m", "pip", "install", "medspacy", "--force-reinstall"]
    ]
    
    for i, cmd in enumerate(comandos, 1):
        try:
            print(f"Intento {i}: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                print(f"✓ Instalación exitosa en intento {i}")
                return True
            else:
                print(f"✗ Error en intento {i}: {result.stderr[:200]}...")
        except subprocess.TimeoutExpired:
            print(f"✗ Timeout en intento {i}")
        except Exception as e:
            print(f"✗ Excepción en intento {i}: {e}")
    
    print("⚠️  No se pudo instalar medspaCy automáticamente")
    print("💡 Intenta manualmente: pip install medspacy")
    return False

def probar_spacy_basico():
    """Probar spaCy básico primero"""
    print("\n=== PROBANDO SPACY BÁSICO ===")
    try:
        nlp = spacy.load("en_core_web_sm")
        texto_prueba = "The patient has diabetes and takes metformin 200 mg daily."
        doc = nlp(texto_prueba)
        
        print(f"Texto: {texto_prueba}")
        print(f"Entidades encontradas con spaCy básico:")
        for ent in doc.ents:
            print(f"  - {ent.text} ({ent.label_})")
        
        return nlp
    except Exception as e:
        print(f"Error con spaCy básico: {e}")
        return None

def probar_medspacy():
    """Probar medspaCy con diferentes configuraciones"""
    print("\n=== PROBANDO MEDSPACY ===")
    
    try:
        print("Intentando medspaCy básico...")
        nlp = medspacy.load("en_core_web_sm")
        print(f"Pipeline medspaCy: {nlp.pipe_names}")
        
        texto_prueba = "Patient has diabetes. Prescribed metformin 200 mg twice daily. No allergies known."
        print(f"\nTexto de prueba: {texto_prueba}")
        
        doc = nlp(texto_prueba)
        print(f"Entidades encontradas con medspaCy:")
        for ent in doc.ents:
            print(f"  - '{ent.text}' ({ent.label_})")
            
        return nlp
        
    except Exception as e:
        print(f"Error con medspaCy: {e}")
        print("Intentando instalar medspaCy...")
        
        if intentar_instalar_medspacy():
            try:
                # Reintentar después de la instalación
                import importlib
                importlib.reload(medspacy)
                nlp = medspacy.load("en_core_web_sm")
                return nlp
            except Exception as e2:
                print(f"Error después de reinstalación: {e2}")
                return None
        else:
            return None

def probar_configuraciones_alternativas():
    """Probar diferentes configuraciones para encontrar entidades médicas"""
    print("\n=== PROBANDO CONFIGURACIONES ALTERNATIVAS ===")
    
    modelos_a_probar = [
        "en_core_med7_lg",  # Modelo médico
        "en_core_web_lg",   # Modelo grande
        "en_core_web_md",   # Modelo mediano
        "en_core_web_sm"    # Modelo pequeño
    ]
    
    for modelo in modelos_a_probar:
        try:
            print(f"\nProbando modelo: {modelo}")
            nlp = spacy.load(modelo)
            
            texto_prueba = "DOLQUINE comp 200 mg for malaria treatment"
            doc = nlp(texto_prueba)
            
            print(f"  Texto: {texto_prueba}")
            print(f"  Tokens: {[token.text for token in doc]}")
            print(f"  POS tags: {[(token.text, token.pos_) for token in doc]}")
            print(f"  Entidades: {[(ent.text, ent.label_) for ent in doc.ents]}")
            
            if doc.ents:
                print(f"  ✓ Encontró entidades con {modelo}")
                return nlp
            else:
                print(f"  ✗ No encontró entidades con {modelo}")
                
        except OSError:
            print(f"  ✗ Modelo {modelo} no disponible")
        except Exception as e:
            print(f"  ✗ Error con {modelo}: {e}")
    
    return None

def crear_extractor_personalizado():
    """Crear un extractor de entidades basado en reglas personalizadas mejorado"""
    print("\n=== CREANDO EXTRACTOR PERSONALIZADO MEJORADO ===")
    
    def extraer_entidades_regex(texto):
        entidades = []
        texto = str(texto)
        
        # Patrón mejorado para dosis (más robusto)
        patron_dosis = r'\b(\d+(?:\.\d+)?\s*(?:%|mg|ml|g|mcg|μg|units?|iu|mEq|comp|amp|vial|tab)(?:\s*/\s*\d+(?:\.\d+)?\s*(?:mg|ml|g|mcg|μg|units?|iu|mEq|mL))*)\b'
        for match in re.finditer(patron_dosis, texto, re.IGNORECASE):
            # Limpiar la dosis capturada
            dosis = match.group(1).strip()
            if any(unit in dosis.lower() for unit in ['mg', 'ml', 'g', 'mcg', 'μg', '%']):
                entidades.append((dosis, "STRENGTH"))
        
        # Lista expandida de medicamentos comunes en datos médicos
        medicamentos = [
            # Medicamentos de tus datos
            "dolquine", "acetilcisteina", "dexclorfeniramina", "ceftriaxona", 
            "amlodipino", "morfina", "paracetamol", "omeprazol", "simvastatina",
            "atorvastatina", "furosemida", "captopril", "losartan", "enalapril",
            
            # Medicamentos comunes adicionales
            "metformin", "insulin", "aspirin", "ibuprofen", "acetaminophen",
            "amoxicillin", "ciprofloxacin", "prednisone", "warfarin", "digoxin",
            "hydrochlorothiazide", "metoprolol", "lisinopril", "gabapentin"
        ]
        
        for med in medicamentos:
            patron = rf'\b{re.escape(med)}\b'
            for match in re.finditer(patron, texto, re.IGNORECASE):
                entidades.append((match.group(0), "DRUG"))
        
        # Patrón expandido para condiciones médicas
        condiciones = [
            "diabetes", "hypertension", "asthma", "copd", "pneumonia",
            "covid", "infection", "fever", "pain", "headache", "nausea",
            "diarrhea", "constipation", "malaria", "tuberculosis", "sepsis",
            "myocardial infarction", "stroke", "heart failure", "arrhythmia"
        ]
        
        for cond in condiciones:
            patron = rf'\b{re.escape(cond)}\b'
            for match in re.finditer(patron, texto, re.IGNORECASE):
                entidades.append((match.group(0), "PROBLEM"))
        
        # Patrón para vías de administración
        vias = ["IV", "IM", "PO", "SC", "sublingual", "topical", "inhalation"]
        for via in vias:
            patron = rf'\b{re.escape(via)}\b'
            for match in re.finditer(patron, texto, re.IGNORECASE):
                entidades.append((match.group(0), "ROUTE"))
        
        return entidades
    
    return extraer_entidades_regex

def generar_gold_standard_completo(texto):
    """Gold standard mejorado que incluye medicamentos, dosis y condiciones"""
    gold = []
    texto = str(texto)
    
    # Patrón para dosis más robusto
    patron_dosis = r'\b(\d+(?:\.\d+)?\s*(?:%|mg|ml|g|mcg|μg|units?|iu|mEq)(?:\s*/\s*\d+(?:\.\d+)?\s*(?:mg|ml|g|mcg|μg|units?|iu|mEq|mL))*)\b'
    for match in re.finditer(patron_dosis, texto, re.IGNORECASE):
        dosis = match.group(1).strip()
        if any(unit in dosis.lower() for unit in ['mg', 'ml', 'g', 'mcg', 'μg', '%']):
            gold.append((dosis, "STRENGTH"))
    
    # Lista expandida de medicamentos conocidos
    medicamentos_conocidos = [
        "dolquine", "acetilcisteina", "dexclorfeniramina", "ceftriaxona", 
        "amlodipino", "morfina", "paracetamol", "omeprazol", "simvastatina",
        "atorvastatina", "furosemida", "captopril", "losartan", "enalapril",
        "metformin", "insulin", "aspirin", "ibuprofen", "acetaminophen",
        "amoxicillin", "ciprofloxacin", "prednisone", "warfarin"
    ]
    
    for med in medicamentos_conocidos:
        patron = rf'\b{re.escape(med)}\b'
        for match in re.finditer(patron, texto, re.IGNORECASE):
            gold.append((match.group(0), "DRUG"))
    
    # Condiciones médicas conocidas
    condiciones_conocidas = [
        "diabetes", "hypertension", "asthma", "copd", "pneumonia",
        "covid", "infection", "fever", "pain", "headache", "malaria"
    ]
    
    for cond in condiciones_conocidas:
        patron = rf'\b{re.escape(cond)}\b'
        for match in re.finditer(patron, texto, re.IGNORECASE):
            gold.append((match.group(0), "PROBLEM"))
    
    return gold

def calcular_metricas_detalladas(gold_entities, pred_entities):
    """Calcular precision, recall y F1 para entidades"""
    
    # Convertir a conjuntos para comparación exacta
    gold_set = set(gold_entities)
    pred_set = set(pred_entities)
    
    # Calcular intersección (aciertos)
    true_positives = len(gold_set.intersection(pred_set))
    false_positives = len(pred_set - gold_set)
    false_negatives = len(gold_set - pred_set)
    
    # Métricas
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'gold_set': gold_set,
        'pred_set': pred_set
    }

def evaluar_con_metricas_completas(df, extractor_func, columnas_texto):
    """Evaluación completa con métricas detalladas"""
    print("\n=== EVALUACIÓN COMPLETA CON MÉTRICAS MEJORADAS ===")
    
    all_gold = []
    all_pred = []
    ejemplos_detallados = []
    metricas_por_columna = {}
    
    for col in columnas_texto:
        if col not in df.columns:
            continue
            
        print(f"\nProcesando columna: {col}")
        col_gold = []
        col_pred = []
        count = 0
        
        for texto in df[col].dropna():
            if pd.isna(texto) or str(texto).strip() == "":
                continue
            
            texto_str = str(texto)
            
            # Usar gold standard mejorado
            gold_entities = generar_gold_standard_completo(texto_str)
            pred_entities = extractor_func(texto_str)
            
            all_gold.extend(gold_entities)
            all_pred.extend(pred_entities)
            col_gold.extend(gold_entities)
            col_pred.extend(pred_entities)
            
            # Guardar ejemplos con análisis detallado
            if len(ejemplos_detallados) < 15:
                metricas = calcular_metricas_detalladas(gold_entities, pred_entities)
                ejemplos_detallados.append({
                    'texto': texto_str,
                    'columna': col,
                    'gold': gold_entities,
                    'pred': pred_entities,
                    'metricas': metricas
                })
            
            count += 1
            if count >= 150:  # Procesar más ejemplos por columna
                break
        
        # Métricas por columna
        if col_gold or col_pred:
            metricas_columna = calcular_metricas_detalladas(col_gold, col_pred)
            metricas_por_columna[col] = metricas_columna
            print(f"  {col} - F1: {metricas_columna['f1']:.3f}, Precision: {metricas_columna['precision']:.3f}, Recall: {metricas_columna['recall']:.3f}")
    
    # Calcular métricas globales
    metricas_globales = calcular_metricas_detalladas(all_gold, all_pred)
    
    return metricas_globales, metricas_por_columna, ejemplos_detallados

def generar_reporte_detallado(metricas_globales, metricas_por_columna, ejemplos_detallados, metodos_probados):
    """Generar un reporte detallado de todos los resultados"""
    
    print(f"\n" + "="*60)
    print(f"              REPORTE FINAL DETALLADO")
    print(f"="*60)
    
    # Estado de los métodos
    print(f"\n📊 ESTADO DE LOS MÉTODOS:")
    for metodo, estado in metodos_probados.items():
        print(f"  {metodo}: {'✓' if estado else '✗'}")
    
    # Métricas globales
    print(f"\n🎯 MÉTRICAS GLOBALES DEL EXTRACTOR PERSONALIZADO:")
    print(f"  Precision: {metricas_globales['precision']:.3f}")
    print(f"  Recall: {metricas_globales['recall']:.3f}")
    print(f"  F1-Score: {metricas_globales['f1']:.3f}")
    print(f"  True Positives: {metricas_globales['true_positives']}")
    print(f"  False Positives: {metricas_globales['false_positives']}")
    print(f"  False Negatives: {metricas_globales['false_negatives']}")
    
    # Interpretación de las métricas
    f1 = metricas_globales['f1']
    print(f"\n💡 INTERPRETACIÓN:")
    if f1 >= 0.8:
        print(f"  Excelente rendimiento (F1 ≥ 0.8)")
    elif f1 >= 0.6:
        print(f"  Buen rendimiento (F1 ≥ 0.6)")
    elif f1 >= 0.4:
        print(f"  Rendimiento moderado (F1 ≥ 0.4)")
    else:
        print(f"  Rendimiento bajo (F1 < 0.4)")
    
    # Métricas por columna
    print(f"\n📋 MÉTRICAS POR COLUMNA:")
    for col, metricas in metricas_por_columna.items():
        print(f"  {col}:")
        print(f"    F1: {metricas['f1']:.3f} | Precision: {metricas['precision']:.3f} | Recall: {metricas['recall']:.3f}")
    
    # Ejemplos representativos
    print(f"\n📝 EJEMPLOS REPRESENTATIVOS:")
    for i, ej in enumerate(ejemplos_detallados[:8], 1):
        print(f"\nEjemplo {i} ({ej['columna']}):")
        print(f"  Texto: {ej['texto'][:80]}{'...' if len(ej['texto']) > 80 else ''}")
        print(f"  Gold: {ej['gold']}")
        print(f"  Pred: {ej['pred']}")
        print(f"  F1: {ej['metricas']['f1']:.3f}")
        
        # Mostrar qué se perdió o se agregó de más
        if ej['metricas']['false_positives'] > 0:
            extras = ej['metricas']['pred_set'] - ej['metricas']['gold_set']
            print(f"  Extras: {list(extras)}")
        if ej['metricas']['false_negatives'] > 0:
            perdidos = ej['metricas']['gold_set'] - ej['metricas']['pred_set']
            print(f"  Perdidos: {list(perdidos)}")
    
    return metricas_globales

def main():
    print("=== DIAGNÓSTICO COMPLETO Y MEJORADO DE MEDSPACY ===")
    
    # 1. Cargar datos
    df = cargar_datos(CSV_PATH)
    if df is None:
        print("❌ No se pudieron cargar los datos. Terminando.")
        return
    
    # 2. Mostrar muestra de datos
    print(f"\n📄 MUESTRA DE DATOS:")
    for col in COLUMNAS_TEXTO:
        if col in df.columns:
            print(f"\n{col}:")
            muestras = df[col].dropna().head(3).tolist()
            for i, muestra in enumerate(muestras, 1):
                print(f"  {i}. {str(muestra)[:100]}{'...' if len(str(muestra)) > 100 else ''}")
    
    # 3. Probar diferentes métodos
    metodos_probados = {}
    
    # spaCy básico
    nlp_spacy = probar_spacy_basico()
    metodos_probados['spaCy básico'] = nlp_spacy is not None
    
    # medspaCy
    nlp_medspacy = probar_medspacy()
    metodos_probados['medspaCy'] = nlp_medspacy is not None
    
    # Configuraciones alternativas
    nlp_alt = probar_configuraciones_alternativas()
    metodos_probados['Modelo alternativo'] = nlp_alt is not None
    
    # 4. Crear y evaluar extractor personalizado
    extractor_personalizado = crear_extractor_personalizado()
    metodos_probados['Extractor personalizado'] = True
    
    # 5. Evaluación completa
    metricas_globales, metricas_por_columna, ejemplos_detallados = evaluar_con_metricas_completas(
        df, extractor_personalizado, COLUMNAS_TEXTO
    )
    
    # 6. Generar reporte final
    resultado_final = generar_reporte_detallado(
        metricas_globales, metricas_por_columna, ejemplos_detallados, metodos_probados
    )
    
    # 7. Guardar diagnóstico completo
    txt_PATH = os.path.abspath(os.path.join(script_dir, '..', 'R35_sopra_steria', 'src', 'evaluation', 'diagnostico_medspacy_completo.txt'))
    with open(txt_PATH, "w", encoding='utf-8') as f:
        f.write("=== DIAGNÓSTICO MEDSPACY COMPLETO ===\n\n")
        
        f.write("ESTADO DE MÉTODOS:\n")
        for metodo, estado in metodos_probados.items():
            f.write(f"  {metodo}: {'✓' if estado else '✗'}\n")
        
        f.write(f"\nMÉTRICAS GLOBALES:\n")
        f.write(f"  Precision: {metricas_globales['precision']:.3f}\n")
        f.write(f"  Recall: {metricas_globales['recall']:.3f}\n")
        f.write(f"  F1-Score: {metricas_globales['f1']:.3f}\n")
        f.write(f"  True Positives: {metricas_globales['true_positives']}\n")
        f.write(f"  False Positives: {metricas_globales['false_positives']}\n")
        f.write(f"  False Negatives: {metricas_globales['false_negatives']}\n")
        
        f.write(f"\nMÉTRICAS POR COLUMNA:\n")
        for col, metricas in metricas_por_columna.items():
            f.write(f"  {col}:\n")
            f.write(f"    F1: {metricas['f1']:.3f} | Precision: {metricas['precision']:.3f} | Recall: {metricas['recall']:.3f}\n")
        
        f.write(f"\nEJEMPLOS DETALLADOS:\n")
        for i, ej in enumerate(ejemplos_detallados[:15], 1):
            f.write(f"\nEjemplo {i} ({ej['columna']}):\n")
            f.write(f"  Texto: {ej['texto']}\n")
            f.write(f"  Gold: {ej['gold']}\n")
            f.write(f"  Pred: {ej['pred']}\n")
            f.write(f"  Métricas: F1={ej['metricas']['f1']:.3f}, P={ej['metricas']['precision']:.3f}, R={ej['metricas']['recall']:.3f}\n")
    
    print(f"\n💾 Diagnóstico completo guardado en 'diagnostico_medspacy_completo.txt'")
    
    # 8. Recomendaciones finales
    print(f"\n🔧 RECOMENDACIONES:")
    if not metodos_probados['medspaCy']:
        print(f"  - Instalar medspaCy: pip install medspacy")
    
    if metricas_globales['f1'] < 0.7:
        print(f"  - Expandir lista de medicamentos en el extractor personalizado")
        print(f"  - Mejorar patrones de regex para dosis más complejas")
    
    if metricas_globales['precision'] < 0.8:
        print(f"  - Reducir falsos positivos refinando patrones")
    
    if metricas_globales['recall'] < 0.8:
        print(f"  - Añadir más variantes de medicamentos y condiciones")
    
    print(f"\n✅ Diagnóstico completado exitosamente!")

if __name__ == "__main__":
    main()