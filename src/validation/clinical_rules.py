import os
import json
import datetime
# Analgésicos y Antiinflamatorios
analgesicos_antiinflamatorios = [ 'PARACETAMOLL', 'PARACETAMOL', 'NORAGESL', 'METAMIZOL', 'METAMIZOLL', 'IBUPROFENO', 'ENANTYUM', 'ENANTYUML', 'ASPIRINA', 'ADIRO', 'INYESPRIN', 'NORAGES']
opioides_potentes = ['MORFINALL', 'MSTCONTINUS', 'TRAMADOLL', 'CODEISAN', 'TOSEINALFL', 'TARGIN', 'MORFINA', 'TOSEINA']
psicofarmacos = ['DIAZEPAN', 'LORAZEPAM', 'BROMAZEPAM', 'MIDAZOLAML', 'HALOPERIDOLL', 'HALOPERIDOLLFL', 'DEPRAX', 'AKINETON', 'APOGOL', 'NEUPRO', 'LYRICA', 'EXELON']
# Fármacos Cardiovasculares
farmacos_cardiovasculares = ['BISOPROLOL', 'CARVEDILOL', 'TENORMIN', 'CORPITOLL', 'BELOKEN', 'SUMIAL', 'TRANDATEL', 'TIMOFTOL', 'TIMOFTOLLFL', 'CAPTOPRIL', 'ENALAPRIL', 'DIOVAN', 'AMLODIPINO', 'ADALAT', 'CLOPIDOGREL', 'XARELTO', 'CLEXANEL', 'HIBOR','HIBORUIL', 'FUROSEMIDA', 'FUROSEMIDAL', 'ATORVASTATINA', 'APOCARD']
# Fármacos del Aparato Respiratorio
farmacos_respiratorios = ['RELVARELLIPTA', 'RELVAR ELLIPTA', 'VENTOLIN', 'VENTOLINL', 'SALBUAIR', 'SALBUAIRL', 'BROMUROIPRATROPIOL', 'ATROVENTL', 'ULUNAR INH', 'ULUNARINH', 'SYMBICORTFORTE', 'ATROALDO', 'AMNIOLINAT', 'BUDESONIDA', 'BUDESONIDAL']
# Agentes Infecciosos (Antibióticos)
antibioticos = ['AMOXICILINACLAV', 'PIPERACILINATAZOBACTAM', 'CEFTRIAXONA', 'CEFTRIAXONAIV', 'AZITROMICINA', 'CIPROFLOXACINOL', 'LEVOFLOXACINO', 'LEVOFLOXACINOL', 'METRONIDAZOL', 'METRONIDAZOLL', 'SEPTRINFORTE', 'AMICACINAL', 'AUREOMICINAT', '"AUREOMICINA']
# Agentes Infecciosos (Antivirales)
antivirales = [ 'ACICLOVIR', 'KALETRAALUVIA', 'ALUVIA', 'VEKLURY']
# Agentes Infecciosos (Antifúngicos)
antifungicos = ['MYCOSTATINUILFL']
# Corticosteroides
corticosteroides = ['PREDNISONA', 'METILPREDNISOLONA', 'URBASON', 'DEXAMETASONAL', 'DEXAMETASONATAD', 'ACTOCORTINA', 'SYNALARRECTALT', 'DEXAMETASONA']
# Fármacos del Aparato Digestivo
farmacos_digestivos = ['OMEPRAZOL', 'RANITIDINAL', 'ONDANSETRONL', 'ONDANSETRON', 'BUSCAPINAL', 'BUSCAPINA', 'DUPHALACL', 'MOVICOL', 'ENEMACASENFL', 'CLISTERANL', 'CLAVERSAL', 'SALOFALKL', 'SALAZOPYRINA', 'ULTRALEVURA']
# Anestésicos Locales
anestesicos_locales = ['LIDOCAINALL', 'BUPIVACAINALSA', 'MEPIVACAINALL']
# Agentes Vasoactivos (Simpaticomiméticos)
agentes_vasoactivos = ['EFEDRINAL', 'ADRENALINAL', 'NORADRENALINAL', 'DOPAMINAL', 'DOBUTAMINAL', 'ALEUDRINAL', 'DOBUTAMINA', 'EFEDRINA', 'ADRENALINA', 'ALEUDRINA']
# Inmunomoduladores e Inmunosupresores
inmunomoduladores = ['DOLQUINE', 'HIDROXICLOROQUINA', 'RESOCHIN', 'SANDIMMUNNEORAL', 'SANDIMMUN NEORAL','ROACTEMRAL', 'BETAFERONL', 'BETAFERON', 'ROACTEMRA']
# Hormonas y Metabolismo
hormonas_y_metabolismo = ['EUTIROX', 'LEVOTIROXINA', 'LEVOTIROXINA', 'METFORMINA', 'ACTRAPIDUILFL', 'ACTRAPID']
# Antihistamínicos
antihistaminicos = ['DEXCLORFENIRAMINAL', 'POLARAMINELFL', 'POLARAMINE', 'DEXCLORFENIRAMINA']
# Antisépticos y Desinfectantes
antisepticos_desinfectantes = ['CLORHEXIDINA ACUOSA', 'CLORHEXIDINA', 'CLORHEXIDINAL', 'DESINCLOR', 'DESINCLORJABONOSOFL', 'ALCOHOL', 'ALCOHOLÂL', 'AGUA OXIGENADA', 'AGUAOXIGENADAL', 'BETADINEAMARILLOFL', 'DESINCLORFL', 'CLORHEXIDINAACUOSAL', 'BETADINE', 'GEL PURELL ADVANCE', 'GELPURELLADVANCEL', 'GELPURELLADVANCETFXOPTICOL', 'CLORHEXIDINAJABONOSAL', 'ACETONAL']
# Soluciones, Suplementos y Productos Sanitarios
soluciones_suplementos_y_productos_sanitarios = ['AGUA ESTERIL', 'AGUA DESTILADA', 'AGUABIDESTILADAL','SUERO GLUCOSADO', 'SUERO ClNa 100 mL I.V.','SUEROCNLIV', 'AGUA BIDESTILADA', 'AGUADESTILADALIRRIGACION', 'CLORURO SODICO', 'CLORUROSODICOL', 'ACETATOPOTASICOML', 'AGUAESTERILLIRRIGACION', 'SUEROGLUCOSADOL', 'ENSURE PLUS ADVANCE VAINILLA', 'ENSUREPLUSADVANCECHOCOLATEFL', 'RESOURCEESPESANTE', 'CLORUROPOTASICOML', 'CLORUROCALCICOLE', 'ALBUMINA', 'ALBUMINALL', 'ENSUREPLUSADVANCEVAINILLAFL', 'OXIGENO PLANTA', 'OXIGENOPLANTA', 'ULTRAVISTLFL', 'LINITULAPOSITO', 'INSTRUNETENZIMATICOEZTPLUSL']
# Otros Fármacos y Tratamientos
otros_tratamientos = ['ACFOL', 'AMCHAFIBRINL', 'AMCHAFIBRIN', 'ATROPINAL', 'ATROPINA', 'LEDERFOLIN', 'GANFORTLFL', 'TAMSULOSINA', 'ACETILCISTEINA', 'ACCOFILMUL', 'ARANESP', 'ARANESPL', 'CISATRACURIO', 'CISATRACURIOL', 'COLCHICINA', 'ALOPURINOL', 'ACIDO ASCORBICO', 'ACIDOASCORBICOL']

def warning_pcr_baja():
    """Warning común que se repite en múltiples casos"""
    return ('ATENCIÓN: Paciente sin inflamación sistémica significativa (PCR <= 10 mg/L). '
            'La evidencia indica que este uso no aporta beneficios y puede aumentar la mortalidad. '
            'El diagnóstico de COVID-19 puede corresponder a un caso leve, asintomático o en fase de recuperación. '
            'Reevaluar urgentemente la indicación del tratamiento.')

def warning_pcr_baja():
    """Warning común para PCR <= 10.0"""
    return ('ATENCIÓN: Paciente sin inflamación sistémica significativa (PCR <= 10 mg/L). '
            'La evidencia indica que este uso no aporta beneficios y puede aumentar la mortalidad. '
            'El diagnóstico de COVID-19 puede corresponder a un caso leve, asintomático o en fase de recuperación. '
            'Reevaluar urgentemente la indicación del tratamiento.')

def evaluar_analgesicos_antiinflamatorios(pcr, saturacion_oxigeno, temperatura):
    if pcr > 10.0:
        return ('ALERTA CLÍNICA: Paciente con COVID-19 y PCR elevada recibe un anti-inflamatorio (ej. Ibuprofeno, Metamizol). '
                'Aunque la evidencia sobre el empeoramiento viral no es concluyente, se asocia a riesgos de lesión renal y gastrointestinal en enfermedad aguda. '
                'Considere el uso de paracetamol como primera opción y monitorice la función renal.')
    else:
        return warning_pcr_baja()

def evaluar_opioides_potentes(pcr, saturacion_oxigeno, temperatura):
    if pcr > 10.0:
        return ('ALERTA DE MANEJO DE PACIENTE CON COVID-19: Paciente con alta carga viral (PCR elevada) recibe un opioide potente. '
                'Esto puede aumentar el riesgo de depresión respiratoria y complicaciones. '
                'Recomendación: Reevaluar la necesidad del opioide y vigilar posible depresión respiratoria.')
    else:
        return warning_pcr_baja()

def evaluar_psicofarmacos(pcr, saturacion_oxigeno, temperatura):
    if pcr > 10.0:
        return ('ALERTA DE SEGURIDAD FARMACOLÓGICA: Paciente con COVID-19 recibe un psicofármaco. '
                'Riesgos potenciales de prolongación del QTc (arritmia), interacciones con antivirales (ej. Paxlovid) y/o sedación excesiva. '
                'Recomendación: 1) Realizar ECG para evaluar intervalo QTc. 2) Revisar compatibilidad con tratamiento antiviral. 3) Vigilar nivel de conciencia y función respiratoria.')
    else:
        return warning_pcr_baja()

def evaluar_farmacos_cardiovasculares(pcr, saturacion_oxigeno, temperatura):
    if pcr > 10.0:
        return ('ADVERTENCIA DE MANEJO CARDIOVASCULAR: Paciente con COVID-19 en tratamiento con IECA/ARA II. '
                'La evidencia actual desaconseja la suspensión rutinaria de estos fármacos. '
                'Su uso identifica a un paciente de alto riesgo cardiovascular. '
                'Recomendación: 1) Mantener tratamiento si está hemodinámicamente estable. 2) Considerar ajuste o suspensión TEMPORAL solo en caso de hipotensión severa, shock o lesión renal aguda. 3) Monitorizar presión arterial y función renal.')
    else:
        return warning_pcr_baja()

def evaluar_farmacos_respiratorios(pcr, saturacion_oxigeno, temperatura):
    if pcr > 10.0:
        return ('ALERTA DE MANEJO RESPIRATORIO: Paciente con COVID-19 recibe fármacos respiratorios (broncodilatadores/ICS). '
                'Esto sugiere una patología de base (Asma/EPOC), elevando el riesgo de exacerbación y COVID-19 grave. '
                'Recomendaciones: 1) Diferenciar entre la disnea por neumonitis viral y la disnea por broncoespasmo. 2) NO suspender el tratamiento respiratorio de base; puede ser necesario intensificarlo. 3) Considerar el uso de corticoides inhalados (ej. Budesonida) como tratamiento para la propia COVID-19, según guías.')
    else:
        return warning_pcr_baja()

def evaluar_antivirales(pcr, saturacion_oxigeno, temperatura):
    if pcr > 10.0:
        return ('AVISO DE MANEJO DE ANTIVIRAL: Paciente con COVID-19 en tratamiento antiviral específico. '
                'Fármaco indicado. '
                'Acción Requerida: Revisar urgentemente el resto de la medicación del paciente por posibles interacciones farmacológicas graves (especialmente con Paxlovid).')
    else:
        return warning_pcr_baja()

def evaluar_antifungicos(pcr, saturacion_oxigeno, temperatura):
    if pcr > 10.0:
        return ('ALERTA DE ALTA GRAVEDAD: Paciente con COVID-19 recibe un antifúngico. '
                'Esto es un fuerte indicador de una sobreinfección fúngica grave (ej. CAPA), asociada a una alta mortalidad. '
                'Requiere manejo agresivo y probable ingreso en UCI.')
    else:
        return warning_pcr_baja()

def evaluar_antibioticos(pcr, saturacion_oxigeno, temperatura):
    if pcr > 10.0:
        return ('ADVERTENCIA DE USO DE ANTIBIÓTICOS: Paciente con COVID-19 en tratamiento antibiótico. '
                'Recordar que no tratan la infección viral. '
                'Recomendación: Asegurar que su uso está justificado por sospecha o confirmación de sobreinfección bacteriana para promover la buena práctica (antibiotic stewardship).')
    else:
        return warning_pcr_baja()

def evaluar_corticosteroides(pcr, saturacion_oxigeno, temperatura):
    """Caso especial con condición de saturación de oxígeno"""
    if saturacion_oxigeno < 94 and pcr > 10.0:
        return ('AVISO DE MANEJO: Paciente con COVID-19 grave (SatO2 < 94%) recibe corticosteroides. '
                'Tratamiento indicado y recomendado por la OMS para reducir la mortalidad. '
                'Recomendaciones: 1) Asegurar dosis estándar (ej. Dexametasona 6mg/día). 2) Monitorizar glucemia y vigilar sobreinfecciones.')
    else:
        return warning_pcr_baja()

def evaluar_farmacos_digestivos(pcr, saturacion_oxigeno, temperatura):
    if pcr > 10.0:
        return ('ALERTA DE VULNERABILIDAD GASTROINTESTINAL: Paciente con COVID-19 recibe fármacos gastroprotectores (ej. Omeprazol). '
                'Esto indica una patología digestiva de base, aumentando el riesgo de complicaciones por el "doble impacto" del virus y sus tratamientos. '
                'El virus puede afectar directamente el sistema digestivo[3, 5], y fármacos como los corticosteroides, antivirales y antibióticos son frecuentemente gastrolesivos. '
                'Recomendaciones: 1) Máxima vigilancia de síntomas como dolor abdominal, náuseas o diarrea. 2) Si se inician corticosteroides, asegurar que la pauta de protección gástrica es adecuada. 3) Ante nuevos síntomas digestivos, evaluar si son por COVID-19, efectos adversos de otros fármacos o descompensación de su enfermedad de base.')
    else:
        return warning_pcr_baja()

def evaluar_anestesicos_locales(pcr, saturacion_oxigeno, temperatura):
    """Caso especial con múltiples condiciones"""
    if (saturacion_oxigeno < 94 and pcr > 10.0) or (temperatura > 39 and pcr > 10.0):
        return ('ALERTA DE ALTO RIESGO PROCEDIMENTAL: Paciente con COVID-19 grave (SatO2 < 94% o fiebre alta) va a someterse a un procedimiento con anestesia local. '
                'Aunque la anestesia local/regional es preferible a la general, el estrés fisiológico del procedimiento en sí puede causar una descompensación clínica en un paciente con reserva limitada. '
                'Recomendaciones: 1) Confirmar que el procedimiento es urgente e inaplazable. 2) Asegurar monitorización continua de signos vitales (SpO2, FC, PA) durante y después. 3) Realizar en un entorno con capacidad de soporte vital de emergencia disponible.')
    else:
        return warning_pcr_baja()

def evaluar_agentes_vasoactivos(pcr, saturacion_oxigeno, temperatura):
    if pcr > 10.0:
        return ('ALERTA DE EXTREMA GRAVEDAD (ESTADO DE SHOCK): Paciente con COVID-19 recibe un agente vasoactivo (ej. Norepinefrina, Dopamina). '
                'Esto indica que el paciente se encuentra en shock circulatorio, una complicación potencialmente mortal que requiere manejo inmediato en una Unidad de Cuidados Intensivos (UCI). '
                'Recomendaciones de manejo: '
                '1) La norepinefrina es el vasopresor de primera línea recomendad. '
                '2) El uso de dopamina está desaconsejado por su perfil de seguridad inferior y mayor riesgo de arritmias. '
                '3) Si se utiliza dobutamina, sospechar disfunción cardíaca (shock cardiogénico) y evaluar función del corazón. '
                '4) Este paciente debe ser considerado de máxima criticidad, con necesidad de monitorización invasiva y soporte vital avanzado.')
    else:
        return warning_pcr_baja()

def evaluar_inmunomoduladores(pcr, saturacion_oxigeno, temperatura):
    if pcr > 10.0:
        return ('Riesgo Principal: Riesgo elevado de progresión a enfermedad grave y de eliminación viral prolongada debido a una respuesta inmunitaria inicial deficiente. '
                'Recomendación: Considerar la modificación o suspensión temporal del tratamiento inmunosupresor de base, siempre en estricta consulta con el especialista correspondiente (reumatólogo, nefrólogo, etc.). Monitorizar de cerca la evolución clínica.')
    else:
        return warning_pcr_baja()

def evaluar_hormonas_y_metabolismo(pcr, saturacion_oxigeno, temperatura):
    if pcr > 10.0:
        return ('ALERTA DE ALTO RIESGO METABÓLICO Y VULNERABILIDAD A COVID-19 GRAVE: Paciente en tratamiento con hormonas/fármacos para el metabolismo. '
                'NO SUSPENDER EL TRATAMIENTO DE BASE: Es fundamental mantener el tratamiento metabólico crónico (insulina, antidiabéticos orales, levotiroxina) para evitar una descompensación aguda. La suspensión debe ser valorada únicamente por un especialista. '
                'MONITORIZACIÓN ESTRICTA DE LA GLUCEMIA: Realizar controles de glucosa capilar frecuentes es obligatorio, especialmente si el paciente requiere ingreso hospitalario o se inicia tratamiento con corticosteroides. '
                'PREPARACIÓN PARA INTENSIFICAR EL TRATAMIENTO: Estar preparado para ajustar las dosis de insulina o iniciar insulinoterapia si el control glucémico se deteriora. '
                'VIGILANCIA TIROIDEA: En pacientes con patología tiroidea conocida, considerar la monitorización de la función tiroidea (TSH, T4L) si se produce un deterioro clínico inexplicado.')
    else:
        return warning_pcr_baja()

def evaluar_antihistaminicos(pcr, saturacion_oxigeno, temperatura):
    if pcr > 10.0:
        return ('AVISO CLÍNICO: Uso de Antihistamínicos en Pacientes con COVID-19: '
                'Riesgo de Enmascaramiento Sintomático y Efectos Secundarios. El uso de antihistamínicos en un paciente con una infección activa por COVID-19 requiere una evaluación cuidadosa por dos motivos principales: '
                '1. Riesgo de Enmascaramiento de Síntomas y Retraso Diagnóstico (Principal Alerta): Ante cualquier síntoma respiratorio superior, incluso en un paciente con historial de alergias, se debe mantener un alto índice de sospecha de COVID-19 y proceder con las pruebas diagnósticas pertinentes. '
                '2. Riesgo Asociado al Tipo de Antihistamínico (Perfil de Seguridad): La advertencia clínica difiere significativamente según la generación del fármaco: '
                '1º GENERACION: Evitar su uso en la medida de lo posible en pacientes con COVID-19 sintomático. Si se necesita un antihistamínico, optar por uno de segunda generación. '
                '2º GENERACION: Preferible su uso, pero con precaución. Son la opción de elección si el paciente requiere tratamiento para una condición alérgica concurrente (como una urticaria o rinitis crónica). No existe evidencia sólida para suspenderlos.')
    else:
        return warning_pcr_baja()

def evaluar_antisepticos_desinfectantes(pcr, saturacion_oxigeno, temperatura):
    if pcr > 10.0:
        return ('ALERTA DE MANEJO DE ANTISÉPTICOS/DESINFECTANTES: Paciente con COVID-19 en tratamiento con antisépticos o desinfectantes. '
                'Aunque estos productos son seguros para la higiene de manos y superficies, su uso excesivo puede causar irritación cutánea o reacciones alérgicas.')
    else:
        return warning_pcr_baja()

def evaluar_soluciones_suplementos_y_productos_sanitarios(pcr, saturacion_oxigeno, temperatura):
    if pcr > 10.0:
        return ('ALERTA DE MANEJO DE SOLUCIONES Y PRODUCTOS SANITARIOS: Paciente con COVID-19 en tratamiento con soluciones, suplementos o productos sanitarios. '
                'Estos productos son generalmente seguros, pero es importante asegurarse de que no interfieran con el tratamiento antiviral o con la función renal del paciente. '
                'Recomendación: Revisar la composición y posibles interacciones con otros medicamentos.')
    else:
        return warning_pcr_baja()

def evaluar_otros_tratamientos(pcr, saturacion_oxigeno, temperatura):
    if pcr > 10.0:
        return ('ALERTA DE MANEJO DE OTROS TRATAMIENTOS: Paciente con COVID-19 en tratamiento con otros fármacos no clasificados. '
                'Es importante revisar la indicación y posibles interacciones con el tratamiento antiviral. '
                'Recomendación: Asegurarse de que estos fármacos no interfieran con el tratamiento antiviral.')
    else:
        return warning_pcr_baja()

# Diccionario de mapeo principal
EVALUADORES_MEDICAMENTOS = {
    'analgesicos_antiinflamatorios': evaluar_analgesicos_antiinflamatorios,
    'opioides_potentes': evaluar_opioides_potentes,
    'psicofarmacos': evaluar_psicofarmacos,
    'farmacos_cardiovasculares': evaluar_farmacos_cardiovasculares,
    'farmacos_respiratorios': evaluar_farmacos_respiratorios,
    'antivirales': evaluar_antivirales,
    'antifungicos': evaluar_antifungicos,
    'antibioticos': evaluar_antibioticos,
    'corticosteroides': evaluar_corticosteroides,
    'farmacos_digestivos': evaluar_farmacos_digestivos,
    'anestesicos_locales': evaluar_anestesicos_locales,
    'agentes_vasoactivos': evaluar_agentes_vasoactivos,
    'inmunomoduladores': evaluar_inmunomoduladores,
    'hormonas_y_metabolismo': evaluar_hormonas_y_metabolismo,
    'antihistaminicos': evaluar_antihistaminicos,
    'antisepticos_desinfectantes': evaluar_antisepticos_desinfectantes,
    'soluciones_suplementos_y_productos_sanitarios': evaluar_soluciones_suplementos_y_productos_sanitarios,
    'otros_tratamientos': evaluar_otros_tratamientos
}

def validate_patient_case(patient):
    """
    La función validate_patient_case toma un diccionario que representa el historial
    clínico de un paciente y genera warnings si se detectan condiciones específicas.
    en los tratamientos y diagnosticos del paciente. Aplica reglas clínicas para
    validar coherencia y compatibilidad diagnóstica-terapéutica. 
    parámetros:
    - patient: un diccionario que contiene el historial clínico del paciente. Contiene
    información sobre el diagnóstico, medicaciones y métricas de salud.
    return:
    - warnings: una lista de advertencias generadas a partir de las condiciones
    """
    warnings = []
    
    # Extraer datos del paciente
    farmaco_paciente = patient.get('FARMACO/DRUG_NOMBRE_COMERCIAL/COMERCIAL_NAME', '')
    diagnostico = patient.get('DIAG ING/INPAT', '')
    saturacion_oxigeno = patient.get('SAT_02_ING/INPAT', 99)
    temperatura = float(patient.get('TEMP_ING/INPAT', 37))
    pcr = float(patient.get('RESULTADO/VAL_RESULT', 10.0))
    
    # Solo procesar si es COVID-19 positivo
    if diagnostico == 'COVID19 - POSITIVO':
        warning_agregado = False
        
        # Verificar si el farmaco_paciente es string
        if isinstance(farmaco_paciente, str):
            # Iterar sobre todos los evaluadores de medicamentos
            for grupo_medicamentos, evaluador in EVALUADORES_MEDICAMENTOS.items():
                if any(medicamento in farmaco_paciente.upper() 
                    for medicamento in globals()[grupo_medicamentos]):
                        warnings.append(evaluador(pcr, saturacion_oxigeno, temperatura))
                        warning_agregado = True
                        break
        
        # Si no se encontró ningún medicamento específico
        if not warning_agregado:
            warnings.append('No se han encontrado interacciones o alertas específicas para el tratamiento actual del paciente con COVID-19.')
    
    return warnings


script_dir = os.getcwd()
JSON_PATH = os.path.abspath(os.path.join(script_dir, 'data', 'synthetic', 'datos_sinteticos_sdv.json'))
JSON_PATH_2 = os.path.abspath(os.path.join(script_dir, 'data', 'synthetic', 'datos_sinteticos_tvae.json'))

def procesar_archivo_json(nombre_archivo):
    log_lines = []
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_lines.append(f"Reporte de validación - {timestamp}")
    log_lines.append(f"Archivo: {nombre_archivo}")
    log_lines.append("-" * 50)
    
    print(f"Procesando el archivo: {nombre_archivo}\n")
    
    # Función auxiliar para generar el log
    def guardar_log():
        # Extraer nombre base del archivo JSON (sin extensión ni ruta)
        archivo_base = os.path.splitext(os.path.basename(nombre_archivo))[0]
        
        # Generar nombre de log único basado en el archivo JSON
        base_name = f'log_clinical_rules_{archivo_base}.txt'
        log_name = base_name
        count = 1
        
        # Verificar en el directorio outputs
        output_dir = os.path.abspath(os.path.join(script_dir, 'outputs'))
        os.makedirs(output_dir, exist_ok=True)
        
        while os.path.exists(os.path.join(output_dir, log_name)):
            log_name = f"log_clinical_rules_{archivo_base}_{count}.txt"
            count += 1
        
        # Escribir el archivo log
        log_path = os.path.join(output_dir, log_name)
        with open(log_path, 'w', encoding='utf-8') as log_file:
            for line in log_lines:
                log_file.write(line + '\n')
        
        print(f"✅ Log guardado en '{log_path}'")
        return log_path
    
    try:
        with open(nombre_archivo, 'r', encoding='utf-8') as f:
            for numero_linea, linea in enumerate(f, 1):
                if not linea.strip():
                    continue
                
                try:
                    datos_paciente = json.loads(linea)
                    
                    # Captura la lista de warnings devuelta por la función
                    warnings = validate_patient_case(datos_paciente)
                    
                    # Si la lista no está vacía, imprime cada warning
                    if warnings:
                        print(f"Alertas para la línea {numero_linea}:")
                        log_lines.append(f"Línea {numero_linea} - ALERTAS:")
                        
                        for warning in warnings:
                            print(f"  - {warning}")
                            log_lines.append(f"  {warning}")
                        
                        log_lines.append("")  # Línea en blanco para separar
                    else:
                        log_lines.append(f"Línea {numero_linea}: Sin alertas")

                except json.JSONDecodeError as e:
                    error_msg = f"❌ Error al decodificar JSON en la línea {numero_linea}: {e}"
                    print(error_msg)
                    log_lines.append(error_msg)
                
                print("-" * 20)

        # Guardar log al finalizar correctamente
        guardar_log()

    except FileNotFoundError:
        error_msg = f"Error: El archivo '{nombre_archivo}' no fue encontrado."
        print(error_msg)
        log_lines.append(error_msg)
        
        # Guardar log incluso en caso de error
        guardar_log()

# Ejecutar para ambos archivos
procesar_archivo_json(JSON_PATH)
procesar_archivo_json(JSON_PATH_2)
