import os
import json

# Analgésicos y Antiinflamatorios
analgesicos_antiinflamatorios = [
    'PARACETAMOLL', 'PARACETAMOL', 'NORAGESL', 'METAMIZOL', 'METAMIZOLL', 
    'IBUPROFENO', 'ENANTYUM', 'ENANTYUML', 'ASPIRINA', 'ADIRO', 'INYESPRIN'
]

opioides_potentes = [
        'MORFINALL',        # Morfina
        'MSTCONTINUS',      # Morfina de liberación prolongada
        'TRAMADOLL',        # Tramadol
        'CODEISAN',         # Codeína
        'TOSEINALFL',       # Codeína (antitusígeno)
        'TARGIN'            # Oxicodona/Naloxona
    ]

# Psicofármacos
psicofarmacos = [
    'DIAZEPAN', 'LORAZEPAM', 'BROMAZEPAM', 'MIDAZOLAML', 'HALOPERIDOLL', 
    'HALOPERIDOLLFL', 'DEPRAX', 'AKINETON', 'APOGOL', 'NEUPRO', 'LYRICA', 
    'EXELON'
]

# Fármacos Cardiovasculares
farmacos_cardiovasculares = [
    'BISOPROLOL', 'CARVEDILOL', 'TENORMIN', 'CORPITOLL', 'BELOKEN', 'SUMIAL', 
    'TRANDATEL', 'TIMOFTOLLFL', 'CAPTOPRIL', 'ENALAPRIL', 'DIOVAN', 
    'AMLODIPINO', 'ADALAT', 'CLOPIDOGREL', 'XARELTO', 'CLEXANEL', 'HIBORUIL', 
    'FUROSEMIDAL', 'ATORVASTATINA', 'APOCARD'
]

# Fármacos del Aparato Respiratorio
farmacos_respiratorios = [
    'RELVARELLIPTA', 'VENTOLIN', 'VENTOLINL', 'SALBUAIRL', 'BROMUROIPRATROPIOL', 
    'ATROVENTL', 'ULUNARINH', 'SYMBICORTFORTE', 'ATROALDO', 'AMNIOLINAT', 
    'BUDESONIDA', 'BUDESONIDAL'
]

# Agentes Infecciosos (Antibióticos)
antibioticos = [
    'AMOXICILINACLAV', 'PIPERACILINATAZOBACTAM', 'CEFTRIAXONAIV', 'AZITROMICINA',
    'CIPROFLOXACINOL', 'LEVOFLOXACINOL', 'METRONIDAZOLL', 'SEPTRINFORTE', 
    'AMICACINAL', 'AUREOMICINAT'
]

# Agentes Infecciosos (Antivirales)
antivirales = [ 'ACICLOVIR', 'KALETRAALUVIA', 'ALUVIA', 
    'VEKLURY'
]

# Agentes Infecciosos (Antifúngicos)
antifungicos = ['MYCOSTATINUILFL']

# Corticosteroides
corticosteroides = [
    'PREDNISONA', 'METILPREDNISOLONA', 'URBASON', 'DEXAMETASONAL', 
    'DEXAMETASONATAD', 'ACTOCORTINA', 'SYNALARRECTALT'
]

# Fármacos del Aparato Digestivo
farmacos_digestivos = [
    'OMEPRAZOL', 'RANITIDINAL', 'ONDANSETRONL', 'BUSCAPINAL', 'BUSCAPINA', 
    'DUPHALACL', 'MOVICOL', 'ENEMACASENFL', 'CLISTERANL', 'CLAVERSAL', 
    'SALOFALKL', 'SALAZOPYRINA', 'ULTRALEVURA'
]

# Anestésicos Locales
anestesicos_locales = ['LIDOCAINALL', 'BUPIVACAINALSA', 'MEPIVACAINALL']

# Agentes Vasoactivos (Simpaticomiméticos)
agentes_vasoactivos = [
    'EFEDRINAL', 'ADRENALINAL', 'NORADRENALINAL', 'DOPAMINAL', 'DOBUTAMINAL', 
    'ALEUDRINAL'
]

# Inmunomoduladores e Inmunosupresores
inmunomoduladores = [
    'DOLQUINE', 'HIDROXICLOROQUINA', 'RESOCHIN', 'SANDIMMUNNEORAL', 'ROACTEMRAL',
    'BETAFERONL'
]

# Hormonas y Metabolismo
hormonas_y_metabolismo = ['EUTIROX', 'LEVOTIROXINA', 'METFORMINA', 'ACTRAPIDUILFL']

# Antihistamínicos
antihistaminicos = ['DEXCLORFENIRAMINAL', 'POLARAMINELFL', 'POLARAMINE']

# Antisépticos y Desinfectantes
antisepticos_desinfectantes = [
    'CLORHEXIDINAL', 'DESINCLORJABONOSOFL', 'ALCOHOLÂL', 'AGUAOXIGENADAL',
    'BETADINEAMARILLOFL', 'DESINCLORFL', 'CLORHEXIDINAACUOSAL', 'BETADINE',
    'GELPURELLADVANCEL', 'GELPURELLADVANCETFXOPTICOL', 'CLORHEXIDINAJABONOSAL', 
    'ACETONAL'
]

# Soluciones, Suplementos y Productos Sanitarios
soluciones_suplementos_y_productos_sanitarios = [
    'AGUABIDESTILADAL', 'SUEROCNLIV', 'AGUADESTILADALIRRIGACION', 
    'CLORUROSODICOL', 'ACETATOPOTASICOML', 'AGUAESTERILLIRRIGACION', 
    'SUEROGLUCOSADOL', 'ENSUREPLUSADVANCECHOCOLATEFL', 
    'RESOURCEESPESANTE', 'CLORUROPOTASICOML', 'CLORUROCALCICOLE', 
    'ALBUMINALL', 'ENSUREPLUSADVANCEVAINILLAFL', 'OXIGENOPLANTA', 
    'ULTRAVISTLFL', 'LINITULAPOSITO', 'INSTRUNETENZIMATICOEZTPLUSL'
]

# Otros Fármacos y Tratamientos
otros_tratamientos = [
    'ACFOL', 'AMCHAFIBRINL', 'ATROPINAL', 'LEDERFOLIN', 'GANFORTLFL', 
    'TAMSULOSINA', 'ACETILCISTEINA', 'ACCOFILMUL', 'ARANESPL', 'CISATRACURIOL', 
    'COLCHICINA', 'ALOPURINOL', 'ACIDOASCORBICOL'
]

def validate_patient_case(patient):

    ''' 
    La función validate_patient_case toma un diccionario que representa el historial
    clínico de un paciente y genera warnings si se detectan condiciones específicas.
    en los tratamientos y diagnosticos del paciente. Aplica reglas clínicas para
    validar coherencia y compatibilidad diagnóstica-terapéutica. 
    parámetros:
    - patient: un diccionario que contiene el historial clínico del paciente. Contiene
    información sobre el diagnóstico, medicaciones y métricas de salud.add()
    return:
    - warnings: una lista de advertencias generadas a partir de las condiciones
    '''
    warnings = [] # lista vacía para almacenar advertencias
    farmaco_paciente = patient.get('FARMACO/DRUG_NOMBRE_COMERCIAL/COMERCIAL_NAME', '')
    diagnostico = patient.get('DIAG ING/INPAT', '')
    saturacion_oxigeno = patient.get('SAT_02_ING/INPAT', 99)
    temperatura = float(patient.get('TEMP_ING/INPAT', 37)) # Temperatura, valor normal por defecto
    if diagnostico == 'COVID19 - POSITIVO':        
        # Se verifica si el fármaco administrado es uno de los opioides potentes.
        if isinstance(farmaco_paciente, str) and any(medicamento in farmaco_paciente.upper() for medicamento in analgesicos_antiinflamatorios):
            warnings.append(
            'ALERTA CLÍNICA: Paciente con COVID-19 recibe un AINE (ej. Ibuprofeno, Metamizol). '
            'Aunque la evidencia sobre el empeoramiento viral no es concluyente, se asocia a riesgos '
            'de lesión renal y gastrointestinal en enfermedad aguda. '
            'Considere el uso de paracetamol como primera opción y monitorice la función renal.'
            )
        elif isinstance(farmaco_paciente, str) and any(medicamento in farmaco_paciente.upper() for medicamento in opioides_potentes):
            warnings.append(
            'Paciente con COVID-19 y alta carga viral está recibiendo un analgésico opioide potente. '
            'Revisar la indicación y vigilar posible depresión respiratoria.'
            )
        elif isinstance(farmaco_paciente, str) and any(medicamento in farmaco_paciente.upper() for medicamento in psicofarmacos):
            warnings.append(
            'ALERTA DE SEGURIDAD FARMACOLÓGICA: Paciente con COVID-19 recibe un psicofármaco. '
            'Riesgos potenciales de prolongación del QTc (arritmia), interacciones con antivirales (ej. Paxlovid) y/o sedación excesiva. '
            'Recomendación: 1) Realizar ECG para evaluar intervalo QTc. 2) Revisar compatibilidad con tratamiento antiviral. 3) Vigilar nivel de conciencia y función respiratoria.'
            )
        elif isinstance(farmaco_paciente, str) and any(medicamento in farmaco_paciente.upper() for medicamento in farmacos_cardiovasculares):
            warnings.append(
            'ADVERTENCIA DE MANEJO CARDIOVASCULAR: Paciente con COVID-19 en tratamiento con IECA/ARA II. '
            'La evidencia actual desaconseja la suspensión rutinaria de estos fármacos. '
            'Su uso identifica a un paciente de alto riesgo cardiovascular. '
            'Recomendación: 1) Mantener tratamiento si está hemodinámicamente estable. 2) Considerar ajuste o suspensión TEMPORAL solo en caso de hipotensión severa, shock o lesión renal aguda. 3) Monitorizar presión arterial y función renal.'
            )
        elif isinstance(farmaco_paciente, str) and any(medicamento in farmaco_paciente.upper() for medicamento in farmacos_respiratorios):
            warnings.append(
            'ALERTA DE MANEJO RESPIRATORIO: Paciente con COVID-19 recibe fármacos respiratorios (broncodilatadores/ICS). '
            'Esto sugiere una patología de base (Asma/EPOC), elevando el riesgo de exacerbación y COVID-19 grave. '
            'Recomendaciones: 1) Diferenciar entre la disnea por neumonitis viral y la disnea por broncoespasmo. 2) NO suspender el tratamiento respiratorio de base; puede ser necesario intensificarlo. 3) Considerar el uso de corticoides inhalados (ej. Budesonida) como tratamiento para la propia COVID-19, según guías.'
            )
        elif isinstance(farmaco_paciente, str) and any(medicamento in farmaco_paciente.upper() for medicamento in antivirales):
            warnings.append(
            'AVISO DE MANEJO DE ANTIVIRAL: Paciente con COVID-19 en tratamiento antiviral específico. '
            'Fármaco indicado. '
            'Acción Requerida: Revisar urgentemente el resto de la medicación del paciente por posibles interacciones farmacológicas graves (especialmente con Paxlovid).'
            )
        elif isinstance(farmaco_paciente, str) and any(medicamento in farmaco_paciente.upper() for medicamento in antifungicos):
            warnings.append(
            'ALERTA DE ALTA GRAVEDAD: Paciente con COVID-19 recibe un antifúngico. '
            'Esto es un fuerte indicador de una sobreinfección fúngica grave (ej. CAPA), asociada a una alta mortalidad. '
            'Requiere manejo agresivo y probable ingreso en UCI.'
            )
        elif isinstance(farmaco_paciente, str) and any(medicamento in farmaco_paciente.upper() for medicamento in antibioticos):
            warnings.append(
            'ADVERTENCIA DE USO DE ANTIBIÓTICOS: Paciente con COVID-19 en tratamiento antibiótico. '
            'Recordar que no tratan la infección viral. '
            'Recomendación: Asegurar que su uso está justificado por sospecha o confirmación de sobreinfección bacteriana para promover la buena práctica (antibiotic stewardship).'
            )
        elif isinstance(farmaco_paciente, str) and any(medicamento in farmaco_paciente.upper() for medicamento in corticosteroides):
            if saturacion_oxigeno < 94:
                warnings.append(
                    'AVISO DE MANEJO: Paciente con COVID-19 grave (SatO2 < 94%) recibe corticosteroides. '
                    'Tratamiento indicado y recomendado por la OMS para reducir la mortalidad. '
                    'Recomendaciones: 1) Asegurar dosis estándar (ej. Dexametasona 6mg/día). 2) Monitorizar glucemia y vigilar sobreinfecciones.'
                )
            else:
                warnings.append(
                    'ALERTA DE USO INAPROPIADO: Paciente con COVID-19 no grave (SatO2 ≥ 94%) recibe corticosteroides. '
                    'La evidencia indica que este uso no aporta beneficios y puede aumentar la mortalidad. '
                    'Reevaluar urgentemente la indicación del tratamiento.'
                )
        elif isinstance(farmaco_paciente, str) and any(medicamento in farmaco_paciente.upper() for medicamento in farmacos_digestivos):
            warnings.append(
            'ALERTA DE VULNERABILIDAD GASTROINTESTINAL: Paciente con COVID-19 recibe fármacos gastroprotectores (ej. Omeprazol). '
            'Esto indica una patología digestiva de base, aumentando el riesgo de complicaciones por el "doble impacto" del virus y sus tratamientos. '
            'El virus puede afectar directamente el sistema digestivo[3, 5], y fármacos como los corticosteroides, antivirales y antibióticos son frecuentemente gastrolesivos. '
            'Recomendaciones: 1) Máxima vigilancia de síntomas como dolor abdominal, náuseas o diarrea. 2) Si se inician corticosteroides, asegurar que la pauta de protección gástrica es adecuada. 3) Ante nuevos síntomas digestivos, evaluar si son por COVID-19, efectos adversos de otros fármacos o descompensación de su enfermedad de base.'
            )
        elif isinstance(farmaco_paciente, str) and any(medicamento in farmaco_paciente.upper() for medicamento in anestesicos_locales):
            if saturacion_oxigeno < 94 or temperatura > 39:
                warnings.append('ALERTA DE ALTO RIESGO PROCEDIMENTAL: Paciente con COVID-19 grave (SatO2 < 94% o fiebre alta) va a someterse a un procedimiento con anestesia local. '
                    'Aunque la anestesia local/regional es preferible a la general, el estrés fisiológico del procedimiento en sí puede causar una descompensación clínica en un paciente con reserva limitada. '
                    'Recomendaciones: 1) Confirmar que el procedimiento es urgente e inaplazable. 2) Asegurar monitorización continua de signos vitales (SpO2, FC, PA) durante y después. 3) Realizar en un entorno con capacidad de soporte vital de emergencia disponible.')
        elif isinstance(farmaco_paciente, str) and any(medicamento in farmaco_paciente.upper() for medicamento in agentes_vasoactivos):
            alerta = (
            'ALERTA DE EXTREMA GRAVEDAD (ESTADO DE SHOCK): Paciente con COVID-19 recibe un agente vasoactivo (ej. Norepinefrina, Dopamina). '
            'Esto indica que el paciente se encuentra en shock circulatorio, una complicación potencialmente mortal que requiere manejo inmediato en una Unidad de Cuidados Intensivos (UCI). '
            'Recomendaciones de manejo: '
            '1) La norepinefrina es el vasopresor de primera línea recomendad. '
            '2) El uso de dopamina está desaconsejado por su perfil de seguridad inferior y mayor riesgo de arritmias. '
            '3) Si se utiliza dobutamina, sospechar disfunción cardíaca (shock cardiogénico) y evaluar función del corazón. '
            '4) Este paciente debe ser considerado de máxima criticidad, con necesidad de monitorización invasiva y soporte vital avanzado.'
            )
            warnings.append(alerta)
        elif isinstance(farmaco_paciente, str) and any(medicamento in farmaco_paciente.upper() for medicamento in inmunomoduladores):
            warnings.append(
            'Riesgo Principal: Riesgo elevado de progresión a enfermedad grave y de eliminación viral prolongada debido a una respuesta inmunitaria inicial deficiente.'
            'Recomendación: Considerar la modificación o suspensión temporal del tratamiento inmunosupresor de base, siempre en estricta consulta con el especialista correspondiente (reumatólogo, nefrólogo, etc.). Monitorizar de cerca la evolución clínica.')
            warnings.append(
            'ALERTA DE ALTO RIESGO METABÓLICO Y VULNERABILIDAD A COVID-19 GRAVE: Paciente en tratamiento con hormonas/fármacos para el metabolismo. '
            'NO SUSPENDER EL TRATAMIENTO DE BASE: Es fundamental mantener el tratamiento metabólico crónico (insulina, antidiabéticos orales, levotiroxina) para evitar una descompensación aguda. La suspensión debe ser valorada únicamente por un especialista. '
            'MONITORIZACIÓN ESTRICTA DE LA GLUCEMIA: Realizar controles de glucosa capilar frecuentes es obligatorio, especialmente si el paciente requiere ingreso hospitalario o se inicia tratamiento con corticosteroides. '
            'PREPARACIÓN PARA INTENSIFICAR EL TRATAMIENTO: Estar preparado para ajustar las dosis de insulina o iniciar insulinoterapia si el control glucémico se deteriora. '
            'VIGILANCIA TIROIDEA: En pacientes con patología tiroidea conocida, considerar la monitorización de la función tiroidea (TSH, T4L) si se produce un deterioro clínico inexplicado.'
            )
            warnings.append(
            'AVISO CLÍNICO: Uso de Antihistamínicos en Pacientes con COVID-19: '
            'Riesgo de Enmascaramiento Sintomático y Efectos Secundarios. El uso de antihistamínicos en un paciente con una infección activa por COVID-19 requiere una evaluación cuidadosa por dos motivos principales: '
            '1. Riesgo de Enmascaramiento de Síntomas y Retraso Diagnóstico (Principal Alerta): Ante cualquier síntoma respiratorio superior, incluso en un paciente con historial de alergias, se debe mantener un alto índice de sospecha de COVID-19 y proceder con las pruebas diagnósticas pertinentes. '
            '2. Riesgo Asociado al Tipo de Antihistamínico (Perfil de Seguridad): La advertencia clínica difiere significativamente según la generación del fármaco: '
            '1º GENERACION: Evitar su uso en la medida de lo posible en pacientes con COVID-19 sintomático. Si se necesita un antihistamínico, optar por uno de segunda generación. '
            '2º GENERACION: Preferible su uso, pero con precaución. Son la opción de elección si el paciente requiere tratamiento para una condición alérgica concurrente (como una urticaria o rinitis crónica). No existe evidencia sólida para suspenderlos.'
            )
            warnings.append(
            'ALERTA DE MANEJO DE ANTISÉPTICOS/DESINFECTANTES: Paciente con COVID-19 en tratamiento con antisépticos o desinfectantes. '
            'Aunque estos productos son seguros para la higiene de manos y superficies, su uso excesivo puede causar irritación cutánea o reacciones alérgicas.'
            )
        elif isinstance(farmaco_paciente, str) and any(medicamento in farmaco_paciente.upper() for medicamento in soluciones_suplementos_y_productos_sanitarios):
            warnings.append(
            'ALERTA DE MANEJO DE SOLUCIONES Y PRODUCTOS SANITARIOS: Paciente con COVID-19 en tratamiento con soluciones, suplementos o productos sanitarios. '
            'Estos productos son generalmente seguros, pero es importante asegurarse de que no interfieran con el tratamiento antiviral o con la función renal del paciente. '
            'Recomendación: Revisar la composición y posibles interacciones con otros medicamentos.'
            )
        elif isinstance(farmaco_paciente, str) and any(medicamento in farmaco_paciente.upper() for medicamento in otros_tratamientos):
            warnings.append(
                'ALERTA DE MANEJO DE OTROS TRATAMIENTOS: Paciente con COVID-19 en tratamiento con otros fármacos no clasificados. '
                'Es importante revisar la indicación y posibles interacciones con el tratamiento antiviral. '
                'Recomendación: Asegurarse de que estos fármacos no interfieran con el tratamiento antiviral.'
            )
        else:
            warnings.append(
                'No se han encontrado interacciones o alertas específicas para el tratamiento actual del paciente con COVID-19.'
            )

    return warnings

script_dir = os.getcwd()
CSV_PATH = os.path.abspath(os.path.join(script_dir, '..', '..', 'data', 'synthetic', 'datos_sinteticos_sdv.json'))
CSV_PATH_2 = os.path.abspath(os.path.join(script_dir, '..', '..', 'data', 'synthetic', 'datos_sinteticos_tvae.json'))

def procesar_archivo_jsonl(nombre_archivo):
    
    print(f"Procesando el archivo: {nombre_archivo}\n")
    try:
        with open(nombre_archivo, 'r', encoding='utf-8') as f:
            for numero_linea, linea in enumerate(f, 1):
                if not linea.strip():
                    continue
                
                try:
                    datos_paciente = json.loads(linea)
                    
                    # --- CORRECCIÓN AQUÍ ---
                    # 1. Captura la lista de warnings devuelta por la función.
                    warnings = validate_patient_case(datos_paciente)
                    
                    # 2. Si la lista no está vacía, imprime cada warning.
                    if warnings:
                        print(f"Alertas para la línea {numero_linea}:")
                        for warning in warnings:
                            print(f"  - {warning}")

                except json.JSONDecodeError as e:
                    print(f"❌ Error al decodificar JSON en la línea {numero_linea}: {e}")
                
                print("-" * 20)

    except FileNotFoundError:
        print(f"Error: El archivo '{nombre_archivo}' no fue encontrado.")


procesar_archivo_jsonl(CSV_PATH)
procesar_archivo_jsonl(CSV_PATH_2)
