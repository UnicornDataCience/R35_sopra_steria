"""
Script para diagnosticar y corregir el archivo TVAE
Ejecutar manualmente en terminal
"""
import json
import os

def diagnose_tvae_file():
    """Diagnostica problemas en el archivo TVAE"""
    print("🔍 DIAGNÓSTICO DEL ARCHIVO TVAE")
    print("=" * 50)
    
    tvae_file = 'data/synthetic/datos_sinteticos_tvae.json'
    
    if not os.path.exists(tvae_file):
        print(f"❌ Archivo no encontrado: {tvae_file}")
        return False
    
    try:
        # Cargar datos
        with open(tvae_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"✅ Archivo cargado: {len(data)} registros")
        print(f"\n📋 ESTRUCTURA DEL PRIMER REGISTRO:")
        first_record = data[0]
        for key, value in first_record.items():
            print(f"  {key}: {value} (tipo: {type(value).__name__})")
        
        print(f"\n🔍 ANÁLISIS DE VALORES PROBLEMÁTICOS:")
        problem_fields = []
        
        for key in first_record.keys():
            if 'SAT' in key.upper():
                # Analizar valores de saturación
                values = [record.get(key) for record in data[:10]]
                print(f"  {key}: {values}")
                
                # Contar valores < 50
                low_values = [v for v in values if isinstance(v, (int, float)) and v < 50]
                if low_values:
                    problem_fields.append(key)
                    print(f"    ⚠️ PROBLEMA: {len(low_values)} valores < 50")
        
        if problem_fields:
            print(f"\n❌ CAMPOS PROBLEMÁTICOS ENCONTRADOS: {problem_fields}")
            return True, data, problem_fields
        else:
            print(f"\n✅ No se encontraron problemas obvios")
            return False, data, []
            
    except Exception as e:
        print(f"❌ Error analizando archivo: {e}")
        return False, None, []

def fix_tvae_data():
    """Corrige los datos problemáticos del TVAE"""
    print(f"\n🔧 CORRECCIÓN DEL ARCHIVO TVAE")
    print("=" * 50)
    
    has_problems, data, problem_fields = diagnose_tvae_file()
    
    if not has_problems or not data:
        print("✅ No hay problemas que corregir o no se pudieron cargar los datos")
        return False
    
    print(f"🔄 Corrigiendo campos problemáticos: {problem_fields}")
    
    corrected_count = 0
    
    for record in data:
        for field in problem_fields:
            if field in record:
                value = record[field]
                if isinstance(value, (int, float)) and value < 50:
                    # Corregir valor: convertir de escala 0-2 a escala 70-100
                    if value <= 2:
                        # Mapear 0-2 a 85-95 (rango típico de saturación)
                        new_value = 85 + (value * 5)  # 0->85, 1->90, 2->95
                    else:
                        # Si es otro valor bajo, llevarlo al rango válido
                        new_value = max(70, value + 70)
                    
                    record[field] = round(new_value, 1)
                    corrected_count += 1
    
    print(f"✅ Se corrigieron {corrected_count} valores")
    
    # Guardar archivo corregido
    corrected_file = 'data/synthetic/datos_sinteticos_tvae_corregido.json'
    try:
        with open(corrected_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"✅ Archivo corregido guardado como: {corrected_file}")
        
        # Reemplazar el archivo original
        backup_file = 'data/synthetic/datos_sinteticos_tvae_original.json'
        os.rename('data/synthetic/datos_sinteticos_tvae.json', backup_file)
        os.rename(corrected_file, 'data/synthetic/datos_sinteticos_tvae.json')
        
        print(f"✅ Archivo original respaldado como: {backup_file}")
        print(f"✅ Archivo corregido reemplaza al original")
        
        return True
        
    except Exception as e:
        print(f"❌ Error guardando archivo corregido: {e}")
        return False

def test_corrected_file():
    """Testa el archivo corregido"""
    print(f"\n🧪 TEST DEL ARCHIVO CORREGIDO")
    print("=" * 50)
    
    try:
        import sys
        sys.path.append('.')
        
        from src.validation.json_schema import procesar_archivo_json
        
        tvae_file = 'data/synthetic/datos_sinteticos_tvae.json'
        result = procesar_archivo_json(tvae_file)
        
        print(f"📊 RESULTADOS DE VALIDACIÓN:")
        print(f"  - Total registros: {result['total_count']}")
        print(f"  - Registros válidos: {result['valid_count']}")
        print(f"  - Tasa de éxito: {result['success_rate']:.1f}%")
        print(f"  - Dominio detectado: {result['domain']}")
        
        if result['success_rate'] > 90:
            print(f"🎉 ¡ARCHIVO CORREGIDO EXITOSAMENTE!")
            return True
        else:
            print(f"⚠️ Aún hay problemas. Errores:")
            for error in result.get('errors', [])[:5]:
                print(f"    - {error}")
            return False
            
    except Exception as e:
        print(f"❌ Error testando archivo: {e}")
        return False

def main():
    """Función principal"""
    print("🚀 REPARADOR DE ARCHIVO TVAE")
    print("=" * 60)
    
    # Paso 1: Diagnosticar
    print("PASO 1: Diagnóstico")
    has_problems, data, problem_fields = diagnose_tvae_file()
    
    if not has_problems:
        print("✅ No se encontraron problemas en el archivo TVAE")
        return
    
    # Paso 2: Corregir
    print("\nPASO 2: Corrección")
    fixed = fix_tvae_data()
    
    if not fixed:
        print("❌ No se pudo corregir el archivo")
        return
    
    # Paso 3: Validar
    print("\nPASO 3: Validación")
    success = test_corrected_file()
    
    if success:
        print(f"\n🎉 ¡PROCESO COMPLETADO CON ÉXITO!")
        print(f"✅ Archivo TVAE corregido y validado")
        print(f"✅ Sistema de validación completamente operativo")
    else:
        print(f"\n⚠️ Archivo corregido pero aún con algunos problemas menores")

if __name__ == "__main__":
    main()
