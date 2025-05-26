from transformers import pipeline

pipe = pipeline(
    "text-generation",
    model = "mistralai/Mistral-7B-Instruct-v0.2")

def generate_clinical_note(prompt):
    ''' 
    la función generate_clinical_note genera una nota clínica sintética
    con un LLM de lenguaje natural de Hugging Face.
    param:
    - prompt: texto de entrada que describe la situación clínica del paciente
    return:
    - texto generado por el modelo de lenguaje natural
    '''
    return pipe(prompt, max_new_tokens=100)[0]['generated_text']