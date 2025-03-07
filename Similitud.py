!pip install pymupdf
import nltk
import fitz
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

def calcular_similitud(texto1, texto2):
    stop_words = set(stopwords.words("spanish"))
    lemmatizer = WordNetLemmatizer()

    def preprocesar(texto):
        tokens = [
            lemmatizer.lemmatize(word.lower())
            for word in word_tokenize(texto)
            if word.isalnum() and word.lower() not in stop_words
        ]
        return ' '.join(tokens)

    texto_preprocesado1 = preprocesar(texto1)
    texto_preprocesado2 = preprocesar(texto2)

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([texto_preprocesado1, texto_preprocesado2])
    similitud = (tfidf_matrix * tfidf_matrix.T).toarray()[0, 1]

    return similitud

def extraer_texto_pdf(ruta_pdf):
    try:
        doc = fitz.open(ruta_pdf)
        texto_completo = "".join(pagina.get_text() for pagina in doc)
        return texto_completo
    except FileNotFoundError:
        return "Archivo no encontrado."
    except Exception as e:
        return f"Error al procesar el PDF: {e}"

ruta_pdf1 = 'ADA.pdf'
ruta_pdf2 = "ADA'.pdf"

texto_pdf1 = extraer_texto_pdf(ruta_pdf1)
texto_pdf2 = extraer_texto_pdf(ruta_pdf2)

if "Error" not in texto_pdf1 and "Error" not in texto_pdf2:
    similitud = calcular_similitud(texto_pdf1, texto_pdf2)
    print(f"Similitud entre los textos de los PDFs: {similitud}")
else:
    print("No se pudo calcular la similitud debido a errores en la lectura de los PDFs.")
