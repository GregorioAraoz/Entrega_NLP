import re
import pandas as pd
import spacy
import sys
import subprocess

# Lazy loading setup
_nlp_instance = None

def get_nlp():
    global _nlp_instance
    if _nlp_instance is None:
        try:
            import es_core_news_sm
            _nlp_instance = es_core_news_sm.load()
        except ImportError:
            print("Model not found. Installing via pip...")
            try:
                # Install directly via pip at runtime
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", 
                    "https://github.com/explosion/spacy-models/releases/download/es_core_news_sm-3.5.0/es_core_news_sm-3.5.0.tar.gz"
                ])
                # Re-import after install
                import es_core_news_sm
                import importlib
                importlib.reload(es_core_news_sm)
                _nlp_instance = es_core_news_sm.load()
            except Exception as e:
                print(f"Failed to install model: {e}")
                raise e
                    
    return _nlp_instance

# Pre-defined phrase replacements for normalization
PHRASE_MAP = {
    "poco satisfactoria": "insatisfactoria",
    "poco satisfactorio": "insatisfactorio",
    "poco agradable": "desagradable",
    "poco bueno": "malo",
    "poco rico": "feo",
    "poco amable": "antipatico",
    "poco profesional": "incompetente",
    "jamás volvería": "nunca_volver",
    "jamas volveria": "nunca_volver",
    "no volvería": "nunca_volver",
    "no volveria": "nunca_volver",
    "nada bueno": "malo",
    "nada rico": "feo",
    "deja mucho que desear": "malo",
    "esperar 3 horas": "espera_larga",
    "esperar mucho": "espera_larga",
    "tardaron mucho": "lento",
    "sin dudas": "definitivamente",
    "sin duda": "definitivamente",
    "lo que me paso fue increible": "experiencia_increible",
    "no me gusto": "no_gusto",
    "gracias por la demora": "queja_demora",
    "me encanto esperar": "queja_demora",
    "si te gusta esperar": "queja_demora",
    "ideal para perder el tiempo": "queja_demora",
    "genial si te gusta comer frio": "comida_fria",
    "buenisimo si te gusta esperar": "queja_demora",
    "precios elevados": "caro",
    "precios altos": "caro",
    "precio alto": "caro",
    "un poco caro": "caro",
    "algo caro": "caro",
    "bastante caro": "caro",
    "para lo que ofrecen": "decepcionante",
    "elevados": "caro",
    "elevado": "caro",
    "precios son altos": "caro",
    "precios estan altos": "caro"
}

def clean_text(text: str) -> str:
    """
    Preprocesses the input text for sentiment analysis.
    Includes Lexical Normalization for phrases like 'poco agradable'.
    """
    # 1. Handle Nulls/Empty
    if pd.isna(text) or not isinstance(text, str) or text.strip() == "":
        return ""
    
    # 2. Lowercase
    text = text.lower()
    
    # 3. Phrase Mapping (before regex removes numbers/chars)
    for phrase, replacement in PHRASE_MAP.items():
        text = text.replace(phrase, replacement)
    
    # 4. Regex Cleaning
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    # Remove non-alphabetic characters (keep spaces AND UNDERSCORES)
    text = re.sub(r'[^a-záéíóúñü_\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    if not text:
        return ""

    # 5. SpaCy Processing
    nlp = get_nlp()
    doc = nlp(text)
    
    cleaned_tokens = []
    
    # Advanced negation handling
    negation_pending = False
    
    for token in doc:
        word = token.text.lower()
        
        # Check for negation triggers
        if word in ['no', 'nunca', 'jamás', 'ni', 'tampoco', 'nada']:
            negation_pending = True
            continue 
            
        # Skip pronouns/connectors AND copular verbs if we are waiting to negate the next meaningful word
        skip_words = [
            'me', 'le', 'se', 'nos', 'te', 'la', 'lo', 'los', 'las', 
            'es', 'fue', 'era', 'son', 'estaba', 'estaban', 'estuvo', 
            'parecio', 'pareció', 'parece', 'parecen', 'creo', 'creer'
        ]
        
        if negation_pending and word in skip_words:
            continue
            
        # Process meaningful words
        # Treat underscores as valid tokens to keep
        if not token.is_stop or negation_pending or '_' in word: 
            if (token.is_alpha or '_' in word) and (not token.is_stop or word in ['bien', 'mal'] or '_' in word):
                lemma = token.lemma_.lower()
                if '_' in word: lemma = word # preserve mapped tokens
                
                if negation_pending:
                    lemma = f"no_{lemma}"
                    negation_pending = False
                
                cleaned_tokens.append(lemma)
            
    return " ".join(cleaned_tokens)

if __name__ == "__main__":
    test_sentences = [
        "nos llego todo frio, no me parecio rico",
        "esperar 3 horas fue una locura",
        "jamas volveria a este lugar",
        "poco satisfactoria la experiencia"
    ]
    for s in test_sentences:
        print(f"Original: '{s}' -> Cleaned: '{clean_text(s)}'")
