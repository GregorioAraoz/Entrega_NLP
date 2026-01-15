import pandas as pd
import random

def generate_data():
    places = [
        "Pizzería Guerrin", "El Cuartito", "Don Julio", "La Cabrera", 
        "Antares", "La Mezzetta", "Rapanui", "Kansas", "La Parolaccia",
        "Sottovoce", "El Pobre Luis", "Las Violetas"
    ]

    # Explicit POS/NEG lists - Very clear polarity
    food_pos = [
        "la comida estaba exquisita", "que rica comida", "los platos son abundantes y deliciosos", 
        "la carne era una manteca", "la pizza es la mejor de buenos aires", "excelente sabor",
        "todo muy rico", "me encantó la comida", "sabroso y fresco", "me gusto mucho", "estaba rico"
    ]
    food_neg = [
        "la comida llegó fría", "la carne estaba dura", "todo tenía sabor feo", 
        "no me gustó la comida", "la pizza estaba cruda", "horrible sabor",
        "la comida un desastre", "incomible", "todo muy grasoso", "nos llego todo frio", 
        "no me pareció rico", "no estaba rico", "teniamos frio", "todo helado", "vino frio", "muy frio",
        "poco bueno", "poco rico", "nada bueno", "genial si te gusta comer frio",
        "bien si te gusta lo crudo", "la comida una obra de arte abstracto del horror"
    ]
    food_neu = [
        "la comida normal", "platos estándar", "nada del otro mundo la comida", 
        "se deja comer", "comida aceptable", "ni muy muy ni tan tan", "regular la comida"
    ]

    serv_pos = [
        "la atención fue genial", "nos atendieron de maravilla", "el mozo super amable", 
        "atención rápida y cordial", "servicio excelente", "muy atentos en todo momento",
        "destaco la atención"
    ]
    serv_neg = [
        "la atención fue pésima", "los mozos te tratan mal", "tardaron mil años en atender", 
        "nadie nos prestaba atención", "servicio lento y malo", "muy mala onda los empleados",
        "falta personal", "gracias por la demora", "me encanto esperar 2 horas", "ideal para perder el tiempo",
        "si te gusta esperar este es tu lugar", "brillante atencion si no tienes prisa"
    ]
    serv_neu = [
        "la atención normal", "el servicio correcto", "tiempos de espera normales", 
        "atención sin destacar", "cumplieron con el pedido"
    ]

    price_pos = [
        "buen precio", "bastante barato", "excelente relación precio calidad", 
        "precios accesibles", "no es caro"
    ]
    price_neg = [
        "muy caro", "precios por las nubes", "me pareció carísimo", 
        "te arrancan la cabeza", "una estafa el precio", "precios elevados", 
        "precios altos", "un poco caro", "costoso", "no vale lo que cuesta",
        "precios un poco elevados para lo que ofrecen"
    ]
    price_neu = [
        "precio acorde", "precios de mercado", "lo que esperas pagar", "no es barato ni caro"
    ]

    # Aspect: Ambience (affects Global, but NEU for others)
    amb_pos = ["me gusto el lugar", "lindo lugar", "el lugar es hermoso", "buen ambiente", "muy linda decoracion", "el lugar esta bueno"]
    amb_neg = ["lugar feo", "no me gusto el lugar", "lugar sucio", "mala ambientacion", "el lugar se cae a pedazos", "no me gusto el ambiente"]
    amb_neu = ["lugar normal", "ambiente estandar", "nada especial el lugar"]

    data = []
    
    # Generate 5000 clean samples
    for _ in range(5000):
        place = random.choice(places)
        
        # We will build specific sentences where we KNOW the label for each part
        # To avoid confusion, we construct the global label based on sum of parts
        
        aspects = ['food', 'service', 'price', 'ambience']
        random.shuffle(aspects)
        
        # Decide how many aspects to mention (1, 2, or 3)
        num_aspects = random.choices([1, 2, 3], weights=[0.4, 0.4, 0.2])[0]
        selected_aspects = aspects[:num_aspects]
        
        segments = []
        labels = {'food': 'NEU', 'service': 'NEU', 'price': 'NEU'} # Default to NEU
        
        score_sum = 0
        
        for aspect in selected_aspects:
            sentiment = random.choice(['POS', 'NEG', 'NEU'])
            
            text = ""
            if aspect == 'food': 
                text = random.choice(food_pos if sentiment=='POS' else food_neg if sentiment=='NEG' else food_neu)
                labels['food'] = sentiment
            elif aspect == 'service': 
                text = random.choice(serv_pos if sentiment=='POS' else serv_neg if sentiment=='NEG' else serv_neu)
                labels['service'] = sentiment
            elif aspect == 'price': 
                text = random.choice(price_pos if sentiment=='POS' else price_neg if sentiment=='NEG' else price_neu)
                labels['price'] = sentiment
            elif aspect == 'ambience':
                text = random.choice(amb_pos if sentiment=='POS' else amb_neg if sentiment=='NEG' else amb_neu)
                # Ambience only affects global score, doesn't set specific aspect labels
            
            segments.append(text)
            
            if sentiment == 'POS': score_sum += 1
            elif sentiment == 'NEG': score_sum -= 1
            
        # Join with natural connectors
        if len(segments) == 1:
            full_text = segments[0]
        elif len(segments) == 2:
            full_text = f"{segments[0]} y {segments[1]}"
        else:
            full_text = f"{segments[0]}, {segments[1]} y {segments[2]}"
            
        full_text = full_text.capitalize() + "."
        
        # Global Label Logic
        # Explicit heuristic: 2 POS > 1 NEG -> POS. Tie -> NEU.
        if score_sum > 0: global_label = 'POS'
        elif score_sum < 0: global_label = 'NEG'
        else: global_label = 'NEU'
        
        data.append({
            "place_name": place,
            "review_text": full_text,
            "sentiment_food": labels['food'],
            "sentiment_service": labels['service'],
            "sentiment_price": labels['price'],
            "sentiment_global": global_label
        })

    # Shared Sentiment / Ellipsis Structure (e.g. "Me gusto el lugar y la comida")
    # This teaches the model that one verb applies to multiple objects
    shared_verb_pos = ["me gusto", "nos encanto", "disfrutamos mucho", "amamos", "excelente", "muy bueno", "estuvo rico"]
    shared_verb_neg = ["odie", "no me gusto", "detestamos", "muy malo", "pesimo", "horrible", "fue un desastre"]
    
    aspect_terms = {
        'food': 'la comida', 
        'service': 'la atencion', 
        'price': 'el precio', 
        'ambience': 'el lugar'
    }
    
    for _ in range(1500): # Heavy boost for this pattern
        verb_type = random.choice(['POS', 'NEG'])
        verb = random.choice(shared_verb_pos if verb_type == 'POS' else shared_verb_neg)
        
        # Pick 2 different aspects
        a1, a2 = random.sample(list(aspect_terms.keys()), 2)
        t1, t2 = aspect_terms[a1], aspect_terms[a2]
        
        # Formats: "Me gusto A y B", "A y B fueron horribles"
        structure = random.choice([1, 2])
        if structure == 1:
            full_text = f"{verb} {t1} y {t2}."
        else:
            full_text = f"{t1} y {t2} {verb}."
            
        labels = {'food': 'NEU', 'service': 'NEU', 'price': 'NEU', 'global': 'NEU'}
        
        # Assign labels
        for a in [a1, a2]:
            if a != 'ambience':
                labels[a] = verb_type
        
        # Global is same as verb
        labels['global'] = verb_type
        
        data.append({
            "place_name": random.choice(places),
            "review_text": full_text.capitalize(),
            "sentiment_food": labels['food'],
            "sentiment_service": labels['service'],
            "sentiment_price": labels['price'],
            "sentiment_global": labels['global']
        })

    df = pd.DataFrame(data)
    df.to_csv('data/train_labeled.csv', index=False)
    print(f"Generated {len(df)} strictly labeled reviews.")

if __name__ == "__main__":
    generate_data()
