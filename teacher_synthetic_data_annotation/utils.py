def extract_noun_chunks(nlp, text):
    doc = nlp(text)
    noun_chunks = []
    
    for chunk in doc.noun_chunks:
        if chunk.root.pos_ != 'PRON':
            # Exclude determiners (DET) from the chunk text
            filtered_chunk = ' '.join(token.text for token in chunk if token.pos_ != 'DET')
            noun_chunks.append(filtered_chunk)
    
    return noun_chunks