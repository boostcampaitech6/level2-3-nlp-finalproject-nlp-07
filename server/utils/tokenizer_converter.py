def nnn_to_mmm(nnn_ids, tokenizer):
    mmm_tokens = []
    
    nnn_tokens = [tokenizer[i] for i in nnn_ids]
    for tk in nnn_tokens:
        if tk.startswith('Pitch_'):
            new_tks = tk.split('+')
            mmm_tokens.append(new_tks[0])
            mmm_tokens.append(new_tks[1])
            mmm_tokens.append(new_tks[2])
        else:
            mmm_tokens.append(tk)
    
    mmm_ids = [tokenizer[t] for t in mmm_tokens]
    return mmm_ids

def mmm_to_nnn(mmm_ids, tokenizer):
    nnn_tokens = []
    
    mmm_tokens = [tokenizer[i] for i in mmm_ids]
    i = 0
    while i < len(mmm_tokens):
        tk = mmm_tokens[i]
        if tk.startswith('Pitch_'):
            new_tk = f'{mmm_tokens[i]}+{mmm_tokens[i+1]}+{mmm_tokens[i+2]}'
            nnn_tokens.append(new_tk)
            i += 3
        else:
            nnn_tokens.append(tk)
            i += 1
            
    nnn_ids = [tokenizer[t] for t in nnn_tokens]
    return nnn_ids