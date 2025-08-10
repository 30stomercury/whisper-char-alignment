import torch
import string
import re
from num2words import num2words

def encode(text, tokenizer, aligned_unit_type='subword'):
    assert aligned_unit_type in ['char', 'subword']
    if aligned_unit_type == 'subword':
        return tokenizer.text_to_ids(text, 'en')
    space_id = tokenizer.token_to_id(u"\u2581", 'en')
    tokens = []
    wrds = text.split()
    for i in range(len(wrds)):
        # tokens += tokenizer.encode(' '.join([char for char in wrds[i]]))
        for j, c in enumerate(wrds[i]):
            prefix = u"\u2581" if j == 0 else ''
            # if j == 0:
            #     tokens += [tokenizer.token_to_id(u"\u2581", 'en')]
            tokens += [tokenizer.token_to_id(prefix+c, 'en')]
    return tokens

def split_tokens_on_spaces(tokens, tokenizer, aligned_unit_type='subword'):
    assert aligned_unit_type in ['char', 'subword']
    blank_symbol = u"\u2581"
    if isinstance(tokens, torch.Tensor):
        tokens = tokens.detach().cpu().tolist()
    # tokens here are list of index numbers
    words = tokenizer.ids_to_text(tokens).split()
    pieces = tokenizer.ids_to_tokens(tokens)     
    print(pieces)
    start_inds = [i for i, p in enumerate(pieces) if p.startswith(blank_symbol)]
    word_tokens = []
    if aligned_unit_type == 'subword':
        for i in range(len(start_inds)):
            if i == len(start_inds) - 1:
                word_tokens.append(tokens[start_inds[i]:])
            else:
                word_tokens.append(tokens[start_inds[i]: start_inds[i+1]])
    else:
        for i in range(len(start_inds)):
            chars = []
            if i == len(start_inds) - 1:
                for j in range(start_inds[i], len(pieces)):
                    chars.append(tokenizer.token_to_id(pieces[j], 'en'))
            else:
                for j in range(start_inds[i], start_inds[i+1]):
                    chars.append(tokenizer.token_to_id(pieces[j], 'en'))
            word_tokens.append(chars)
    # append eos manually to fit the whisper alignment function
    words.append('<|endoftext|>')
    word_tokens.append(tokenizer.text_to_ids('<|endoftext|>', 'en'))
    return words, word_tokens

def remove_punctuation(text):
    text = text.translate(str.maketrans('', '', string.punctuation.replace("'", "")))
    normalized_text = []
    for wrd in text.split():
        if wrd.isdigit():
            wrd = num2words(int(wrd))
        normalized_text.append(wrd)

    text = ' '.join(normalized_text)
    return text.translate(str.maketrans('', '', string.punctuation.replace("'", "")))

def normalizer(text):
    text = re.sub("[^\w\d'\s]+",'', text)
    text = text[0].upper() + text[1:]
    return text
