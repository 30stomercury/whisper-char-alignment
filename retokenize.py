import string
from num2words import num2words


def encode(text, tokenizer, aligned_unit_type='subword'):
    assert aligned_unit_type in ['char', 'subword']
    if aligned_unit_type == 'subword':
        return tokenizer.encode(text)
    tokens = []
    space_id = tokenizer.encode(' ')
    wrds = text.split()
    for i in range(len(wrds)):
        for c in wrds[i]:
            tokens += tokenizer.encode(c)
        if i < len(wrds) - 1:
            tokens += space_id
    return tokens

def split_tokens_on_spaces(tokens, tokenizer, aligned_unit_type='subword'):
    assert aligned_unit_type in ['char', 'subword']
    if aligned_unit_type == 'subword':
        return tokenizer.split_to_word_tokens(tokens)

    subwords, subword_tokens_list = tokenizer.split_tokens_on_unicode(tokens)
    words = []
    word_tokens = []

    for subword, subword_tokens in zip(subwords, subword_tokens_list):
        special = subword_tokens[0] >= tokenizer.eot
        with_space = subword == " "
        punctuation = subword.strip() in string.punctuation
        if special or with_space or punctuation or len(words) == 0:
            words.append(subword)
            word_tokens.append(subword_tokens)
        else:
            words[-1] = words[-1] + subword
            word_tokens[-1].extend(subword_tokens)

    return words, word_tokens

def remove_punctuation(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    normalized_text = []
    for wrd in text.split():
        if wrd.isdigit():
            wrd = num2words(int(wrd))
        normalized_text.append(wrd)

    text = ' '.join(normalized_text)
    return text.translate(str.maketrans('', '', string.punctuation))
