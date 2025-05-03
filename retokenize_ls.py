import string

# support different char tokenize
def char_tokenizer_encode(text, tokenizer, sep_space=True):
    space_id = tokenizer.encode(' ')
    tokens = []
    wrds = text.split()
    for i in range(len(wrds)):
        #tokens += tokenizer.encode(' '.join([char for char in wrds[i]]))
        for j, c in enumerate(wrds[i]):
            if j == 0 and i == 0:
                tokens += tokenizer.encode(c)
            else:
                p = '' if sep_space else ' '
                tokens += tokenizer.encode(p+c)
        if sep_space:
            if i < len(wrds) - 1:
                tokens += space_id

    return tokens

def split_chars_on_spaces(tokens, tokenizer, hypothesis, sep_space=True):
    subwords, subword_tokens = tokenizer.split_tokens_on_unicode(tokens)
    # print(subwords, subword_tokens)
    words = []
    word_tokens = []

    s = 0
    for i, word in enumerate(hypothesis.split()):
        subword = ""
        subword_token = []
        if sep_space:
            if i > 0:
                end = s+len(word)+1
            else:
                end = s+len(word)
        else:
            end = s+len(word)
        for j in range(s, end):
            subword += subwords[j]
            subword_token.extend(subword_tokens[j])
        words.append(subword)
        word_tokens.append(subword_token)
        s = end

    # append eos
    words.append(subwords[-1])
    word_tokens.append([subword_tokens[-1]])

    return words, word_tokens

def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

def normalizer(text):
    text = text.replace(",", "").replace(".", "")
    text = text[0].upper() + text[1:]
    return text