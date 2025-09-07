from typing import List
import nltk
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def normalize(sentence) -> List[str]:
    valid = ["NN", "NNS", "NNP", "NNPS", "JJ", "JJR", "JJS"]

    logging.debug(f'Start normalize sentence: {sentence}')
    tokenizer = nltk.TweetTokenizer()
    stemmer = nltk.stem.LancasterStemmer()

    sentence = tokenizer.tokenize(sentence)
    sentence = [word for word in sentence if nltk.pos_tag([word])[0][1] in valid]
    sentence = [stemmer.stem(word).lower() for word in sentence]

    logging.debug(f'Normalized tokens: {sentence}')
    return sentence

