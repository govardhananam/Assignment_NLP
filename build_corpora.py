from nltk.corpus import gutenberg
import nltk
print(nltk.__version__)
nltk.download('gutenberg')

def build_corpus():
    """
    Instructions:
        - first, install the NLTK package
        - to acquire the data, open a python terminal and run:
            - import nltk
            -nltk.download('gutenberg')
    """
    
    for _fn in ['shakespeare-hamlet.txt', 'shakespeare-macbeth.txt']:
        _sents = gutenberg.sents(_fn)
        out_name = _fn.split('-')[1]
        print('Writing corpus to {}'.format(out_name))
        with open(out_name, 'w') as f:
            for s in _sents:
                outs = " ".join(s).strip("[").strip("]").lstrip().rstrip()
                f.write(outs + "\n")


if __name__ == "__main__":
    build_corpus()
