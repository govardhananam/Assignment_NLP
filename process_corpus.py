

def preprocessing(_filename):
    """
    A function for building a simple text corpus from an input txt file.

    :param _filename: Path to the txt file of interest.
    :return: _sentences: list of lists of parsed sentences.
    """
    with open(_filename, 'r', errors='ignore') as f:
        corpus = [x.strip().strip(',') for x in f]
    corpus = [x for x in corpus if x is not None]
    _sentences = []
    for doc in corpus:
        d = doc.lower().strip('.')
        sent = d.split(',')
        for s in sent:
            out_sent = s.split()
            out_sent = ['<s>'] + out_sent + ['<\\s>']
            out_sent = [w.strip('.') for w in out_sent]
            _sentences.append(out_sent)
    return _sentences


if __name__ == "__main__":
    preprocessing('hamlet.txt')
