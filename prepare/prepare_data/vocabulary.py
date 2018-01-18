# -*- coding： utf-8

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from collections import Counter
from functools import reduce


class Vocabulary:
    def __init__(self, corpus_file):
        self._df = pd.read_csv(corpus_file)
        self._vocabulary_size = None
        self._vocabulary = None

    @property
    def vocabulary(self):
        if self._vocabulary is not None:
            return self._vocabulary
        if self._vocabulary_size is None:
            self._vocabulary_size = 70000
        # save the vocabulary
        sentences = map(lambda s: str(s).split(), list(self._df.comment.values))
        words = reduce(lambda l1, l2: l1.extend(l2) or l1, sentences)
        # append UNK to the end.
        self._vocabulary = [word for (word, num) in Counter(words).most_common(self._vocabulary_size - 1)]
        self._vocabulary.append('UNK')
        return self._vocabulary

    def analyze(self):
        sentences = map(lambda s: str(s).split(), list(self._df.comment.values))
        words = reduce(lambda l1, l2: l1.extend(l2) or l1, sentences)

        counter = Counter(words)
        most_unique_words_num = []
        most_words_num = []
        for i in range(0, len(counter), 10000):
            most = counter.most_common(i)
            count = reduce(lambda n1, n2: n1 + n2, [n for (w, n) in most]) if len(most) != 0 else 0
            most_unique_words_num.append(i)
            most_words_num.append(count)

        X = np.array(most_unique_words_num)
        Y = np.array(most_words_num, dtype=np.float32)
        Y /= Y[-1] / 100

        np.save('./prepare/prepare_data/voca_size.npy', X)
        np.save('./prepare/prepare_data/corpus_ratio.npy', Y)

        # X = np.load('./prepare/prepare_data/voca_size.npy')
        # Y = np.load('./prepare/prepare_data/corpus_ratio.npy')

        fig, ax = plt.subplots()
        ax.plot(X, Y, 'bo')

        fmt = '%.0f%%'
        yticks = mtick.FormatStrFormatter(fmt)
        ax.yaxis.set_major_formatter(yticks)

        ax.annotate('coverage: %.2f%%' % Y[10], xy=(X[10], Y[10]), xytext=(-20, 20),
                    textcoords='offset points', ha='center', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.5',
                                    color='red'))

        plt.vlines(X[10], 0, Y[10], colors='g', linestyles='dashed')
        plt.hlines(Y[10], 0, X[10], colors='g', linestyles='dashed')

        plt.xlabel('vocabulary size')
        plt.ylabel('corpus ratio')

        # we can see that the words almost cover 98% of the whole corpus with vocabulary size 200,000.
        plt.plot(X, Y, 'r--')
        plt.show()

        # after analyzing the corpus, decide vocabulary = 150,000


if __name__ == '__main__':
    '''generate vocabulary'''
    vocabulary = Vocabulary('./data/corpus/corpus.sample.csv')
    # vocabulary.analyze()
    np.save('./data/vocabulary.npy', np.array(vocabulary.vocabulary))
