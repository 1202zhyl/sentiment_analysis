# - * -coding: utf-8

import jieba
import pandas as pd
from functools import reduce


class Segmentation:
    DF_COLUMNS = ['rating', 'comment']

    def __init__(self, raw_data=None, dest_data=None,
                 user_dict='data/dict/user_dict.dic',
                 stop_dict='data/dict/stop_words.dic'):
        self.raw_data = raw_data
        self.dest_data = dest_data
        jieba.load_userdict(user_dict)
        with open(stop_dict, encoding='utf-8') as stops:
            self._stopwords = [word.strip() for word in stops]
        self._df = None

    @property
    def stop_words(self):
        return self._stopwords

    def to_csv(self):
        def segment(sentence):
            def concat(str1, str2):
                return '{} {}'.format(str1.strip(), str2.strip())

            def filter_stop_word(word):
                def is_number(s):
                    try:
                        float(s)
                        return True
                    except ValueError:
                        return False

                stripped = word.strip()

                return len(stripped) != 0 and not stripped in self._stopwords and not is_number(stripped)

            filtered = list(filter(filter_stop_word, jieba.cut(sentence, cut_all=False)))
            if len(filtered) == 0:
                return None
            else:
                return reduce(concat, filtered)

        data_frames = list()
        with open(self.raw_data, encoding='utf-8') as raw:
            for raw_line in raw:
                raw_line = raw_line.lower().strip()
                if len(raw_line) == 0:
                    continue
                try:
                    rating, comment = raw_line[0].strip(), raw_line[1:].strip()
                except Exception as e:
                    continue
                comment = segment(comment)
                if comment is None or len(comment.strip()) == 0:
                    continue
                df = pd.DataFrame([[rating, comment.strip()]], columns=self.DF_COLUMNS)
                data_frames.append(df)

        if len(data_frames) != 0:
            self._df = pd.concat(data_frames, ignore_index=True)
            # save the corpus
            self._df.to_csv(self.dest_data, index=False, encoding='utf-8')


if __name__ == '__main__':
    '''generate corpus'''
    seg = Segmentation(raw_data='./data/corpus/corpus.dat', dest_data='./data/corpus/corpus.csv')
    seg.to_csv()