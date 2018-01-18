# coding=utf-8

# python 2.7.13

from bs4 import BeautifulSoup
import os
from urllib.request import *
import sys
import struct


class SogouCrawler:
    def __init__(self, saved_path='.'):
        self.base_url = 'http://pinyin.sogou.com'
        self.home_page = '{}/dict/'.format(self.base_url)
        self.saved_path = saved_path

    def crawl(self):
        html = urlopen(self.home_page).read()
        soup = BeautifulSoup(html)
        soup = soup.find(id='dict_category_show').find_all('div', class_='dict_category_list')

        for top_level in soup:

            for secondary_level in top_level.find(class_='catewords').find_all('a'):
                second_class = secondary_level.contents[0]
                second_class_url = '{}{}'.format(self.base_url, secondary_level['href'])

                s_soup = BeautifulSoup(urlopen(second_class_url).read())

                try:
                    page_num = s_soup.find(id='dict_page_list').find('ul').find_all('span')[-2].a.contents[0]
                except Exception as e:
                    page_num = 1

                for pageind in range(1, int(page_num) + 1):
                    t_soup = BeautifulSoup(
                        urlopen('%s/default/%d' % (second_class_url.replace('?rf=dictindex', ''), pageind)).read())
                    for third_level in t_soup.find_all('div', class_='dict_detail_block'):
                        third_class = third_level.find(class_='detail_title').find('a').contents[0]
                        if os.path.exists(
                                '{}/{}-{}.scel'.format(self.saved_path, second_class, third_class)):
                            continue
                        third_class_url = third_level.find(class_='dict_dl_btn').a['href']
                        third_class = third_class.replace('/', '')
                        urlretrieve(third_class_url,
                                    '{}/{}-{}.scel'.format(self.saved_path, second_class, third_class))


class Scel2Txt:
    # 拼音表偏移，
    STARTPY = 0x1540
    # 汉语词组表偏移
    STARTCHINESE = 0x2628

    def __init__(self, src_path, dest_dict):
        # 全局拼音表
        self._GPy_Table = {}

        # 解析结果
        # 元组(词频,拼音,中文词组)的列表
        self._GTable = []

        self._src_path = src_path
        self._dest_dict = dest_dict

    def _byte2str(self, data):
        '''将原始字节码转为字符串'''
        i = 0
        length = len(data)
        ret = u''
        while i < length:
            x = data[i:i + 1] + data[i + 1:i + 2]
            t = chr(struct.unpack('H', x)[0])
            if t == u'\r':
                ret += u'\n'
            elif t != u' ':
                ret += t
            i += 2
        return ret

    # 获取拼音表
    def _getPyTable(self, data):
        if data[0:4] != '\x9D\x01\x00\x00':
            return None
        data = data[4:]
        pos = 0
        length = len(data)
        while pos < length:
            index = struct.unpack('H', data[pos] + data[pos + 1])[0]

            pos += 2
            l = struct.unpack('H', data[pos] + data[pos + 1])[0]

            pos += 2
            py = self._byte2str(data[pos:pos + l])

            self._GPy_Table[index] = py
            pos += l

    # 获取一个词组的拼音
    def _getWordPy(self, data):
        pos = 0
        length = len(data)
        ret = u''
        while pos < length:
            index = struct.unpack('H', data[pos] + data[pos + 1])[0]
            ret += self._GPy_Table[index]
            pos += 2
        return ret

    # 获取一个词组
    def _getWord(self, data):
        pos = 0
        length = len(data)
        ret = u''
        while pos < length:
            index = struct.unpack('H', data[pos] + data[pos + 1])[0]
            ret += self._GPy_Table[index]
            pos += 2
        return ret

    # 读取中文表
    def _getChinese(self, data):

        pos = 0
        length = len(data)
        while pos < length:
            # 同音词数量
            same = struct.unpack('H', data[pos:pos + 1] + data[pos + 1:pos + 2])[0]

            # 拼音索引表长度
            pos += 2
            py_table_len = struct.unpack('H', data[pos:pos + 1] + data[pos + 1:pos + 2])[0]
            # 拼音索引表
            pos += 2

            # 中文词组
            pos += py_table_len
            for i in range(same):
                # 中文词组长度
                c_len = struct.unpack('H', data[pos:pos + 1] + data[pos + 1:pos + 2])[0]
                # 中文词组
                pos += 2
                word = self._byte2str(data[pos: pos + c_len])
                # 扩展数据长度
                pos += c_len
                ext_len = struct.unpack('H', data[pos:pos + 1] + data[pos + 1:pos + 2])[0]
                # 词频
                pos += 2

                self._GTable.append(word)
                # 到下个词的偏移位置
                pos += ext_len

    def generate_dict(self):
        def deal(fn):
            with open(fn, 'rb') as f:
                data = f.read()

            if data[0:12] != b'\x40\x15\x00\x00\x44\x43\x53\x01\x01\x00\x00\x00':
                print('确认你选择的是搜狗(.scel)词库?')
                raise OSError('')

            self._getPyTable(data[self.STARTPY:self.STARTCHINESE])
            self._getChinese(data[self.STARTCHINESE:])

        for filename in os.listdir(self._src_path):
            if os.path.exists('{}'.format(self._dest_dict)):
                continue

            deal('{}/{}'.format(self._src_path, filename))

        with open(self._dest_dict, 'w') as dic:
            # 删除相同元素
            GTable_filter = sorted(set(self._GTable), key=self._GTable.index)

            for word in GTable_filter:
                dic.write(word)
                dic.write('\n')


if __name__ == '__main__':
    SogouCrawler('dicts/').crawl()
    Scel2Txt('dicts/', 'user_dict.dic').generate_dict()
