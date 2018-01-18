# -*- coding: utf-8 -*-
from bs4 import BeautifulSoup
from time import sleep
import random
import os
import requests


class ShortComment:
    URL = 'https://douban.com/accounts/login'
    FORM_DATA = {
        'redir': 'https://www.douban.com',
        'form_email': 'username',
        'form_password': 'password',
        'login': u'登录',
        'source': 'None',
    }
    HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.3; WOW64; Trident/7.0; LCJB; rv:11.0) like Gecko',
               'Referer': 'https://douban.com/accounts/login',
               'Host': 'accounts.douban.com',
               'Connection': 'Keep-Alive',
               'Content-Type': 'application/x-www-form-urlencoded'
               }

    def __init__(self, ids_file, comment_dir='.'):
        self._comment_dir = comment_dir
        self._ids = self._douban_ids(ids_file)
        self.sess = requests.session()
        #####以下代码 如果不要求 验证码，注释掉 start######
        # r_ = s.post(self.URL, data=self.FORM_DATA, headers=self.HEADERS)
        # a = r_.text
        # soup_ = BeautifulSoup(a, 'html.parser')
        # captchaAddr = soup_.find('img', id='captcha_image')['src']
        # reCaptchaID = r'<input type='hidden' name='captcha-id' value='(.*?)'/'
        # captchaID = re.findall(reCaptchaID, a)
        # urllib.request.urlretrieve(captchaAddr, 'captcha.jpg')
        # captcha = input('please input the captcha:')
        # self.FORM_DATA['captcha-solution'] = captcha
        # self.FORM_DATA['captcha-id'] = captchaID
        #####end######
        self.sess.post(self.URL, data=self.FORM_DATA, headers=self.HEADERS)

    def _douban_ids(self, ids_file):
        ids = set()
        with open(ids_file) as f:
            for line in f:
                id = line.strip()
                ids.add(id)
        excluded_ids = set([f.split('.')[0] for f in os.listdir(self._comment_dir) if f.endswith('txt')])
        return list(ids - excluded_ids)

    def crawl(self):
        def process_h3(soup, fp):
            h3s = soup.findAll('h3')
            for i in h3s:
                aa = i.span.next_siblings
                bb = next(aa).next()
                if len(bb) == 4:
                    fp.write(bb[2].attrs['class'][0][-2:-1])
                    fp.write(' ')
                    fp.flush()
                    cc = i.next_siblings
                    next(cc)
                    dd = next(cc).get_text().strip()
                    ee = dd.replace('\n', ' ')
                    fp.write(ee)
                    fp.write('\n')

        def find_next(soup):
            line = soup.findAll('a', {'class', 'next'})
            if len(line) == 0:
                return None
            else:
                href = line[0].attrs['href']
                return target + href

        for id in self._ids:
            target = 'https://movie.douban.com/subject/%s/comments' % id

            movie = self.sess.get(target)
            page_movie = movie.text
            soup = BeautifulSoup(page_movie, 'lxml')
            movie_name = soup.find('title').get_text()[:-3]
            with open('{}/{}.{}.txt'.format(self._comment_dir, id, movie_name), 'w', encoding='utf-8') as fp:
                process_h3(soup, fp)
                while True:
                    inter = random.gauss(9, 2)
                    time = inter if inter > 2.1 else 2.1
                    sleep(time)
                    next_ = find_next(soup)
                    if next_ is None:
                        break
                    try:
                        soup = BeautifulSoup(self.sess.get(next_, timeout=10).text, 'lxml')
                        process_h3(soup, fp)
                    except:
                        sleep(100)
                        try:
                            soup = BeautifulSoup(self.sess.get(next_, timeout=10).text, 'lxml')
                            process_h3(soup, fp)
                        except:
                            break


if __name__ == '__main__':
    ShortComment('ids.txt', './comments').crawl()
