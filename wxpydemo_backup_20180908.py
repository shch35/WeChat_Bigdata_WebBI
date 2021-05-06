# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 18:31:21 2018

@author: shch35
"""

import itchat
#import numpy as np
#import os
#from collections import Counter
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 绘图时可以显示中文
plt.rcParams['axes.unicode_minus'] = False  # 绘图时可以显示中文
#import TencentYoutuyun
#from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
#from PIL import Image
#import time
import re
#import snownlp
import jieba
import jieba.analyse
#import pyecharts
#import pyecharts_javascripthon


# 登陆

if __name__ == "__main__":
    itchat.auto_login(hotReload=True)
    friends = itchat.get_friends(update=True)

"""
male = 0
female = 0
other = 0
# friends[0]是自己的信息，因此我们要从[1:]开始
for i in friends[1:]:
    sex = i['Sex']  # 注意大小写，2 是女性， 1 是男性
    if sex == 1:
        male += 1
    elif sex == 2:
        female += 1
    else:
        other += 1
# 计算好友总数

# 提出好友的昵称、性别、省份、城市、个性签名，生成一个数据框
import pandas as pd

# friends[0]是自己的信息，因此我们要从[1:]开始
data = pd.DataFrame()
columns = ['NickName', 'Sex', 'Province', 'City', 'Signature']
for col in columns:
    val = []
    for i in friends[1:]:
        val.append(i[col])
    data[col] = pd.Series(val)


# 查看好友的分布情况
pr = data[data['Sex'] <= 2]['Province']
gd = data[data['Province'] == '北京']['City']



#import pyecharts
from pyecharts import Bar, Pie , Page
#from pyecharts import Pie,Grid,Bar




attr = ["男性", "女性", "其他"]
v1 = [male,female,other]
v12 = [male/(male+female+other),female/(male+female+other),other/(male+female+other)]

attr2 = data['Province'].value_counts().index
v2 = data['Province'].value_counts()


attr3 = gd.value_counts().index
v3 = gd.value_counts()


pie = Pie("我的好友-比例分析", title_pos='center', width=1200)
bar = Bar(title_pos='center', width=1200)




pie.add(
    "性别",
    attr,
    v1,
    center=[25, 50],
    is_random=True,
    radius=[10, 30],
    label_text_color=None,
    is_label_show=True,
    legend_orient="vertical",
    legend_pos="left",
)



pie.add(
    "区县",
    attr3,
    v3,
    center=[60, 55],
    is_random=True,
    radius=[30, 80],
    rosetype="area",
    label_text_color=None,
    is_label_show=True,
    legend_orient="vertical",
    legend_pos="left",
   # legend_pos="right",
)

bar.add("省份", attr2, v2, is_label_show=True, is_datazoom_show=True)







#import re

siglist = []
for i in data['Signature']:
    signature = i.strip().replace('emoji', '').replace('span', '').replace('class', '')
    rep = re.compile('1f\d+\w*|[<>/=]')  # 具体含义另行查看
    signature = rep.sub('', signature)
    siglist.append(signature)
text = ''.join(siglist)

import jieba  # 没有安装的话，先使用 pip install jieba 安装

word_list = jieba.cut(text, cut_all=True)
word_space_split = ' '.join(word_list)

import matplotlib.pyplot as plt
from wordcloud import WordCloud, ImageColorGenerator
import numpy as np
import PIL.Image as Image




coloring = np.array(Image.open("D:/python/demo/wechat/timg.jpg"))
cloud_mask = np.array(Image.open("D:/python/demo/wechat/timg.jpg"))

my_wordcloud = WordCloud(background_color="black", 
                         max_words=200,
                         mask=cloud_mask,
                         max_font_size=100, 
                         random_state=42, 
                         scale=2,
                         font_path="C:/Windows/Fonts/simkai.ttf",
                         width=1200,#宽
                         height=800
                         ).generate(word_space_split)







image_colors = ImageColorGenerator(coloring)

plt.imshow(my_wordcloud.recolor(color_func=image_colors))
plt.imshow(my_wordcloud)
plt.axis("off")
plt.show()

from os import path
d = path.dirname("C:\\Users\\shch3\\Desktop\\")
my_wordcloud.to_file(path.join(d,"wcrender.png"))



from pyecharts import WordCloud


wc = word_space_split.split()



import re
from collections import Counter
 
txt = word_space_split
new_txt = txt.replace('\n', ' ').lower()
strings = re.split('\W+', new_txt)
result = Counter(strings)
#每个单词出现的次数
#print(result)
#出现次数前10的单词
#print(result.most_common(50))
#某个单词出现的次数
#print("flower 出现的次数:%d" % result["flower"])




wc222 = result.most_common(50)



wcname = result.keys()

wcvalue = result.values()





wordcloud = WordCloud(width=1200, height=880)
wordcloud.add("", wcname, wcvalue, word_gap=20,word_size_range=[20, 100],shape='pentagon')








page = Page()

page.add(pie)
page.add(bar)
page.add(wordcloud)

bar.use_theme("dark")
pie.use_theme("dark")

page.render(path = "d:\\pie.render.html")




"""















