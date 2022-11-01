import re
from tkinter import W
#以 utf-8 的编码格式打开指定文件
f = open("test.txt",encoding = "utf-8")
f_out = open("test_out.txt","w",encoding = 'utf-8')
#输出读取到的数据
txt = f.read()
a = 0
for c in txt:
    res = re.findall('[\u4e00-\u9fa5]', c)
    if(len(res) != 0):
        a += 1
        f_out.write(res[0])
        # print(res)
    # if c == "'":
    #     a += 1
print(a)
f.close()
