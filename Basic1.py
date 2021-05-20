import random
import pandas as pd
import openpyxl
import datetime
import os

name = random.choice(['Wendy', 'Yang', 'Jan'])
tm = datetime.datetime.today()
date_str = tm.strftime("%Y/%m/%d")
tm_str = tm.strftime("%H:%M:%S")

data = pd.read_excel(os.path.join('Staffprofile.xlsx'),engine='openpyxl')
Signdata = pd.read_excel(os.path.join('SignIn.xlsx'),engine='openpyxl')

data.loc[(data['員工姓名'] == name) & (data['部門代號'] == 'A02'), '權限'] = "Pass"
data.loc[(data['權限'] == 'Pass') & (data['員工姓名'] == name),
         '打卡:%s' % (date_str)] = tm_str


Signdata.loc[(Signdata['員工姓名'] == name) & (data['權限'] == 'Pass' ) , '打卡:%s' %(date_str) ] = tm_str


#data.loc[data['權限']=="Error"]
pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.unicode.east_asian_width', True)
pd.set_option('display.width', 200) # 设置打印宽度(**重要**)


df = pd.DataFrame(data)
data.to_excel('Staffprofile.xlsx',index=0)
Signdata.to_excel('SignIn.xlsx',index=0)
print(data)
print(Signdata)
print(name)
