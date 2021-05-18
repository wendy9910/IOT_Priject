import random
import pandas as pd
import openpyxl
import datetime

name = random.choice(['Wendy', 'Yang', 'Jan'])
tm = datetime.datetime.today()
date_str = tm.strftime("%Y/%m/%d")
tm_str = tm.strftime("%H:%M:%S")

data = pd.read_excel('Staffprofile.xlsx')

data.loc[(data['員工姓名'] == name) & (data['部門代號'] == 'A02'), '權限'] = "Pass"
data.loc[(data['權限'] == 'Pass') & (data['員工姓名'] == name),
         '打卡:%s' % (date_str)] = tm_str
data.loc[data['員工姓名'] != name, '權限'] = "Error"

df = pd.DataFrame(data)

# data.to_excel('Staffprofile.xlsx')
print(data)
print(name)
