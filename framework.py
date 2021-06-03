import random
import pandas as pd
import openpyxl
import os
import numpy as np
import datetime
import tkinter as tk  # 使用Tkinter前需要先匯入
from tkinter import *


name = random.choice(['Wendy', 'Yang', 'Jan'])
tm = datetime.datetime.today()
date_str = tm.strftime("%Y/%m/%d")
tm_str = tm.strftime("%H:%M:%S")

data = pd.read_excel(os.path.join('Staffprofile.xlsx'), engine='openpyxl')

Signdata = pd.read_excel(os.path.join('SignIn.xlsx'), engine='openpyxl')
get_name = data['員工姓名'].tolist()
k = ['Pass']


def readData():

    data.loc[(data['權限'] == 'Pass') & (data['員工姓名'] == name),
             '打卡:%s' % (date_str)] = tm_str
    Signdata.loc[(Signdata['員工姓名'] == name) & (
        data['權限'] == 'Pass'), '打卡:%s' % (date_str)] = tm_str


def signIN():
    if name in get_name:

        print('公司員工')

        data.loc[(data['員工姓名'] == name) & (
            data['部門代號'] == 'A02'), '權限'] = "Pass"

        card = data.loc[(data['員工姓名'] == name) & (
            data['部門代號'] == 'A02'), '權限'].tolist()

        if card == k:
            print(' %s 擁有辦公室權限' % (name))
            readData()

        else:
            print(' %s 沒有辦公室權限' % (name))

    else:
        print('非公司員工')


window = tk.Tk()
window.title('My Window')
window.geometry('1400x800')


code = data.loc[(data['員工姓名'] == name), '部門代號'].to_string()
n = code.lstrip('01234  ')

label1 = Label(window, text='部門：%s' % (n), bg='#D2E9FF', font=(
    '標楷體', 28),  width=15, height=1, anchor=NW).place(x=1000, y=340, anchor='n')

#n = code.lstrip('  ')


signIN()
window.mainloop()
