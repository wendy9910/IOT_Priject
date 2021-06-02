import tkinter as tk  # 使用Tkinter前需要先匯入
from tkinter import *
import cv2
from PIL import*
from PIL import Image, ImageTk
import random
import pandas as pd
import openpyxl
import datetime
import os

camera = cv2.VideoCapture(0)  # 摄像头 


def video_loop():
    success, img = camera.read()  # 从摄像头读取照片
    if success:
        # cv2.waitKey(1000)
        cv2image = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)  # 转换颜色从BGR到RGBA
        #image = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        current_image = Image.fromarray(cv2image)  # 将图像转换成Image对象
        imgtk = ImageTk.PhotoImage(image=current_image)
        panel.ImageTk = imgtk
        panel.config(image=imgtk)
        window.after(1, video_loop)

# 讀取
name = random.choice(['Wendy', 'Yang', 'Jan'])
tm = datetime.datetime.today()
date_str = tm.strftime("%Y/%m/%d")
tm_str = tm.strftime("%H:%M:%S")

data = pd.read_excel(os.path.join('Staffprofile.xlsx'), engine='openpyxl')
Signdata = pd.read_excel(os.path.join('SignIn.xlsx'), engine='openpyxl')

def readData():
    data.loc[(data['員工姓名'] == name) & (
        data['部門代號'] == 'A02'), '權限'] = "Pass"
    data.loc[(data['權限'] == 'Pass') & (data['員工姓名'] == name),
                '打卡:%s' % (date_str)] = tm_str

    Signdata.loc[(Signdata['員工姓名'] == name) & (
        data['權限'] == 'Pass'), '打卡:%s' % (date_str)] = tm_str

readData()

window = tk.Tk()
window.title('My Window')
window.geometry('1400x800')
# window.geometry(1200,800)

panel = Label(window, bg='#D2E9FF')  # initialize image panel
panel.place(x=80, y=50)

label1 = Label(window, text='日期：%s' % (date_str), bg='#D2E9FF', font=(
    '標楷體', 28), width=15, height=1, anchor=NW).place(x=1000, y=100, anchor='n')
label1 = Label(window, text='時間：%s' % (tm_str), bg='#D2E9FF', font=(
    '標楷體', 28),  width=15, height=1, anchor=NW).place(x=1000, y=220, anchor='n')
label1 = Label(window, text='部門：A02', bg='#D2E9FF', font=(
    '標楷體', 28),  width=15, height=1, anchor=NW).place(x=1000, y=340, anchor='n')
label1 = Label(window, text='姓名：%s' % (name), bg='#D2E9FF', font=(
    '標楷體', 28), width=15, height=1, anchor=NW).place(x=1000, y=460, anchor='n')


get_name = data['員工姓名'].tolist()

if name in get_name:
    p = Label(window, text='Pass', bg='#93FF93', fg='dark green', font=(
        'Arial', 50), width=15, height=1).place(x=650, y=600, anchor='n')  # Pass
else:
    e = tk.Label(window, text='Error', bg='pink', fg='dark red', font=(
        'Arial', 50), width=15, height=1).place(x=650, y=595, anchor='n')  # Error

video_loop()

window.configure(bg='#D2E9FF')

window.mainloop()

camera.release()
cv2.destroyAllWindows()
