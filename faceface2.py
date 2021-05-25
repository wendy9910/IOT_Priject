import tkinter as tk  # 使用Tkinter前需要先匯入
from tkinter import *
import cv2
from PIL import*
from PIL import Image,ImageTk


camera = cv2.VideoCapture(0)    #摄像头

def video_loop():
    success, img = camera.read()  # 从摄像头读取照片
    if success:
        #cv2.waitKey(1000)
        cv2image = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)#转换颜色从BGR到RGBA
        current_image = Image.fromarray(cv2image)#将图像转换成Image对象
        imgtk = ImageTk.PhotoImage(image=current_image)
        panel.ImageTk = imgtk
        panel.config(image=imgtk)
        window.after(1, video_loop)

# 例項化object，建立視窗window
window = tk.Tk()

# 給視窗的視覺化起名字
window.title('My Window')

# 設定視窗的大小(長 x 寬)
window.geometry('1275x750') 

canvas = tk.Canvas(window, bg='#D2E9FF', width=1280 , height=800) # 畫布當背景

panel = Label(window)  # initialize image panel
panel.pack(padx=10, pady=10, anchor=W)


canvas.create_text(700,100,text='日期：' , font=('標楷體', 28))
canvas.create_text(700,220,text='時間：' , font=('標楷體', 28))
canvas.create_text(700,340,text='代碼：' , font=('標楷體', 28))
canvas.create_text(700,460,text='姓名：' , font=('標楷體', 28))

# 在圖形介面上設定標籤
p = tk.Label(window, text='Pass', bg='#93FF93', fg='dark green' , font=('Arial', 50), width=15, height=1).place(x=650,y=595,anchor='n') # Pass
#e = tk.Label(window, text='Error', bg='pink', fg='dark red' , font=('Arial', 50), width=15, height=1).place(x=650,y=595,anchor='n') # Error
# 說明： bg為背景，fg為字型顏色，font為字型，width為長，height為高，這裡的長和高是字元的長和高，比如height=2,就是標籤有2個字元這麼高
canvas.pack(anchor=E)

video_loop()

# 主視窗迴圈顯示
window.mainloop()

camera.release()
cv2.destroyAllWindows()