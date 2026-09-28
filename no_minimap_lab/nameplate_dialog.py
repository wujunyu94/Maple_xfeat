"""Frozen-frame nameplate and feet calibration in its own window."""
import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk


class NameplateDialog(tk.Toplevel):
    def __init__(self, parent, frame, on_save):
        super().__init__(parent)
        self.title('名牌与脚底标定 · 固定截图')
        self.transient(parent)
        self.resizable(False, False)
        self.frame = frame.copy()
        self.on_save = on_save
        self.roi = self.feet = self.start_point = None
        h, w = frame.shape[:2]
        ratio = min(1., (self.winfo_screenwidth()-100)/w, (self.winfo_screenheight()-210)/h)
        width, height = max(1,round(w*ratio)), max(1,round(h*ratio))
        self.sx, self.sy = width/w, height/h
        self.hint = tk.StringVar(value='① 拖框选中自己的名牌 → ② 点击角色脚底中心 → ③ 保存')
        ttk.Label(self, textvariable=self.hint, padding=10).pack(fill='x')
        self.canvas = tk.Canvas(self, width=width, height=height, highlightthickness=0)
        self.canvas.pack()
        image = cv2.resize(self.frame, (width,height), interpolation=cv2.INTER_AREA)
        self.photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
        self.canvas.create_image(0,0,image=self.photo,anchor='nw')
        bar = ttk.Frame(self,padding=10);bar.pack(fill='x')
        ttk.Button(bar,text='重新选择',command=self.reset).pack(side='left')
        self.save_button = ttk.Button(bar,text='保存标定',command=self.save,state='disabled')
        self.save_button.pack(side='right',padx=5)
        ttk.Button(bar,text='取消',command=self.destroy).pack(side='right',padx=5)
        self.canvas.bind('<ButtonPress-1>',self.press)
        self.canvas.bind('<B1-Motion>',self.drag)
        self.canvas.bind('<ButtonRelease-1>',self.release)
        self.bind('<Escape>',lambda event:self.destroy())
        self.grab_set()

    def point(self,event):
        return (max(0,min(self.frame.shape[1]-1,round(event.x/self.sx))),
                max(0,min(self.frame.shape[0]-1,round(event.y/self.sy))))

    def reset(self):
        self.roi = self.feet = self.start_point = None
        self.canvas.delete('selection');self.canvas.delete('feet')
        self.save_button.configure(state='disabled')
        self.hint.set('① 拖框选中自己的名牌，不要包含旁边人物的名牌')

    def press(self,event):
        if self.roi is None:
            self.start_point = self.point(event)
        else:
            self.feet = self.point(event)
            self.canvas.delete('feet')
            x,y=self.feet[0]*self.sx,self.feet[1]*self.sy
            self.canvas.create_line(x-8,y,x+8,y,fill='#ff6633',width=3,tags='feet')
            self.canvas.create_line(x,y-8,x,y+8,fill='#ff6633',width=3,tags='feet')
            self.hint.set('③ 确认黄色名牌框和橙色脚底十字，点击“保存标定”；可再次点击调整脚底')
            self.save_button.configure(state='normal')

    def drag(self,event):
        if self.roi is not None or self.start_point is None:
            return
        x,y=self.start_point;end=self.point(event)
        self.canvas.delete('selection')
        self.canvas.create_rectangle(x*self.sx,y*self.sy,end[0]*self.sx,end[1]*self.sy,
                                     outline='#ffe66d',width=2,tags='selection')

    def release(self,event):
        if self.roi is not None or self.start_point is None:
            return
        self.drag(event)
        x0,y0=self.start_point;x1,y1=self.point(event)
        x,y,w,h=min(x0,x1),min(y0,y1),abs(x1-x0),abs(y1-y0)
        self.start_point=None
        if w<15 or h<8:
            self.canvas.delete('selection')
            self.hint.set('名牌框太小，请重新拖框（至少15×8原图像素）')
            return
        self.roi=(x,y,w,h)
        self.hint.set('② 在这张固定截图上，点击角色脚底的中心位置')

    def save(self):
        if self.roi is None or self.feet is None:
            return
        try:
            self.on_save(self.frame,self.roi,self.feet)
        except Exception as exc:
            messagebox.showerror('标定失败',str(exc),parent=self)
            return
        self.destroy()
