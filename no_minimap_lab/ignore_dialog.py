"""Multiple exclusion rectangles edited against one frozen screenshot."""
import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk
from .ignore_regions import DEFAULTS, load


class IgnoreRegionsDialog(tk.Toplevel):
    def __init__(self,parent,frame,on_save):
        super().__init__(parent)
        self.title('忽略区域 · 可连续框选多个区域')
        self.transient(parent);self.resizable(False,False)
        self.frame=frame.copy();self.on_save=on_save
        self.regions=list(load());self.start_point=None
        h,w=frame.shape[:2]
        ratio=min(1.,(self.winfo_screenwidth()-110)/w,(self.winfo_screenheight()-300)/h)
        self.width,self.height=max(1,round(w*ratio)),max(1,round(h*ratio))
        ttk.Label(self,text='拖动添加忽略区域；可框选多个。下方选择编号后删除。清空后表示不忽略任何区域。',padding=8).pack()
        self.canvas=tk.Canvas(self,width=self.width,height=self.height,highlightthickness=0)
        self.canvas.pack()
        image=cv2.resize(self.frame,(self.width,self.height),interpolation=cv2.INTER_AREA)
        self.photo=ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB)))
        self.canvas.create_image(0,0,image=self.photo,anchor='nw')
        bar=ttk.Frame(self,padding=8);bar.pack(fill='x')
        self.listbox=tk.Listbox(bar,height=3,width=40,exportselection=False)
        self.listbox.pack(side='left');self.listbox.bind('<<ListboxSelect>>',lambda e:self.draw())
        for text,action in [('删除选中',self.remove),('清空全部',self.clear),('恢复默认',self.defaults),('保存',self.save),('取消',self.destroy)]:
            ttk.Button(bar,text=text,command=action).pack(side='left',padx=4)
        self.canvas.bind('<ButtonPress-1>',self.press)
        self.canvas.bind('<B1-Motion>',self.drag)
        self.canvas.bind('<ButtonRelease-1>',self.release)
        self.bind('<Escape>',lambda e:self.destroy())
        self.refresh();self.grab_set()

    def point(self,event):
        return max(0,min(1,event.x/self.width)),max(0,min(1,event.y/self.height))

    def press(self,event):
        self.start_point=self.point(event)

    def drag(self,event):
        if self.start_point is None:return
        x,y=self.start_point;a,b=self.point(event)
        self.canvas.delete('draft')
        self.canvas.create_rectangle(x*self.width,y*self.height,a*self.width,b*self.height,outline='#ffda50',width=2,tags='draft')

    def release(self,event):
        if self.start_point is None:return
        x,y=self.start_point;a,b=self.point(event);self.start_point=None
        self.canvas.delete('draft')
        rect=(min(x,a),min(y,b),abs(a-x),abs(b-y))
        if rect[2]*self.frame.shape[1]>=4 and rect[3]*self.frame.shape[0]>=4:
            self.regions.append(rect);self.refresh()

    def refresh(self):
        self.listbox.delete(0,'end')
        for i,(x,y,w,h) in enumerate(self.regions,1):
            self.listbox.insert('end',f'{i}: X={x:.1%} Y={y:.1%} 宽={w:.1%} 高={h:.1%}')
        self.draw()

    def draw(self):
        self.canvas.delete('region');selected=self.listbox.curselection()
        for i,(x,y,w,h) in enumerate(self.regions):
            color='#ffda50' if i in selected else '#ff6655'
            self.canvas.create_rectangle(x*self.width,y*self.height,(x+w)*self.width,(y+h)*self.height,outline=color,width=2,fill='black',stipple='gray50',tags='region')
            self.canvas.create_text(x*self.width+5,y*self.height+5,text=str(i+1),fill=color,anchor='nw',tags='region')

    def remove(self):
        for i in reversed(self.listbox.curselection()):self.regions.pop(i)
        self.refresh()

    def clear(self):
        self.regions=[];self.refresh()

    def defaults(self):
        self.regions=list(DEFAULTS);self.refresh()

    def save(self):
        try:self.on_save(self.regions)
        except Exception as exc:
            messagebox.showerror('保存失败',str(exc),parent=self);return
        self.destroy()
