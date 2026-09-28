"""Lightweight Canvas overlays, refreshed independently of recognition.

Extrapolation is display-only and expires after 250ms. It never feeds control.
"""
import math
import cv2
from PIL import Image, ImageTk


def projected_player(record, now):
    player = record.get('player_state', {})
    world = player.get('world')
    age = now-record['timestamp']
    if world is None or not 0 <= age <= .25 or age+player.get('observation_age', 0) > .25:
        return None
    velocity = player.get('velocity', (0, 0))
    if not all(math.isfinite(v) for v in (*world, *velocity)):
        return None
    return [world[i]+velocity[i]*age for i in range(2)]


class MapPreview:
    def __init__(self, canvas):
        self.canvas = canvas
        self.background = self.record = None
        self.photo = None
        self.size = None
        self.camera_velocity = (0, 0)

    def setup(self, background, origin, shape):
        self.background, self.origin, self.shape = background, origin, shape
        self.size = None
        self.record = None
        self.camera_velocity = (0, 0)
        self.canvas.delete('all')

    def update(self, record, frame_shape, screen_scale):
        previous = self.record
        self.camera_velocity = (0, 0)
        if previous and previous.get('camera') and record.get('camera'):
            dt = record['timestamp']-previous['timestamp']
            if .005 <= dt <= .25:
                velocity = tuple((record['camera'][i]-previous['camera'][i])/dt for i in range(2))
                if math.hypot(*velocity) <= 1200:
                    self.camera_velocity = velocity
        self.record, self.frame_shape, self.screen_scale = record, frame_shape, screen_scale

    def clear(self):
        self.record = None
        self.canvas.delete('overlay')

    def item(self, tag, kind, coords, **options):
        items = self.canvas.find_withtag(tag)
        if items:
            self.canvas.coords(items[0], *coords)
            self.canvas.itemconfigure(items[0], state='normal', **options)
        else:
            getattr(self.canvas, 'create_'+kind)(*coords, tags=('overlay', tag), **options)

    def draw(self, now):
        if self.background is None:
            return
        size = (max(10,self.canvas.winfo_width()), max(10,self.canvas.winfo_height()))
        if size != self.size:
            h, w = self.background.shape[:2]
            ratio = min(size[0]/w, size[1]/h)
            width, height = max(1,round(w*ratio)), max(1,round(h*ratio))
            image = cv2.resize(self.background, (width,height), interpolation=cv2.INTER_AREA)
            self.photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
            self.canvas.delete('base')
            self.canvas.create_image(0,0,image=self.photo,anchor='nw',tags='base')
            self.canvas.tag_lower('base')
            self.sx, self.sy = width/self.shape[1], height/self.shape[0]
            self.size = size
        self.canvas.itemconfigure('overlay', state='hidden')
        record = self.record
        if not record:
            return
        age = max(0, now-record['timestamp'])
        def point(p):
            return ((p[0]-self.origin[0])*self.sx, (p[1]-self.origin[1])*self.sy)
        def dot(tag, p, color, radius=4):
            x,y=point(p)
            self.item(tag,'oval',(x-radius,y-radius,x+radius,y+radius),fill=color,outline=color)
        camera = record.get('camera')
        if camera:
            def box(tag, camera, color, dash):
                far = [camera[0]+self.frame_shape[1]/self.screen_scale,
                       camera[1]+self.frame_shape[0]/self.screen_scale]
                self.item(tag,'rectangle',(*point(camera),*point(far)),outline=color,width=2,dash=dash)
            box('measured-camera',camera,'#ffe04a' if age<=.25 else '#777777',())
            if age<=.25:
                projected = [camera[i]+self.camera_velocity[i]*age for i in range(2)]
                box('predicted-camera',projected,'#ff9d35',(4,3))
        player = record.get('player_state', {})
        world = player.get('world')
        if world is not None:
            dot('player',world,'#ef4545' if player.get('status')!='PREDICTED' and age<=.25 else '#888888')
        predicted = projected_player(record, now)
        if predicted is not None:
            dot('predicted-player',predicted,'#ff9d35',3)
            x,y=point(predicted)
            radius=max(7,(player.get('uncertainty_px',3)+180*age)*max(self.sx,self.sy))
            self.item('uncertainty','oval',(x-radius,y-radius,x+radius,y+radius),outline='#ff9d35')
        yellow = record.get('yellow', {})
        if yellow.get('detected') and age<=.25:
            x,y=point(yellow['raw_world'])
            self.item('yellow-x','line',(x-5,y,x+5,y),fill='#30eeee',width=2)
            self.item('yellow-y','line',(x,y-5,x,y+5),fill='#30eeee',width=2)
        if age > .25:
            state = '预测已过期，等待识别'
        elif not camera:
            state = '地图定位未建立，暂无世界坐标'
        elif predicted is None:
            state = '人物预测不可用，等待可靠观测'
        else:
            state = '橙色：显示预测'
        text = f"定位结果 {age*1000:.0f}ms | "+state
        self.item('age','text',(8,8),text=text,fill='white',anchor='nw')
