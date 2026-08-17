"""
RD Simulator Pro MAX ++  —  Pygame Edition v3
Converted from PySide6 to Pygame.

Controls:
  SPACE  — Start / Pause          R — Reset with current seed
  C      — Clear                  F — Fit image to viewer
  0      — Reset zoom to 1:1      ESC / Q — Quit
  S      — Save state             L — Load state
  E      — Export snapshot        Scroll — Zoom image (toward cursor)
  +/-    — Zoom in/out            H — Toggle help overlay
  MMB/RMB drag — Pan image        LMB drag on image — Paint
  DblClick viewer — Fit           1-7 — Select seed type
"""

import sys, os, time, datetime, logging, threading, warnings
import numpy as np, cv2, pygame

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)-5s] %(name)s - %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("RD-Sim")

# ============================================================================
# CONFIG
# ============================================================================

PRESETS = {
    "Coral": (0.0545,0.0620), "Mitosis": (0.0367,0.0649), "Spirals": (0.0260,0.0510),
    "Waves": (0.0140,0.0450), "Spots": (0.0350,0.0650), "Stripes": (0.0420,0.0590),
    "Maze": (0.0290,0.0570),
}
SEEDS = ["Multi-Random","Single Circle","Ring","Grid Dots","Noise Field","Spiral","Image Import"]
BOUNDARIES = ["No-Flux","Periodic (Toroidal)"]
COLORMAPS = {
    "Inferno":cv2.COLORMAP_INFERNO,"Viridis":cv2.COLORMAP_VIRIDIS,"Plasma":cv2.COLORMAP_PLASMA,
    "Magma":cv2.COLORMAP_MAGMA,"Cividis":cv2.COLORMAP_CIVIDIS,"Turbo":cv2.COLORMAP_TURBO,
    "Jet":cv2.COLORMAP_JET,"Rainbow":cv2.COLORMAP_RAINBOW,"Hot":cv2.COLORMAP_HOT,
    "Cool":cv2.COLORMAP_COOL,"Ocean":cv2.COLORMAP_OCEAN,"Twilight":cv2.COLORMAP_TWILIGHT,
    "DeepGreen":cv2.COLORMAP_DEEPGREEN,"Bone":cv2.COLORMAP_BONE,
}

# Colours
BG=(30,30,30); PANEL_BG=(37,37,37); GROUP_BG=(42,42,42); GROUP_BORDER=(58,58,58)
TEXT=(220,220,220); ACCENT=(0,120,215); ACCENT_HOV=(30,140,235); ACCENT_ACT=(0,100,0)
BTN_DIS=(34,34,34); DIM=(136,136,136); SL_TRACK=(68,68,68); SL_FILL=(0,120,215)
BORDER=(58,58,58); RED=(255,50,50); GREEN=(50,255,50); YELLOW=(255,200,50)
WHITE=(255,255,255); SB_BG=(40,40,40); SB_KNOB=(120,120,120)
VIEWER_BG=(15,15,15); STATUS_BG=(25,25,25); HELP_BG=(0,0,0,200)
ORANGE=(255,165,0)

# ============================================================================
# BACKEND
# ============================================================================

try:
    import cupy as cp; GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

try:
    from numba import jit, prange; NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def jit(*a,**k):
        def d(f): return f
        return d
    def prange(*a, **kw): return range(*a)

warnings.filterwarnings("ignore")
log.info("Backend: CuPy=%s Numba=%s", GPU_AVAILABLE, NUMBA_AVAILABLE)

# ============================================================================
# NUMBA KERNEL
# ============================================================================

@jit(nopython=True, parallel=True, cache=True, fastmath=True)
def _nb_step(Us,Vs,Ud,Vd,Dux,Duy,Dvx,Dvy,Ff,kf,dt,bc,u9):
    R,C=Us.shape
    for i in prange(R):
        for j in range(C):
            if bc==1: ip=(i+1)%R;im=(i-1+R)%R;jp=(j+1)%C;jm=(j-1+C)%C
            else: ip=min(i+1,R-1);im=max(i-1,0);jp=min(j+1,C-1);jm=max(j-1,0)
            u=Us[i,j];Uip, Uim, Ujp, Ujm = Us[ip,j],Us[im,j],Us[i,jp],Us[i,jm]
            v=Vs[i,j];Vip, Vim, Vjp, Vjm = Vs[ip,j],Vs[im,j],Vs[i,jp],Vs[i,jm]
            Lu=Dux*(Ujp-2*u+Ujm)+Duy*(Uip-2*u+Uim)
            Lv=Dvx*(Vjp-2*v+Vjm)+Dvy*(Vip-2*v+Vim)
            if u9:
                Lu+=(Dux+Duy)/12.0*(Us[ip,jp]+Us[im,jm]+Us[ip,jm]+Us[im,jp]-4*u)
                Lv+=(Dvx+Dvy)/12.0*(Vs[ip,jp]+Vs[im,jm]+Vs[ip,jm]+Vs[im,jp]-4*v)
            uvv=u*v*v;F=Ff[i,j];k=kf[i,j]
            Ud[i,j]=u+(Lu-uvv+F*(1-u))*dt; Vd[i,j]=v+(Lv+uvv-(F+k)*v)*dt

# ============================================================================
# ENGINE
# ============================================================================

class Engine:
    def __init__(self, w=256, h=256, gpu=True):
        self.width=w; self.height=h; self.use_gpu=gpu and GPU_AVAILABLE; self.lock=threading.Lock()
        self.F=0.0545; self.k=0.062; self.Dux=0.2; self.Duy=0.2; self.Dvx=0.1; self.Dvy=0.1
        self.boundary=0; self.u9=False; self.noise=0.0; self.F_grad=None; self.k_grad=None
        self.iteration=0; self.reset()

    def lib(self): return cp if self.use_gpu else np

    def _alloc(self):
        L=self.lib(); s=(self.height,self.width)
        self.Us=L.ones(s,np.float32); self.Vs=L.zeros(s,np.float32)
        self.Ud=L.ones(s,np.float32); self.Vd=L.zeros(s,np.float32)
        self._grads()

    def _grads(self):
        L=self.lib(); s=(self.height,self.width)
        if self.F_grad is not None:
            t=L.tile(L.linspace(0,1,self.width),(self.height,1))
            self.Ff=L.maximum(self.F+(t-0.5)*self.F_grad, 0.001)
        else: self.Ff=L.full(s,self.F,np.float32)
        if self.k_grad is not None:
            t=L.tile(L.linspace(0,1,self.width),(self.height,1))
            self.kf=L.maximum(self.k+(t-0.5)*self.k_grad, 0.001)
        else: self.kf=L.full(s,self.k,np.float32)

    def reset(self, seed="Multi-Random", img_path=None):
        with self.lock:
            self._alloc(); L=self.lib(); self.iteration=0; self._mesh=None
            y,x=L.meshgrid(L.arange(self.height),L.arange(self.width),indexing="ij")
            cx,cy=self.width//2,self.height//2; r=self.height//8
            if seed=="Multi-Random":
                for _ in range(15):
                    xi,yi,ri=int(L.random.randint(0,self.width)),int(L.random.randint(0,self.height)),int(L.random.randint(5,25))
                    m=(x-xi)**2+(y-yi)**2<=ri**2; self.Vs[m]=L.random.uniform(.2,.5); self.Us[m]=.5
            elif seed=="Single Circle":
                m=(x-cx)**2+(y-cy)**2<=r**2; self.Vs[m]=.25; self.Us[m]=.5
            elif seed=="Ring":
                d=L.sqrt((x-cx)**2+(y-cy)**2); m=(d>r*.7)&(d<r); self.Vs[m]=.25; self.Us[m]=.5
            elif seed=="Grid Dots":
                sp=self.width//10
                for gi in range(5,self.width,sp):
                    for gj in range(5,self.height,sp):
                        m=(x-gi)**2+(y-gj)**2<=9; self.Vs[m]=.25; self.Us[m]=.5
            elif seed=="Noise Field":
                n=L.random.rand(self.height,self.width).astype(np.float32); m=n>.98; self.Vs[m]=.25; self.Us[m]=.5
            elif seed=="Spiral":
                th=L.arctan2(y-cy,x-cx); d=L.sqrt((x-cx)**2+(y-cy)**2)
                m=(L.sin(th*3+d*.1)>.7)&(d<r*2); self.Vs[m]=.25; self.Us[m]=.5
            elif seed=="Image Import" and img_path:
                im=cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
                if im is not None:
                    im=cv2.resize(im,(self.width,self.height)); v=im.astype(np.float32)/255*.5
                    self.Vs=L.array(v); self.Us=L.ones_like(v)*(1-v*.5)
            L.copyto(self.Ud,self.Us); L.copyto(self.Vd,self.Vs)
            log.info("Reset: seed='%s' F=%.4f k=%.4f", seed, self.F, self.k)

    def clear(self):
        with self.lock:
            self.Us.fill(1);self.Vs.fill(0);self.Ud.fill(1);self.Vd.fill(0);self.iteration=0

    def disturb(self, n=5):
        with self.lock:
            L=self.lib()
            if not hasattr(self,'_mesh') or self._mesh is None or self._mesh.shape != (self.height, self.width):
                y,x=L.meshgrid(L.arange(self.height),L.arange(self.width),indexing="ij")
                self._mesh=(y,x)
            y,x=self._mesh
            for _ in range(n):
                cx=int(L.random.randint(0,self.width));cy=int(L.random.randint(0,self.height));r=int(L.random.randint(3,10))
                m=(x-cx)**2+(y-cy)**2<=r**2; self.Vs[m]=L.random.uniform(.2,.4); self.Us[m]=.5

    def paint(self, x, y, rad=10, stren=.5):
        with self.lock:
            if hasattr(self,'_undo_stack') and len(self._undo_stack)>=self._undo_max: self._undo_stack.pop(0)
            if not hasattr(self,'_undo_stack'): self._undo_stack=[]
            self._undo_stack.append((self.Us.copy(),self.Vs.copy()))
            L=self.lib()
            if not hasattr(self,'_mesh') or self._mesh is None or self._mesh.shape != (self.height, self.width):
                yy,xx=L.meshgrid(L.arange(self.height),L.arange(self.width),indexing="ij")
                self._mesh=(yy,xx)
            yc,xc=self._mesh
            m=L.sqrt((xc-x)**2+(yc-y)**2)<rad; self.Vs[m]=stren; self.Vd[m]=stren

    def step(self, steps=10, dt=1.0):
        mD=max(self.Dux,self.Duy,self.Dvx,self.Dvy)
        dt=min(dt,.9/(4*mD)) if mD>0 else dt
        with self.lock:
            (self._gpu if self.use_gpu else self._cpu)(steps,dt)
            self.iteration+=steps

    def _gpu(self, steps, dt):
        pm="wrap" if self.boundary==1 else "edge"
        for _ in range(steps):
            Up=cp.pad(self.Us,1,mode=pm);Vp=cp.pad(self.Vs,1,mode=pm)
            Uc=Up[1:-1,1:-1];Un,Us_=Up[:-2,1:-1],Up[2:,1:-1];Ue,Uw=Up[1:-1,2:],Up[1:-1,:-2]
            Vc=Vp[1:-1,1:-1];Vn,Vs2=Vp[:-2,1:-1],Vp[2:,1:-1];Ve,Vw=Vp[1:-1,2:],Vp[1:-1,:-2]
            Lu=self.Dux*(Ue+Uw-2*Uc)+self.Duy*(Un+Us_-2*Uc)
            Lv=self.Dvx*(Ve+Vw-2*Vc)+self.Dvy*(Vn+Vs2-2*Vc)
            if self.u9:
                Lu+=(self.Dux+self.Duy)/12.0*(Up[:-2,2:]+Up[:-2,:-2]+Up[2:,2:]+Up[2:,:-2]-4*Uc)
                Lv+=(self.Dvx+self.Dvy)/12.0*(Vp[:-2,2:]+Vp[:-2,:-2]+Vp[2:,2:]+Vp[2:,:-2]-4*Vc)
            uvv=self.Us*self.Vs*self.Vs
            self.Ud=self.Us+(Lu-uvv+self.Ff*(1-self.Us))*dt
            self.Vd=self.Vs+(Lv+uvv-(self.Ff+self.kf)*self.Vs)*dt
            if self.noise>0:
                self.Ud+=cp.random.normal(0,self.noise,self.Us.shape).astype(cp.float32)*dt
                self.Vd+=cp.random.normal(0,self.noise,self.Vs.shape).astype(cp.float32)*dt
            cp.clip(self.Ud,0,1,out=self.Ud);cp.clip(self.Vd,0,1,out=self.Vd)
            self.Us,self.Ud=self.Ud,self.Us;self.Vs,self.Vd=self.Vd,self.Vs

    def _cpu(self, steps, dt):
        for _ in range(steps):
            _nb_step(self.Us,self.Vs,self.Ud,self.Vd,self.Dux,self.Duy,self.Dvx,self.Dvy,
                     self.Ff,self.kf,dt,self.boundary,self.u9)
            if self.noise>0:
                self.Ud+=np.random.normal(0,self.noise,self.Ud.shape).astype(np.float32)*dt
                self.Vd+=np.random.normal(0,self.noise,self.Vd.shape).astype(np.float32)*dt
            np.clip(self.Ud,0,1,out=self.Ud);np.clip(self.Vd,0,1,out=self.Vd)
            self.Us,self.Ud=self.Ud,self.Us;self.Vs,self.Vd=self.Vd,self.Vs

    def get_image(self, cmap=cv2.COLORMAP_INFERNO, gamma=.6, sharp=False,
                  bloom=0., dual=False, cmap_u=cv2.COLORMAP_COOL):
        with self.lock:
            vi=cp.asnumpy(self.Vs) if self.use_gpu else self.Vs.copy()
            ui=cp.asnumpy(self.Us) if self.use_gpu else self.Us.copy()
        if dual:
            un=np.zeros_like(ui,dtype=np.uint8) if ui.max()<=ui.min() else cv2.normalize(ui,None,0,255,cv2.NORM_MINMAX,cv2.CV_8U)
            vn=np.zeros_like(vi,dtype=np.uint8) if vi.max()<=vi.min() else cv2.normalize(vi,None,0,255,cv2.NORM_MINMAX,cv2.CV_8U)
            uc=cv2.applyColorMap(un,cmap_u);vc=cv2.applyColorMap(vn,cmap)
            if gamma!=1.0:
                uc=cv2.normalize((uc.astype(np.float32)/255)**gamma,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
                vc=cv2.normalize((vc.astype(np.float32)/255)**gamma,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
            a=vn.astype(np.float32)/255;a3=np.stack([a]*3,-1)
            col=(uc*(1-a3)+vc*a3).astype(np.uint8)
        else:
            if vi.max()<=vi.min():
                vn=np.full_like(vi,128,dtype=np.uint8)
            else:
                vn=cv2.normalize(vi,None,0,255,cv2.NORM_MINMAX,cv2.CV_8U)
            if gamma!=1.0:
                vn=cv2.normalize((vn.astype(np.float32)/255)**gamma,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
            col=cv2.applyColorMap(vn,cmap)
        if sharp: col=cv2.filter2D(col,-1,np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]]))
        if bloom>0:
            g=cv2.cvtColor(col,cv2.COLOR_BGR2GRAY); gm=g.max()
            thresh=int(np.percentile(g,92)) if gm>0 else 180; _,bm=cv2.threshold(g,thresh,255,cv2.THRESH_BINARY)
            bl=cv2.GaussianBlur(col,(0,0),15);m3=(bm.astype(np.float32)/255)[:,:,np.newaxis]
            col=np.clip(col*(1-m3*bloom)+bl*m3*bloom,0,255).astype(np.uint8)
        return cv2.cvtColor(col,cv2.COLOR_BGR2RGB)

# ============================================================================
# WIDGETS
# ============================================================================

class Button:
    def __init__(self, rect, text, toggle=False, font=None):
        self.rect=pygame.Rect(rect); self.text=text; self.toggle=toggle
        self.checked=False; self.enabled=True; self.hovered=False; self.font=font; self.clicked=False
    def handle_event(self, e):
        self.clicked=False
        if not self.enabled: return
        if e.type==pygame.MOUSEMOTION: self.hovered=self.rect.collidepoint(e.pos)
        elif e.type==pygame.MOUSEBUTTONDOWN and e.button==1 and self.rect.collidepoint(e.pos):
            if self.toggle: self.checked=not self.checked
            self.clicked=True
    def draw(self, s):
        if not self.enabled:   c,tc=BTN_DIS,DIM
        elif self.checked:     c,tc=ACCENT_ACT,WHITE
        elif self.hovered:     c,tc=ACCENT_HOV,WHITE
        else:                  c,tc=(51,51,51),TEXT
        pygame.draw.rect(s,c,self.rect,border_radius=4)
        pygame.draw.rect(s,(85,85,85),self.rect,1,border_radius=4)
        t=self.font.render(self.text,True,tc)
        s.blit(t,(self.rect.x+(self.rect.width-t.get_width())//2,self.rect.y+(self.rect.height-t.get_height())//2))

class Slider:
    def __init__(self, rect, mn, mx, val, label="", font=None, fmt="{}"):
        self.rect=pygame.Rect(rect); self.mn=mn; self.mx=mx; self.value=val
        self.label=label; self.font=font; self.fmt=fmt; self.dragging=False; self.changed=False
    def handle_event(self, e):
        self.changed=False
        if e.type==pygame.MOUSEBUTTONDOWN and e.button==1 and self._tr().inflate(10,18).collidepoint(e.pos):
            self.dragging=True; self._upd(e.pos[0]); self.changed=True
        elif e.type==pygame.MOUSEBUTTONUP and e.button==1: self.dragging=False
        elif e.type==pygame.MOUSEMOTION and self.dragging: self._upd(e.pos[0]); self.changed=True
    def _tr(self): return pygame.Rect(self.rect.x+8,self.rect.y+self.rect.height-8,self.rect.width-16,6)
    def _upd(self, mx):
        t=self._tr(); r=max(0,min(1,(mx-t.x)/t.width)) if t.width>0 else 0; self.value=self.mn+r*(self.mx-self.mn)
    def draw(self, s):
        if self.label: s.blit(self.font.render(self.label,True,TEXT),(self.rect.x,self.rect.y))
        vt=self.font.render(self.fmt.format(self.value),True,DIM)
        s.blit(vt,(self.rect.right-vt.get_width(),self.rect.y))
        tr=self._tr(); pygame.draw.rect(s,SL_TRACK,tr,border_radius=3)
        t=(self.value-self.mn)/(self.mx-self.mn) if self.mx!=self.mn else 0
        fw=int(t*tr.width); pygame.draw.rect(s,SL_FILL,(tr.x,tr.y,fw,tr.height),border_radius=3)
        hx=int(tr.x+max(0,min(1,t))*tr.width);hy=tr.centery
        pygame.draw.circle(s,ACCENT,(hx,hy),7); pygame.draw.circle(s,WHITE,(hx,hy),7,1)

class Dropdown:
    def __init__(self, rect, options, default=0, font=None):
        self.rect=pygame.Rect(rect); self.options=options; self.index=default
        self.font=font; self.open=False; self.hover=-1; self.changed=False
    def handle_event(self, e):
        self.changed=False
        if e.type==pygame.MOUSEBUTTONDOWN and e.button==1:
            if self.open:
                for i in range(len(self.options)):
                    ir=pygame.Rect(self.rect.x,self.rect.y+(i+1)*self.rect.height,self.rect.width,self.rect.height)
                    if ir.collidepoint(e.pos): self.index=i; self.changed=True
                self.open=False
            elif self.rect.collidepoint(e.pos): self.open=True
        elif e.type==pygame.MOUSEMOTION and self.open:
            self.hover=-1
            for i in range(len(self.options)):
                ir=pygame.Rect(self.rect.x,self.rect.y+(i+1)*self.rect.height,self.rect.width,self.rect.height)
                if ir.collidepoint(e.pos): self.hover=i
    def close(self): self.open=False; self.hover=-1
    def draw_box(self, s):
        pygame.draw.rect(s,(51,51,51),self.rect,border_radius=3)
        pygame.draw.rect(s,(85,85,85),self.rect,1,border_radius=3)
        t=self.font.render(self.options[self.index],True,TEXT)
        s.blit(t,(self.rect.x+6,self.rect.y+(self.rect.height-t.get_height())//2))
        ax,ay=self.rect.right-18,self.rect.centery
        arrow=[(ax-4,ay+3),(ax+4,ay+3),(ax,ay-3)] if self.open else [(ax-4,ay-3),(ax+4,ay-3),(ax,ay+3)]
        pygame.draw.polygon(s,TEXT,arrow)
    @property
    def current_text(self): return self.options[self.index]

class Checkbox:
    def __init__(self, rect, text, checked=False, font=None):
        self.rect=pygame.Rect(rect); self.text=text; self.checked=checked
        self.font=font; self.changed=False
    def handle_event(self, e):
        self.changed=False
        if e.type==pygame.MOUSEBUTTONDOWN and e.button==1 and self.rect.collidepoint(e.pos):
            self.checked=not self.checked; self.changed=True
    def draw(self, s):
        b=pygame.Rect(self.rect.x,self.rect.y+1,15,15)
        pygame.draw.rect(s,(51,51,51),b,border_radius=2); pygame.draw.rect(s,(85,85,85),b,1,border_radius=2)
        if self.checked: pygame.draw.rect(s,ACCENT,b.inflate(-6,-6),border_radius=1)
        t=self.font.render(self.text,True,TEXT)
        s.blit(t,(b.right+6,self.rect.y+(self.rect.height-t.get_height())//2))

class SpinBox:
    def __init__(self, rect, mn, mx, val, step=.001, dec=4, font=None):
        self.rect=pygame.Rect(rect); self.mn=mn; self.mx=mx; self.value=val
        self.step=step; self.dec=dec; self.font=font; self.changed=False
        self._drag=False; self._dx=0; self._dv=0.0
    def handle_event(self, e):
        self.changed=False; bw=18
        if e.type==pygame.MOUSEBUTTONDOWN and e.button==1:
            plus=pygame.Rect(self.rect.right-bw,self.rect.y,bw,self.rect.height//2)
            minus=pygame.Rect(self.rect.right-bw,self.rect.y+self.rect.height//2,bw,self.rect.height//2)
            if plus.collidepoint(e.pos): self.value=min(self.mx,round(self.value+self.step,self.dec+2)); self.changed=True
            elif minus.collidepoint(e.pos): self.value=max(self.mn,round(self.value-self.step,self.dec+2)); self.changed=True
            elif self.rect.collidepoint(e.pos) and pygame.key.get_mods()&pygame.KMOD_SHIFT:
                self._drag=True; self._dx=e.pos[0]; self._dv=self.value
        elif e.type==pygame.MOUSEBUTTONUP and e.button==1: self._drag=False
        elif e.type==pygame.MOUSEMOTION and self._drag:
            self.value=max(self.mn,min(self.mx,self._dv+(e.pos[0]-self._dx)*self.step*.5)); self.changed=True
        elif e.type==pygame.MOUSEWHEEL and self.rect.inflate(4,4).collidepoint(e.pos):
            self.value=max(self.mn,min(self.mx,self.value+e.y*self.step)); self.changed=True
    def draw(self, s):
        pygame.draw.rect(s,(51,51,51),self.rect,border_radius=3)
        pygame.draw.rect(s,(85,85,85),self.rect,1,border_radius=3)
        t=self.font.render(f"{{:.{self.dec}f}}".format(self.value),True,TEXT)
        s.blit(t,(self.rect.x+4,self.rect.y+(self.rect.height-t.get_height())//2))
        bw=18
        mp=pygame.mouse.get_pos()
        for r,ch in [(pygame.Rect(self.rect.right-bw,self.rect.y,bw,self.rect.height//2),"+"),
                      (pygame.Rect(self.rect.right-bw,self.rect.y+self.rect.height//2,bw,self.rect.height//2),"-")]:
            hov=r.collidepoint(mp)
            pygame.draw.rect(s,(90,90,90) if hov else (65,65,65),r)
            ts=self.font.render(ch,True,WHITE if hov else TEXT)
            s.blit(ts,(r.x+(r.width-ts.get_width())//2,r.y+(r.height-ts.get_height())//2))

# ============================================================================
# FILE DIALOGS
# ============================================================================

def _fdsave(title,name,filt):
    try:
        import tkinter as tk; from tkinter import filedialog
        r=tk.Tk();r.withdraw();r.attributes("-topmost",True)
        p=filedialog.asksaveasfilename(title=title,initialfile=name,filetypes=[filt]);r.destroy()
        return p or None
    except Exception: return None

def _fdopen(title,filt):
    try:
        import tkinter as tk; from tkinter import filedialog
        r=tk.Tk();r.withdraw();r.attributes("-topmost",True)
        p=filedialog.askopenfilename(title=title,filetypes=[filt]);r.destroy()
        return p or None
    except Exception: return None

# ============================================================================
# APP
# ============================================================================

class RDApp:
    STATUS_CLEAR_TIME = 3.0  # seconds before status auto-clears

    def __init__(self):
        pygame.init()
        info=pygame.display.Info()
        self.W=max(800,min(1400,info.current_w-80)); self.H=max(500,min(900,info.current_h-80))
        self.screen=pygame.display.set_mode((self.W,self.H),pygame.RESIZABLE)
        pygame.display.set_caption("RD Simulator Pro MAX ++")
        self.clock=pygame.time.Clock()
        self.fsm=pygame.font.SysFont(None,13); self.fm=pygame.font.SysFont(None,15)
        self.flg=pygame.font.SysFont(None,17); self.ft=pygame.font.SysFont(None,18,bold=True)
        self.use_gpu=GPU_AVAILABLE; self.res=512
        self.engine=Engine(self.res,self.res,self.use_gpu)
        self.running=True; self.paused=True; self.tfps=60; self.spf=30
        self.cmap=cv2.COLORMAP_INFERNO; self.cmap_u=cv2.COLORMAP_COOL
        self.gamma=.6; self.sharp=False; self.bloom=0.; self.dual=False
        self.autodist=True; self.brad=10; self.bstr=.9
        self.recording=False; self.vw=None; self.rec_start=0; self.rec_dur=0
        self.rgb=None; self.zoom=1.; self.pan=[0.,0.]
        self.panning=False; self.pan0=(0,0); self.drawing=False
        self.PW=320; self.scroll_y=0; self.max_scroll=0; self.in_panel=False
        self.fps=0.; self.status="Ready"; self._status_t=time.time()
        self.show_help=False
        self._phase_cache=None; self._phase_ck=None; self._blink=0
        self._last_click=0; self._last_click_pos=(0,0); self._rgb_lock=threading.Lock()
        self._tb_fit=pygame.Rect(0,0,0,0); self._tb_1to1=pygame.Rect(0,0,0,0)
        self._confirm_clear=False; self._confirm_time=0; self._stop_rec_pending=False; self._undo_max=10; self.engine._undo_stack=[]; self.engine._undo_max=10
        # Cached panel background (groups don't change)
        self._panel_bg_cache=None; self._panel_bg_ck=None
        self._build_ui()
        threading.Thread(target=self._worker,daemon=True).start()
        with self._rgb_lock: self.rgb=self.engine.get_image(self.cmap,self.gamma)
        self._fit()

    # ---- UI BUILD ----
    def _build_ui(self):
        pw=self.PW; x=10; w=pw-20; y=10; g=6
        self.groups=[]
        def grp(t):
            nonlocal y; y+=g; return y
        def adv(h):
            nonlocal y
            if self.groups: t,s,_=self.groups[-1]; self.groups[-1]=(t,s,y+h)
            y+=h+g
        self.widgets=[]; self.sliders=[]; self.buttons=[]; self.drops=[]
        self.checks=[]; self.spins=[]
        def add(w):
            self.widgets.append(w)
            for L,C in [(self.sliders,Slider),(self.buttons,Button),(self.drops,Dropdown),
                        (self.checks,Checkbox),(self.spins,SpinBox)]:
                if isinstance(w,C): L.append(w)
            return w

        # System Info
        gy=grp("System Info"); self.groups.append(("System Info",gy,gy+40))
        self._mode_y=gy+18; adv(40)

        # File
        gy=grp("File / State"); bw=(w-10)//3
        self.btn_save=add(Button((x,gy+18,bw,28),"Save",font=self.fsm))
        self.btn_load=add(Button((x+bw+5,gy+18,bw,28),"Load",font=self.fsm))
        self.btn_snap=add(Button((x+2*(bw+5),gy+18,bw,28),"Snap",font=self.fsm))
        adv(55)

        # Sim Controls
        gy=grp("Simulation Controls"); bw2=(w-5)//2
        self.btn_start=add(Button((x,gy+18,bw2,28),"Start",toggle=True,font=self.fsm))
        self.btn_reset=add(Button((x+bw2+5,gy+18,bw2,28),"Reset",font=self.fsm))
        self.btn_clear=add(Button((x,gy+52,bw2,28),"Clear",font=self.fsm))
        self.sl_speed=add(Slider((x,gy+86,w,30),1,100,30,"Speed:",self.fsm,"{:.0f}"))
        self.chk_auto=add(Checkbox((x,gy+120,w,20),"Auto-Disturbance",True,self.fsm))
        self.cb_res=add(Dropdown((x,gy+146,w,24),["256 (Fast)","512 (Balanced)","1024 (Detail)"],1,self.fsm))
        adv(180)

        # Paint
        gy=grp("Paint Tools")
        self.sl_brush=add(Slider((x,gy+18,w,30),2,50,10,"Brush Size:",self.fsm,"{:.0f}"))
        self.sl_str=add(Slider((x,gy+52,w,30),1,100,90,"Strength %:",self.fsm,"{:.0f}"))
        adv(90)

        # Recording
        gy=grp("Video Recording")
        self.sp_dur=add(SpinBox((x+80,gy+18,w-80,24),0,3600,10,step=1,dec=0,font=self.fsm))
        self._dur_y=gy+20; bw3=(w-5)//2
        self.btn_rec=add(Button((x,gy+46,bw3,26),"Rec",font=self.fsm))
        self.btn_recstop=add(Button((x+bw3+5,gy+46,bw3,26),"Stop",font=self.fsm))
        self.btn_recstop.enabled=False; self._rec_y=gy+76
        adv(100)

        # Physics
        gy=grp("Physics Parameters")
        self.sp_F=add(SpinBox((x+80,gy+18,w-80,22),.001,.1,.0545,step=.001,dec=4,font=self.fsm))
        self._F_y=gy+20
        self.sp_k=add(SpinBox((x+80,gy+44,w-80,22),.001,.1,.062,step=.001,dec=4,font=self.fsm))
        self._k_y=gy+46
        self.cb_bc=add(Dropdown((x+80,gy+70,w-80,22),BOUNDARIES,0,self.fsm))
        self._bc_y=gy+72
        self.chk_9pt=add(Checkbox((x,gy+96,w,20),"9-Point Laplacian",False,self.fsm))
        ly=gy+120; self._aniso_y=ly; dw=(w-10)//2
        self.sp_Dux=add(SpinBox((x,ly+18,dw,20),.01,1,.20,step=.01,dec=2,font=self.fsm))
        self.sp_Duy=add(SpinBox((x+dw+10,ly+18,dw,20),.01,1,.20,step=.01,dec=2,font=self.fsm))
        self.sp_Dvx=add(SpinBox((x,ly+42,dw,20),.01,1,.10,step=.01,dec=2,font=self.fsm))
        self.sp_Dvy=add(SpinBox((x+dw+10,ly+42,dw,20),.01,1,.10,step=.01,dec=2,font=self.fsm))
        self._du_y=ly+20; self._dv_y=ly+44
        adv(175)

        # Visual
        gy=grp("Visual Settings")
        self.cb_cmap=add(Dropdown((x+80,gy+18,w-80,22),list(COLORMAPS.keys()),0,self.fsm))
        self._cmap_y=gy+20
        self.sl_gamma=add(Slider((x,gy+46,w,30),10,300,60,"Gamma:",self.fsm,"{:.2f}"))
        self.chk_sharp=add(Checkbox((x,gy+80,w,20),"Sharpen",False,self.fsm))
        self.sl_bloom=add(Slider((x,gy+102,w,30),0,100,0,"Bloom %:",self.fsm,"{:.0f}"))
        self.chk_dual=add(Checkbox((x,gy+136,w,20),"Dual Channel (U+V)",False,self.fsm))
        self.cb_cmapu=add(Dropdown((x+80,gy+162,w-80,22),list(COLORMAPS.keys()),list(COLORMAPS.keys()).index("Cool"),self.fsm))
        self._cmapu_y=gy+164
        adv(195)

        # Advanced
        gy=grp("Advanced / Noise / Gradient")
        self.sl_noise=add(Slider((x,gy+18,w,30),0,50,0,"Noise (x1e4):",self.fsm,"{:.0f}"))
        self.sl_fg=add(Slider((x,gy+52,w,30),0,50,0,"F Grad (x.01):",self.fsm,"{:.0f}"))
        self.sl_kg=add(Slider((x,gy+86,w,30),0,50,0,"k Grad (x.01):",self.fsm,"{:.0f}"))
        adv(120)

        # Presets
        gy=grp("Presets & Phase Space")
        self.cb_preset=add(Dropdown((x,gy+18,w,22),list(PRESETS.keys()),0,self.fsm))
        self._seed_lbl_y=gy+42
        self.cb_seed=add(Dropdown((x,gy+56,w,22),SEEDS,0,self.fsm))
        self.btn_import=add(Button((x,gy+84,w,24),"Load Image Seed",font=self.fsm))
        self.phase_r=pygame.Rect(x,gy+116,w,160)
        self.pFr=(.01,.10); self.pKr=(.03,.075)
        if self.groups: t,s,_=self.groups[-1]; self.groups[-1]=(t,s,gy+282)
        adv(292)
        self._recalc_scroll()

    def _recalc_scroll(self):
        if self.groups: _,_,b=self.groups[-1]; self.max_scroll=max(0,b-self.H+30)
        else: self.max_scroll=0
        self._panel_bg_cache=None  # invalidate

    # ---- STATUS AUTO-CLEAR ----
    def _set_status(self, msg):
        self.status=msg; self._status_t=time.time()

    # ---- EVENTS ----
    def _events(self):
        dd_open=any(d.open for d in self.drops)
        for e in pygame.event.get():
            if not self.running: break
            if e.type==pygame.QUIT: self.running=False; return
            if e.type==pygame.WINDOWLEAVE:
                for w in self.widgets:
                    if hasattr(w,'hovered'): w.hovered=False
                for d in self.drops: d.close()
            if e.type==pygame.VIDEORESIZE:
                self.W=max(800,e.w); self.H=max(500,e.h)
                self.screen=pygame.display.set_mode((self.W,self.H),pygame.RESIZABLE)
                self._recalc_scroll()
            if e.type==pygame.KEYDOWN:
                if self.show_help:
                    if e.key in (pygame.K_h,pygame.K_ESCAPE): self._key(e)
                else: self._key(e)
            # Wheel on panel -> scroll; also close dropdowns
            if e.type==pygame.MOUSEWHEEL:
                mx,my=pygame.mouse.get_pos()
                if mx<self.PW:
                    self.scroll_y=max(0,min(self.max_scroll,self.scroll_y-e.y*20))
                    for d in self.drops: d.close()
                    dd_open=False
                    if not any(s.dragging for s in self.sliders): continue

            mx,my=pygame.mouse.get_pos()
            self.in_panel=mx<self.PW
            if self.in_panel:
                if e.type in (pygame.MOUSEBUTTONDOWN,pygame.MOUSEBUTTONUP,pygame.MOUSEMOTION,pygame.MOUSEWHEEL):
                    op=e.pos; e.pos=(mx,my+self.scroll_y)
                    for w in self.widgets: w.handle_event(e)
                    e.pos=op
                    if e.type==pygame.MOUSEBUTTONDOWN and e.button==1:
                        self._phase_click((mx,my+self.scroll_y))
                else:
                    for w in self.widgets: w.handle_event(e)
            else:
                if dd_open:
                    for d in self.drops: d.close(); dd_open=False
                self._viewer_evt(e)

        # Close dropdowns that weren't interacted with this frame
        if dd_open:
            mx,my=pygame.mouse.get_pos()
            if mx<self.PW:
                sp=(mx,my+self.scroll_y)
                for d in self.drops:
                    if d.open:
                        hit=d.rect.collidepoint(sp)
                        for i in range(len(d.options)):
                            if pygame.Rect(d.rect.x,d.rect.y+(i+1)*d.rect.height,d.rect.width,d.rect.height).collidepoint(sp):
                                hit=True
                        if not hit: d.close()

    def _key(self, e):
        m=pygame.key.get_mods()
        k=e.key
        if k==pygame.K_ESCAPE:
            if self.show_help: self.show_help=False; return
            self.running=False
        elif k==pygame.K_q: self.running=False
        elif k==pygame.K_SPACE:
            self.paused=not self.paused; self.btn_start.checked=not self.paused
            self.btn_start.text="Pause" if not self.paused else "Start"
        elif k==pygame.K_r and not(m&pygame.KMOD_CTRL): self._do_reset()
        elif k==pygame.K_c and not(m&pygame.KMOD_CTRL):
            if not self._confirm_clear or time.time()-self._confirm_time>2:
                self._confirm_clear=True; self._confirm_time=time.time()
                self._set_status("Press C again to confirm clear")
            else:
                self.engine.clear(); self._set_status("Cleared"); self._confirm_clear=False
        elif k==pygame.K_z and (m&pygame.KMOD_CTRL):
            if hasattr(self.engine,'_undo_stack') and self.engine._undo_stack:
                u,v=self.engine._undo_stack.pop(); L=self.engine.lib()
                with self.engine.lock: L.copyto(self.engine.Us,u); L.copyto(self.engine.Vs,v)
                self._set_status("Undo")
        elif k==pygame.K_s and not(m&pygame.KMOD_CTRL): self._save()
        elif k==pygame.K_l: self._load()
        elif k==pygame.K_e: self._snap()
        elif k==pygame.K_f: self._fit()
        elif k==pygame.K_0: self.zoom=1.; self.pan=[0.,0.]
        elif k==pygame.K_h: self.show_help=not self.show_help
        elif k in (pygame.K_EQUALS,pygame.K_PLUS,pygame.K_KP_PLUS):
            self.zoom=min(20.,self.zoom*1.2)
        elif k in (pygame.K_MINUS,pygame.K_KP_MINUS):
            self.zoom=max(.1,self.zoom/1.2)
        elif k==pygame.K_TAB:
            open_dd=[d for d in self.drops if d.open]
            if open_dd:
                cur=open_dd[0]; idx=self.drops.index(cur)
                cur.close(); self.drops[(idx+1)%len(self.drops)].open=True
        elif pygame.K_1<=k<=pygame.K_7:
            idx=k-pygame.K_1
            if idx<len(SEEDS):
                self.cb_seed.index=idx; self._do_reset(); self._set_status(f"Seed: {SEEDS[idx]}")

    def _do_reset(self):
        self.paused=True; self.btn_start.checked=False; self.btn_start.text="Start"
        self.engine.reset(seed=self.cb_seed.current_text); self._render(); self._set_status("Reset")

    def _fit(self):
        with self._rgb_lock: rgb_f = self.rgb
        if rgb_f is None: return
        h,w=rgb_f.shape[:2]
        vw=self.W-self.PW-20; vh=self.H-28
        self.zoom=min(vw/w,vh/h); self.pan=[0.,0.]

    def _viewer_evt(self, e):
        vr=pygame.Rect(self.PW,0,self.W-self.PW,self.H-28)
        # Toolbar Fit / 1:1 clicks (check first, before general handling)
        if e.type==pygame.MOUSEBUTTONDOWN and e.button==1 and hasattr(self,'_tb_fit'):
            if self._tb_fit.collidepoint(e.pos): self._fit(); return
            if self._tb_1to1.collidepoint(e.pos): self.zoom=1.; self.pan=[0.,0.]; return
        if e.type==pygame.MOUSEWHEEL and vr.collidepoint(pygame.mouse.get_pos()):
            mx,my=pygame.mouse.get_pos(); oz=self.zoom
            self.zoom*=1.15 if e.y>0 else 1/1.15
            self.zoom=max(.1,min(self.zoom,20.))
            r=self.zoom/oz; cx=self.PW+(self.W-self.PW)/2
            self.pan[0]=mx-r*(mx-cx-self.pan[0])-cx
            vy_center=(self.H-28)/2
            self.pan[1]=my-r*(my-vy_center-self.pan[1])-vy_center
        elif e.type==pygame.MOUSEBUTTONDOWN:
            now=time.time(); pos=e.pos
            # Double-click detection
            if e.button==1 and now-self._last_click<0.35 and abs(pos[0]-self._last_click_pos[0])<10 and abs(pos[1]-self._last_click_pos[1])<10:
                self._fit(); self._last_click=0; return
            if e.button==1: self._last_click=now; self._last_click_pos=pos
            if e.button in (2,3) and vr.collidepoint(pos):
                self.panning=True; self.pan0=pos
            elif e.button==1 and vr.collidepoint(pos):
                self.drawing=True; self._paint(pos)
        elif e.type==pygame.MOUSEBUTTONUP:
            if e.button in (2,3): self.panning=False
            elif e.button==1: self.drawing=False
        elif e.type==pygame.MOUSEMOTION:
            if self.panning:
                self.pan[0]+=e.pos[0]-self.pan0[0]; self.pan[1]+=e.pos[1]-self.pan0[1]; self.pan0=e.pos
            elif self.drawing: self._paint(e.pos)

    def _paint(self, sp):
        with self._rgb_lock: rgb_p = self.rgb
        if rgb_p is None: return
        ih,iw=rgb_p.shape[:2]; vw=self.W-self.PW
        sw,sh=iw*self.zoom,ih*self.zoom
        dx=(vw-sw)/2+self.pan[0]; dy=(self.H-28-sh)/2+self.pan[1]
        lx=(sp[0]-self.PW-dx)/self.zoom; ly=(sp[1]-dy)/self.zoom
        if 0<=lx<iw and 0<=ly<ih:
            self.engine.paint(int(lx),int(ly),self.brad,self.bstr)

    # ---- SYNC & ACTIONS ----
    def _sync(self):
        E=self.engine
        E.F=self.sp_F.value; E.k=self.sp_k.value
        E.Dux=self.sp_Dux.value; E.Duy=self.sp_Duy.value
        E.Dvx=self.sp_Dvx.value; E.Dvy=self.sp_Dvy.value
        E.boundary=self.cb_bc.index; E.u9=self.chk_9pt.checked
        E.noise=self.sl_noise.value/10000
        fg=self.sl_fg.value/100 if self.sl_fg.value>0 else None
        kg=self.sl_kg.value/100 if self.sl_kg.value>0 else None
        if E.F_grad!=fg or E.k_grad!=kg: E.F_grad=fg; E.k_grad=kg; E._grads()
        self.spf=int(self.sl_speed.value)
        self.cmap=COLORMAPS.get(self.cb_cmap.current_text,cv2.COLORMAP_INFERNO)
        self.cmap_u=COLORMAPS.get(self.cb_cmapu.current_text,cv2.COLORMAP_COOL)
        self.autodist=self.chk_auto.checked; self.sharp=self.chk_sharp.checked
        self.gamma=self.sl_gamma.value/100; self.bloom=self.sl_bloom.value/100
        self.dual=self.chk_dual.checked
        self.brad=int(self.sl_brush.value); self.bstr=self.sl_str.value/100

    def _actions(self):
        if self.btn_start.clicked:
            self.paused=not self.btn_start.checked
            self.btn_start.text="Pause" if self.btn_start.checked else "Start"
        if self.btn_reset.clicked: self._do_reset()
        if self.btn_clear.clicked:
                if not hasattr(self,'_confirm_clear') or not self._confirm_clear or time.time()-self._confirm_time>2:
                    self._confirm_clear=True; self._confirm_time=time.time()
                    self._set_status("Press Clear again to confirm")
                else:
                    self.engine.clear(); self._set_status("Cleared"); self._confirm_clear=False
        if self.btn_save.clicked: self._save()
        if self.btn_load.clicked: self._load()
        if self.btn_snap.clicked: self._snap()
        if self.btn_import.clicked: self._import_seed()
        if self.cb_preset.changed:
            n=self.cb_preset.current_text
            if n in PRESETS:
                f,k=PRESETS[n]; self.sp_F.value=f; self.sp_k.value=k; self._sync()
                self.engine.reset(seed=self.cb_seed.current_text)
                self.paused=True; self.btn_start.checked=False; self.btn_start.text="Start"
                self._render(); self._set_status(f"Preset: {n} applied - press Start")
        if self.cb_res.changed:
            rm={0:256,1:512,2:1024}; self.res=rm.get(self.cb_res.index,512)
            old_E=self.engine
            self.engine=Engine(self.res,self.res,self.use_gpu)
            self.engine.F=old_E.F;self.engine.k=old_E.k
            self.engine.Dux=old_E.Dux;self.engine.Duy=old_E.Duy
            self.engine.Dvx=old_E.Dvx;self.engine.Dvy=old_E.Dvy
            self.engine.boundary=old_E.boundary;self.engine.u9=old_E.u9
            self.engine.noise=old_E.noise
            self.engine.F_grad=old_E.F_grad;self.engine.k_grad=old_E.k_grad
            self.engine._grads(); self.engine._undo_stack=[]; self.engine._undo_max=10
            self._render(); self._set_status(f"Res: {self.res} (physics preserved)"); self._fit()
        if self.btn_rec.clicked: self._start_rec()
        if self.btn_recstop.clicked: self._stop_rec()
        if getattr(self,'_stop_rec_pending',False): self._stop_rec_pending=False; self._stop_rec()

    def _render(self):
        try:
            img=self.engine.get_image(self.cmap,self.gamma,self.sharp,self.bloom,self.dual,self.cmap_u)
            with self._rgb_lock: self.rgb=img
        except Exception as ex: log.error("Render: %s",ex)

    # ---- WORKER ----
    def _worker(self):
        lt=time.time(); frames=0; dt_=0
        while self.running:
            if not self.paused:
                t0=time.time(); self.engine.step(self.spf)
                if self.autodist:
                    dt_+=1
                    if dt_>150: self.engine.disturb(2); dt_=0
                if np.any(np.isnan(self.engine.Us)) or np.any(np.isinf(self.engine.Us)) or np.any(np.isnan(self.engine.Vs)) or np.any(np.isinf(self.engine.Vs)):
                    log.warning("NaN/Inf detected, resetting"); self.engine.clear(); self.paused=True; self.btn_start.checked=False; self.btn_start.text="Start"; continue
                self._render()
                if self.recording and self.vw and self.rgb:
                    vr=pygame.Rect(self.PW+1,0,self.W-self.PW,self.H-28)
                    sub=self.screen.subsurface(vr).copy()
                    frame=cv2.cvtColor(pygame.surfarray.array3d(sub).swapaxes(0,1),cv2.COLOR_RGB2BGR)
                    self.vw.write(frame)
                    if self.rec_dur>0 and time.time()-self.rec_start>=self.rec_dur: self._stop_rec_pending=True
                frames+=1; now=time.time(); el=now-lt
                if el>=1.: self.fps=self.fps*0.7+(frames/el)*0.3; frames=0; lt=now
                wt=time.time()-t0; sl=1./self.tfps-wt
                time.sleep(max(sl,.001))
            else: time.sleep(.05); dt_=0

    # ---- FILE I/O ----
    def _save(self):
        p=_fdsave("Save State","rd_state.npz",("NumPy","*.npz"))
        if p:
            if not p.endswith(".npz"): p+=".npz"
            u=cp.asnumpy(self.engine.Us) if self.engine.use_gpu else self.engine.Us
            v=cp.asnumpy(self.engine.Vs) if self.engine.use_gpu else self.engine.Vs
            E=self.engine
            np.savez_compressed(p,U=u,V=v,F=E.F,k=E.k,Dux=E.Dux,Duy=E.Duy,Dvx=E.Dvx,Dvy=E.Dvy,
                noise=E.noise,bc=E.boundary,u9=E.u9,
                Fg=E.F_grad if E.F_grad is not None else -1.,kg=E.k_grad if E.k_grad is not None else -1.,
                cmap_name=self.cb_cmap.current_text,gamma=self.gamma,sharp=self.sharp,
                bloom=self.bloom,dual=self.dual,cmap_u_name=self.cb_cmapu.current_text)
            self._set_status(f"Saved: {os.path.basename(p)}")

    def _load(self):
        p=_fdopen("Load State",("NumPy","*.npz"))
        if p:
            try:
                d=np.load(p, allow_pickle=False); E=self.engine
                if "U" not in d or "V" not in d:
                    self._set_status("Invalid state file"); return
                if d["U"].shape!=(E.height,E.width): self._set_status("Error: res mismatch"); return
                L=cp if self.engine.use_gpu else np
                E.Us=L.array(d["U"]);E.Vs=L.array(d["V"]);E.Ud=L.array(d["U"]);E.Vd=L.array(d["V"])
                E.F=float(d["F"]);E.k=float(d["k"])
                for k2,a in [("Dux","Dux"),("Duy","Duy"),("Dvx","Dvx"),("Dvy","Dvy")]:
                    if k2 in d: setattr(E,a,float(d[k2]))
                if "noise" in d: E.noise=float(d["noise"])
                if "bc" in d: E.boundary=int(d["bc"])
                if "u9" in d: E.u9=bool(d["u9"])
                if "Fg" in d:
                    fg=float(d["Fg"]);E.F_grad=fg if fg>=0 else None
                if "kg" in d:
                    kg=float(d["kg"]);E.k_grad=kg if kg>=0 else None
                E._grads()
                self.sp_F.value=E.F;self.sp_k.value=E.k
                self.sp_Dux.value=E.Dux;self.sp_Duy.value=E.Duy
                self.sp_Dvx.value=E.Dvx;self.sp_Dvy.value=E.Dvy
                if "cmap_name" in d:
                    cn=str(d["cmap_name"])
                    if cn in COLORMAPS: self.cb_cmap.index=list(COLORMAPS.keys()).index(cn)
                if "gamma" in d: self.sl_gamma.value=float(d["gamma"])*100
                if "sharp" in d: self.chk_sharp.checked=bool(d["sharp"])
                if "bloom" in d: self.sl_bloom.value=float(d["bloom"])*100
                if "dual" in d: self.chk_dual.checked=bool(d["dual"])
                if "cmap_u_name" in d:
                    cn2=str(d["cmap_u_name"])
                    if cn2 in COLORMAPS: self.cb_cmapu.index=list(COLORMAPS.keys()).index(cn2)
                self._render();self._set_status(f"Loaded: {os.path.basename(p)}")
            except Exception as ex:
                log.error("Load: %s",ex);self._set_status("Load error")

    def _snap(self):
        p=_fdsave("Export PNG","rd_snapshot.png",("PNG","*.png"))
        if p:
            with self._rgb_lock: rgb_snap = self.rgb
            if rgb_snap is None: return
            if not p.endswith(".png"): p+=".png"
            cv2.imwrite(p,cv2.cvtColor(rgb_snap,cv2.COLOR_RGB2BGR))
            self._set_status(f"Snap: {os.path.basename(p)}")

    def _import_seed(self):
        p=_fdopen("Import Seed",("Images","*.png *.jpg *.bmp"))
        if p:
            idx_img = SEEDS.index("Image Import") if "Image Import" in SEEDS else 6
            self.cb_seed.index = idx_img
            self.engine.reset(seed="Image Import",img_path=p)
            self._render();self._set_status(f"Seed: {os.path.basename(p)}")

    def _start_rec(self):
        if self.recording: return
        ts=datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        p=_fdsave("Save Video",f"rd_sim_{ts}.mp4",("MP4","*.mp4"))
        if not p: return
        if not p.endswith(".mp4"): p+=".mp4"
        if self.paused:
            self._set_status("Cannot record while paused"); return
        vr_tmp=pygame.Rect(self.PW+1,0,self.W-self.PW,self.H-28)
        self._rec_fps=self.tfps; self.vw=cv2.VideoWriter(p,cv2.VideoWriter_fourcc(*"mp4v"),self._rec_fps,(vr_tmp.width,vr_tmp.height))
        self.recording=self.vw.isOpened();self.rec_start=time.time();self.rec_dur=int(self.sp_dur.value)
        if not self.recording: self.vw=None; self._set_status("Video codec not available"); return
        self.btn_rec.enabled=False;self.btn_recstop.enabled=True
        if self.paused:
            self.paused=False
            self.btn_start.checked=True; self.btn_start.text="Pause"
        self._set_status("Recording...");log.info("Rec: %s",p)

    def _stop_rec(self):
        if self.recording and self.vw:
            self.vw.release();self.vw=None;self.recording=False
            self.btn_rec.enabled=True;self.btn_recstop.enabled=False
            self._set_status("Recording saved");log.info("Rec stopped")

    # ---- DRAWING ----
    def _draw(self):
        # Auto-clear status
        if self.status!="Ready" and time.time()-self._status_t>self.STATUS_CLEAR_TIME:
            self.status="Ready"
        self.screen.fill(BG)
        self._draw_panel()
        pygame.draw.line(self.screen,BORDER,(self.PW,0),(self.PW,self.H),1)
        self._draw_viewer()
        self._draw_dropdowns()
        if self.show_help: self._draw_help()
        self._draw_status()
        mx,my=pygame.mouse.get_pos()
        if self.show_help: pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_ARROW)
        elif mx<self.PW: pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_HAND)
        elif self.panning: pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_SIZEALL)
        elif self.drawing: pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_CROSSHAIR)
        else: pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_ARROW)
        pygame.display.flip()

    def _build_panel_bg(self, sy):
        """Build a surface with just the group boxes (cached)."""
        key=(self.PW, self.H, self.max_scroll)
        if self._panel_bg_cache and self._panel_bg_ck==key:
            return self._panel_bg_cache
        s=pygame.Surface((self.PW,self.H)); s.fill(PANEL_BG)
        for title,ys,ye in self.groups:
            r=pygame.Rect(4,ys-sy,self.PW-8,ye-ys)
            if r.bottom>0 and r.top<self.H:
                pygame.draw.rect(s,GROUP_BG,r,border_radius=5)
                pygame.draw.rect(s,GROUP_BORDER,r,1,border_radius=5)
                ts=self.fsm.render(title,True,TEXT)
                tx=r.x+10; ty=r.y-1
                pygame.draw.rect(s,GROUP_BG,(tx-4,ty-1,ts.get_width()+8,ts.get_height()+2))
                s.blit(ts,(tx,ty))
        self._panel_bg_cache=s; self._panel_bg_ck=key
        return s

    def _draw_panel(self):
        sy=self.scroll_y
        bg=self._build_panel_bg(sy)
        # Draw dynamic content directly on a new surface with bg as base
        panel=pygame.Surface((self.PW,self.H)); panel.blit(bg,(0,0))
        panel.set_clip(pygame.Rect(0,0,self.PW,self.H))

        # Labels
        mode="GPU (CuPy)" if self.engine.use_gpu else "CPU (Numba)"
        panel.blit(self.fsm.render(f"Engine: {mode}",True,GREEN),(10,self._mode_y-sy))
        panel.blit(self.fsm.render("Duration (s):",True,DIM),(10,self._dur_y-sy))
        rc=RED if self.recording and (self._blink//15)%2==0 else (RED if self.recording else DIM)
        if self.recording:
            elapsed=time.time()-self.rec_start
            rec_txt=f"REC {int(elapsed)}s"
            if self.rec_dur>0: rec_txt+=f"/{self.rec_dur}s"
            panel.blit(self.fsm.render(rec_txt,True,rc),(10,self._rec_y-sy))
        else:
            panel.blit(self.fsm.render("Idle",True,DIM),(10,self._rec_y-sy))

        for t,yo in [("Feed (F):",self._F_y),("Kill (k):",self._k_y),("Boundary:",self._bc_y)]:
            panel.blit(self.fsm.render(t,True,TEXT),(10,yo-sy))
        panel.blit(self.fsm.render("Anisotropy (X/Y):",True,TEXT),(10,self._aniso_y-sy))
        dw=(self.PW-30)//2
        for t,yo,dx in [("Du_x",self._du_y,0),("Du_y",self._du_y,dw+10),("Dv_x",self._dv_y,0),("Dv_y",self._dv_y,dw+10)]:
            panel.blit(self.fsm.render(t,True,DIM),(10+dx,yo-sy-12))
        for t,yo in [("Colormap:",self._cmap_y),("U Colormap:",self._cmapu_y)]:
            panel.blit(self.fsm.render(t,True,TEXT),(10,yo-sy))
        # Seed Type label (was missing)
        panel.blit(self.fsm.render("Seed Type:",True,TEXT),(10,self._seed_lbl_y-sy))

        # Widgets
        for w in self.widgets:
            oy=w.rect.y; w.rect.y-=sy
            if isinstance(w,Dropdown): w.draw_box(panel)
            else: w.draw(panel)
            w.rect.y=oy

        # Phase diagram
        pr=self.phase_r.copy(); pr.y-=sy
        self._draw_phase(panel,pr)

        # Scrollbar
        if self.max_scroll>0:
            sbx=self.PW-8
            pygame.draw.rect(panel,SB_BG,(sbx,0,6,self.H),border_radius=3)
            vr=self.H/(self.H+self.max_scroll)
            kh=max(20,int(self.H*vr)); ky=int(self.scroll_y/self.max_scroll*(self.H-kh))
            pygame.draw.rect(panel,SB_KNOB,(sbx,ky,6,kh),border_radius=3)

        panel.set_clip(None)
        self.screen.blit(panel,(0,0))

    def _draw_phase(self, surf, rect):
        if rect.bottom<0 or rect.top>self.H: return
        ck=(rect.width,rect.height)
        if self._phase_cache is None or self._phase_ck!=ck:
            self._phase_cache=pygame.Surface(ck)
            for row in range(ck[1]):
                t=row/max(1,ck[1]-1)
                self._phase_cache.fill((int(25+t*15),int(35+t*5),int(45-t*20)),
                                       (0,row,ck[0],1))
            self._phase_ck=ck
        surf.blit(self._phase_cache,rect.topleft)
        fr=self.pFr[1]-self.pFr[0]; kr=self.pKr[1]-self.pKr[0]
        cur_name=self.cb_preset.current_text
        for idx,(name,(f,k)) in enumerate(PRESETS.items()):
            px=rect.x+max(0,min(1,(f-self.pFr[0])/fr))*rect.width
            py=rect.y+max(0,min(1,(k-self.pKr[0])/kr))*rect.height
            # Highlight current preset
            if name==cur_name:
                pygame.draw.circle(surf,ORANGE,(int(px),int(py)),6,2)
            pygame.draw.circle(surf,WHITE,(int(px),int(py)),3)
            label_y=max(int(py)-5+(idx%2)*10, rect.y+2)
            label_x=min(int(px)+5, rect.right-self.fsm.size(name)[0]-2)
            surf.blit(self.fsm.render(name,True,ORANGE if name==cur_name else TEXT),(label_x,label_y))
        # Crosshair + F/k readout
        cx=rect.x+max(0,min(1,(self.engine.F-self.pFr[0])/fr))*rect.width
        cy=rect.y+max(0,min(1,(self.engine.k-self.pKr[0])/kr))*rect.height
        pygame.draw.line(surf,RED,(int(cx)-6,int(cy)),(int(cx)+6,int(cy)),2)
        pygame.draw.line(surf,RED,(int(cx),int(cy)-6),(int(cx),int(cy)+6),2)
        pygame.draw.circle(surf,RED,(int(cx),int(cy)),4,1)
        # F/k value label near crosshair
        fk_txt=self.fsm.render(f"F={self.engine.F:.4f} k={self.engine.k:.4f}",True,YELLOW)
        fx=min(int(cx)+8,rect.right-fk_txt.get_width()-2)
        fy=max(int(cy)-16,rect.y+2)
        surf.blit(fk_txt,(fx,fy))
        # Axis tick labels
        for val in [self.pFr[0], (self.pFr[0]+self.pFr[1])/2, self.pFr[1]]:
            tx=rect.x+max(0,min(1,(val-self.pFr[0])/fr))*rect.width
            tl=self.fsm.render(f"{val:.3f}",True,DIM)
            surf.blit(tl,(int(tx)-tl.get_width()//2, rect.bottom+1))
        for val in [self.pKr[0], (self.pKr[0]+self.pKr[1])/2, self.pKr[1]]:
            ty=rect.y+max(0,min(1,(val-self.pKr[0])/kr))*rect.height
            tl=self.fsm.render(f"{val:.3f}",True,DIM)
            surf.blit(tl,(rect.x-tl.get_width()-2, int(ty)-tl.get_height()//2))
        pygame.draw.rect(surf,GROUP_BORDER,rect,1,border_radius=3)

    def _phase_click(self, sp):
        if self.phase_r.collidepoint(sp):
            self.sp_F.value=self.pFr[0]+(sp[0]-self.phase_r.x)/self.phase_r.width*(self.pFr[1]-self.pFr[0])
            self.sp_k.value=self.pKr[0]+(sp[1]-self.phase_r.y)/self.phase_r.height*(self.pKr[1]-self.pKr[0])
            return True
        return False

    def _draw_dropdowns(self):
        self.screen.set_clip(pygame.Rect(0,0,self.W,self.H-28))
        sy=self.scroll_y
        for dd in self.drops:
            if not dd.open: continue
            for i,opt in enumerate(dd.options):
                sr=pygame.Rect(dd.rect.x,dd.rect.y+(i+1)*dd.rect.height-sy,dd.rect.width,dd.rect.height)
                if sr.bottom<0 or sr.top>self.H: continue
                is_sel = (i == dd.index)
                c=(70,70,70) if is_sel else ((60,60,60) if i==dd.hover else (45,45,45))
                pygame.draw.rect(self.screen,(20,20,20),sr.move(2,2),border_radius=2)
                pygame.draw.rect(self.screen,c,sr,border_radius=2)
                pygame.draw.rect(self.screen,(85,85,85),sr,1,border_radius=2)
                prefix="> " if is_sel else "  "
                t=self.fsm.render(prefix+opt,True,WHITE if i==dd.hover else (ACCENT if is_sel else TEXT))
                self.screen.blit(t,(sr.x+6,sr.y+(sr.height-t.get_height())//2))
        self.screen.set_clip(None)

    def _draw_viewer(self):
        vr=pygame.Rect(self.PW+1,0,self.W-self.PW,self.H-28)
        pygame.draw.rect(self.screen,VIEWER_BG,vr)

        with self._rgb_lock:
            rgb_local = self.rgb.copy() if self.rgb is not None else None
        if rgb_local is not None:
            h,w=rgb_local.shape[:2]; sw,sh=int(w*self.zoom),int(h*self.zoom)
            surf=pygame.surfarray.make_surface(np.ascontiguousarray(rgb_local.swapaxes(0,1)))
            if self.zoom!=1.: surf=pygame.transform.smoothscale(surf,(sw,sh))
            dx=int((vr.width-sw)/2+self.pan[0]); dy=int((vr.height-sh)/2+self.pan[1])
            self.screen.blit(surf,(vr.x+dx,vr.y+dy))

        # Brush cursor (only when hovering viewer and not panning)
        mx,my=pygame.mouse.get_pos()
        if vr.collidepoint(mx,my) and not self.show_help:
            sr=int(self.brad*self.zoom)
            if sr>2:
                cs=pygame.Surface((sr*2+2,sr*2+2),pygame.SRCALPHA)
                pygame.draw.circle(cs,(255,255,255,180),(sr+1,sr+1),sr,1)
                self.screen.blit(cs,(mx-sr-1,my-sr-1))
                pygame.draw.circle(self.screen,WHITE,(mx,my),2)

        # Paused overlay
        if self.paused and self.engine.iteration>0:
            pi=self.flg.render("PAUSED",True,WHITE)
            self.screen.blit(pi,(vr.x+8,vr.bottom-pi.get_height()-8))
        # Viewer toolbar: Fit | 1:1 | zoom%
        tb_y=vr.y+4; bw=32; bh=20
        bx=vr.right-8
        zt=self.fsm.render(f"{self.zoom:.2f}x",True,DIM)
        self.screen.blit(zt,(bx-zt.get_width(),tb_y))
        bx2=bx-zt.get_width()-8
        for label in ["1:1","Fit"]:
            bx2-=bw+4
            r=pygame.Rect(bx2,tb_y-2,bw,bh)
            hov=r.collidepoint(mx,my)
            pygame.draw.rect(self.screen,(60,60,60) if hov else (45,45,45),r,border_radius=3)
            pygame.draw.rect(self.screen,(85,85,85),r,1,border_radius=3)
            t=self.fsm.render(label,True,WHITE if hov else TEXT)
            self.screen.blit(t,(r.x+(r.width-t.get_width())//2,r.y+(r.height-t.get_height())//2))
        # Store toolbar button rects for click detection in _viewer_evt
        self._tb_fit=pygame.Rect(bx2,tb_y-2,bw,bh)
        self._tb_1to1=pygame.Rect(bx2+bw+4,tb_y-2,bw,bh)

    def _draw_help(self):
        # Semi-transparent overlay
        overlay=pygame.Surface((self.W,self.H),pygame.SRCALPHA)
        overlay.fill(HELP_BG)
        self.screen.blit(overlay,(0,0))
        # Help box
        bw,bh=min(520,self.W-40),min(340,self.H-40)
        bx=(self.W-bw)//2; by=(self.H-bh)//2
        pygame.draw.rect(self.screen,(35,35,35),(bx,by,bw,bh),border_radius=8)
        pygame.draw.rect(self.screen,ACCENT,(bx,by,bw,bh),2,border_radius=8)
        # Title
        t=self.ft.render("Keyboard Shortcuts",True,WHITE)
        self.screen.blit(t,(bx+(bw-t.get_width())//2,by+12))
        # Lines
        lines=[
            ("SPACE","Start / Pause"),("R","Reset simulation"),("C","Clear to uniform"),
            ("F","Fit image to viewer"),("0","Reset zoom to 1:1"),("+/-","Zoom in / out"),
            ("S","Save state (.npz)"),("L","Load state (.npz)"),("E","Export snapshot (.png)"),
            ("H","Toggle this help"),("ESC / Q","Quit"),
            ("",""),
            ("LMB drag","Paint on image"),("MMB / RMB drag","Pan image"),
            ("Scroll wheel","Zoom (toward cursor)"),("Double-click","Fit to viewer"),
            ("Shift+drag SpinBox","Scrub value"),("Ctrl+Z","Undo paint"),
            ("Tab","Cycle open dropdowns"),
            ("",""),
            ("1-7","Quick select seed type"),
        ]
        ly=by+42
        for key,desc in lines:
            if not key and not desc: ly+=8; continue
            if not key:
                self.screen.blit(self.fsm.render(desc,True,DIM),(bx+20,ly))
            else:
                kt=self.fsm.render(key,True,ACCENT)
                self.screen.blit(kt,(bx+20,ly))
                dt=self.fsm.render(f"  {desc}",True,TEXT)
                self.screen.blit(dt,(bx+20+kt.get_width(),ly))
            ly+=18
        # Close hint
        ct=self.fsm.render("Press H or ESC to close",True,DIM)
        self.screen.blit(ct,(bx+(bw-ct.get_width())//2,by+bh-24))

    def _draw_status(self):
        bh=28; br=pygame.Rect(0,self.H-bh,self.W,bh)
        pygame.draw.rect(self.screen,STATUS_BG,br)
        pygame.draw.line(self.screen,BORDER,(0,br.y),(self.W,br.y),1)
        self._blink+=1
        mode="GPU" if self.engine.use_gpu else "CPU"
        rec=" [REC]" if self.recording else ""
        hint="SPC:Play  R:Reset  F:Fit  H:Help  ESC:Quit"
        hs=self.fsm.render(hint,True,DIM)
        max_tw=self.W-hs.get_width()-20
        txt=f"{mode} | {self.res}x{self.res} | {self.fps:.0f} FPS | Iter {self.engine.iteration}{rec} | {self.status}"
        while self.fsm.size(txt)[0]>max_tw and len(txt)>10: txt=txt[:-4]+".."
        self.screen.blit(self.fsm.render(txt,True,TEXT),(10,br.y+7))
        self.screen.blit(hs,(self.W-hs.get_width()-10,br.y+7))

    # ---- MAIN LOOP ----
    def run(self):
        log.info("RD Simulator v3 | NumPy %s | OpenCV %s | Pygame %s",np.__version__,cv2.__version__,pygame.version.ver)
        self.paused=False; self.btn_start.checked=True; self.btn_start.text="Pause"
        while self.running:
            self._events()
            self._sync()
            self._actions()
            self._draw()
            self.clock.tick(60)
        if int(time.time())%5==0:
            pygame.display.set_caption(f"RD Simulator | {self.fps:.0f} FPS | Iter {self.engine.iteration}")
        if self.recording and self.vw: self.vw.release()
        self.running=False; pygame.quit(); log.info("Exited")

if __name__ == "__main__":
    app = RDApp()
    app.run()