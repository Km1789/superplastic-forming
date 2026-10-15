"""Построение графиков сравнения режимов давления (рис. 5, 7-9 текста работы).

Входные данные (не входят в репозиторий, см. README):
  data/forming/Free_bulg_test/Reports/Smirnov_parsed/Frictionless/<режим>/increment_<N>/{top,bottom}.txt
      - профили верхней и нижней поверхности заготовки при заполнении H = 0,8;
  data/forming/sr_opt_tp.csv - история давления режима поддержания оптимальной скорости
      деформации (t, p), извлечённая из отчёта Abaqus (см. extract_pressure_history ниже);
  data/forming/model/alg_try.inp - входной файл модели (берётся профиль матрицы).
Запуск из корня репозитория:  python pressure_algorithm/plots/nirs_figures.py
"""
import sys, glob, os, re, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'pressure_algorithm', 'postprocessing'))
from compute_distances import compute_distances_from_folder

plt.rcParams.update({'font.family':['Times New Roman','Liberation Serif','DejaVu Serif'],'font.size':11,'mathtext.fontset':'stix',
                     'axes.grid':True,'grid.alpha':0.35,'grid.linestyle':'--'})
FIG=os.path.join(ROOT,'figures'); os.makedirs(FIG, exist_ok=True)
from matplotlib.ticker import FuncFormatter, FixedLocator, NullLocator
def _fmt(v,pos):
    s=('%g'%round(v,6)); return s.replace('.',',')
COMMA=FuncFormatter(_fmt)
def commas(ax):
    ax.xaxis.set_major_formatter(COMMA); ax.yaxis.set_major_formatter(COMMA)
_orig_savefig=matplotlib.figure.Figure.savefig
def _sf(self,*a,**k):
    for ax in self.axes:
        if ax.get_yscale()=='linear': ax.yaxis.set_major_formatter(COMMA)
        if ax.get_xscale()=='linear': ax.xaxis.set_major_formatter(COMMA)
    k.setdefault('bbox_inches','tight'); return _orig_savefig(self,*a,**k)
matplotlib.figure.Figure.savefig=_sf
base=os.path.join(ROOT,'data/forming/Free_bulg_test/Reports/Smirnov_parsed/Frictionless')
H0=4.0
const = {0.5:'pressure_0_5/increment_11470',0.75:'pressure_0_75/increment_5250',1.0:'pressure_1/increment_3204',
 1.25:'pressure_1_25/increment_2254',1.5:'pressure_1_5/increment_1678',1.75:'pressure_1_75/increment_1328',
 2.0:'pressure_2/increment_1052',2.25:'pressure_2_25/increment_872',2.5:'pressure_2_5/increment_706',
 2.75:'pressure_2_75/increment_610',3.0:'pressure_3/increment_508',3.25:'pressure_3_25/increment_454',
 3.5:'pressure_3_5/increment_386',3.75:'pressure_3_75/increment_358',4.0:'pressure_4/increment_306'}
special={'alg':'algorithm_try_0_8/increment_2034','smooth':'smooth_alg/increment_2136','sr':'sr_opt/increment_3830'}
def load(rel):
    d=os.path.join(base,rel); t=float(open(d+'/top.txt').readline().split(':')[1])
    x,h=compute_distances_from_folder(d,1); return t,x,h
res={}
for p,rel in const.items():
    t,x,h=load(rel); res[p]=(t,h.min(),h.mean(),h.max()-h.min())
sp={k:load(v) for k,v in special.items()}
print('p  time  hmin  thin%  mean  range')
for p,(t,hm,mean,rng) in res.items(): print(p, round(t,1), round(hm,3), round(100*(H0-hm)/H0,2), round(mean,3), round(rng,3))
for k,(t,x,h) in sp.items(): print(k, round(t,1), round(h.min(),3), round(100*(H0-h.min())/H0,2), round(h.mean(),3), round(h.max()-h.min(),3))

# Smirnov
s0,ss,kv,mv=8.61,142.80,2.702e5,1.081
def m_of(e):
    X=kv*e**mv; return mv*X*(ss-s0)/((s0+X)*(ss+X))
eopt=(np.sqrt(s0*ss)/kv)**(1/mv)
B=(mv/0.5)*(ss-s0)-s0-ss; D=B*B-4*s0*ss
emin=((B-np.sqrt(D))/(2*kv))**(1/mv); emax=((B+np.sqrt(D))/(2*kv))**(1/mv)
print('eopt',eopt,'m(eopt)',m_of(eopt),'range',emin,emax)

# pole strain-rate along algorithm path
bp=np.genfromtxt(os.path.join(ROOT,'pressure_algorithm/example_run/run_19')+'/fill_pressure_optimizer_best_path.csv',delimiter=',',names=True,dtype=None,encoding=None)
T=np.r_[0,bp['frame_value']]; Hh=np.r_[H0,bp['min_thickness']]
rates=np.diff(np.log(H0/Hh))/np.diff(T)
print('stage pressures',bp['selected_pressure'])
print('pole eq strain rates',rates, 'm', [round(m_of(r),3) for r in rates])
print('mean rate stages2-8', rates[1:].mean(), m_of(rates[1:].mean()))
# eps_opt run pole rate between 1202.7 and 1907.7
print('sr_opt pole rate', np.log(3.0888/2.5940)/(1907.666-1202.666))

# ---------- Figure: pressure regimes
sch=np.genfromtxt(os.path.join(ROOT,'pressure_algorithm/example_run/run_19')+'/fill_pressure_optimizer_pressure_schedule.csv',delimiter=',',names=True)
Tend=sch['time'][-1]
tk=np.array([0,Tend/3,2*Tend/3,Tend]); pk=np.interp(tk,sch['time'],sch['pressure'])
print('smooth knots',tk,pk)
srp=np.loadtxt(os.path.join(ROOT,'data/forming/sr_opt_tp.csv'),delimiter=',',skiprows=1)
fig,ax=plt.subplots(figsize=(7.2,3.9))
m=srp[:,0]<=1907.7
ax.plot(srp[m,0],srp[m,1],color='#1f5fa8',lw=2,label=r'Поддержание $\dot\varepsilon_{opt}$ (ПИД-регулятор)')
ax.plot(sch['time'],sch['pressure'],color='#c0392b',lw=2,ls='--',label='Итерационный алгоритм (ступенчатый режим)')
tt=np.linspace(0,Tend,400); ax.plot(tt,PchipInterpolator(tk,pk)(tt),color='#27ae60',lw=2,label='Сглаженный режим алгоритма')
ax.plot(tk,pk,'o',color='#27ae60',ms=6)
ax.axvline(Tend,color='#c0392b',lw=0.8,alpha=0.6); ax.axvline(1907.7,color='#1f5fa8',lw=0.8,alpha=0.6)
ax.text(Tend+20,0.15,'H = 0,8\n1006 с',color='#c0392b',fontsize=9); ax.text(1907.7-150,0.15,'H = 0,8\n1908 с',color='#1f5fa8',fontsize=9)
ax.set_xlabel('Время, с'); ax.set_ylabel('Давление, МПа'); ax.set_xlim(0,2000); ax.set_ylim(0,1.65)
ax.legend(loc='center right',fontsize=9.5); fig.tight_layout(); fig.savefig(FIG+'/fig_pressure.png',dpi=250); plt.close()

# ---------- Figure: min thickness & time vs constant pressure
P=np.array(list(res)); tm=np.array([res[p][0] for p in P]); hm=np.array([res[p][1] for p in P])
fig,axs=plt.subplots(1,2,figsize=(7.4,3.3))
ax=axs[0]; ax.plot(P,hm,'o-',color='#555555',ms=4.5,label='Постоянное давление')
styles={'sr':('#1f5fa8','-',r'$\dot\varepsilon_{opt}$'),'alg':('#c0392b','--','Алгоритм'),'smooth':('#27ae60',':','Сглаженный алгоритм')}
for k,(c,ls,l) in styles.items(): ax.axhline(sp[k][2].min(),color=c,ls=ls,lw=1.8,label=l)
ax.set_xlabel('Давление, МПа'); ax.set_ylabel(r'$h_{min}$, мм'); ax.set_title('а) минимальная толщина',fontsize=11)
ax.legend(fontsize=8.5,loc='lower right')
ax=axs[1]; ax.semilogy(P,tm,'o-',color='#555555',ms=4.5)
for k,(c,ls,l) in styles.items(): ax.axhline(sp[k][0],color=c,ls=ls,lw=1.8)
ax.yaxis.set_major_locator(FixedLocator([100,200,500,1000,2000,5000])); ax.yaxis.set_minor_locator(NullLocator()); ax.yaxis.set_major_formatter(FuncFormatter(lambda v,p:'%d'%v)); ax.xaxis.set_major_formatter(COMMA)
ax.set_xlabel('Давление, МПа'); ax.set_ylabel('Время до H = 0,8, с'); ax.set_title('б) продолжительность формовки',fontsize=11)
fig.tight_layout(); fig.savefig(FIG+'/fig_minthick.png',dpi=250); plt.close()

# ---------- Figure: thickness distribution
fig,ax=plt.subplots(figsize=(7.2,3.8))
sel=[(0.5,'#8e44ad'),(1.25,'#7f8c8d'),(4.0,'#e67e22')]
for p,c in sel:
    t,x,h=load(const[p]); ax.plot(x,h,color=c,lw=1.3,label=f'{str(p).replace(".",",")} МПа ({t:.0f} с)')
for k,(c,ls,l) in styles.items():
    t,x,h=sp[k]; ax.plot(x,h,color=c,ls=ls,lw=1.8,label=f'{l} ({t:.0f} с)')
ax.set_xlabel('Радиальная координата r, мм'); ax.set_ylabel('Толщина, мм'); ax.legend(fontsize=8.5,ncol=2,loc='upper left')
ax.set_xlim(0,200)
ins=ax.inset_axes([0.62,0.08,0.35,0.42])
for p,c in sel:
    t,x,h=load(const[p]); ins.plot(x,h,color=c,lw=1.2)
for k,(c,ls,l) in styles.items():
    t,x,h=sp[k]; ins.plot(x,h,color=c,ls=ls,lw=1.6)
ins.set_xlim(0,40); ins.set_ylim(2.44,2.66); ins.tick_params(labelsize=8); ins.text(0.03,0.5,'зона полюса',transform=ins.transAxes,fontsize=8.5)
fig.tight_layout(); fig.savefig(FIG+'/fig_thickness.png',dpi=250); plt.close()

# ---------- Figure: geometry (die + blank initial + at H=0.8)
L=open(os.path.join(ROOT,'data/forming/model/alg_try.inp')).read().split('\n')
i=[k for k,l in enumerate(L) if l.startswith('*Surface, type=SEGMENTS, name=Instrument')][0]
pts=[]; cur=None
for l in L[i+1:]:
    if l.startswith('*'): break
    a=[s.strip() for s in l.split(',')]
    if a[0]=='START': cur=np.array([float(a[1]),float(a[2])]); pts.append(cur)
    elif a[0]=='LINE': cur=np.array([float(a[1]),float(a[2])]); pts.append(cur)
    elif a[0]=='CIRCL':
        e=np.array([float(a[1]),float(a[2])]); c=np.array([float(a[3]),float(a[4])])
        a0=np.arctan2(*(cur-c)[::-1]); a1=np.arctan2(*(e-c)[::-1]); r=np.linalg.norm(cur-c)
        da=(a1-a0+np.pi)%(2*np.pi)-np.pi
        for s in np.linspace(0,1,60)[1:]: pts.append(c+r*np.array([np.cos(a0+s*da),np.sin(a0+s*da)]))
        cur=e
pts=np.array(pts)
fig,ax=plt.subplots(figsize=(7.2,3.6))
ax.fill_between(pts[:,0],pts[:,1],145,color='#d5d8dc',step=None,label='Матрица')
ax.plot(pts[:,0],pts[:,1],color='k',lw=1.4)
ax.fill_between([0,200],[0,0],[4,4],color='#f5b041',alpha=0.9,label='Исходная заготовка, 4 мм')
d=os.path.join(base,special['alg'])
xt,yt=np.loadtxt(d+'/top.txt',skiprows=1).T; xb,yb=np.loadtxt(d+'/bottom.txt',skiprows=1).T
ax.fill(np.r_[xt,xb[::-1]],np.r_[yt,yb[::-1]],color='#c0392b',alpha=0.85,label='Заготовка при H = 0,8')
ax.annotate('давление p(t)',xy=(60,10),xytext=(60,25),ha='center',arrowprops=dict(arrowstyle='<-',lw=1.2),fontsize=10)
ax.annotate('',xy=(100,10),xytext=(100,25),arrowprops=dict(arrowstyle='<-',lw=1.2))
ax.annotate('',xy=(20,10),xytext=(20,25),arrowprops=dict(arrowstyle='<-',lw=1.2))
ax.text(-3.5,60,'ось симметрии',rotation=90,fontsize=9,va='center',ha='right')
ax.axvline(0,color='k',ls='-.',lw=0.8)
ax.set_xlim(-12,205); ax.set_ylim(-5,145); ax.set_aspect('equal')
ax.set_xlabel('r, мм'); ax.set_ylabel('z, мм'); ax.legend(fontsize=9,loc='upper right'); ax.grid(False)
fig.tight_layout(); fig.savefig(FIG+'/fig_geometry.png',dpi=250); plt.close()
print('die top', pts[:,1].max())


def extract_pressure_history(rpt_path, out_csv):
    """Извлекает историю давления (t, p) из отчёта Abaqus с полем P по кадрам."""
    t = None; out = []; seen = True
    for line in open(rpt_path, errors='ignore'):
        m = re.search(r'Step Time =\s*([-\d.Ee+]+)', line)
        if m:
            t = float(m.group(1)); seen = False; continue
        if not seen:
            p = line.split()
            if len(p) == 4 and p[0].isdigit():
                out.append((t, float(p[3]))); seen = True
    np.savetxt(out_csv, np.array(out), delimiter=',', header='t,p', comments='')
