"""Figures of the AidFuse paper. Run from this folder: RESULTS=<path to results> python figs.py"""
import os, sys, json, math, random, numpy as np
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..','aidfuse'))
import augs as A
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
def person(T=64):
    """Synthetic customer facing the camera: walks in (frames 0-31), reaches to a shelf with the right hand (32-63).
    Box-normalized: y down, box height 1. COCO left joints on image right."""
    x = np.zeros((T, 17, 3), np.float32)
    base = {0:(0,-.42),1:(.02,-.44),2:(-.02,-.44),3:(.045,-.43),4:(-.045,-.43),5:(.09,-.30),6:(-.09,-.30),
            7:(.12,-.15),8:(-.12,-.15),9:(.12,-.02),10:(-.12,-.02),11:(.06,0.0),12:(-.06,0.0),
            13:(.06,.22),14:(-.06,.22),15:(.06,.44),16:(-.06,.44)}
    for t in range(T):
        ph = 2*math.pi*t/16
        for j,(a,b) in base.items():
            x[t,j,:2] = (a + 0.004*t if t < 32 else a + 0.128, b)
        if t < 32:                                     # walking: legs and arms swing
            for k,s in ((13,1),(15,1),(14,-1),(16,-1)):
                x[t,k,1] -= 0.03*max(0, s*math.sin(ph)) * (1 if k in (13,14) else 1.6)
                x[t,k,0] += 0.01*s*math.sin(ph)
            for k,s in ((9,1),(10,-1)): x[t,k,1] -= 0.03*s*math.sin(ph)
        else:                                          # right hand reaches to the shelf
            r = min(1,(t-32)/12)
            x[t,8,:2] += (-0.05*r, -0.10*r); x[t,10,:2] += (-0.12*r, -0.30*r)
        x[t,:,2] = 0.9
    return x

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.edgecolor':'#9a9993','axes.linewidth':0.6,
                     'xtick.color':'#52514e','ytick.color':'#52514e','axes.labelcolor':'#0b0b0b','text.color':'#0b0b0b'})
INK='#0b0b0b'; INK2='#52514e'; GRID='#e6e5e0'
BLUE='#2a78d6'; ORANGE='#eb6834'; AQUA='#1baf7a'; YEL='#eda100'; GRAY='#9a9993'
EDGES=[(0,1),(0,2),(1,3),(2,4),(0,5),(0,6),(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]
def sk(ax,p,c,lw=1.6):
    for a,b in EDGES:
        if p[a,2]>0 and p[b,2]>0: ax.plot([p[a,0],p[b,0]],[p[a,1],p[b,1]],color=c,lw=lw,solid_capstyle='round')
    v=p[:,2]>0; ax.scatter(p[v,0],p[v,1],s=9,color=c,zorder=3,edgecolor='white',linewidth=0.5)
# ---------------- Fig 1: concept
random.seed(11); np.random.seed(11)
st=person(); st[:]=st[40]
se=A.u1_seated(person()); se[:]=se[40]
os.makedirs('fig',exist_ok=True)
fig=plt.figure(figsize=(7.2,2.9))
ax1=fig.add_axes([0.01,0.08,0.22,0.8]); ax2=fig.add_axes([0.25,0.08,0.22,0.8]); ax3=fig.add_axes([0.49,0.02,0.51,0.90])
for ax,(a,b),ttl in [(ax1,(st,se),'(a) Box reference'),(ax2,(A.u5_upper_norm(st.copy()),A.u5_upper_norm(se.copy())),'(b) Body reference')]:
    if ax is ax1: ax.add_patch(Rectangle((-0.5,-0.5),1,1,fill=False,ls='--',lw=0.8,color=INK2))
    else:
        ax.axhline(0,color=GRID,lw=0.6,zorder=0); ax.axvline(0,color=GRID,lw=0.6,zorder=0)
    sk(ax,a[0],GRAY); sk(ax,b[0],BLUE)
    ax.set_xlim(-0.62,0.62); ax.set_ylim(0.62,-0.62) if ax is ax1 else ax.set_ylim(1.3,-0.55)
    if ax is ax2: ax.set_xlim(-0.85,0.85)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(ttl,fontsize=8.5,loc='left')
    for s in ax.spines.values(): s.set_visible(False)
ax1.text(0,0.58,'box = person + aid',ha='center',fontsize=7,color=INK2)

ax1.plot([],[],color=GRAY,label='standing customer'); ax1.plot([],[],color=BLUE,label='seated wheelchair user')
ax1.legend(loc='upper center',bbox_to_anchor=(1.05,-0.0),ncol=2,frameon=False,fontsize=7)
ax3.set_xlim(0,10.2); ax3.set_ylim(0.2,5.6); ax3.axis('off'); ax3.text(0,5.45,'(c) AidFuse',fontsize=8.5,va='top')
def box(x,y,w,h,t,fc='#f3f2ee',ec='#9a9993',fs=7):
    ax3.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.02,rounding_size=0.15',fc=fc,ec=ec,lw=0.8))
    ax3.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=fs,color=INK)
def arr(x0,y0,x1,y1): ax3.annotate('',xy=(x1,y1),xytext=(x0,y0),arrowprops=dict(arrowstyle='-|>',lw=0.8,color=INK2,mutation_scale=8))
box(0.0,2.4,1.6,1.2,'2D pose\nsequence\n(COCO-17)')
box(2.1,3.7,2.4,1.1,'box-referenced\nnormalization')
box(2.1,1.2,2.4,1.1,'body-referenced\nnormalization',fc='#e8f0fb',ec=BLUE)
box(5.0,3.7,1.7,1.1,'skeleton\nnetwork $f_b$')
box(5.0,1.2,1.7,1.1,'skeleton\nnetwork $f_u$',fc='#e8f0fb',ec=BLUE)
box(7.2,2.35,1.5,1.3,'mean of\nclass\nprobabilities')
box(9.0,2.6,1.15,0.8,'activity')
arr(1.6,3.2,2.1,4.25); arr(1.6,2.8,2.1,1.75); arr(4.5,4.25,5.0,4.25); arr(4.5,1.75,5.0,1.75)
arr(6.7,4.25,7.2,3.4); arr(6.7,1.75,7.2,2.6); arr(8.7,3.0,9.0,3.0)
ax3.text(5.1,0.55,'same backbone, trained separately; no extra parameters',ha='center',fontsize=6.5,color=INK2)
fig.savefig('fig/fig1_overview.pdf'); fig.savefig('fig/fig1_overview.png',dpi=200)
# ---------------- Fig 2: failure analysis
q={'no aid':0.024,'crutches':0.043,'cane':0.351,'wheelchair':0.495}; lc={'no aid':0.416,'crutches':0.133,'cane':0.447,'wheelchair':0.686}
f1={'ST-GCN':(54.3,41.5),'ST-GCN++':(54.9,39.5),'CTR-GCN':(54.1,36.1)}
fig,(a,b)=plt.subplots(1,2,figsize=(7.2,2.5),gridspec_kw=dict(width_ratios=[1.2,1]))
g=list(q); xs=np.arange(len(g)); w=0.36
a.bar(xs-w/2-0.01,[q[k]*100 for k in g],w,color=BLUE,label='frames without pose',zorder=2)
a.bar(xs+w/2+0.01,[lc[k]*100 for k in g],w,color=ORANGE,label='low-confidence legs',zorder=2)
for i,k in enumerate(g):
    a.text(i-w/2,q[k]*100+1.5,f'{q[k]*100:.1f}',ha='center',fontsize=6.5,color=INK2)
    a.text(i+w/2,lc[k]*100+1.5,f'{lc[k]*100:.1f}',ha='center',fontsize=6.5,color=INK2)
a.set_xticks(xs); a.set_xticklabels(g); a.set_ylabel('% of frames'); a.set_ylim(0,80)
a.grid(axis='y',color=GRID,lw=0.6,zorder=0); a.set_axisbelow(True); a.legend(frameon=False,fontsize=7,loc='upper left')
a.set_title('(a) Pose quality on CARR by group',fontsize=8.5,loc='left')
m=list(f1); xs=np.arange(len(m))
b.bar(xs-w/2-0.01,[f1[k][0] for k in m],w,color=GRAY,label='CS (mixed users)',zorder=2)
b.bar(xs+w/2+0.01,[f1[k][1] for k in m],w,color=BLUE,label='CG (unseen aid users)',zorder=2)
for i,k in enumerate(m):
    b.text(i-w/2,f1[k][0]+1,f'{f1[k][0]:.1f}',ha='center',fontsize=6.5,color=INK2); b.text(i+w/2,f1[k][1]+1,f'{f1[k][1]:.1f}',ha='center',fontsize=6.5,color=INK2)
b.set_xticks(xs); b.set_xticklabels(m); b.set_ylabel('video macro-F1 (%)'); b.set_ylim(0,80)
b.grid(axis='y',color=GRID,lw=0.6,zorder=0); b.set_axisbelow(True); b.legend(frameon=False,fontsize=7,loc='upper center',ncol=2)
b.set_title('(b) Baseline gap (no extra augmentation)',fontsize=8.5,loc='left')
for ax in (a,b):
    for s in ('top','right'): ax.spines[s].set_visible(False)
plt.tight_layout(); fig.savefig('fig/fig2_failure.pdf'); fig.savefig('fig/fig2_failure.png',dpi=200)
# ---------------- Fig 3: per aid type, CG
import pandas as pd
s5=pd.read_csv(''+os.environ.get('RESULTS','results')+'/S5_per_aid.csv'); s5=s5[s5.protocol=='CG']
V=[('A4',None,GRAY),('OCC','S4:A4+OCC',ORANGE),('ASM','S4:A4+ASM',AQUA),('MIX','S4:A4+MIX',YEL),('AidFuse','FUSE:S3@A4|S3b@A4+U5',BLUE)]
M=['ST-GCN','ST-GCN++','CTR-GCN']; T=['wheelchair','crutches','cane']
fig,axs=plt.subplots(1,3,figsize=(7.2,2.6),sharey=True)
for ax,mdl in zip(axs,M):
    xs=np.arange(3)*1.35; w=0.19
    for k,(lab,var,c) in enumerate(V):
        vals=[]; sig=[]
        for t in T:
            r=s5[(s5.variant=='FUSE:S3@A4|S3b@A4+U5')&(s5.model==mdl)&(s5.aid==t)].iloc[0]
            if var is None: vals.append(r.base_f1); sig.append(0)
            else:
                rr=s5[(s5.variant==var)&(s5.model==mdl)&(s5.aid==t)].iloc[0]; vals.append(rr.variant_f1)
                sig.append((1 if rr['diff']>0 else -1) if rr.significant else 0)
        pos=xs+(k-2)*(w+0.02)
        ax.bar(pos,vals,w,color=c,label=lab,zorder=2)
        for p_,v_,s_ in zip(pos,vals,sig):
            if s_: ax.text(p_,v_+1.2,'▲' if s_>0 else '▼',ha='center',fontsize=6.5,color=INK)
    ax.set_xticks(xs); ax.set_xticklabels(['wheelchair\n(17)','crutches\n(33)','cane\n(23)'],fontsize=6.5); ax.set_title(mdl,fontsize=8.5,loc='left')
    ax.grid(axis='y',color=GRID,lw=0.6,zorder=0); ax.set_axisbelow(True); ax.set_ylim(0,85)
    for s in ('top','right'): ax.spines[s].set_visible(False)
axs[0].set_ylabel('CG video macro-F1 (%)')
axs[1].legend(loc='upper center',bbox_to_anchor=(0.5,-0.2),ncol=5,frameon=False,fontsize=7)
plt.tight_layout(rect=(0,0.06,1,1)); fig.savefig('fig/fig3_per_aid.pdf'); fig.savefig('fig/fig3_per_aid.png',dpi=200)
print('ok')
