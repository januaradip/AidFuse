"""Collect the numbers of the paper tables from the result CSVs: RESULTS=<path> python build_data.py"""
import pandas as pd, json, numpy as np
import os
R=os.environ.get('RESULTS','results')+'/'
M3=['ST-GCN','ST-GCN++','CTR-GCN']
FUSE='FUSE:S3@A4|S3b@A4+U5'; BODY='S3b:A4+U5'; BB='FUSE:S3@A4|S3@A4#1'; TT='FUSE:S3b@A4+U5|S3b@A4+U5#1'
fA4=pd.read_csv(R+'fusi_vs_A4.csv'); fA0=pd.read_csv(R+'fusi_vs_A0.csv'); s4=pd.read_csv(R+'S4_vs_A4.csv')
def g(df,var,m,p,col='variant_f1'):
    x=df[(df.variant==var)&(df.model==m)&(df.protocol==p)]
    return None if x.empty else x.iloc[0][col]
def sig(df,var,m,p):
    x=df[(df.variant==var)&(df.model==m)&(df.protocol==p)]
    if x.empty: return ''
    x=x.iloc[0]; return ('up' if x['diff']>0 else 'down') if x.significant else ''
P=['CS','CS-dis','CL','CL-dis','CG']
main={}
for m in M3:
    rows={}
    rows['A0']={p:(g(fA0,FUSE,m,p,'base'),'') for p in P}
    rows['A4']={p:(g(fA4,FUSE,m,p,'base'),'') for p in P}
    for c in ['OCC','ASM','MIX']:
        rows[c]={p:(g(s4,'S4:A4+'+c,m,p),sig(s4,'S4:A4+'+c,m,p)) for p in P}
    rows['AidFuse']={p:(g(fA4,FUSE,m,p),sig(fA4,FUSE,m,p)) for p in P}
    main[m]=rows
# SAFER
s9=pd.read_csv(R+'S9_summary.csv'); s9c=pd.read_csv(R+'S9_compare_vs_A4.csv'); s9f=pd.read_csv(R+'S9_fusion_compare.csv'); s9e=pd.read_csv(R+'S9_ensemble_vs_base.csv')
saf={}
for m in M3:
    rows={}
    for a,lab in [('A0','A0'),('A4','A4'),('A4+OCC','OCC'),('A4+ASM','ASM'),('A4+MIX','MIX')]:
        x=s9[(s9.protocol=='SZ')&(s9.model==m)&(s9.aug==a)].iloc[0]
        d={}
        for grp,col in [('wheelchair','wheelchair'),('non-wheelchair','non-wheelchair')]:
            mu=float(str(x[col]).split('±')[0])
            sg=''
            if a not in ('A0','A4'):
                y=s9c[(s9c.model==m)&(s9c.aug==a)&(s9c.group==grp)].iloc[0]; sg=('up' if y['diff']>0 else 'down') if y.significant else ''
            d[grp]=(mu,sg)
        rows[lab]=d
    d={}
    for grp in ['wheelchair','non-wheelchair']:
        y=s9f[(s9f.model==m)&(s9f.variant=='FUSE:A4|A4+U5')&(s9f.base=='A4')&(s9f.group==grp)].iloc[0]
        d[grp]=(y.variant_f1,('up' if y['diff']>0 else 'down') if y.significant else '')
    rows['AidFuse']=d
    saf[m]=rows
# consistency counts vs A4
s5=pd.read_csv(R+'S5_per_aid.csv')
def tests(cfg,m):
    out=[]
    carr={'AidFuse':(fA4,FUSE),'Body':(fA4,BODY),'Box+box':(fA4,BB),'Body+body':(fA4,TT),'OCC':(s4,'S4:A4+OCC'),'ASM':(s4,'S4:A4+ASM'),'MIX':(s4,'S4:A4+MIX')}[cfg]
    df,var=carr
    for p in ['CS','CS-dis','CS-nondis','CL','CL-dis','CG']:
        x=df[(df.variant==var)&(df.model==m)&(df.protocol==p)]
        if len(x): out.append(('CARR '+p,x.iloc[0]['diff'],bool(x.iloc[0].significant)))
    v5={'AidFuse':FUSE,'Body':BODY,'Box+box':BB,'Body+body':TT,'OCC':'S4:A4+OCC','ASM':'S4:A4+ASM','MIX':'S4:A4+MIX'}[cfg]
    for _,x in s5[(s5.variant==v5)&(s5.model==m)].iterrows():
        out.append((f'CARR {x.protocol} {x.aid}',x['diff'],bool(x.significant)))
    for grp in ['wheelchair','non-wheelchair']:
        if cfg=='AidFuse': y=s9f[(s9f.model==m)&(s9f.variant=='FUSE:A4|A4+U5')&(s9f.base=='A4')&(s9f.group==grp)]
        elif cfg=='Box+box': y=s9e[(s9e.model==m)&(s9e.variant=='FUSE:A4|A4#1')&(s9e.base=='A4')&(s9e.group==grp)]
        elif cfg=='Body+body': y=s9e[(s9e.model==m)&(s9e.variant=='FUSE:A4+U5|A4+U5#1')&(s9e.base=='A4')&(s9e.group==grp)]
        else: y=s9c[(s9c.model==m)&(s9c.aug=={'Body':'A4+U5','OCC':'A4+OCC','ASM':'A4+ASM','MIX':'A4+MIX'}[cfg])&(s9c.group==grp)]
        y=y.iloc[0]; out.append(('SAFER '+grp,y['diff'],bool(y.significant)))
    return out
cons={}
for cfg in ['Body','Box+box','Body+body','OCC','ASM','MIX','AidFuse']:
    allt=[t for m in M3 for t in tests(cfg,m)]
    up=sum(1 for t in allt if t[2] and t[1]>0); dn=sum(1 for t in allt if t[2] and t[1]<0)
    worst=min(t[1] for t in allt if t[2]) if dn else None
    saf_up=sum(1 for m in M3 for t in tests(cfg,m) if t[0]=='SAFER wheelchair' and t[2] and t[1]>0)
    carr_aid_up=sum(1 for m in M3 for t in tests(cfg,m) if t[0] in ('CARR CG','CARR CS-dis','CARR CL-dis') and t[2] and t[1]>0)
    cons[cfg]=dict(n=len(allt),up=up,down=dn,worst=worst,safer_wc_up=saf_up,carr_aid_up=carr_aid_up,
                   down_list=[(m,t[0],t[1]) for m in M3 for t in tests(cfg,m) if t[2] and t[1]<0])
# ensemble control table
ens={}
for m in M3:
    r={}
    for lab,var,sv in [('Box+box',BB,'FUSE:A4|A4#1'),('Body+body',TT,'FUSE:A4+U5|A4+U5#1'),('AidFuse',FUSE,None)]:
        d={p:g(fA4,var,m,p) for p in ['CS','CL','CG','CL-dis']}
        if sv: d['SAFER']=s9e[(s9e.model==m)&(s9e.variant==sv)&(s9e.base=='A4')&(s9e.group=='wheelchair')].iloc[0].variant_f1
        else: d['SAFER']=s9f[(s9f.model==m)&(s9f.variant=='FUSE:A4|A4+U5')&(s9f.base=='A4')&(s9f.group=='wheelchair')].iloc[0].variant_f1
        r[lab]=d
    ens[m]=r
json.dump(dict(main=main,safer=saf,cons=cons,ens=ens),open('data.json','w'),indent=1,default=float)
for k,v in cons.items(): print(k,{a:b for a,b in v.items() if a!='down_list'}, v['down_list'])
print(json.dumps(ens,indent=0)[:1500])
