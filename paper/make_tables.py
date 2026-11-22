import json
D=json.load(open('data.json'))
M3=['ST-GCN','ST-GCN++','CTR-GCN']
def cell(v,s,bold=False):
    if v is None: return '--'
    t=f'{v:.1f}'
    if bold: t=r'\textbf{'+t+'}'
    if s=='up': t+=r'$^{\uparrow}$'
    if s=='down': t+=r'$^{\downarrow}$'
    return t
P=['CS','CS-dis','CL','CL-dis','CG']
L=[]
L.append(r'''\begin{table}[H]
\tablesize{\footnotesize}
\caption{Video-level macro-F1 (\%) on CARR, pooled over folds and averaged over five seeds. CS: cross-subject; CL: cross-location; CG: training without any mobility-aid user, testing on all 73 videos of the seven aid users; ``-dis'' rows restrict CS/CL to the aid-user videos.}
\label{tab:main}
\newcolumntype{C}{>{\centering\arraybackslash}X}
\begin{tabularx}{\textwidth}{llCCCCC}
\toprule
\textbf{Backbone} & \textbf{Method} & \textbf{CS} & \textbf{CS-dis} & \textbf{CL} & \textbf{CL-dis} & \textbf{CG}\\
\midrule''')
names={'A0':'Baseline (A0)','A4':'Standard aug. (A4)','OCC':'+ OCC [23]','ASM':'+ ASM [24]','MIX':'+ MIX [25]','AidFuse':'AidFuse (ours)'}
for m in M3:
    rows=D['main'][m]
    best={p:max(rows[k][p][0] for k in rows if rows[k][p][0] is not None) for p in P}
    for i,k in enumerate(['A0','A4','OCC','ASM','MIX','AidFuse']):
        lab=names[k].replace('[23]',r'\cite{mathe2025occluded}').replace('[24]',r'\cite{chen2023occluded}').replace('[25]',r'\cite{xiang2025jmda}')
        cs=[cell(rows[k][p][0],rows[k][p][1],rows[k][p][0]==best[p]) for p in P]
        L.append((m if i==0 else '')+' & '+lab+' & '+' & '.join(cs)+r'\\')
    if m!=M3[-1]: L.append(r'\midrule')
L.append(r'''\bottomrule
\end{tabularx}
\footnotesize{Bold: best value per backbone and column. $^{\uparrow}$/$^{\downarrow}$: significantly higher/lower than A4 (paired bootstrap over videos, 95\% CI excluding zero). OCC, ASM and MIX were re-implemented on top of A4 and evaluated on CS and CG only (--: not run).}
\end{table}''')
open('tab_main.tex','w').write('\n'.join(L))
# SAFER
L=[r'''\begin{table}[H]
\tablesize{\footnotesize}
\caption{Macro-F1 (\%) on SAFER-Activities (train without wheelchair data, three seeds), for the wheelchair test subjects and for held-out non-wheelchair subjects.}
\label{tab:safer}
\newcolumntype{C}{>{\centering\arraybackslash}X}
\begin{tabularx}{\textwidth}{lCCCCCC}
\toprule
 & \multicolumn{2}{c}{\textbf{ST-GCN}} & \multicolumn{2}{c}{\textbf{ST-GCN++}} & \multicolumn{2}{c}{\textbf{CTR-GCN}}\\
\textbf{Method} & \textbf{Wheelchair} & \textbf{Other} & \textbf{Wheelchair} & \textbf{Other} & \textbf{Wheelchair} & \textbf{Other}\\
\midrule''']
for k in ['A0','A4','OCC','ASM','MIX','AidFuse']:
    cs=[]
    for m in M3:
        r=D['safer'][m]
        for gname in ['wheelchair','non-wheelchair']:
            best=max(r[x][gname][0] for x in r)
            cs.append(cell(r[k][gname][0],r[k][gname][1],r[k][gname][0]==best))
    lab=names[k].replace('[23]',r'\cite{mathe2025occluded}').replace('[24]',r'\cite{chen2023occluded}').replace('[25]',r'\cite{xiang2025jmda}')
    L.append(lab+' & '+' & '.join(cs)+r'\\')
L.append(r'''\bottomrule
\end{tabularx}
\footnotesize{Bold and arrows as in Table~\ref{tab:main}; significance is tested against A4 over test segments.}
\end{table}''')
open('tab_safer.tex','w').write('\n'.join(L))
# consistency
C=D['cons']
lab={'Body':'Body stream only','Box+box':'Box + box ensemble','Body+body':'Body + body ensemble','OCC':'OCC','ASM':'ASM','MIX':'MIX','AidFuse':'AidFuse (box + body)'}
L=[r'''\begin{table}[H]
\tablesize{\footnotesize}
\caption{Consistency of each configuration relative to A4 across all tested groups (CARR CS, CS-dis, CS-non-dis, CL, CL-dis, CG; CARR per aid type in CG, CS and CL; SAFER wheelchair and non-wheelchair), summed over the three backbones.}
\label{tab:consistency}
\newcolumntype{C}{>{\centering\arraybackslash}X}
\begin{tabularx}{\textwidth}{lCCCCC}
\toprule
\textbf{Configuration} & \textbf{Models} & \textbf{Tests} & \textbf{Sig. gains} & \textbf{Sig. losses} & \textbf{Worst sig. loss}\\
\midrule''']
for k in ['OCC','ASM','MIX','Body','Box+box','Body+body','AidFuse']:
    v=C[k]; n_models=2 if k in ('Box+box','Body+body','AidFuse') else 1
    w='--' if v['worst'] is None else f"$-${abs(v['worst']):.1f}"
    row=f"{lab[k]} & {n_models} & {v['n']} & {v['up']} & {v['down']} & {w}"
    if k=='Box+box': row=f"{lab[k]} & {n_models} & {v['n']} & {v['up']} & "+r'\textbf{0} & --'
    if k=='AidFuse': row=r'\textbf{'+lab[k]+'} & '+f"{n_models} & {v['n']} & "+r'\textbf{'+str(v['up'])+'} & '+r'\textbf{0} & --'
    L.append(row+r'\\')
L.append(r'''\bottomrule
\end{tabularx}
\footnotesize{Bold: most gains, and no significant loss. Sig.: paired bootstrap 95\% CI excluding zero. OCC, ASM and MIX were not run under CL, hence fewer tests. Models: number of trained networks used at inference.}
\end{table}''')
open('tab_consistency.tex','w').write('\n'.join(L))
# ensemble control
E=D['ens']
L=[r'''\begin{table}[H]
\tablesize{\footnotesize}
\caption{Cost-matched ensemble control: macro-F1 (\%) of three two-network ensembles that differ only in the reference frames of their members (seeds paired; SAFER: wheelchair subjects).}
\label{tab:ensemble}
\newcolumntype{C}{>{\centering\arraybackslash}X}
\begin{tabularx}{\textwidth}{llCCCCC}
\toprule
\textbf{Backbone} & \textbf{Ensemble} & \textbf{CARR CS} & \textbf{CARR CL} & \textbf{CARR CL-dis} & \textbf{CARR CG} & \textbf{SAFER}\\
\midrule''']
for m in M3:
    r=E[m]
    cols=['CS','CL','CL-dis','CG','SAFER']
    best={c:max(r[k][c] for k in r) for c in cols}; worst={c:min(r[k][c] for k in r) for c in cols}
    for i,k in enumerate(['Box+box','Body+body','AidFuse']):
        cs=[]
        for c in cols:
            t=f"{r[k][c]:.1f}"
            if r[k][c]==best[c]: t=r'\textbf{'+t+'}'
            if r[k][c]==worst[c]: t=r'\underline{'+t+'}'
            cs.append(t)
        L.append((m if i==0 else '')+' & '+{'Box+box':'Box + box','Body+body':'Body + body','AidFuse':'AidFuse'}[k]+' & '+' & '.join(cs)+r'\\')
    if m!=M3[-1]: L.append(r'\midrule')
L.append(r'''\bottomrule
\end{tabularx}
\footnotesize{Bold: best and underlined: worst of the three ensembles per backbone and column.}
\end{table}''')
open('tab_ensemble.tex','w').write('\n'.join(L))
print(open('tab_consistency.tex').read())
