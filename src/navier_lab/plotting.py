"""Графики второй лабораторной."""

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import networkx as nx
import numpy as np

from .geometry import estimate_gaussian, spd_geodesic

COLORS = ("#2962a3", "#c24b35")


def plot_model_graph():
    """Укрупнённые группы операций исходного графа Ивана."""
    g = nx.DiGraph()
    positions = {"u, v": (0, 1.5), "p": (0, -0.1),
                 "Конвекция C": (1.3, 2.1), "Лапласиан L": (1.3, 0.9),
                 "Градиент p": (1.3, -0.5), "β": (2.6, 2.6), "ν": (3.8, 2.1),
                 "−βC + νL − ∇p": (3.8, 0.8), "Ускорение": (5.3, 0.8)}
    g.add_edges_from([("u, v", "Конвекция C"), ("u, v", "Лапласиан L"),
                      ("p", "Градиент p"), ("Конвекция C", "−βC + νL − ∇p"),
                      ("Лапласиан L", "−βC + νL − ∇p"),
                      ("Градиент p", "−βC + νL − ∇p"), ("β", "−βC + νL − ∇p"),
                      ("ν", "−βC + νL − ∇p"), ("−βC + νL − ∇p", "Ускорение")])
    fig, ax = plt.subplots(figsize=(12, 4), layout="constrained")
    boxes = {}
    for label, (x, y) in positions.items():
        boxes[label] = ax.text(x, y, label, ha="center", va="center", fontsize=11,
                bbox=dict(boxstyle="round,pad=0.5", fc="#fcece8" if label in ("p", "Градиент p") else "#edf2f7", ec="#b6bec6"))
    for a, b in g.edges:
        ax.annotate('', xy=positions[b], xytext=positions[a],
                    arrowprops=dict(arrowstyle='-|>', mutation_scale=14,
                                    patchA=boxes[a].get_bbox_patch(), patchB=boxes[b].get_bbox_patch(),
                                    shrinkA=3, shrinkB=3, lw=1.2,
                                    color=COLORS[1] if a in ('p','Градиент p') else '#8c969e'))
    ax.set(xlim=(-0.7,6), ylim=(-1.1,3.2), title="Интервенция p=0 выключает ветви градиента давления")
    ax.axis("off")
    return fig


def plot_fields(fields):
    frame = 40
    fig, axes = plt.subplots(1,3,figsize=(12,3.6),layout="constrained")
    extent = (fields['x'][0],fields['x'][-1],fields['y'][0],fields['y'][-1])
    for ax,key,title in zip(axes, ('u','v','p'), ('Скорость u','Скорость v','Давление p')):
        im=ax.imshow(fields[key][frame],origin='lower',extent=extent,aspect='auto',cmap='RdBu_r')
        ax.set(title=title,xlabel='x',ylabel='y')
        fig.colorbar(im,ax=ax,shrink=.8)
    fig.suptitle(f"Внешний расчёт течения за цилиндром, t={fields['time'][frame]:.1f}")
    return fig


def _ellipse(ax, mean, covariance, color, label=None, linestyle='-'):
    values,vectors=np.linalg.eigh(covariance)
    angle=np.degrees(np.arctan2(vectors[1,1],vectors[0,1]))
    width,height=2*np.sqrt(5.991*values[::-1])
    ax.add_patch(Ellipse(mean,width,height,angle=angle,fill=False,ec=color,lw=1.5,ls=linestyle,label=label))


def plot_distributions(results):
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for ax,(name,result) in zip(axes,results.items()):
        for index,color,label in ((0,COLORS[0],'Исходные'),(-1,COLORS[1],'p=0')):
            points=result['samples'][index]
            fit=estimate_gaussian(points)
            ax.scatter(points[:,0],points[:,1],s=26,color=color,label=label,zorder=3)
            _ellipse(ax,fit.mean,fit.covariance,color)
        ax.set(title=name,xlabel='β, коэффициент конвекции',ylabel='ν, вязкость')
        ax.autoscale_view()
        ax.ticklabel_format(axis='y',style='sci',scilimits=(-3,3),useOffset=False)
        ax.legend()
        ax.grid(alpha=.2)
    fig.suptitle('12 оценок по временным блокам и 95% эллипсы гауссовой аппроксимации')
    return fig


def plot_sensitivity(results):
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for col,(name,result) in enumerate(results.items()):
        rows=result['rows'];a=[r['alpha'] for r in rows]
        axes[0,col].plot(a,[r['kl'] for r in rows],'o-',color=COLORS[0],label='KL(P₀ || Pα)')
        axes[0,col].plot(a,[r['reverse_kl'] for r in rows],'s--',color=COLORS[1],label='KL(Pα || P₀)')
        axes[0,col].set_yscale('symlog',linthresh=.1)
        axes[0,col].set(title=name,ylabel='KL, нат')
        axes[0,col].legend(fontsize=9)
        axes[1,col].plot(a,[100*r['test_error'] for r in rows],'o-',label='Коэффициенты переоценены',color=COLORS[0])
        axes[1,col].plot(a,[100*r['error_with_frozen_theta'] for r in rows],'s--',label='Исходные коэффициенты',color=COLORS[1])
        axes[1,col].set(xlabel='α: доля убранного давления',ylabel='Ошибка ускорения, %')
        if col == 0:
            axes[1,col].set_yscale('log')
        axes[1,col].legend(fontsize=9)
        for ax in axes[:,col]:ax.grid(alpha=.2);ax.set_xticks(a)
    return fig


def plot_covariance_geometry(result):
    base=estimate_gaussian(result['samples'][0]);end=estimate_gaussian(result['samples'][-1])
    # Один общий переход к единицам стандартного отклонения исходных оценок.
    scale=np.diag(1/np.sqrt(np.diag(base.covariance)))
    a,b=scale@base.covariance@scale,scale@end.covariance@scale
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for t,color in zip((0,.25,.5,.75,1),plt.cm.viridis(np.linspace(.1,.9,5))):
        cov=spd_geodesic(a,b,t)
        _ellipse(axes[0],np.zeros(2),cov,color,label=f's={t:g}')
    axes[0].autoscale_view();axes[0].set_aspect('equal',adjustable='datalim')
    axes[0].set(title='Геодезическая между ковариациями',xlabel='Изменение β / σβ₀',ylabel='Изменение ν / σν₀')
    axes[0].legend(fontsize=9)
    rows=result['rows']
    axes[1].plot([r['alpha'] for r in rows],[r['spd_distance'] for r in rows],'o-',color=COLORS[0])
    axes[1].set(title='Ковариации при интервенции',xlabel='α',ylabel='Расстояние на SPD')
    axes[1].grid(alpha=.2)
    return fig
