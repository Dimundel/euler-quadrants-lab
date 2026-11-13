"""Figures for the saved graph search, with no optimization in plotting code."""
from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from .vendor.ivan_interventions import PARAMS, PARAM_LABELS

COLORS = ["#277DA8", "#E49B36", "#8F97A3", "#277DA8", "#E49B36", "#8F97A3", "#277DA8", "#E49B36", "#8F97A3"]
TITLES = {"synthetic":"Эйлер: синтетика ANU", "control":"Контроль: давление без K", "well":"The Well: внешняя симуляция"}


def _finish(fig, path):
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_gates(run, path, title):
    trajectory = run["trajectory"]
    gates = np.asarray([step["gates"] for step in trajectory])
    fig, ax = plt.subplots(figsize=(10, 4.1))
    for j,p in enumerate(PARAMS):
        ax.plot(gates[:,j], label=PARAM_LABELS[p], color=COLORS[j],
                ls=["-","--",":"][j//3], lw=1.8)
    ax.axhline(.5, color="black", lw=.8, alpha=.4)
    ax.set(xlabel="Шаг оптимизации", ylabel="Коэффициент включения z", ylim=(-.04,1.04), title=title)
    ax.legend(ncol=3, loc="upper right", fontsize=9)
    _finish(fig,path)


def plot_archive(run, path, title):
    rows=run["archive"]
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    scores=np.array([r["score"] for r in rows])
    best=np.minimum.accumulate(scores)
    axes[0].scatter(np.arange(len(rows)),scores,s=9,c="#b2bac4",alpha=.55,label="Кандидат")
    axes[0].plot(best,color="#277DA8",label="Лучший в архиве")
    axes[0].set(xlabel="Номер кандидата",ylabel="Критерий на валидации",title="Выбор графа")
    axes[0].legend()
    edges=np.array([sum(r["mask"]) for r in rows])
    errors=np.array([r["val_nmse"] for r in rows])
    points=axes[1].scatter(edges,errors,c=[r["intervention_penalty"] for r in rows],cmap="viridis",s=25,alpha=.7)
    axes[1].scatter(sum(run["selected_mask"]),run["val_nmse"],marker="*",s=240,c="#e49b36",edgecolor="black",label="Выбранный")
    axes[1].set(xlabel="Число активных групп",ylabel="NMSE на валидации",title="Точность и сложность",xticks=range(10))
    axes[1].legend()
    fig.colorbar(points,ax=axes[1],label="Доля расхождений с диагностикой")
    fig.suptitle(title)
    fig.tight_layout()
    _finish(fig,path)


def _draw_group_graph(ax,mask,title):
    graph=nx.DiGraph()
    graph.add_nodes_from(["U",*PARAMS,"rhs"])
    pos={"U":(-1.5,0),"rhs":(1.5,0)}
    for j,p in enumerate(PARAMS):
        pos[p]=(0,4-j)
        graph.add_edges_from([("U",p),(p,"rhs")])
    alive={"U","rhs"}|{p for j,p in enumerate(PARAMS) if mask[j]}
    active_edges=[(a,b) for a,b in graph.edges if a in alive and b in alive]
    inactive_edges=[(a,b) for a,b in graph.edges if (a,b) not in active_edges]
    nx.draw_networkx_edges(graph,pos,edgelist=inactive_edges,ax=ax,width=.6,edge_color="#e4e7eb",arrowsize=8)
    nx.draw_networkx_edges(graph,pos,edgelist=active_edges,ax=ax,width=1.2,edge_color="#8ea2b0",arrowsize=10)
    nodes=list(graph)
    nx.draw_networkx_nodes(graph,pos,nodelist=nodes,node_size=600,ax=ax,
        node_color=["#dcebf1" if v in alive else "#f2f3f5" for v in nodes],edgecolors="#ffffff")
    labels={p:PARAM_LABELS[p] for p in PARAMS}
    labels.update(U="$U$",rhs="$\\partial_t U$")
    nx.draw_networkx_labels(graph,pos,labels=labels,ax=ax,font_size=10)
    ax.set_title(title,fontsize=11)
    ax.set_xlim(-2.0,2.0);ax.set_ylim(-4.7,4.7);ax.axis("off")


def plot_graphs(run,path,title):
    # Actual chronological candidates, without showing the full graph twice.
    rows=run["archive"]
    intermediate=[]
    final=tuple(run["selected_mask"])
    seen={tuple([1]*9),tuple([0]*9),final}
    for i,row in enumerate(rows):
        mask=tuple(row["mask"])
        if mask==final: break
        if mask not in seen:
            intermediate.append((i,row));seen.add(mask)
        if len(intermediate)==2: break
    panels=[("Исходная библиотека: 9 групп",[1]*9)]
    panels += [(f"Кандидат {i}: {sum(r['mask'])} групп",r["mask"]) for i,r in intermediate]
    panels.append((f"Выбранный: {sum(run['selected_mask'])} групп",run["selected_mask"]))
    fig,axes=plt.subplots(1,len(panels),figsize=(4.4*len(panels),6.8),squeeze=False)
    for ax,(label,mask) in zip(axes.flat,panels): _draw_group_graph(ax,mask,label)
    fig.suptitle(title+". Группы вычислительного графа",fontsize=13)
    fig.text(.5,.015,"Светлые группы исключены. Внутренние операции каждой группы заданы уравнением Эйлера.",ha="center",fontsize=10)
    _finish(fig,path)


def plot_fit(data,run,path,title):
    true=data.test.y
    pred=data.test.X @ np.asarray(run["coefficients"])
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.2))
    rng=np.random.default_rng(0)
    idx=rng.choice(len(true),min(6000,len(true)),replace=False)
    points=axes[0].scatter(true[idx],pred[idx],c=data.test.equation[idx],cmap="tab10",vmin=0,vmax=4,s=3,alpha=.35)
    handles, _ = points.legend_elements(num=range(5), alpha=.8)
    axes[0].legend(handles, ["ρ", "mₓ", "mᵧ", "m_z", "E"], ncol=5, fontsize=8,
                   loc="upper left", handletextpad=.2, columnspacing=.6)
    bounds=np.quantile(np.r_[true,pred],[.001,.999])
    axes[0].plot(bounds,bounds,"k--",lw=1)
    axes[0].set(xlabel="Наблюдаемая производная",ylabel="Предсказанная производная",title="Тестовые кадры после нормировки")
    eq_errors=[]
    for eq in range(5):
        m=data.test.equation==eq
        den=float(true[m] @ true[m])
        eq_errors.append(np.sqrt(np.sum((true[m]-pred[m])**2)/den) if den>1e-20 else np.nan)
    axes[1].bar(["ρ","mₓ","mᵧ","m_z","E"],np.asarray(eq_errors)*100,color="#277da8")
    axes[1].set(ylabel="Относительная L2-ошибка, %",title="Ошибка по уравнениям")
    for eq, value in enumerate(eq_errors):
        if not np.isfinite(value):
            axes[1].text(eq, .02, "0/0\nне определена", ha="center", va="bottom",
                         transform=axes[1].get_xaxis_transform(), fontsize=8)
    fig.suptitle(title+f". Тест: {100*np.sqrt(run['test_nmse']):.2f}% L2")
    fig.text(.5,.015,"Для m_z наблюдение и прогноз равны нулю. Относительная ошибка этого уравнения не определена.",ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.06,1,.96))
    _finish(fig,path)


def plot_diagnostics(diag,path,title):
    info=np.asarray(diag["info"]);shift=np.asarray(diag["shift"]);threshold=np.asarray(diag["null_q"])
    ratio=np.divide(shift,np.maximum(threshold,1e-12))
    labels=[PARAM_LABELS[p] for p in PARAMS]
    fig,axes=plt.subplots(1,2,figsize=(11,4.3))
    colors=["#277DA8" if k else "#acb3bd" for k in diag["keep"]]
    axes[0].bar(labels,info,color=colors)
    axes[0].axhline(1,color="#d45744",ls="--",label="Порог 1 нат")
    axes[0].set(ylabel="KL(post || prior), нат",title="Информация о весах")
    axes[0].legend(fontsize=9)
    axes[1].bar(labels,np.minimum(ratio,1e7),color=colors)
    axes[1].set_yscale("symlog",linthresh=1)
    axes[1].axhline(1,color="#d45744",ls="--",label="Сдвиг равен порогу")
    axes[1].set(ylabel="Jeffreys / 95%-квантиль",title="Реакция на условное перемешивание")
    axes[1].legend(fontsize=9)
    fig.suptitle(title);fig.tight_layout()
    _finish(fig,path)


def plot_comparison(summary,path):
    fig,axes=plt.subplots(1,2,figsize=(11,4.3))
    names=list(summary["datasets"])
    x=np.arange(len(names));width=.35
    for off,method,color,label in [(-.5,"intervention","#277DA8","С критерием Ивана"),(.5,"ablation","#E49B36","Точность + сложность")]:
        errors=[];counts=[]
        for name in names:
            rows=[r for r in summary["datasets"][name]["runs"] if r["method"]==method]
            errors.append(100*np.sqrt(rows[0]["test_nmse"]))
            counts.append(sum(rows[0]["selected_mask"]))
        axes[0].bar(x+off*width,errors,width,color=color,label=label)
        axes[1].bar(x+off*width,counts,width,color=color,label=label)
    for ax in axes:
        ax.set_xticks(x,["Синтетика","Без K → p","The Well"])
        ax.legend(fontsize=9)
    axes[0].set(ylabel="Относительная L2-ошибка, %",title="Качество на тесте (seed=0)")
    axes[1].set(ylabel="Число активных групп",title="Сложность выбранной структуры",yticks=range(10))
    fig.tight_layout();_finish(fig,path)


def plot_dataset(data,result,output):
    output=Path(output)
    title=TITLES[data.name]
    run=next(r for r in result["runs"] if r["method"]=="intervention")
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    plot_gates(run,output/f"{data.name}_gates.png",title)
    plot_archive(run,output/f"{data.name}_archive.png",title)
    plot_graphs(run,output/f"{data.name}_graphs.png",title)
    plot_fit(data,run,output/f"{data.name}_fit.png",title)
    plot_diagnostics(result["reference_diagnostics"],output/f"{data.name}_diagnostics.png",title)
