"""Лабораторная работа 2: интервенции в данные и распределения параметров графа.

Модуль продолжает лабораторную работу 1 (уравнения Эйлера). Каждой группе рёбер
графа ``equation_graph()`` сопоставлен вес. При всех весах, равных истинным,
получаются обычные уравнения Эйлера:

    d_t U = -sum_d [ a_d * d_d(U v_d)                      # адвекция, ребро v_d -> F_d
                    + bE_d * d_d(P^E_d) - bK_d * d_d(P^K_d) ]  # давление, ребро p -> F_d

где P^E_d = (0, E e_d, E v_d), P^K_d = (0, K e_d, K v_d). Если подставить
p = (gamma-1)(E-K), истинные значения равны a_d = 1 и bE_d = bK_d = gamma-1.
Модель линейна по весам. Давление в данных не используется, поэтому gamma
оценивается вместе с остальными весами.

Регрессия строится в слабой форме: обе части уравнения сглаживаются одним
гауссовым ядром. Сглаживание линейно и перестановочно с производной, поэтому
это равносильно интегрированию уравнения с гладкими пробными функциями
(weak SINDy, Messenger & Bortz, 2021).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy import stats
from scipy.ndimage import gaussian_filter

AXES = ("x", "y", "z")
PARAMS = tuple(
    [f"a_{d}" for d in AXES] + [f"bE_{d}" for d in AXES] + [f"bK_{d}" for d in AXES]
)
PARAM_LABELS = {
    **{f"a_{d}": rf"$a_{d}$" for d in AXES},
    **{f"bE_{d}": rf"$\beta^E_{d}$" for d in AXES},
    **{f"bK_{d}": rf"$\beta^K_{d}$" for d in AXES},
}
EQUATIONS = ("rho", "mx", "my", "mz", "E")
INPUT_NODES = ("rho", "mx", "my", "mz", "E")
DERIVED_NODES = ("u", "v", "w", "kinetic")

# Какие элементы графа ЛР1 «обслуживает» каждый параметр.
PARAM_GRAPH_ELEMENTS = {
    "a_x": [("u", "Fx"), ("mx", "Fx"), ("my", "Fx"), ("mz", "Fx"), ("E", "Fx")],
    "a_y": [("v", "Fy"), ("mx", "Fy"), ("my", "Fy"), ("mz", "Fy"), ("E", "Fy")],
    "a_z": [("w", "Fz"), ("mx", "Fz"), ("my", "Fz"), ("mz", "Fz"), ("E", "Fz")],
    "bE_x": [("pressure", "Fx"), ("E", "pressure"), ("gamma", "pressure")],
    "bE_y": [("pressure", "Fy"), ("E", "pressure"), ("gamma", "pressure")],
    "bE_z": [("pressure", "Fz"), ("E", "pressure"), ("gamma", "pressure")],
    "bK_x": [("pressure", "Fx"), ("kinetic", "pressure")],
    "bK_y": [("pressure", "Fy"), ("kinetic", "pressure")],
    "bK_z": [("pressure", "Fz"), ("kinetic", "pressure")],
}
# Узел-родитель, в который делается интервенция для проверки параметра.
PARAM_PARENT = {
    "a_x": "u", "a_y": "v", "a_z": "w",
    "bE_x": "E", "bE_y": "E", "bE_z": "E",
    "bK_x": "kinetic", "bK_y": "kinetic", "bK_z": "kinetic",
}


def true_params(gamma: float) -> NDArray:
    """Значения весов, при которых модель совпадает с уравнениями Эйлера."""
    return np.array([1.0] * 3 + [gamma - 1.0] * 6)


# ---------------------------------------------------------------------------
# Построение библиотеки слагаемых (прямая задача → регрессия)
# ---------------------------------------------------------------------------


def _smooth(field: NDArray, sigma: float) -> NDArray:
    """Сгладить по x, y (outflow -> 'nearest') и z (periodic -> 'wrap')."""
    if sigma <= 0:
        return field
    out = gaussian_filter(field, sigma=(sigma, sigma, 0), mode="nearest")
    return out


def _grad(field: NDArray, axis: int, spacing: float) -> NDArray:
    """Центральная разность; по z — периодическая (для плоских данных = 0)."""
    if axis == 2:
        if field.shape[2] == 1:
            return np.zeros_like(field)
        return (np.roll(field, -1, axis=2) - np.roll(field, 1, axis=2)) / (2 * spacing)
    return np.gradient(field, spacing, axis=axis)


def node_values(U: NDArray, do: dict | None = None) -> dict[str, NDArray]:
    """Значения узлов графа ЛР1 в одной ячейке; ``do`` — интервенции в узлы.

    U имеет форму (nx, ny, nz, 5) с компонентами (rho, mx, my, mz, E).
    ``do`` отображает имя узла в функцию f(values) -> новое значение узла.
    Входные узлы изменяются первыми, затем пересчитываются потомки
    (u, v, w, kinetic) — так интервенция распространяется по графу.
    """
    do = do or {}
    nodes = {name: U[..., i].copy() for i, name in enumerate(INPUT_NODES)}
    for name in INPUT_NODES:
        if name in do:
            nodes[name] = do[name](nodes[name])
    for vel, mom in (("u", "mx"), ("v", "my"), ("w", "mz")):
        nodes[vel] = nodes[mom] / nodes["rho"]
        if vel in do:
            nodes[vel] = do[vel](nodes[vel])
    nodes["kinetic"] = 0.5 * nodes["rho"] * (nodes["u"] ** 2 + nodes["v"] ** 2 + nodes["w"] ** 2)
    if "kinetic" in do:
        nodes["kinetic"] = do["kinetic"](nodes["kinetic"])
    return nodes


def library_frame(U: NDArray, spacing, sigma: float, do: dict | None = None) -> NDArray:
    """Матрица слагаемых в одном кадре: форма (nx, ny, nz, 5 уравнений, 9 параметров).

    Столбец j содержит -d_d(слагаемое), умножаемое на параметр j.
    """
    n = node_values(U, do)
    cons = [n[name] for name in INPUT_NODES]  # компоненты U после интервенции
    vel = [n["u"], n["v"], n["w"]]
    lib = np.zeros((*U.shape[:3], 5, 9))
    for d in range(3):
        h = spacing[d]
        # адвекция: d_d(U_k v_d)
        for k in range(5):
            lib[..., k, d] = -_grad(_smooth(cons[k] * vel[d], sigma), d, h)
        # давление, часть с E: импульс d -> d_d E, энергия -> d_d(E v_d)
        lib[..., 1 + d, 3 + d] = -_grad(_smooth(n["E"], sigma), d, h)
        lib[..., 4, 3 + d] = -_grad(_smooth(n["E"] * vel[d], sigma), d, h)
        # давление, часть с K (входит со знаком минус)
        lib[..., 1 + d, 6 + d] = _grad(_smooth(n["kinetic"], sigma), d, h)
        lib[..., 4, 6 + d] = _grad(_smooth(n["kinetic"] * vel[d], sigma), d, h)
    return lib


def time_derivative(frames: NDArray, times: NDArray, centers, sigma: float) -> NDArray:
    """Сглаженная d_t U в кадрах ``centers`` (центральная разность, допускает неравный шаг)."""
    out = []
    for c in centers:
        t0, t1, t2 = times[c - 1], times[c], times[c + 1]
        h0, h1 = t1 - t0, t2 - t1
        # формула второго порядка для неравномерной сетки
        w0 = -h1 / (h0 * (h0 + h1))
        w1 = (h1 - h0) / (h0 * h1)
        w2 = h0 / (h1 * (h0 + h1))
        dU = w0 * frames[c - 1] + w1 * frames[c] + w2 * frames[c + 1]
        out.append(np.stack([_smooth(dU[..., k], sigma) for k in range(5)], axis=-1))
    return np.stack(out)


@dataclass
class RegressionData:
    """Строки регрессии y = X w, разбитые на пространственно-временные блоки."""

    X: NDArray            # (n_rows, 9)
    y: NDArray            # (n_rows,)
    block: NDArray        # (n_rows,) номер блока для бутстрепа
    equation: NDArray     # (n_rows,) номер уравнения 0..4
    scales: NDArray       # (5,) нормировки уравнений
    meta: dict = field(default_factory=dict)


def build_regression(
    frames: NDArray,
    times: NDArray,
    centers,
    spacing=(1 / 64, 1 / 64, 1.0),
    sigma: float = 1.5,
    margin: int = 4,
    block_size: int = 8,
    do: dict | None = None,
    scales: NDArray | None = None,
    equations=(0, 1, 2, 3, 4),
) -> RegressionData:
    """Собрать регрессию по кадрам консервативных полей (nt, nx, ny, nz, 5).

    Интервенция ``do`` применяется только к регрессорам — к значениям узлов
    в центральных кадрах. Наблюдаемый отклик d_t U остаётся прежним, как и в
    ноутбуке doInt: мы меняем вход графа и смотрим, как перестраиваются веса.
    """
    frames = np.asarray(frames, dtype=float)
    times = np.asarray(times, dtype=float)
    target = time_derivative(frames, times, centers, sigma)
    libs = np.stack([library_frame(frames[c], spacing, sigma, do) for c in centers])
    nx, ny = frames.shape[1:3]
    sl = (slice(None), slice(margin, nx - margin), slice(margin, ny - margin))
    target, libs = target[sl], libs[sl]
    if scales is None:
        rms = np.sqrt(np.mean(target**2, axis=(0, 1, 2, 3)))
        fallback = np.mean(rms[1:3])
        scales = np.where(rms > 1e-12 * max(rms.max(), 1e-300), rms, fallback)
    target = target / scales
    libs = libs / scales[:, None]
    nc, bx, by, nz = target.shape[:4]
    ix = np.arange(bx)[None, :, None, None] // block_size
    iy = np.arange(by)[None, None, :, None] // block_size
    it = np.arange(nc)[:, None, None, None]
    nbx, nby = int(ix.max()) + 1, int(iy.max()) + 1
    block = np.broadcast_to(it * nbx * nby + ix * nby + iy, (nc, bx, by, nz))
    eq_list = list(equations)
    X = libs[..., eq_list, :].reshape(-1, 9)
    y = target[..., eq_list].reshape(-1)
    blk = np.repeat(block.reshape(-1), len(eq_list))
    eqn = np.tile(np.array(eq_list), block.size)
    return RegressionData(X, y, blk, eqn, scales, {"centers": list(centers), "sigma": sigma})


# ---------------------------------------------------------------------------
# Оценка распределения параметров
# ---------------------------------------------------------------------------


@dataclass
class GaussianPosterior:
    mean: NDArray
    cov: NDArray
    noise_var: float
    n: int

    @property
    def std(self):
        return np.sqrt(np.diag(self.cov))


PRIOR_PRECISION = 1e-2  # априорно w ~ N(0, 10^2 I): слабое, но собственное


def laplace_posterior(data: RegressionData, alpha: float = PRIOR_PRECISION) -> GaussianPosterior:
    """Байесовская линейная регрессия; ковариация = обратный гессиан (матрица Фишера + prior).

    Для линейно-гауссовой модели аппроксимация Лапласа точна:
    Sigma = (beta X^T X + alpha I)^-1, mu = beta Sigma X^T y.
    """
    X, y = data.X, data.y
    XtX, Xty = X.T @ X, X.T @ y
    p = X.shape[1]
    w_ols = np.linalg.lstsq(XtX + 1e-12 * np.eye(p), Xty, rcond=None)[0]
    resid = y - X @ w_ols
    noise_var = float(resid @ resid / max(len(y) - p, 1))
    beta = 1.0 / noise_var
    precision = beta * XtX + alpha * np.eye(p)
    cov = np.linalg.inv(precision)
    mean = cov @ (beta * Xty)
    return GaussianPosterior(mean, cov, noise_var, len(y))


def fisher_matrix(data: RegressionData) -> NDArray:
    """Матрица Фишера I(w) = X^T X / sigma^2 — метрика риманова пространства параметров."""
    post = laplace_posterior(data)
    return data.X.T @ data.X / post.noise_var


def block_stats(data: RegressionData):
    """Достаточные статистики X^T X и X^T y по каждому блоку."""
    ids, inv = np.unique(data.block, return_inverse=True)
    nb, p = len(ids), data.X.shape[1]
    XtX = np.zeros((nb, p, p))
    Xty = np.zeros((nb, p))
    np.add.at(Xty, inv, data.X * data.y[:, None])
    for j in range(p):
        np.add.at(XtX[:, :, j], inv, data.X * data.X[:, [j]])
    return XtX, Xty


def block_bootstrap(
    data: RegressionData, n_boot: int = 300, alpha: float = PRIOR_PRECISION, seed: int = 0,
    n_blocks: int | None = None,
) -> NDArray:
    """Блочный бутстреп: блоки 8x8 ячеек сохраняют пространственную корреляцию остатков.

    Возвращает (n_boot, 9) оценок MAP. ``n_blocks`` < числа блоков — бутстреп
    подвыборки малого объёма (одна из интервенций).
    """
    rng = np.random.default_rng(seed)
    XtX_b, Xty_b = block_stats(data)
    nb, p = Xty_b.shape
    m = nb if n_blocks is None else n_blocks
    # prior масштабирован так же, как в laplace_posterior (в единицах шума)
    noise_var = laplace_posterior(data).noise_var
    samples = np.empty((n_boot, p))
    for b in range(n_boot):
        counts = np.bincount(rng.integers(0, nb, m), minlength=nb).astype(float)
        A = np.tensordot(counts, XtX_b, axes=1) + alpha * noise_var * np.eye(p)
        rhs = counts @ Xty_b
        samples[b] = np.linalg.solve(A, rhs)
    return samples


def gaussian_from_samples(samples: NDArray, floor: float = 1e-10) -> GaussianPosterior:
    cov = np.cov(samples, rowvar=False) + floor * np.eye(samples.shape[1])
    return GaussianPosterior(samples.mean(axis=0), cov, float("nan"), len(samples))


def student_t_loglik(w, X, y, scale, nu):
    r = (y - X @ w) / scale
    return float(np.sum(stats.t.logpdf(r, df=nu)) - len(y) * np.log(scale))


def metropolis_hastings(
    data: RegressionData,
    n_steps: int = 6000,
    nu: float = 3.0,
    temper: float = 1.0,
    proposal_cov: NDArray | None = None,
    start: NDArray | None = None,
    alpha: float = PRIOR_PRECISION,
    max_rows: int = 20000,
    active: NDArray | None = None,
    seed: int = 0,
    adapt_steps: int = 1500,
):
    """Случайное блуждание Метрополиса — Хастингса, правдоподобие Стьюдента.

    ``temper`` ∈ (0,1] — степень правдоподобия (coarsened posterior, Miller &
    Dunson 2019): компенсирует корреляцию соседних ячеек, иначе апостериорное
    распределение получается слишком узким. Неидентифицируемые параметры
    (``active`` = False) совпадают с prior, их можно не семплировать.
    """
    rng = np.random.default_rng(seed)
    X, y = data.X, data.y
    if len(y) > max_rows:
        idx = rng.choice(len(y), max_rows, replace=False)
        temper = temper * len(y) / max_rows
        X, y = X[idx], y[idx]
    p = X.shape[1]
    active = np.ones(p, bool) if active is None else np.asarray(active, bool)
    lap = laplace_posterior(data)
    w = lap.mean.copy() if start is None else np.asarray(start, float).copy()
    resid = y - X @ w
    scale = 1.4826 * np.median(np.abs(resid - np.median(resid)))  # MAD — устойчивый масштаб
    C = lap.cov if proposal_cov is None else proposal_cov
    k = int(active.sum())

    def logpost(w):
        return temper * student_t_loglik(w, X, y, scale, nu) - 0.5 * alpha * w @ w

    def run(w, Ca, steps):
        La = np.linalg.cholesky(Ca * 2.38**2 / k + 1e-14 * np.eye(k))
        lp = logpost(w)
        out = np.empty((steps, p))
        acc = 0
        for i in range(steps):
            prop = w.copy()
            prop[active] += La @ rng.standard_normal(k)
            lpp = logpost(prop)
            if np.log(rng.random()) < lpp - lp:
                w, lp = prop, lpp
                acc += 1
            out[i] = w
        return out, acc / steps

    # пилотный прогон (адаптация масштаба предложения, Haario et al. 2001), затем основной
    pilot, _ = run(w, C[np.ix_(active, active)], adapt_steps)
    Ca = np.cov(pilot[adapt_steps // 2:, active], rowvar=False)
    chain, rate = run(pilot[-1], Ca, n_steps)
    # неактивные параметры: точные выборки из prior
    chain[:, ~active] = rng.normal(0, 1 / np.sqrt(alpha), (n_steps, (~active).sum()))
    return chain, rate


# ---------------------------------------------------------------------------
# Интервенции
# ---------------------------------------------------------------------------


def _bins(cond: NDArray, n_bins: int) -> NDArray:
    q = np.quantile(cond, np.linspace(0, 1, n_bins + 1)[1:-1])
    return np.digitize(cond, q)


def _conditional_moments(x, labels, n_bins):
    mu = np.zeros_like(x)
    sd = np.zeros_like(x)
    for b in range(n_bins):
        m = labels == b
        if m.any():
            mu[m] = x[m].mean()
            sd[m] = x[m].std()
    return mu, sd


@dataclass
class Intervention:
    """Интервенция в узел графа; условное распределение — по бинам узла ``given``.

    kind: zero | noise | shuffle | outliers | attack.
    level: для noise — множитель условного СКО (гиперпараметр λ);
           для outliers — доля испорченных ячеек;
           для attack — множитель безусловного СКО узла ``given`` (шум,
           не согласованный с условным распределением, «атака»).
    """

    node: str
    kind: str
    level: float = 1.0
    given: str = "rho"
    n_bins: int = 10
    seed: int = 0

    def label(self) -> str:
        if self.kind == "zero":
            return f"do({self.node}=0)"
        if self.kind == "shuffle":
            return f"shuffle({self.node} | {self.given})"
        if self.kind == "noise":
            return f"noise({self.node} | {self.given}, λ={self.level:g})"
        if self.kind == "attack":
            return f"attack({self.node}, λ={self.level:g}·std({self.given}))"
        return f"3σ-outliers({self.node} | {self.given}, q={self.level:g})"

    def as_do(self, U_center: NDArray) -> dict:
        """Функция для node_values; условие строится по исходному кадру."""
        rng = np.random.default_rng(self.seed)
        base = node_values(U_center)
        cond_node = self.given if self.given != self.node else "E"
        labels = _bins(base[cond_node], self.n_bins)
        positive = self.node in ("rho", "E", "kinetic")

        def apply(x):
            x = x.copy()
            if self.kind == "zero":
                if self.node == "rho":
                    raise ValueError("Зануление плотности делает скорость бесконечной")
                return np.zeros_like(x)
            mu, sd = _conditional_moments(x, labels, self.n_bins)
            if self.kind == "noise":
                x = x + self.level * sd * rng.standard_normal(x.shape)
            elif self.kind == "shuffle":
                for b in range(self.n_bins):
                    m = np.flatnonzero(labels == b)
                    x.flat[m] = x.flat[rng.permutation(m)]
            elif self.kind == "attack":
                x = x + self.level * base[self.given].std() * rng.standard_normal(x.shape)
            elif self.kind == "outliers":
                m = rng.random(x.shape) < self.level
                sign = rng.choice([-1.0, 1.0], size=x.shape)
                x[m] = (mu + sign * (3.0 + rng.exponential(1.0, x.shape)) * np.maximum(sd, 1e-12))[m]
            else:
                raise ValueError(self.kind)
            if positive:
                floor = 1e-3 * np.median(np.abs(base[self.node]))
                x = np.maximum(x, floor)
            return x

        return {self.node: apply}


def build_with_intervention(frames, times, centers, intervention: Intervention | None, scales=None, **kw):
    """Регрессия, где интервенция применяется к каждому центральному кадру отдельно."""
    if intervention is None:
        return build_regression(frames, times, centers, scales=scales, **kw)
    parts = []
    for i, c in enumerate(centers):
        iv = Intervention(**{**intervention.__dict__, "seed": intervention.seed + i})
        do = iv.as_do(frames[c])
        parts.append(build_regression(frames, times, [c], do=do, scales=scales, **kw))
    X = np.concatenate([p.X for p in parts])
    y = np.concatenate([p.y for p in parts])
    nblk = 0
    blocks = []
    for p in parts:
        blocks.append(p.block + nblk)
        nblk += int(p.block.max()) + 1
    return RegressionData(X, y, np.concatenate(blocks), np.concatenate([p.equation for p in parts]),
                          parts[0].scales, {"centers": list(centers)})


# ---------------------------------------------------------------------------
# Расстояния между распределениями
# ---------------------------------------------------------------------------


def _sub(mean, cov, idx):
    idx = np.atleast_1d(idx)
    return mean[idx], cov[np.ix_(idx, idx)]


def kl_gauss(m0, S0, m1, S1) -> float:
    """KL(N0 || N1)."""
    m0, m1 = np.atleast_1d(m0), np.atleast_1d(m1)
    S0, S1 = np.atleast_2d(S0), np.atleast_2d(S1)
    k = len(m0)
    S1i = np.linalg.inv(S1)
    d = m1 - m0
    _, ld0 = np.linalg.slogdet(S0)
    _, ld1 = np.linalg.slogdet(S1)
    return float(0.5 * (np.trace(S1i @ S0) - k + d @ S1i @ d + ld1 - ld0))


def _sqrtm_psd(S):
    vals, vecs = np.linalg.eigh(S)
    return (vecs * np.sqrt(np.clip(vals, 0, None))) @ vecs.T


def w2_gauss(m0, S0, m1, S1) -> float:
    """Расстояние Вассерштейна — Добрушина W2 между гауссианами (формула Гивенса — Шорта)."""
    m0, m1 = np.atleast_1d(m0), np.atleast_1d(m1)
    S0, S1 = np.atleast_2d(S0), np.atleast_2d(S1)
    r = _sqrtm_psd(S1)
    cross = _sqrtm_psd(r @ S0 @ r)
    val = np.sum((m0 - m1) ** 2) + np.trace(S0 + S1 - 2 * cross)
    return float(np.sqrt(max(val, 0.0)))


def hellinger_gauss(m0, S0, m1, S1) -> float:
    m0, m1 = np.atleast_1d(m0), np.atleast_1d(m1)
    S0, S1 = np.atleast_2d(S0), np.atleast_2d(S1)
    S = 0.5 * (S0 + S1)
    d = m0 - m1
    _, ld0 = np.linalg.slogdet(S0)
    _, ld1 = np.linalg.slogdet(S1)
    _, ld = np.linalg.slogdet(S)
    log_bc = 0.25 * ld0 + 0.25 * ld1 - 0.5 * ld - 0.125 * d @ np.linalg.solve(S, d)
    return float(np.sqrt(max(1 - np.exp(log_bc), 0.0)))


def s_score_gauss(m0, S0, m1, S1) -> float:
    """Адекватная мера сходства Адуенко (s-score) для гауссиан.

    s(g0, g1) = ∫ g0 g1 dw / max_b ∫ g0(w - b) g1(w) dw.
    Для N(m0,S0), N(m1,S1): s = exp(-1/2 (m0-m1)^T (S0+S1)^-1 (m0-m1)) ∈ (0, 1].
    s = 1 при совпадении центров; расстояние — 1 - s.
    """
    m0, m1 = np.atleast_1d(m0), np.atleast_1d(m1)
    S = np.atleast_2d(S0) + np.atleast_2d(S1)
    d = m0 - m1
    return float(np.exp(-0.5 * d @ np.linalg.solve(S, d)))


def gaussian_distances(p0: GaussianPosterior, p1: GaussianPosterior, idx) -> dict[str, float]:
    a = _sub(p0.mean, p0.cov, idx)
    b = _sub(p1.mean, p1.cov, idx)
    return {
        "KL": kl_gauss(*a, *b),
        "W2": w2_gauss(*a, *b),
        "Hellinger": hellinger_gauss(*a, *b),
        "1 - s": 1 - s_score_gauss(*a, *b),
    }


def sample_distances(x0: NDArray, x1: NDArray, bins: int = 40) -> dict[str, float]:
    """Одномерные выборочные меры: КС, W1, полная вариация и Дженсен — Шеннон по гистограмме."""
    ks = stats.ks_2samp(x0, x1)
    lo, hi = min(x0.min(), x1.min()), max(x0.max(), x1.max())
    if hi <= lo:
        hi = lo + 1e-12
    h0, edges = np.histogram(x0, bins=bins, range=(lo, hi))
    h1, _ = np.histogram(x1, bins=edges)
    p, q = h0 / h0.sum(), h1 / h1.sum()
    m = 0.5 * (p + q)

    def _kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log(a[mask] / b[mask])))

    return {
        "KS": float(ks.statistic),
        "KS p-value": float(ks.pvalue),
        "W1": float(stats.wasserstein_distance(x0, x1)),
        "TV": float(0.5 * np.abs(p - q).sum()),
        "JS": 0.5 * _kl(p, m) + 0.5 * _kl(q, m),
    }


def info_gain(post: GaussianPosterior, alpha: float = PRIOR_PRECISION) -> NDArray:
    """KL(posterior_j || prior_j) для каждого параметра: сколько данные сообщили о нём."""
    prior_var = 1 / alpha
    out = []
    for j in range(len(post.mean)):
        out.append(kl_gauss(post.mean[j], post.cov[j, j], 0.0, prior_var))
    return np.array(out)


def null_distances(data: RegressionData, n_splits: int = 40, n_boot: int = 200, seed: int = 0):
    """Расстояния между оценками по двум случайным половинам блоков.

    Это «нулевой» уровень: так различаются распределения без всякой
    интервенции, только из-за конечной выборки. Возвращает (n_splits, 9)
    одномерных KL симметризованных (Jeffreys) и 1-s.
    """
    rng = np.random.default_rng(seed)
    ids = np.unique(data.block)
    out_kl, out_s = [], []
    for s in range(n_splits):
        perm = rng.permutation(ids)
        half = set(perm[: len(ids) // 2].tolist())
        mask = np.array([b in half for b in data.block])
        res = []
        for m in (mask, ~mask):
            sub = RegressionData(data.X[m], data.y[m], data.block[m], data.equation[m], data.scales)
            res.append(gaussian_from_samples(block_bootstrap(sub, n_boot, seed=seed + s)))
        kl, ss = [], []
        for j in range(9):
            a = _sub(res[0].mean, res[0].cov, j)
            b = _sub(res[1].mean, res[1].cov, j)
            kl.append(0.5 * (kl_gauss(*a, *b) + kl_gauss(*b, *a)))
            ss.append(1 - s_score_gauss(*a, *b))
        out_kl.append(kl)
        out_s.append(ss)
    return np.array(out_kl), np.array(out_s)


# ---------------------------------------------------------------------------
# Синтетика из «урезанного» графа (положительный контроль критерия)
# ---------------------------------------------------------------------------


def solve_relaxed(initial_primitive, times, spacing, gamma=1.4, theta_k=1.0, cfl=0.4):
    """Схема Русанова из ЛР1 для модели с весом ребра kinetic -> pressure.

    p = (gamma - 1) (E - theta_k K). При theta_k = 1 — обычные уравнения Эйлера
    (совпадает с EulerSolver для плоской задачи), при theta_k = 0 ребро удалено.
    Возвращает консервативные поля (nt, nx, ny, 1, 5). Граница — outflow.
    """
    W = np.asarray(initial_primitive, float)
    if W.ndim != 4 or W.shape[2] != 1:
        raise ValueError("Нужно плоское начальное состояние (nx, ny, 1, 5)")
    g1 = gamma - 1.0

    def to_cons(W):
        U = np.empty_like(W)
        U[..., 0] = W[..., 0]
        U[..., 1:4] = W[..., :1] * W[..., 1:4]
        K = 0.5 * W[..., 0] * np.sum(W[..., 1:4] ** 2, axis=-1)
        U[..., 4] = W[..., 4] / g1 + theta_k * K
        return U

    def prim(U):
        rho = U[..., 0]
        vel = U[..., 1:4] / rho[..., None]
        K = 0.5 * rho * np.sum(vel**2, axis=-1)
        p = g1 * (U[..., 4] - theta_k * K)
        if np.any(rho <= 0) or np.any(p <= 0):
            raise RuntimeError("Нефизическое состояние в solve_relaxed")
        return rho, vel, p

    def phys_flux(U, vel, p, axis):
        F = U * vel[..., axis, None]
        F[..., 1 + axis] += p
        F[..., 4] += p * vel[..., axis]
        return F

    def rhs(U):
        out = np.zeros_like(U)
        for axis in (0, 1):
            first = np.take(U, [0], axis=axis)
            last = np.take(U, [-1], axis=axis)
            P = np.concatenate((first, U, last), axis=axis)
            rho, vel, p = prim(P)
            c = np.sqrt(gamma * p / rho)
            F = phys_flux(P, vel, p, axis)
            s = np.abs(vel[..., axis]) + c
            L = [slice(None)] * 4
            R = [slice(None)] * 4
            L[axis], R[axis] = slice(None, -1), slice(1, None)
            L, R = tuple(L), tuple(R)
            a = np.maximum(s[L[:3]], s[R[:3]])[..., None]
            face = 0.5 * (F[L] + F[R]) - 0.5 * a * (P[R] - P[L])
            out -= np.diff(face, axis=axis) / spacing[axis]
        return out

    U = to_cons(W)
    times = np.asarray(times, float)
    snaps = np.empty((len(times), *U.shape))
    snaps[0] = U
    t = float(times[0])
    for i, target in enumerate(times[1:], start=1):
        while t < target:
            rho, vel, p = prim(U)
            c = np.sqrt(gamma * p / rho)
            rate = sum(np.max(np.abs(vel[..., a]) + c) / spacing[a] for a in (0, 1))
            dt = min(cfl / rate, target - t)
            U = U + dt * rhs(U)
            t += dt
        snaps[i] = U
    return snaps


# ---------------------------------------------------------------------------
# Обёртки для ноутбука
# ---------------------------------------------------------------------------


def jeffreys_1d(p0: GaussianPosterior, p1: GaussianPosterior, j: int) -> float:
    """Симметризованная KL (Джеффрис) между одномерными маргиналами параметра j."""
    a = _sub(p0.mean, p0.cov, j)
    b = _sub(p1.mean, p1.cov, j)
    return 0.5 * (kl_gauss(*a, *b) + kl_gauss(*b, *a))


def run_interventions(frames, times, centers, base: RegressionData, interventions,
                      n_boot: int = 200, seed: int = 0, **kw):
    """Для каждой интервенции: пересобрать регрессию, оценить бутстрепом распределение весов."""
    out = []
    for k, item in enumerate(interventions):
        data = build_with_intervention(frames, times, centers, item, scales=base.scales, **kw)
        samples = block_bootstrap(data, n_boot, seed=seed + k)
        out.append({"intervention": item, "label": item.label(), "samples": samples,
                    "gauss": gaussian_from_samples(samples)})
    return out


def effect_matrix(base: GaussianPosterior, results, metric: str = "Jeffreys") -> NDArray:
    """Матрица (интервенция × параметр) одномерных расстояний."""
    M = np.zeros((len(results), len(PARAMS)))
    for i, r in enumerate(results):
        for j in range(len(PARAMS)):
            if metric == "Jeffreys":
                M[i, j] = jeffreys_1d(base, r["gauss"], j)
            else:
                M[i, j] = gaussian_distances(base, r["gauss"], j)[metric]
    return M


def decide(info: NDArray, shift: NDArray, null_q: NDArray, info_min: float = 1.0):
    """Правило: удалить, если данные не информативны о параметре (KL(post||prior) < info_min)
    или интервенция в родительский узел сдвигает распределение не больше нулевого уровня."""
    decision = []
    for j in range(len(PARAMS)):
        if info[j] < info_min:
            decision.append("удалить: не идентифицируем")
        elif shift[j] <= null_q[j]:
            decision.append("удалить: не реагирует на интервенцию")
        else:
            decision.append("оставить")
    return decision


def graph_positions(graph):
    """Та же раскладка, что и в plot_equation_graph из ЛР1."""
    positions = {}
    for layer in range(7):
        nodes = [n for n, a in graph.nodes(data=True) if a["layer"] == layer]
        for row, node in enumerate(nodes):
            positions[node] = (1.9 * layer, (len(nodes) - 1) / 2 - row)
    return positions


def plot_decision_graph(graph, removed_nodes, removed_edges, title=""):
    """Граф ЛР1: серым пунктиром — удаляемые элементы, цветом — сохраняемые."""
    import matplotlib.pyplot as plt
    import networkx as nx

    pos = graph_positions(graph)
    fig, ax = plt.subplots(figsize=(15, 6), layout="constrained")
    kept = [e for e in graph.edges if e not in removed_edges]
    nx.draw_networkx_edges(graph, pos, edgelist=kept, ax=ax, arrowsize=12, width=1.0,
                           alpha=0.45, node_size=2000, edge_color="#2b6a9e")
    nx.draw_networkx_edges(graph, pos, edgelist=list(removed_edges), ax=ax, arrowsize=10,
                           width=1.0, alpha=0.6, node_size=2000, edge_color="#c0392b",
                           style="dashed")
    colors = {"input": "#a8cbea", "parameter": "#f0bf72", "operation": "#cae2cd", "output": "#e8b9b5"}
    for node, attrs in graph.nodes(data=True):
        x, y = pos[node]
        removed = node in removed_nodes
        ax.text(x, y, attrs["label"], ha="center", va="center", fontsize=9,
                color="#888888" if removed else "black",
                bbox={"boxstyle": "round,pad=0.4",
                      "facecolor": "#eeeeee" if removed else colors[attrs["kind"]],
                      "edgecolor": "#c0392b" if removed else "#777777",
                      "linestyle": "--" if removed else "-", "linewidth": 0.9})
    ax.set_xlim(-0.45, 12.4)
    ys = [p[1] for p in pos.values()]
    ax.set_ylim(min(ys) - 0.35, max(ys) + 0.35)
    ax.set_title(title)
    ax.axis("off")
    return fig
