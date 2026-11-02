
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ПОСЛЕДОВАТЕЛЬНАЯ слепая проверка H1→H2→H3→H4→H5.

Не импортирует generate_traces.py и НЕ читает model_parameters.json.
Все параметры оцениваются только по временам событий в CSV.
Наблюдаемый столбец state_after_event НЕ используется.

Запуск:
    pip install numpy scipy
    python analyze_traces.py --traces traces
    python analyze_traces.py --traces traces --output analysis_results.json

Применяются тест независимости Льюнга–Бокса, бутстрэп-тест KS и BIC.
При недостаточном свидетельстве итог: НЕОПРЕДЕЛЕННО.
Математически универсальное однозначное восстановление невозможно.
"""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import gamma as gamma_dist, weibull_min, chi2, kstest

try:
    from numba import njit
except ImportError:
    def njit(fn):
        return fn

MODELS = ("gamma_renewal", "weibull_renewal", "mmpp_2", "mmpp_3", "map_2")
PARAMETER_NAMES = {
    "gamma_renewal": ("shape", "scale"),
    "weibull_renewal": ("shape", "scale"),
    "mmpp_2": ("lambda_1", "lambda_2", "q_12", "q_21"),
    "mmpp_3": ("lambda_1", "lambda_2", "lambda_3", "q_12", "q_23", "q_31"),
    "map_2": ("q_12", "q_21", "r_11", "r_12", "r_21", "r_22"),
}


def load_intervals(path):
    """Используем только моменты событий; интервалы пересчитываем сами."""
    times = []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if "event_time" not in (reader.fieldnames or []):
            raise ValueError(f"{path}: отсутствует столбец event_time")
        for row in reader:
            times.append(float(row["event_time"]))
    t = np.asarray(times, dtype=float)
    if len(t) < 50 or not np.all(np.isfinite(t)) or t[0] <= 0 or np.any(np.diff(t) <= 0):
        raise ValueError(f"{path}: нужно >= 50 строго возрастающих положительных моментов")
    return np.diff(np.r_[0.0, t])


def diagnostic_statistics(x):
    m = float(np.mean(x))
    return {
        "events": int(len(x)), "mean_interval": m,
        "cv_squared": float(np.var(x) / m**2),
        "lag1_correlation": float(np.corrcoef(x[:-1], x[1:])[0, 1]),
        "lag2_correlation": float(np.corrcoef(x[:-2], x[2:])[0, 1]),
    }


def stationary(q):
    """Стационарное распределение фоновой цепи с генератором Q."""
    n = q.shape[0]
    A = q.T.copy()
    A[-1, :] = 1.0
    b = np.zeros(n)
    b[-1] = 1.0
    pi = np.linalg.solve(A, b)
    if np.min(pi) <= 0:
        raise ValueError("Некорректное стационарное распределение")
    return pi


def markov_matrices(name, p):
    """Матрицы по параметрам, ОЦЕНЕННЫМ на анализируемой трассе."""
    if name == "mmpp_2":
        l1, l2, q12, q21 = p
        d1 = np.diag([l1, l2])
        d0 = np.array([[-q12-l1, q12], [q21, -q21-l2]])
    elif name == "mmpp_3":
        l1, l2, l3, q12, q23, q31 = p
        d1 = np.diag([l1, l2, l3])
        d0 = np.array([[-q12-l1, q12, 0.0],
                       [0.0, -q23-l2, q23],
                       [q31, 0.0, -q31-l3]])
    elif name == "map_2":
        q12, q21, r11, r12, r21, r22 = p
        d1 = np.array([[r11, r12], [r21, r22]])
        d0 = np.array([[-q12-r11-r12, q12], [q21, -q21-r21-r22]])
    else:
        raise ValueError(name)
    return d0, d1, stationary(d0 + d1)


@njit
def forward_likelihood_2(kernels, a0, a1):
    score = 0.0
    for i in range(kernels.shape[0]):
        b0 = a0 * kernels[i, 0, 0] + a1 * kernels[i, 1, 0]
        b1 = a0 * kernels[i, 0, 1] + a1 * kernels[i, 1, 1]
        s = b0 + b1
        if s <= 0.0 or b0 < -1e-10 or b1 < -1e-10:
            return -1e100
        a0, a1 = max(0.0, b0) / s, max(0.0, b1) / s
        score += math.log(s)
    return score


@njit
def forward_likelihood_3(kernels, a0, a1, a2):
    score = 0.0
    for i in range(kernels.shape[0]):
        b0 = a0*kernels[i, 0, 0] + a1*kernels[i, 1, 0] + a2*kernels[i, 2, 0]
        b1 = a0*kernels[i, 0, 1] + a1*kernels[i, 1, 1] + a2*kernels[i, 2, 1]
        b2 = a0*kernels[i, 0, 2] + a1*kernels[i, 1, 2] + a2*kernels[i, 2, 2]
        s = b0 + b1 + b2
        if s <= 0.0 or min(b0, b1, b2) < -1e-10:
            return -1e100
        a0, a1, a2 = max(0.0, b0)/s, max(0.0, b1)/s, max(0.0, b2)/s
        score += math.log(s)
    return score


def markov_loglikelihood(x, name, p):
    """log[pi * prod(expm(D0*t_i)*D1) * ones], с нормировкой."""
    try:
        d0, d1, pi = markov_matrices(name, p)
        eig, vectors = np.linalg.eig(d0)
        if np.linalg.cond(vectors) > 1e10:
            return -1e100
        w = np.linalg.solve(vectors, d1)
        exponents = np.exp(np.outer(x, eig))
        kernels = np.einsum("ik,tk,kj->tij", vectors, exponents, w, optimize=True)
        if np.max(np.abs(np.imag(kernels))) > 1e-7:
            return -1e100
        kernels = np.ascontiguousarray(np.real(kernels), dtype=np.float64)
        if len(pi) == 2:
            return float(forward_likelihood_2(kernels, float(pi[0]), float(pi[1])))
        return float(forward_likelihood_3(kernels, float(pi[0]), float(pi[1]), float(pi[2])))
    except (np.linalg.LinAlgError, ValueError, FloatingPointError, OverflowError):
        return -1e100


def rate_guesses(name, x):
    """Стартовые приближения из статистик ТРАССЫ, не из файла генерации."""
    rate = 1.0 / float(np.mean(x))
    if name == "mmpp_2":
        return [rate * np.array([0.4, 2.0, 0.12, 0.12]),
                rate * np.array([0.2, 3.0, 0.05, 0.10])]
    if name == "mmpp_3":
        return [rate * np.array([0.25, 0.9, 2.2, 0.12, 0.12, 0.12]),
                rate * np.array([0.18, 1.4, 2.8, 0.06, 0.09, 0.08])]
    if name == "map_2":
        return [rate * np.array([0.12, 0.12, 0.4, 0.15, 0.15, 1.6]),
                rate * np.array([0.04, 0.06, 0.2, 0.14, 0.12, 2.3])]
    raise ValueError(name)


def fit_model(x, name, maxiter=110):
    """Максимизация правдоподобия; возвращает (logL, параметры, BIC)."""
    n = len(x)
    if name == "gamma_renewal":
        shape, loc, scale = gamma_dist.fit(x, floc=0)
        param = [shape, scale]
        ll = float(np.sum(gamma_dist.logpdf(x, shape, loc=0, scale=scale)))
    elif name == "weibull_renewal":
        shape, loc, scale = weibull_min.fit(x, floc=0)
        param = [shape, scale]
        ll = float(np.sum(weibull_min.logpdf(x, shape, loc=0, scale=scale)))
    else:
        rate = 1.0 / float(np.mean(x))
        bounds = [(math.log(rate * 0.002), math.log(rate * 60.0))]
        bounds *= len(PARAMETER_NAMES[name])

        def objective(z):
            return -markov_loglikelihood(x, name, np.exp(z))

        best = None
        for guess in rate_guesses(name, x):
            result = minimize(objective, np.log(guess), method="L-BFGS-B",
                              bounds=bounds, options={"maxiter": maxiter, "ftol": 2e-9,
                                                       "maxls": 20})
            if best is None or result.fun < best.fun:
                best = result
        param = np.exp(best.x).tolist()
        ll = -float(best.fun)
    if not math.isfinite(ll) or ll < -1e90:
        raise RuntimeError(f"Не удалось подобрать параметры модели {name}")
    bic = len(PARAMETER_NAMES[name]) * math.log(n) - 2 * ll
    return {"log_likelihood": ll, "bic": bic,
            "parameters_estimated_from_trace": {
                key: float(value) for key, value in zip(PARAMETER_NAMES[name], param)
            }}


def ljung_box_test(x, lags=10):
    """H0: корреляции интервалов до заданного лага равны нулю.

    Большое p НЕ доказывает независимость, а лишь не отвергает её.
    """
    n = len(x)
    lag_count = min(lags, n - 2)
    v = x - np.mean(x)
    den = float(np.dot(v, v))
    r = np.asarray([float(np.dot(v[:-j], v[j:]) / den)
                    for j in range(1, lag_count + 1)])
    q = n * (n + 2) * float(np.sum(r**2 / (n - np.arange(1, lag_count + 1))))
    return {"lags": lag_count, "statistic": q, "p_value": float(chi2.sf(q, lag_count))}


def fitted_ks_statistic(x, model):
    """Статистика KS для распределения с параметрами, подобранными по x."""
    d = gamma_dist if model == "gamma_renewal" else weibull_min
    shape, _, scale = d.fit(x, floc=0)
    return float(kstest(x, d.cdf, args=(shape, 0, scale)).statistic), (shape, scale)


def bootstrap_ks(x, model, repeats=99, seed=2026):
    """Параметрический бутстрэп корректирует KS при оценке параметров."""
    d = gamma_dist if model == "gamma_renewal" else weibull_min
    observed, (shape, scale) = fitted_ks_statistic(x, model)
    rng = np.random.default_rng(seed)
    exceedances = 0
    for _ in range(repeats):
        sample = d.rvs(shape, loc=0, scale=scale, size=len(x), random_state=rng)
        current, _ = fitted_ks_statistic(sample, model)
        exceedances += (current >= observed)
    return {"statistic": observed, "bootstrap_p": (exceedances + 1) / (repeats + 1),
            "bootstrap_repeats": repeats}


def sequential_decision(x, maxiter=110, independence_alpha=0.05,
                        ks_alpha=0.05, min_delta_bic=10.0,
                        bootstrap_repeats=99, bootstrap_seed=2026):
    """Проверка H1 -> H5 с остановкой при первом выполненном условии."""
    test = ljung_box_test(x)
    independent_not_rejected = test["p_value"] >= independence_alpha
    fits, checks = {}, []

    def fit(name):
        if name not in fits:
            fits[name] = fit_model(x, name, maxiter=maxiter)
        return fits[name]

    def append(hid, model, accepted, rationale):
        checks.append({"hypothesis": hid, "model": model,
                       "accepted": bool(accepted), "reason": rationale})

    # H1: Gamma
    if independent_not_rejected:
        g = fit("gamma_renewal")["bic"]
        w = fit("weibull_renewal")["bic"]
        g_ks = bootstrap_ks(x, "gamma_renewal", bootstrap_repeats, bootstrap_seed)
        if g_ks["bootstrap_p"] >= ks_alpha and w - g >= min_delta_bic:
            append("H1", "gamma_renewal", True,
                   f"p_LB={test['p_value']:.4g}, p_KS={g_ks['bootstrap_p']:.4g}, "
                   f"BIC(Weibull)-BIC(Gamma)={w-g:.2f}")
            return "gamma_renewal", checks, fits, test
        append("H1", "gamma_renewal", False,
               f"p_LB={test['p_value']:.4g}, p_KS={g_ks['bootstrap_p']:.4g}, "
               f"ΔBIC(Weibull-Gamma)={w-g:.2f}")
    else:
        append("H1", "gamma_renewal", False,
               f"обнаружена зависимость интервалов (p_LB={test['p_value']:.4g})")

    # H2: Weibull
    if independent_not_rejected:
        g = fit("gamma_renewal")["bic"]
        w = fit("weibull_renewal")["bic"]
        w_ks = bootstrap_ks(x, "weibull_renewal", bootstrap_repeats, bootstrap_seed+1)
        if w_ks["bootstrap_p"] >= ks_alpha and g - w >= min_delta_bic:
            append("H2", "weibull_renewal", True,
                   f"p_LB={test['p_value']:.4g}, p_KS={w_ks['bootstrap_p']:.4g}, "
                   f"BIC(Gamma)-BIC(Weibull)={g-w:.2f}")
            return "weibull_renewal", checks, fits, test
        append("H2", "weibull_renewal", False,
               f"p_LB={test['p_value']:.4g}, p_KS={w_ks['bootstrap_p']:.4g}, "
               f"ΔBIC(Gamma-Weibull)={g-w:.2f}")
    else:
        append("H2", "weibull_renewal", False,
               f"обнаружена зависимость интервалов (p_LB={test['p_value']:.4g})")

    # H3-H5: сравнение марковских моделей
    if independent_not_rejected:
        for hid, model in (("H3", "mmpp_2"), ("H4", "mmpp_3"), ("H5", "map_2")):
            append(hid, model, False,
                   "тест не выявил зависимости; ни одна рекуррентная гипотеза уверенно не принята")
        return None, checks, fits, test

    markov_names = ("mmpp_2", "mmpp_3", "map_2")
    for name in markov_names:
        fit(name)
    for hid, model in (("H3", "mmpp_2"), ("H4", "mmpp_3"), ("H5", "map_2")):
        diff = min(fits[other]["bic"] for other in markov_names if other != model) - fits[model]["bic"]
        accept = diff >= min_delta_bic
        append(hid, model, accept,
               f"BIC(лучшая альтернативная марковская)-BIC({model})={diff:.2f}; "
               f"порог {min_delta_bic}")
        if accept:
            return model, checks, fits, test
    return None, checks, fits, test


def analyze_csv(path, maxiter=110, bootstrap_repeats=99):
    x = load_intervals(path)
    best, checks, fits, lb = sequential_decision(
        x, maxiter=maxiter, bootstrap_repeats=bootstrap_repeats)
    sorted_models = sorted(fits, key=lambda name: fits[name]["bic"])
    runner_up = sorted_models[1] if len(sorted_models) > 1 else None
    delta = (float(fits[runner_up]["bic"] - fits[sorted_models[0]]["bic"])
             if runner_up else None)
    return {"file": path.name, "statistics": diagnostic_statistics(x),
            "best_model": best, "second_model": runner_up,
            "delta_bic": delta,
            "interpretation": "последовательная проверка приняла гипотезу" if best
                              else "нет достаточно убедительной гипотезы",
            "ljung_box": lb, "sequential_checks": checks, "all_fits": fits}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path,
                        default=Path(__file__).resolve().parent / "traces",
                        help="папка с CSV; имя файла не используется при определении модели")
    parser.add_argument("--output", type=Path, default=None,
                        help="необязательный JSON с протоколом проверок H1→H5")
    parser.add_argument("--maxiter", type=int, default=110,
                        help="лимит итераций подбора параметров марковских моделей")
    parser.add_argument("--bootstrap", type=int, default=99,
                        help="число повторов бутстрэп-KS (по умолчанию 99)")
    args = parser.parse_args()
    if args.bootstrap < 19:
        parser.error("для KS-бутстрэпа нужно не менее 19 повторов")
    sources = sorted(args.traces.glob("*.csv"))
    if not sources:
        parser.error("В указанной папке нет CSV-файлов")
    results = []
    print("Последовательная проверка H1→H2→H3→H4→H5; "
          "параметры генератора НЕ читаются.", flush=True)
    for path in sources:
        result = analyze_csv(path, args.maxiter, args.bootstrap)
        results.append(result)
        s = result["statistics"]
        model = result["best_model"] or "НЕОПРЕДЕЛЕННО"
        print(f"\n{path.name}: n={s['events']}, r1={s['lag1_correlation']:+.3f}, "
              f"p_LB={result['ljung_box']['p_value']:.4g}", flush=True)
        for step in result["sequential_checks"]:
            label = "ПРИНЯТА" if step["accepted"] else "отклонена"
            print(f"  {step['hypothesis']} ({step['model']}): {label}; {step['reason']}", flush=True)
        print(f"  ИТОГ: {model}", flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2)+"\n",
                               encoding="utf-8")
        print("Результаты сохранены:", args.output)


if __name__ == "__main__":
    main()
