"""Генерация пяти моделей потоков событий.
Во всех CSV:
    event_id            -- номер события (начиная с 1)
    event_time          -- время события от t=0
    interarrival_time   -- интервал с момента предыдущего события (от 0 для первого)
    state_after_event   -- состояние после события; пустое для рекуррентных потоков
"""

import argparse
import csv
import math
from pathlib import Path

import numpy as np


# Параметры генерации намеренно хранятся отдельно от кода.
# Анализатор НЕ загружает этот файл и оценивает параметры только по трассам.
MODEL_NAMES = ("gamma_renewal", "weibull_renewal", "mmpp_2", "mmpp_3", "map_2")


def load_parameters(path):
    """Читает единственный источник параметров из JSON и проверяет структуру."""
    import json
    with Path(path).open(encoding="utf-8") as fh:
        params = json.load(fh)
    if not isinstance(params, dict) or set(params) != set(MODEL_NAMES):
        raise ValueError("JSON должен содержать параметры ровно пяти моделей: "
                         + ", ".join(MODEL_NAMES))
    keys = {
        "gamma_renewal": {"shape", "scale"},
        "weibull_renewal": {"shape", "scale"},
        "mmpp_2": {"lambda_1", "lambda_2", "q_12", "q_21"},
        "mmpp_3": {"lambda_1", "lambda_2", "lambda_3", "q_12", "q_23", "q_31"},
        "map_2": {"q_12", "q_21", "r_11", "r_12", "r_21", "r_22"},
    }
    for name, required in keys.items():
        if set(params[name]) != required:
            raise ValueError(f"{name}: ожидаются поля {sorted(required)}")
        for key, value in params[name].items():
            if type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name}.{key}: требуется положительное число")
    return params


# Для обратной совместимости с небольшими тестами оставлен этот атрибут.
# При использовании как скрипта параметры перечитываются из --config.
PARAMS = load_parameters(Path(__file__).resolve().parent / "model_parameters.json")


def stationary_two_state(rate_12, rate_21):
    """Стационарные доли времени в фазах CTMC (не доли событий)."""
    denom = rate_12 + rate_21
    return np.array([rate_21 / denom, rate_12 / denom], dtype=float)


def gamma_renewal(n, rng, p):
    """Независимые интервалы T_i ~ Gamma(shape, scale)."""
    intervals = rng.gamma(shape=p["shape"], scale=p["scale"], size=n)
    times = np.cumsum(intervals)
    return times, intervals, None


def weibull_renewal(n, rng, p):
    """Независимые интервалы T_i ~ Weibull(shape, scale)."""
    intervals = p["scale"] * rng.weibull(a=p["shape"], size=n)
    times = np.cumsum(intervals)
    return times, intervals, None


def simulate_markov_arrivals(n, rng, stationary, transitions):
    """Событийное моделирование непрерывной марковской фазы.

    transitions[state] = [(rate, next_state, is_arrival), ...].
    Возможные 'часы' в состоянии независимы и экспоненциальны.
    Следующее изменение наступает через Exp(сумма всех скоростей).
    Вид изменения выбирается пропорционально его скорости.

    Если is_arrival=False, фаза меняется, но событие НЕ записывается.
    Если is_arrival=True, фиксируется событие; фаза может сохраниться
    или поменяться, в зависимости от next_state.
    """
    compiled = []
    for options in transitions:
        rates = np.array([item[0] for item in options], dtype=float)
        if np.any(rates < 0) or rates.sum() <= 0:
            raise ValueError("Скорости переходов должны быть неотрицательными")
        compiled.append((float(rates.sum()), np.cumsum(rates), options))

    state = int(rng.choice(len(stationary), p=stationary))
    times = np.empty(n, dtype=float)
    phases = np.empty(n, dtype=np.int8)
    clock = 0.0
    saved = 0

    while saved < n:
        total, cumulative, options = compiled[state]
        clock += float(rng.exponential(scale=1.0 / total))
        choice = int(np.searchsorted(cumulative, rng.random() * total, side="right"))
        _, next_state, is_arrival = options[choice]
        state = next_state
        if is_arrival:
            times[saved] = clock
            phases[saved] = state + 1
            saved += 1

    intervals = np.diff(np.concatenate(([0.0], times)))
    return times, intervals, phases


def mmpp_2(n, rng, p):
    """Двухфазный MMPP: переходы фаз не являются событиями."""
    pi = stationary_two_state(p["q_12"], p["q_21"])
    transitions = [
        [(p["lambda_1"], 0, True), (p["q_12"], 1, False)],
        [(p["lambda_2"], 1, True), (p["q_21"], 0, False)],
    ]
    return simulate_markov_arrivals(n, rng, pi, transitions)


def mmpp_3(n, rng, p):
    """Трёхфазный MMPP с циклом 1 -> 2 -> 3 -> 1 (шесть параметров)."""
    holding = np.array([1.0 / p["q_12"], 1.0 / p["q_23"],
                        1.0 / p["q_31"]])
    pi = holding / holding.sum()
    transitions = [
        [(p["lambda_1"], 0, True), (p["q_12"], 1, False)],
        [(p["lambda_2"], 1, True), (p["q_23"], 2, False)],
        [(p["lambda_3"], 2, True), (p["q_31"], 0, False)],
    ]
    return simulate_markov_arrivals(n, rng, pi, transitions)


def map_matrices(p):
    """Матрицы MAP: D0 -- без события, D1 -- с событием."""
    d0 = np.array([
        [-(p["q_12"] + p["r_11"] + p["r_12"]), p["q_12"]],
        [p["q_21"], -(p["q_21"] + p["r_21"] + p["r_22"])],
    ])
    d1 = np.array([
        [p["r_11"], p["r_12"]],
        [p["r_21"], p["r_22"]],
    ])
    return d0, d1


def map_2(n, rng, p):
    """MAP(2), допускающий события как со сменой фазы, так и без неё."""
    d0, d1 = map_matrices(p)
    q = d0 + d1
    if not np.allclose(q.sum(axis=1), 0.0):
        raise ValueError("Некорректные матрицы MAP: суммы строк D0+D1 не ноль")

    pi = stationary_two_state(q[0, 1], q[1, 0])
    transitions = [
        [(p["q_12"], 1, False), (p["r_11"], 0, True),
         (p["r_12"], 1, True)],
        [(p["q_21"], 0, False), (p["r_21"], 0, True),
         (p["r_22"], 1, True)],
    ]
    return simulate_markov_arrivals(n, rng, pi, transitions)


def theory(name, p):
    """Теоретические средние интервалы и средние интенсивности."""
    if name == "gamma_renewal":
        mean = p["shape"] * p["scale"]
        pi = None
    elif name == "weibull_renewal":
        mean = p["scale"] * math.gamma(1.0 + 1.0 / p["shape"])
        pi = None
    elif name == "mmpp_2":
        pi = stationary_two_state(p["q_12"], p["q_21"])
        lam = float(np.dot(pi, [p["lambda_1"], p["lambda_2"]]))
        mean = 1.0 / lam
    elif name == "mmpp_3":
        holding = np.array([1/p["q_12"], 1/p["q_23"], 1/p["q_31"]])
        pi = holding / holding.sum()
        lam = float(np.dot(pi, [p["lambda_1"], p["lambda_2"], p["lambda_3"]]))
        mean = 1.0 / lam
    elif name == "map_2":
        d0, d1 = map_matrices(p)
        q = d0 + d1
        pi = stationary_two_state(q[0, 1], q[1, 0])
        lam = float(pi @ d1 @ np.ones(2))
        mean = 1.0 / lam
    else:
        raise KeyError(name)
    return mean, None if pi is None else pi.tolist()


def save_csv(path, times, intervals, states):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "event_id",
            "event_time",
            "interarrival_time"
        ])

        for i, (t, dt) in enumerate(
            zip(times, intervals), start=1
        ):
            writer.writerow([
                i,
                f"{t:.12f}",
                f"{dt:.12f}"
            ])


def get_statistics(name, times, intervals, states, params, seed):
    if len(times) == 0 or np.any(intervals <= 0):
        raise ValueError(f"{name}: события должны иметь положительные интервалы")
    if np.any(np.diff(times) <= 0):
        raise ValueError(f"{name}: время событий не строго возрастает")
    if not np.allclose(np.diff(np.concatenate(([0.0], times))),
                       intervals, rtol=1e-10, atol=1e-10):
        raise ValueError(f"{name}: интервалы не согласованы со временем")

    target, pi = theory(name, params)
    empirical = float(np.mean(intervals))
    cv2 = float(np.var(intervals) / empirical ** 2)
    corr = float(np.corrcoef(intervals[:-1], intervals[1:])[0, 1]) \
           if len(intervals) >= 3 else None
    result = {
        "model": name,
        "events": len(times),
        "seed": seed,
        "parameters": params,
        "stationary_phase_probabilities": pi,
        "theoretical_mean_interval": target,
        "theoretical_mean_event_rate": 1.0 / target,
        "empirical_mean_interval": empirical,
        "empirical_mean_event_rate": len(times) / float(times[-1]),
        "relative_mean_error_pct": 100.0 * (empirical / target - 1.0),
        "empirical_interval_cv_squared": cv2,
        "empirical_lag1_interval_correlation": corr,
        "last_event_time": float(times[-1]),
    }
    if name == "map_2":
        d0, d1 = map_matrices(params)
        result["D0"] = d0.tolist()
        result["D1"] = d1.tolist()
    return result


def create_plot(traces, path):
    """Необязательная иллюстрация: эмпирические функции распределения."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {
        "gamma_renewal": "Gamma renewal",
        "weibull_renewal": "Weibull renewal",
        "mmpp_2": "MMPP(2)",
        "mmpp_3": "MMPP(3)",
        "map_2": "MAP(2)",
    }
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, intervals in traces.items():
        sample = np.sort(intervals)
        y = np.arange(1, len(sample) + 1) / len(sample)
        ax.plot(sample, y, label=labels[name], lw=1.8)
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Interarrival time")
    ax.set_ylabel("Empirical CDF")
    ax.set_title("Distribution of interarrival times: five event-stream models")
    ax.grid(alpha=0.22)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).resolve().parent / "model_parameters.json",
                        help="параметры генерации (JSON)")
    parser.add_argument("--n", type=int, default=10000,
                        help="число событий для каждой модели (по умолчанию 10000)")
    parser.add_argument("--seed", type=int, default=2026,
                        help="базовый seed для воспроизводимости (по умолчанию 2026)")
    parser.add_argument("--outdir", type=Path, default=Path(__file__).resolve().parent,
                        help="каталог результатов")
    parser.add_argument("--plots", action="store_true",
                        help="дополнительно построить рисунок эмпирических CDF")
    args = parser.parse_args()
    params = load_parameters(args.config)
    if args.n < 3:
        parser.error("--n должен быть не меньше 3")

    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "traces").mkdir(exist_ok=True)
    simulators = {
        "gamma_renewal": gamma_renewal,
        "weibull_renewal": weibull_renewal,
        "mmpp_2": mmpp_2,
        "mmpp_3": mmpp_3,
        "map_2": map_2,
    }
    ecdf_data = {}
    print(f"Generating {args.n:,} events for each of 5 models:")
    for index, name in enumerate(MODEL_NAMES):
        model_seed = args.seed + 1000 * index
        rng = np.random.default_rng(model_seed)
        p = params[name]
        times, intervals, states = simulators[name](args.n, rng, p)
        save_csv(args.outdir / "traces" / f"{name}.csv", times, intervals, states)
        stats = get_statistics(name, times, intervals, states, p, model_seed)
        ecdf_data[name] = intervals
        print(f"  {name:17s}: theor E[dt]={stats['theoretical_mean_interval']:.5f}, "
              f"sim E[dt]={stats['empirical_mean_interval']:.5f}, "
              f"lag1 corr={stats['empirical_lag1_interval_correlation']:+.4f}")

    if args.plots:
        try:
            create_plot(ecdf_data, args.outdir / "figures" / "interarrival_ecdf.png")
        except ImportError as exc:
            print(f"Cannot plot: {exc}; install matplotlib")
    print("Done. Files saved under", args.outdir)


if __name__ == "__main__":
    main()
