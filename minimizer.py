import warnings
from functools import lru_cache
from typing import Generic, Self, Type, TypeVar

import numpy as np
import numpy.typing as npt
import plotly.graph_objs as go
import plotly_express as px
import streamlit as st
from joblib import Parallel, delayed
from scipy.optimize import OptimizeResult, minimize
from tqdm_joblib import tqdm_joblib

warnings.filterwarnings("ignore")


def wandermond_matrix(gammas: npt.NDArray, lambdas: npt.NDArray, n: int) -> npt.NDArray:
    return np.vstack([gammas * lambdas**i for i in range(n)]).T


def approximation_elemnet(
    gammas: npt.NDArray, lambdas: npt.NDArray, n: int
) -> npt.NDArray:
    return np.sum(wandermond_matrix(gammas, lambdas, n), axis=0)


def functional_J(
    f: npt.NDArray, gammas: npt.NDArray, lambdas: npt.NDArray, n: int
) -> float:
    diff = f - approximation_elemnet(gammas, lambdas, n)
    return diff @ diff


class MinimizeResult:
    def __init__(self, result: OptimizeResult, k: int, f: npt.NDArray):
        self.minimum: float = result["fun"]
        x: npt.NDArray = result["x"]
        self.gammas: npt.NDArray = x[:k]
        self.lambdas: npt.NDArray = x[k:]
        self.k = k
        self.f = f

    def __str__(self):
        cond = np.linalg.cond(self.A)
        return (
            f"k={self.k}\n"
            f"min={self.minimum}\n"
            f"gammas={self.gammas}\n"
            f"lambdas={self.lambdas}\n"
            f"cond_number={cond}\n"
        )

    @property
    def A(self):
        return np.vstack([self.lambdas**i for i in range(self.f.size)]).T

    @property
    def ATA(self) -> npt.NDArray:
        return self.A @ self.A.T

    def __eq__(self, other: Self) -> bool:
        return abs(self.minimum - other.minimum) < 1e-5

    def __gt__(self, other: Self) -> bool:
        return self.minimum - other.minimum > 1e-5

    def __lt__(self, other: Self) -> bool:
        return self.minimum - other.minimum < -1e-5

    def __ge__(self, other: Self) -> bool:
        return self.__gt__(other) or self.__eq__(other)

    def __le__(self, other: Self) -> bool:
        return self.__lt__(other) or self.__eq__(other)


class Experiment:
    def __init__(
        self,
        f: npt.NDArray,
        n: int,
        k: int,
        lambdas: list[float] = [],
        gammas: list[float] = [],
    ):
        assert f.size == n
        assert n - 1 > k
        self.f = f
        self.n = n
        self.k = k
        self._lambdas = lambdas
        self._gammas = gammas
        self.result: MinimizeResult | None = None

    @property
    def x0(self) -> list[float]:
        if len(self._lambdas) > 0 and len(self._gammas) > 0:
            return self._gammas + [0.0] + self._lambdas + [-self.k * 1.1]
        return [0, -1.1]

    def objective(self, x: npt.NDArray) -> float:
        assert x.size == 2 * self.k
        gammas, lambdas = x[: self.k], x[self.k :]

        for i in range(self.k):
            for j in range(i + 1, self.k):
                distance = np.abs(lambdas[i] - lambdas[j])
                if distance < 1e-6:
                    return 1e6
        return functional_J(self.f, gammas, lambdas, self.n)


E = TypeVar("E", bound=Experiment)


class MinimizeExperiment(Experiment):
    @lru_cache
    def minimize(self, **kwargs) -> MinimizeResult:
        b_gamma = [-1e9, 1e9]
        b_lambda = [-1e9, -1]
        bounds = [b_gamma] * self.k + [b_lambda] * self.k
        solution = minimize(self.objective, self.x0, bounds=bounds, **kwargs)
        self.result = MinimizeResult(solution, self.k, self.f)
        return self.result


class ExperimentFabric(Generic[E]):
    def __init__(self, f: npt.NDArray, n: int, base_class: Type[E]):
        assert f.size == n
        self.f = f
        self.n = n
        self.current = 0
        self.base_class = base_class
        self.experiment_step = None

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> E:
        if self.current < self.n - 2:
            self.current += 1
            if (self.experiment_step is not None) and (
                self.experiment_step.result is not None
            ):
                self.experiment_step = self.base_class(
                    self.f,
                    self.n,
                    self.current,
                    self.experiment_step.result.lambdas.tolist(),
                    self.experiment_step.result.gammas.tolist(),
                )
            else:
                self.experiment_step = self.base_class(self.f, self.n, self.current)
            return self.experiment_step
        self.current = 0
        raise StopIteration


class ExperimentGenerator(Generic[E]):
    def __init__(self, max_iter: int, vector_size: int, base_class: Type[E]):
        self.max_iter = max_iter
        self.vector_size = vector_size
        self.current = 0
        self.base_class = base_class

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> ExperimentFabric[E]:
        if self.current < self.max_iter:
            self.current += 1
            return ExperimentFabric(
                self.f, n=self.vector_size, base_class=self.base_class
            )
        self.current = 0
        raise StopIteration

    @property
    def f(self) -> npt.NDArray:
        raise NotImplementedError

    def __str__(self) -> str:
        return self.__class__.__name__


class RandomExperimentGenerator(ExperimentGenerator[E]):
    @property
    def f(self) -> npt.NDArray:
        return np.round(
            np.random.randint(-3, 3, self.vector_size)
            + np.random.random(self.vector_size),
            2,
        )


class ExponentialExperimentGenerator(ExperimentGenerator[E]):
    @property
    def f(self) -> npt.NDArray:
        l_true = -np.random.exponential(1, self.vector_size)  # отрицательные l_i
        g_coeffs = np.random.normal(0, 2, self.vector_size)  # коэффициенты g_i

        # Создаем матрицу базисных функций
        t = np.arange(self.vector_size)
        basis_matrix = np.array([lt**t for lt in l_true]).T

        # Генерируем целевой вектор f
        f_true = basis_matrix @ g_coeffs
        f_noisy = f_true + np.random.normal(0, 0.5, self.vector_size)
        return f_noisy


class ExperimentRunner(Generic[E]):
    def __init__(
        self,
        experiment_generator: ExperimentGenerator[E],
        backend: str = "multiprocessing",
        n_jobs: int = -1,
    ):
        self.experiment_generator = experiment_generator
        self.results: list[list[MinimizeResult]] = []
        self.statistic: list[bool] = []
        self.backend = backend
        self.n_jobs = n_jobs

    def step(
        self, experiment: E, verbose: bool = True, **kwargs
    ) -> list[MinimizeResult]:
        experiment_results: list[MinimizeResult] = [
            step.minimize(**kwargs) for step in experiment
        ]
        if verbose:
            self.print_experiment(experiment_results)
            self.plot_experiment(experiment_results)
            self.plot_experiment_parameters(experiment_results)
        return experiment_results, self.check_monotonicity(experiment_results, verbose)

    def run(self, verbose: bool = True, **kwargs):
        self.results: list[list[MinimizeResult]] = []
        self.statistic: list[bool] = []
        with tqdm_joblib(
            desc="Experiment runs", total=self.experiment_generator.max_iter
        ):
            result = Parallel(n_jobs=self.n_jobs, backend=self.backend)(
                delayed(self.step)(experiment, verbose, **kwargs)
                for experiment in self.experiment_generator
            )
        self.results, self.statistic = zip(*result)

    def plot_experiment(self, experiment_results: list[MinimizeResult]):
        index = [step.k for step in experiment_results]
        function_result = [step.minimum for step in experiment_results]
        fig = go.Figure(go.Scatter(x=index, y=function_result, name="min J"))
        fig.update_layout(
            title="min J",
            xaxis_title="Значение 'k'",
            yaxis_title="min J",
            plot_bgcolor="white",
            showlegend=True,
        )
        fig.update_xaxes(
            mirror=True,
            ticks="outside",
            showline=True,
            linecolor="black",
            gridcolor="lightgrey",
        )
        fig.update_yaxes(
            mirror=True,
            ticks="outside",
            showline=True,
            linecolor="black",
            gridcolor="lightgrey",
        )
        self.show(fig)

    def plot_experiment_parameters(self, experiment_results: list[MinimizeResult]):
        fig1 = go.Figure()
        fig2 = go.Figure()
        for i in range(len(experiment_results)):
            index = [step.k for step in experiment_results[i:]]
            lambdas = [step.lambdas[i] for step in experiment_results[i:]]
            gammas = [step.gammas[i] for step in experiment_results[i:]]
            fig1.add_trace(
                go.Scatter(x=index, y=lambdas, name=f"λ<sub>{i + 1}</sub>"),
            )
            fig2.add_trace(
                go.Scatter(x=index, y=gammas, name=f"γ<sub>{i + 1}</sub>"),
            )
        fig1.update_layout(
            title="Параметры λ",
            plot_bgcolor="white",
            colorway=px.colors.qualitative.Plotly,
        )
        fig2.update_layout(
            title="Параметры γ",
            plot_bgcolor="white",
            colorway=px.colors.qualitative.Plotly,
        )

        fig1.update_xaxes(
            title_text="Значение 'k'",
            mirror=True,
            ticks="outside",
            showline=True,
            linecolor="black",
            gridcolor="lightgrey",
            showgrid=True,
        )

        fig1.update_yaxes(
            title_text="Значение λ",
            mirror=True,
            ticks="outside",
            showline=True,
            linecolor="black",
            gridcolor="lightgrey",
            showgrid=True,
        )

        # Настройки осей для второго subplot (γ)
        fig2.update_xaxes(
            title_text="Значение 'k'",
            mirror=True,
            ticks="outside",
            showline=True,
            linecolor="black",
            gridcolor="lightgrey",
            showgrid=True,
        )

        fig2.update_yaxes(
            title_text="Значение γ",
            mirror=True,
            ticks="outside",
            showline=True,
            linecolor="black",
            gridcolor="lightgrey",
            showgrid=True,
        )
        self.show(fig1)
        self.show(fig2)

    def print_experiment(self, experiment_results: list[MinimizeResult]):
        self.print(f"f={experiment_results[0].f}")
        for step in experiment_results:
            self.print(str(step))
        self.print("\n")

    def check_monotonicity(
        self, experiment_results: list[MinimizeResult], verbose: bool
    ) -> bool:
        """Проверяет, что последовательность минимумов не возрастает"""
        for i in range(1, len(experiment_results)):
            # Учитываем численную погрешность
            if experiment_results[i] > experiment_results[i - 1]:
                if verbose:
                    self.print("Минимум возрастает")
                    self.print("\n")
                return False
        if verbose:
            self.print("Минимум убывает")
            self.print("\n")
        return True

    def analyze_results(self):
        self.print("РЕЗУЛЬТАТЫ ЭКСПЕРИМЕНТА")
        experiments_count = len(self.results)
        self.print(f"Всего испытаний: {experiments_count}")
        corrects = sum(self.statistic)
        violations = experiments_count - corrects
        self.print(f"Нарушений монотонности: {violations}")
        violation_rate = violations / experiments_count
        alpha = 0.05
        if violation_rate < alpha:
            self.print(
                f"✓ Гипотеза об убывании минимума ПОДТВЕРЖДЕНА (уровень нарушений < {alpha})"
            )
        else:
            self.print(
                f"✗ Гипотеза о убывании минимума НЕ ПОДТВЕРЖДЕНА (уровень нарушений ≥ {alpha})"
            )
        self.print("\n")

    @property
    def info(self) -> dict[str, int]:
        experiments_count = len(self.results)
        corrects = sum(self.statistic)
        violations = experiments_count - corrects
        violation_rate = violations / experiments_count
        return {
            "violation_rate": violation_rate,
            "experiments_count": experiments_count,
        }

    def print(self, s):
        print("=" * 100)
        print(s)

    def show(self, fig: go.Figure):
        fig.show()


class StreamlitExperimentRunner(ExperimentRunner):
    def print(self, s):
        st.text(s)

    def show(self, fig: go.Figure):
        st.plotly_chart(fig)


def _main():
    n = st.number_input("$N$", value=10)
    method = st.selectbox("method", ["Nelder-Mead", "L-BFGS-B", "TNC"], index=0)
    runner = StreamlitExperimentRunner(
        RandomExperimentGenerator(
            max_iter=1,
            vector_size=n,
            base_class=MinimizeExperiment,
        ),
        n_jobs=1,
    )

    if st.button("Запустить расчет"):
        runner.run(verbose=True, method=method)


if __name__ == "__main__":
    _main()
