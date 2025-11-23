from abc import ABC, abstractmethod
import numpy as np
import polars as pl
import altair as alt



class StochasticProcess(ABC):

    @abstractmethod
    def simulate_paths(self, n_paths: int, n_steps: int, dt: float):
        """
        Returnerer en matrix af størrelse (n_paths, n_steps+1)
        med simulerede scenarier.
        """
        pass

class GBM(StochasticProcess):
    def __init__(self, s0: float, mu: float, sigma: float):
        self.s0 = s0
        self.mu = mu
        self.sigma = sigma

    def simulate_paths(self, n_paths: int, n_steps: int, dt: float):
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = self.s0

        drift = (self.mu - 0.5 * self.sigma**2) * dt
        vol = self.sigma * np.sqrt(dt)

        # Træk alle shocks på én gang – hurtigere
        shocks = np.random.normal(0, 1, size=(n_paths, n_steps))

        for t in range(n_steps):
            paths[:, t+1] = paths[:, t] * np.exp(drift + vol * shocks[:, t])

        return paths


class MonteCarloSimulator:
    def __init__(self, model: StochasticProcess):
        self.model = model

    def run(self, n_paths: int, n_steps: int, dt: float):
        return self.model.simulate_paths(n_paths, n_steps, dt)


class PathStatistics:
    def __init__(self, paths: np.ndarray):
        """
        paths: np.ndarray med shape (n_paths, n_steps+1)
        """
        self.paths = paths
        self.n_paths, self.n_steps = paths.shape

    def compute_statistics(self, p_low=5, p_high=95):
        """
        Beregn mean, median og percentiler for hver tid.
        """

        mean = self.paths.mean(axis=0)
        median = np.median(self.paths, axis=0)
        low = np.percentile(self.paths, p_low, axis=0)
        high = np.percentile(self.paths, p_high, axis=0)

        return {
            "mean": mean,
            "median": median,
            f"p{p_low}": low,
            f"p{p_high}": high,
        }

    def plot(self, p_low=5, p_high=95):
        """
        Altair-plot af mean og percentilbånd.
        """

        stats = self.compute_statistics(p_low=p_low, p_high=p_high)

        df = pl.DataFrame({
            "step": np.arange(self.n_steps),
            "mean": stats["mean"],
            "median": stats["median"],
            "low": stats[f"p{p_low}"],
            "high": stats[f"p{p_high}"],
        })

        pdf = df.to_pandas()

        # Percentilbånd
        band = (
            alt.Chart(pdf)
            .mark_area(opacity=0.3)
            .encode(
                x="step:Q",
                y="low:Q",
                y2="high:Q"
            )
        )

        # Mean-linje
        mean_line = (
            alt.Chart(pdf)
            .mark_line(color="black")
            .encode(
                x="step:Q",
                y="mean:Q"
            )
        )

        # Median-linje (valgfrit)
        median_line = (
            alt.Chart(pdf)
            .mark_line(color="gray", strokeDash=[4,2])
            .encode(
                x="step:Q",
                y="median:Q"
            )
        )

        chart = (band + mean_line + median_line).properties(
            width=800,
            height=400,
            title="Mean og percentilbånd"
        )

        return chart


