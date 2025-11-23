import numpy as np
import altair as alt
import polars as pl


def plot_paths_altair_polars(paths: np.ndarray, n_to_plot: int = 20):
    """
    Visualiserer de første n_to_plot paths fra en Monte Carlo simulation
    med Altair, hvor databehandlingen sker i Polars.
    """

    n_to_plot = min(n_to_plot, paths.shape[0])

    # paths[:n_to_plot] = shape (n_to_plot, n_steps+1)
    # vi transponerer, så:
    # --> rows = steps
    # --> columns = path_i
    arr = paths[:n_to_plot].T

    # Lav Polars DataFrame med kolonner path_0, path_1, ...
    df = pl.DataFrame(arr, orient="row", schema=[f"path_{i}" for i in range(n_to_plot)])

    # Tilføj en step-kolonne
    df = df.with_row_index("step")

    # Gør data long-format (step, path, value)
    df_long = df.unpivot(
      index="step",
        on=[f"path_{i}" for i in range(n_to_plot)],
        variable_name="path",
        value_name="value"
)
    # Altair skal bruge Pandas:
    pdf = df_long.to_pandas()

    chart = (
        alt.Chart(pdf)
        .mark_line()
        .encode(
            x="step:Q",
            y="value:Q",
            color=alt.Color("path:N", legend=None)
        )
        .properties(
            width=800,
            height=400,
            title=f"De første {n_to_plot} simulerede stier"
        )
    )

    return chart


