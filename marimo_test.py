import marimo

__generated_with = "0.17.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from stochastic_models import GBM, MonteCarloSimulator, PathStatistics
    from path_plot import plot_paths_altair_polars
    return (
        GBM,
        MonteCarloSimulator,
        PathStatistics,
        mo,
        plot_paths_altair_polars,
    )


@app.cell
def _(mo):
    n_paths_input = mo.ui.number(value=1000)
    mo.hstack([n_paths_input], justify='center')
    return (n_paths_input,)


@app.cell
def _(mo):
    vol_slider = mo.ui.slider(start=0, stop=1, step=0.01, value=0.05, show_value=True, label='sigma')
    my_slider = mo.ui.slider(start=-1, stop =1, step=0.01, value=0, show_value=True, label='mu' )
    mo.hstack([my_slider,vol_slider], justify='center')

    return my_slider, vol_slider


@app.cell
def _(
    GBM,
    MonteCarloSimulator,
    my_slider,
    n_paths_input,
    plot_paths_altair_polars,
    vol_slider,
):
    gbm = GBM(100, my_slider.value, vol_slider.value)
    sim = MonteCarloSimulator(gbm)

    paths = sim.run(n_paths=n_paths_input.value, n_steps=252, dt=1/252)

    plot_paths_altair_polars(paths, n_to_plot=30)
    return (paths,)


@app.cell
def _(PathStatistics, paths):
    stats = PathStatistics(paths)
    stats.plot(p_low=5, p_high=95)
    return


if __name__ == "__main__":
    app.run()
