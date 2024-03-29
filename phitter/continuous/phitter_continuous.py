import concurrent.futures
import sys
import typing

import numpy
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats

sys.path.append("../..")
from phitter.continuous.continuous_distributions import ALL_CONTINUOUS_DISTRIBUTIONS
from phitter.continuous.continuous_measures import CONTINUOUS_MEASURES
from phitter.continuous.continuous_statistical_tests import evaluate_continuous_test_anderson_darling
from phitter.continuous.continuous_statistical_tests import evaluate_continuous_test_chi_square
from phitter.continuous.continuous_statistical_tests import evaluate_continuous_test_kolmogorov_smirnov


class PHITTER_CONTINUOUS:
    def __init__(
        self,
        data: list[int | float] | numpy.ndarray,
        num_bins: int | None = None,
        confidence_level=0.95,
        minimum_sse=numpy.inf,
        distributions_to_fit: list[str] | typing.Literal["all"] = "all",
    ):
        if distributions_to_fit != "all":
            not_distributions_ids = [dist for dist in distributions_to_fit if dist not in ALL_CONTINUOUS_DISTRIBUTIONS.keys()]
            if len(not_distributions_ids) > 0:
                raise Exception(f"{not_distributions_ids} not founded in continuous disributions")

        self.data = data
        self.continuous_measures = CONTINUOUS_MEASURES(self.data, num_bins, confidence_level)
        self.confidence_level = confidence_level
        self.minimum_sse = minimum_sse
        self.distribution_results = {}
        self.none_results = {"test_statistic": None, "critical_value": None, "p_value": None, "rejected": None}
        self.distributions_to_fit = list(ALL_CONTINUOUS_DISTRIBUTIONS.keys()) if distributions_to_fit == "all" else distributions_to_fit
        self.sorted_distributions_sse = None
        self.not_rejected_distributions = None
        self.distribution_instances = None

    def test(self, test_function, label: str, distribution):
        validation_test = False
        try:
            test = test_function(distribution, self.continuous_measures)
            if numpy.isnan(test["test_statistic"]) == False and numpy.isinf(test["test_statistic"]) == False and test["test_statistic"] > 0:
                self.distribution_results[label] = {
                    "test_statistic": test["test_statistic"],
                    "critical_value": test["critical_value"],
                    "p_value": test["p-value"],
                    "rejected": test["rejected"],
                }
                validation_test = True
            else:
                self.distribution_results[label] = self.none_results
        except:
            self.distribution_results[label] = self.none_results
        return validation_test

    def process_distribution(self, distribution_name: str) -> tuple[str, dict, typing.Any] | None:
        distribution_class = ALL_CONTINUOUS_DISTRIBUTIONS[distribution_name]

        validate_estimation = True
        sse = 0
        try:
            distribution = distribution_class(self.continuous_measures)
            pdf_values = distribution.pdf(self.continuous_measures.central_values)
            sse = numpy.sum(numpy.power(self.continuous_measures.densities_frequencies - pdf_values, 2.0))
        except:
            validate_estimation = False

        self.distribution_results = {}
        if validate_estimation and distribution.parameter_restrictions() and not numpy.isnan(sse) and not numpy.isinf(sse) and sse < self.minimum_sse:
            v1 = self.test(evaluate_continuous_test_chi_square, "chi_square", distribution)
            v2 = self.test(evaluate_continuous_test_kolmogorov_smirnov, "kolmogorov_smirnov", distribution)
            v3 = self.test(evaluate_continuous_test_anderson_darling, "anderson_darling", distribution)

            if v1 or v2 or v3:
                self.distribution_results["sse"] = sse
                self.distribution_results["parameters"] = distribution.parameters
                self.distribution_results["n_test_passed"] = (
                    +int(self.distribution_results["chi_square"]["rejected"] == False)
                    + int(self.distribution_results["kolmogorov_smirnov"]["rejected"] == False)
                    + int(self.distribution_results["anderson_darling"]["rejected"] == False)
                )
                self.distribution_results["n_test_null"] = (
                    +int(self.distribution_results["chi_square"]["rejected"] == None)
                    + int(self.distribution_results["kolmogorov_smirnov"]["rejected"] == None)
                    + int(self.distribution_results["anderson_darling"]["rejected"] == None)
                )
                return distribution_name, self.distribution_results, distribution
        return None

    def fit(self, n_workers: int = 1):
        if n_workers <= 0:
            raise Exception("n_workers must be greater than 1")

        if n_workers == 1:
            processing_results = [self.process_distribution(distribution_name) for distribution_name in self.distributions_to_fit]
        else:
            executor = concurrent.futures.ProcessPoolExecutor(max_workers=n_workers)
            processing_results = list(executor.map(self.process_distribution, self.distributions_to_fit))

        processing_results = [r for r in processing_results if r is not None]
        self.sorted_distributions_sse = {distribution: results for distribution, results, _ in sorted(processing_results, key=lambda x: (-x[1]["n_test_passed"], x[1]["sse"]))}
        self.not_rejected_distributions = {distribution: results for distribution, results in self.sorted_distributions_sse.items() if results["n_test_passed"] > 0}
        self.distribution_instances = {distribution: instance for distribution, _, instance in processing_results}

    def plot_histogram(
        self,
        plot_title: str,
        plot_xaxis_title: str,
        plot_yaxis_title: str,
        plot_legend_title: str,
        plot_height: int,
        plot_width: int,
        plot_bar_color: str,
        plot_bargap: float,
    ):
        central_values = self.continuous_measures.central_values
        densities_frequencies = self.continuous_measures.densities_frequencies

        fig = go.Figure()
        fig.add_trace(go.Bar(x=central_values, y=densities_frequencies, marker_color=plot_bar_color, name="Data", showlegend=True))
        fig.update_layout(
            height=plot_height,
            width=plot_width,
            title=plot_title,
            xaxis_title=plot_xaxis_title,
            yaxis_title=plot_yaxis_title,
            legend_title=plot_legend_title,
            template="ggplot2",
            legend=dict(orientation="v", yanchor="auto", y=1, xanchor="left", font=dict(size=10), title_font_size=10),
            bargap=plot_bargap,
        )
        fig.show()

    def plot_histogram_distributions_pdf(
        self,
        n_distributions: int,
        n_distributions_visible: int,
        plot_title: str,
        plot_xaxis_title: str,
        plot_yaxis_title: str,
        plot_legend_title: str,
        plot_height: int,
        plot_width: int,
        plot_bar_color: str,
        plot_bargap: float,
    ):
        central_values = self.continuous_measures.central_values
        densities_frequencies = self.continuous_measures.densities_frequencies

        fig = go.Figure()
        fig.add_trace(go.Bar(x=central_values, y=densities_frequencies, marker_color=plot_bar_color, showlegend=False, name="Data"))

        x_plot = numpy.linspace(self.continuous_measures.min, self.continuous_measures.max, 1000)
        for idx, (distribution_name, result) in enumerate(list(self.sorted_distributions_sse.items())[:n_distributions]):
            y_plot = self.distribution_instances[distribution_name].pdf(x_plot)
            distribution_sse = result["sse"]
            is_visible = True if idx + 1 <= n_distributions_visible else "legendonly"
            is_rejected = "✅" if distribution_name in self.not_rejected_distributions else ""
            scatter_name = f"{distribution_name}: {distribution_sse:.4E}{is_rejected}"
            scatter_line = dict(color=px.colors.qualitative.G10[idx], width=2) if idx < len(px.colors.qualitative.G10) else dict(width=3)
            try:
                fig.add_trace(go.Scatter(x=x_plot, y=y_plot, mode="lines", visible=is_visible, name=scatter_name, line=scatter_line))
            except Exception:
                fig.add_trace(go.Scatter(x=x_plot, y=numpy.zeros(len(x_plot)), mode="lines", visible=is_visible, name=scatter_name, line=scatter_line))

        fig.update_layout(
            height=plot_height,
            width=plot_width,
            title=f"{plot_title} - PDF DISTRIBUTIONS",
            xaxis_title=plot_xaxis_title,
            yaxis_title=plot_yaxis_title,
            legend_title=plot_legend_title,
            template="ggplot2",
            xaxis=dict(title_font_size=12, tickfont=dict(size=10)),
            yaxis=dict(title_font_size=12, tickfont=dict(size=10)),
            legend=dict(orientation="v", yanchor="auto", y=1, xanchor="left", font=dict(size=10), title_font_size=10),
            bargap=plot_bargap,
        )

        fig.show()

    def plot_distribution_pdf(
        self,
        distribution_name: str,
        plot_title: str,
        plot_xaxis_title: str,
        plot_yaxis_title: str,
        plot_legend_title: str,
        plot_height: int,
        plot_width: int,
        plot_bar_color: str,
        plot_bargap: float,
        plot_line_color: str,
        plot_line_width: int,
    ):
        if distribution_name not in self.distribution_instances:
            raise Exception(f"{distribution_name} distribution not founded")

        central_values = self.continuous_measures.central_values
        densities_frequencies = self.continuous_measures.densities_frequencies

        fig = go.Figure()
        fig.add_trace(go.Bar(x=central_values, y=densities_frequencies, marker_color=plot_bar_color, showlegend=True, name="Data"))

        x_plot = numpy.linspace(self.continuous_measures.min, self.continuous_measures.max, 1000)
        y_plot = self.distribution_instances[distribution_name].pdf(x_plot)
        distribution_sse = self.sorted_distributions_sse[distribution_name]["sse"]
        is_rejected = "✅" if distribution_name in self.not_rejected_distributions else ""
        scatter_name = f"{distribution_name}: {distribution_sse:.4E}{is_rejected}"
        scatter_line = dict(color=plot_line_color, width=plot_line_width)

        try:
            fig.add_trace(go.Scatter(x=x_plot, y=y_plot, mode="lines", name=scatter_name, line=scatter_line))
        except Exception:
            fig.add_trace(go.Scatter(x=x_plot, y=numpy.zeros(len(x_plot)), mode="lines", name=scatter_name, line=scatter_line))

        fig.update_layout(
            height=plot_height,
            width=plot_width,
            title=f"{plot_title} - PDF {distribution_name.upper().replace('_', ' ')} DISTRIBUTION",
            xaxis_title=plot_xaxis_title,
            yaxis_title=plot_yaxis_title,
            legend_title=plot_legend_title,
            template="ggplot2",
            xaxis=dict(title_font_size=12, tickfont=dict(size=10)),
            yaxis=dict(title_font_size=12, tickfont=dict(size=10)),
            legend=dict(orientation="v", yanchor="auto", y=1, xanchor="left", font=dict(size=10), title_font_size=10),
            bargap=plot_bargap,
        )

        fig.show()

    def plot_ecdf(
        self,
        plot_title: str,
        plot_xaxis_title: str,
        plot_xaxis_min_offset: float,
        plot_xaxis_max_offset: float,
        plot_yaxis_title: str,
        plot_legend_title: str,
        plot_height: int,
        plot_width: int,
        plot_line_color: str,
        plot_line_width: int,
        plot_line_name: str,
    ):
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=numpy.repeat(self.continuous_measures.data_unique, 2)[1:-1],
                y=numpy.repeat(self.continuous_measures.ecdf_frequencies, 2)[:-1],
                mode="lines",
                name=plot_line_name,
                line=dict(color=plot_line_color, width=plot_line_width),
                showlegend=True,
            )
        )
        fig.update_layout(
            height=plot_height,
            width=plot_width,
            title=plot_title,
            xaxis_title=plot_xaxis_title,
            yaxis_title=plot_yaxis_title,
            legend_title=plot_legend_title,
            template="ggplot2",
            xaxis=dict(title_font_size=12, tickfont=dict(size=10)),
            yaxis=dict(title_font_size=12, tickfont=dict(size=10)),
            legend=dict(orientation="v", yanchor="auto", y=1, xanchor="left", font=dict(size=10), title_font_size=10),
            xaxis_range=[self.continuous_measures.min - plot_xaxis_min_offset, self.continuous_measures.max + plot_xaxis_max_offset],
        )
        fig.show()

    def plot_ecdf_distribution(
        self,
        distribution_name: str,
        plot_title: str,
        plot_xaxis_title: str,
        plot_xaxis_min_offset: float,
        plot_xaxis_max_offset: float,
        plot_yaxis_title: str,
        plot_legend_title: str,
        plot_height: int,
        plot_width: int,
        plot_empirical_line_color: str,
        plot_empirical_line_width: int,
        plot_empirical_line_name: str,
        plot_distribution_line_color: str,
        plot_distribution_line_width: int,
    ):
        if distribution_name not in self.distribution_instances:
            raise Exception(f"{distribution_name} distribution not founded")

        fig = go.Figure()

        # Línea de la distribución
        x_plot = numpy.linspace(self.continuous_measures.min, self.continuous_measures.max, 1000)
        y_plot = self.distribution_instances[distribution_name].cdf(x_plot)
        distribution_sse = self.sorted_distributions_sse[distribution_name]["sse"]
        is_rejected = "✅" if distribution_name in self.not_rejected_distributions else ""
        try:
            fig.add_trace(
                go.Scatter(
                    x=x_plot,
                    y=y_plot,
                    mode="lines",
                    name=f"{distribution_name}: {distribution_sse:.4E}{is_rejected}",
                    line=dict(color=plot_distribution_line_color, width=plot_distribution_line_width),
                )
            )
        except Exception:
            fig.add_trace(go.Scatter(x=x_plot, y=numpy.zeros(len(x_plot)), mode="lines", name=f"{distribution_name}: {distribution_sse:.4E}{is_rejected}"))

        # Línea empírica
        fig.add_trace(
            go.Scatter(
                x=numpy.repeat(self.continuous_measures.data_unique, 2)[1:-1],
                y=numpy.repeat(self.continuous_measures.ecdf_frequencies, 2)[:-1],
                mode="lines",
                name=plot_empirical_line_name,
                line=dict(color=plot_empirical_line_color, width=plot_empirical_line_width),
                showlegend=True,
            )
        )

        fig.update_layout(
            height=plot_height,
            width=plot_width,
            title=f"{plot_title} - CDF {distribution_name.upper().replace('_', ' ')} DISTRIBUTION",
            xaxis_title=plot_xaxis_title,
            yaxis_title=plot_yaxis_title,
            legend_title=plot_legend_title,
            template="ggplot2",
            xaxis=dict(title_font_size=12, tickfont=dict(size=10)),
            yaxis=dict(title_font_size=12, tickfont=dict(size=10)),
            legend=dict(orientation="v", yanchor="auto", y=1, xanchor="left", font=dict(size=10), title_font_size=10),
            xaxis_range=[self.continuous_measures.min - plot_xaxis_min_offset, self.continuous_measures.max + plot_xaxis_max_offset],
        )
        fig.show()

    def qq_plot(
        self,
        distribution_name: str,
        plot_title: str,
        plot_xaxis_title: str,
        plot_yaxis_title: str,
        plot_legend_title: str,
        plot_height: int,
        plot_width: int,
        qq_marker_name: str,
        qq_marker_color: str,
        qq_marker_size: int,
    ):
        if distribution_name not in self.distribution_instances:
            raise Exception(f"{distribution_name} distribution not founded")

        x = self.distribution_instances[distribution_name].ppf(self.continuous_measures.qq_arr)
        y = self.continuous_measures.data

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name=qq_marker_name, marker=dict(color=qq_marker_color, size=qq_marker_size), showlegend=True))
        fig.update_layout(
            height=plot_height,
            width=plot_width,
            title=f"{plot_title} {distribution_name.upper().replace('_', ' ')} DISTRIBUTION",
            xaxis_title=plot_xaxis_title,
            yaxis_title=plot_yaxis_title,
            legend_title=plot_legend_title,
            template="ggplot2",
            xaxis=dict(title_font_size=12, tickfont=dict(size=10)),
            yaxis=dict(title_font_size=12, tickfont=dict(size=10)),
            legend=dict(orientation="v", yanchor="auto", y=1, xanchor="left", font=dict(size=10), title_font_size=10),
        )
        fig.show()

    def qq_plot_regression(
        self,
        distribution_name: str,
        plot_title: str,
        plot_xaxis_title: str,
        plot_yaxis_title: str,
        plot_legend_title: str,
        plot_height: int,
        plot_width: int,
        qq_marker_name: str,
        qq_marker_color: str,
        qq_marker_size: int,
        regression_line_name: str,
        regression_line_color: str,
        regression_line_width: int,
    ):
        if distribution_name not in self.distribution_instances:
            raise Exception(f"{distribution_name} distribution not founded")

        x = self.distribution_instances[distribution_name].ppf(self.continuous_measures.qq_arr)
        y = self.continuous_measures.data

        linear_regression = scipy.stats.linregress(x, y)
        y_reg = linear_regression.intercept + x * linear_regression.slope

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y_reg, mode="lines", name=regression_line_name, line=dict(color=regression_line_color, width=regression_line_width)))
        fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name=qq_marker_name, marker=dict(color=qq_marker_color, size=qq_marker_size)))
        fig.update_layout(
            height=plot_height,
            width=plot_width,
            title=f"{plot_title} {distribution_name.upper().replace('_', ' ')} DISTRIBUTION <br><br><sup>Regression: {linear_regression.intercept:.4g} + x * {linear_regression.slope:.4g} • r = {linear_regression.rvalue:.4g}</sup>",
            xaxis_title=plot_xaxis_title,
            yaxis_title=plot_yaxis_title,
            legend_title=plot_legend_title,
            template="ggplot2",
            xaxis=dict(title_font_size=12, tickfont=dict(size=10)),
            yaxis=dict(title_font_size=12, tickfont=dict(size=10)),
            legend=dict(orientation="v", yanchor="auto", y=1, xanchor="left", font=dict(size=10), title_font_size=10),
        )
        fig.show()


if __name__ == "__main__":

    path = "../../datasets_test/continuous/data_1000/data_alpha.txt"
    # path = "../datasets_test/discrete/book2.txt"
    # path = "../datasets_test/continuous/data_1000/data_beta.txt"
    sample_distribution_file = open(path, "r", encoding="utf-8-sig")
    data = [float(x.replace(",", ".")) for x in sample_distribution_file.read().splitlines()]
    sample_distribution_file.close()

    phitter_continuous = PHITTER_CONTINUOUS(data, distributions_to_fit="all")
    phitter_continuous.fit(n_workers=2)

    for distribution, results in phitter_continuous.sorted_distributions_sse.items():
        print(f"Distribution: {distribution}, SSE: {results['sse']}, Aprobados: {results['n_test_passed']}")
