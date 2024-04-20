import sys
import typing

import numpy
import pandas

sys.path.append("../")
from phitter.continuous.phitter_continuous import PHITTER_CONTINUOUS
from phitter.discrete.phitter_discrete import PHITTER_DISCRETE


class PHITTER:
    def __init__(
        self,
        data: list[int | float] | numpy.ndarray,
        fit_type: typing.Literal["continuous", "discrete"] = "continuous",
        num_bins: int | None = None,
        confidence_level=0.95,
        minimum_sse=numpy.inf,
        subsample_estimation_size: int | None = None,
        distributions_to_fit: list[str] | typing.Literal["all"] = "all",
        exclude_distributions: list[str] | typing.Literal["any"] = "any",
    ):
        self.data = data
        self.fit_type = fit_type
        self.num_bins = num_bins if num_bins != None else len(numpy.histogram_bin_edges(self.data, bins="doane"))
        self.confidence_level = confidence_level
        self.minimum_sse = minimum_sse
        self.subsample_estimation_size = subsample_estimation_size
        self.distributions_to_fit = distributions_to_fit
        self.exclude_distributions = exclude_distributions

    def fit(self, n_workers: int = 1):
        if self.fit_type == "continuous":
            self.phitter_continuous = PHITTER_CONTINUOUS(
                data=self.data,
                num_bins=self.num_bins,
                confidence_level=self.confidence_level,
                minimum_sse=self.minimum_sse,
                subsample_estimation_size=self.subsample_estimation_size,
                distributions_to_fit=self.distributions_to_fit,
                exclude_distributions=self.exclude_distributions,
            )
            self.phitter_continuous.fit(n_workers=n_workers)

        if self.fit_type == "discrete":
            self.phitter_discrete = PHITTER_DISCRETE(
                data=self.data,
                confidence_level=self.confidence_level,
                minimum_sse=self.minimum_sse,
                subsample_estimation_size=self.subsample_estimation_size,
                distributions_to_fit=self.distributions_to_fit,
                exclude_distributions=self.exclude_distributions,
            )
            self.phitter_discrete.fit(n_workers=n_workers)

    @property
    def best_distribution(self):
        id = list(self.sorted_distributions_sse.keys())[0]
        return {"id": id, "parameters": self.sorted_distributions_sse[id]["parameters"]}

    @property
    def sorted_distributions_sse(self):
        if self.fit_type == "continuous":
            return self.phitter_continuous.sorted_distributions_sse

        if self.fit_type == "discrete":
            return self.phitter_discrete.sorted_distributions_sse

    @property
    def not_rejected_distributions(self):
        if self.fit_type == "continuous":
            return self.phitter_continuous.not_rejected_distributions

        if self.fit_type == "discrete":
            return self.phitter_discrete.not_rejected_distributions

    def dict_to_dataframe(self, data: dict[str, dict]) -> pandas.DataFrame:
        flat_data = []
        for distribution, values in data.items():
            is_rejected = "✅" if values["n_test_passed"] >= 1 else "❌"
            flat_entry = {("distribution", ""): distribution}
            flat_entry[("passed", "")] = is_rejected
            for test, results in values.items():
                if test == "parameters":
                    flat_entry[("parameters", "")] = ", ".join([f"{k}: {v:.4g}" for k, v in results.items()])
                elif isinstance(results, dict):
                    for key, value in results.items():
                        flat_entry[(test, key)] = value
                else:
                    flat_entry[(test, "")] = results
            flat_data.append(flat_entry)

        df = pandas.DataFrame(flat_data)
        df.columns = pandas.MultiIndex.from_tuples(df.columns)
        return df

    @property
    def df_sorted_distributions_sse(self) -> pandas.DataFrame | None:
        if self.fit_type == "continuous":
            if self.phitter_continuous.sorted_distributions_sse is None:
                return None
            if len(self.phitter_continuous.sorted_distributions_sse) == 0:
                return pandas.DataFrame(
                    columns=pandas.MultiIndex.from_tuples(
                        [
                            ("distribution", ""),
                            ("passed", ""),
                            ("sse", ""),
                            ("parameters", ""),
                            ("chi_square", "test_statistic"),
                            ("chi_square", "critical_value"),
                            ("chi_square", "p_value"),
                            ("chi_square", "rejected"),
                            ("kolmogorov_smirnov", "test_statistic"),
                            ("kolmogorov_smirnov", "critical_value"),
                            ("kolmogorov_smirnov", "p_value"),
                            ("kolmogorov_smirnov", "rejected"),
                            ("anderson_darling", "test_statistic"),
                            ("anderson_darling", "critical_value"),
                            ("anderson_darling", "p_value"),
                            ("anderson_darling", "rejected"),
                        ]
                    )
                )
            df = self.dict_to_dataframe(self.phitter_continuous.sorted_distributions_sse)
            return df[["distribution", "passed", "sse", "parameters", "chi_square", "kolmogorov_smirnov", "anderson_darling"]]

        if self.fit_type == "discrete":
            if self.phitter_discrete.sorted_distributions_sse is None:
                return None
            if len(self.phitter_discrete.sorted_distributions_sse) == 0:
                return pandas.DataFrame(
                    columns=pandas.MultiIndex.from_tuples(
                        [
                            ("distribution", ""),
                            ("passed", ""),
                            ("sse", ""),
                            ("parameters", ""),
                            ("chi_square", "test_statistic"),
                            ("chi_square", "critical_value"),
                            ("chi_square", "p_value"),
                            ("chi_square", "rejected"),
                            ("kolmogorov_smirnov", "test_statistic"),
                            ("kolmogorov_smirnov", "critical_value"),
                            ("kolmogorov_smirnov", "p_value"),
                            ("kolmogorov_smirnov", "rejected"),
                        ]
                    )
                )
            df = self.dict_to_dataframe(self.phitter_discrete.sorted_distributions_sse)
            return df[["distribution", "passed", "sse", "parameters", "chi_square", "kolmogorov_smirnov"]]

    @property
    def df_not_rejected_distributions(self) -> pandas.DataFrame | None:
        if self.fit_type == "continuous":
            if self.phitter_continuous.not_rejected_distributions is None:
                return None
            if len(self.phitter_continuous.not_rejected_distributions) == 0:
                return pandas.DataFrame(
                    columns=pandas.MultiIndex.from_tuples(
                        [
                            ("distribution", ""),
                            ("passed", ""),
                            ("sse", ""),
                            ("parameters", ""),
                            ("chi_square", "test_statistic"),
                            ("chi_square", "critical_value"),
                            ("chi_square", "p_value"),
                            ("chi_square", "rejected"),
                            ("kolmogorov_smirnov", "test_statistic"),
                            ("kolmogorov_smirnov", "critical_value"),
                            ("kolmogorov_smirnov", "p_value"),
                            ("kolmogorov_smirnov", "rejected"),
                            ("anderson_darling", "test_statistic"),
                            ("anderson_darling", "critical_value"),
                            ("anderson_darling", "p_value"),
                            ("anderson_darling", "rejected"),
                        ]
                    )
                )
            df = self.dict_to_dataframe(self.phitter_continuous.not_rejected_distributions)
            return df[["distribution", "sse", "parameters", "chi_square", "kolmogorov_smirnov", "anderson_darling"]]

        if self.fit_type == "discrete":
            if self.phitter_discrete.not_rejected_distributions is None:
                return None
            if len(self.phitter_discrete.not_rejected_distributions) == 0:
                return pandas.DataFrame(
                    columns=pandas.MultiIndex.from_tuples(
                        [
                            ("distribution", ""),
                            ("passed", ""),
                            ("sse", ""),
                            ("parameters", ""),
                            ("chi_square", "test_statistic"),
                            ("chi_square", "critical_value"),
                            ("chi_square", "p_value"),
                            ("chi_square", "rejected"),
                            ("kolmogorov_smirnov", "test_statistic"),
                            ("kolmogorov_smirnov", "critical_value"),
                            ("kolmogorov_smirnov", "p_value"),
                            ("kolmogorov_smirnov", "rejected"),
                        ]
                    )
                )
            df = self.dict_to_dataframe(self.phitter_discrete.not_rejected_distributions)
            return df[["distribution", "sse", "parameters", "chi_square", "kolmogorov_smirnov"]]

    def plot_histogram(
        self,
        plot_title="HISTOGRAM",
        plot_xaxis_title=None,
        plot_yaxis_title=None,
        plot_legend_title: str | None = None,
        plot_height=400,
        plot_width=600,
        plot_bar_color="rgba(128,128,128,1)",
        plot_bargap=0.15,
        plot_renderer: str | None = None,
        plot_engine: typing.Literal["plotly", "matplotlib"] = "plotly",
    ):
        if self.fit_type == "continuous":
            if plot_engine == "matplotlib":
                self.phitter_continuous.plot_histogram_matplotlib(
                    plot_title=plot_title,
                    plot_xaxis_title=plot_xaxis_title,
                    plot_yaxis_title=plot_yaxis_title,
                    plot_legend_title=plot_legend_title,
                    plot_height=plot_height,
                    plot_width=plot_width,
                    plot_bar_color=plot_bar_color,
                    plot_bargap=plot_bargap,
                    plot_renderer=plot_renderer,
                )
            else:
                self.phitter_continuous.plot_histogram_plotly(
                    plot_title=plot_title,
                    plot_xaxis_title=plot_xaxis_title,
                    plot_yaxis_title=plot_yaxis_title,
                    plot_legend_title=plot_legend_title,
                    plot_height=plot_height,
                    plot_width=plot_width,
                    plot_bar_color=plot_bar_color,
                    plot_bargap=plot_bargap,
                    plot_renderer=plot_renderer,
                )

        if self.fit_type == "discrete":
            if plot_engine == "matplotlib":
                self.phitter_discrete.plot_histogram_matplotlib(
                    plot_title=plot_title,
                    plot_xaxis_title=plot_xaxis_title,
                    plot_yaxis_title=plot_yaxis_title,
                    plot_legend_title=plot_legend_title,
                    plot_height=plot_height,
                    plot_width=plot_width,
                    plot_bar_color=plot_bar_color,
                    plot_bargap=plot_bargap,
                    plot_renderer=plot_renderer,
                )
            else:
                self.phitter_discrete.plot_histogram_plotly(
                    plot_title=plot_title,
                    plot_xaxis_title=plot_xaxis_title,
                    plot_yaxis_title=plot_yaxis_title,
                    plot_legend_title=plot_legend_title,
                    plot_height=plot_height,
                    plot_width=plot_width,
                    plot_bar_color=plot_bar_color,
                    plot_bargap=plot_bargap,
                    plot_renderer=plot_renderer,
                )

    def plot_histogram_distributions(
        self,
        n_distributions=10,
        n_distributions_visible=1,
        plot_title="HISTOGRAM",
        plot_xaxis_title="Domain",
        plot_yaxis_title="Probability Density/Mass Function",
        plot_legend_title: str | None = "DISTRIBUTIONS",
        plot_height=400,
        plot_width=600,
        plot_bar_color="rgba(128,128,128,1)",
        plot_bargap=0.15,
        plot_renderer: str | None = None,
        plot_engine: typing.Literal["plotly", "matplotlib"] = "plotly",
    ):
        if self.fit_type == "continuous":
            if plot_engine == "matplotlib":
                self.phitter_continuous.plot_histogram_distributions_pdf_matplotlib(
                    n_distributions=n_distributions,
                    n_distributions_visible=n_distributions_visible,
                    plot_title=plot_title,
                    plot_xaxis_title=plot_xaxis_title,
                    plot_yaxis_title=plot_yaxis_title,
                    plot_legend_title=plot_legend_title,
                    plot_height=plot_height,
                    plot_width=plot_width,
                    plot_bar_color=plot_bar_color,
                    plot_bargap=plot_bargap,
                    plot_renderer=plot_renderer,
                )
            else:
                self.phitter_continuous.plot_histogram_distributions_pdf_plotly(
                    n_distributions=n_distributions,
                    n_distributions_visible=n_distributions_visible,
                    plot_title=plot_title,
                    plot_xaxis_title=plot_xaxis_title,
                    plot_yaxis_title=plot_yaxis_title,
                    plot_legend_title=plot_legend_title,
                    plot_height=plot_height,
                    plot_width=plot_width,
                    plot_bar_color=plot_bar_color,
                    plot_bargap=plot_bargap,
                    plot_renderer=plot_renderer,
                )

        if self.fit_type == "discrete":
            if plot_engine == "matplotlib":
                self.phitter_discrete.plot_histogram_distributions_pmf_matplotlib(
                    n_distributions=n_distributions,
                    n_distributions_visible=n_distributions_visible,
                    plot_title=plot_title,
                    plot_xaxis_title=plot_xaxis_title,
                    plot_yaxis_title=plot_yaxis_title,
                    plot_legend_title=plot_legend_title,
                    plot_height=plot_height,
                    plot_width=plot_width,
                    plot_bar_color=plot_bar_color,
                    plot_bargap=plot_bargap,
                    plot_renderer=plot_renderer,
                )
            else:
                self.phitter_discrete.plot_histogram_distributions_pmf_plotly(
                    n_distributions=n_distributions,
                    n_distributions_visible=n_distributions_visible,
                    plot_title=plot_title,
                    plot_xaxis_title=plot_xaxis_title,
                    plot_yaxis_title=plot_yaxis_title,
                    plot_legend_title=plot_legend_title,
                    plot_height=plot_height,
                    plot_width=plot_width,
                    plot_bar_color=plot_bar_color,
                    plot_bargap=plot_bargap,
                    plot_renderer=plot_renderer,
                )

    def plot_distribution(
        self,
        distribution_name: str,
        plot_title="HISTOGRAM",
        plot_xaxis_title="Domain",
        plot_yaxis_title="Probability Density/Mass Function",
        plot_legend_title: str | None = None,
        plot_height=400,
        plot_width=600,
        plot_bar_color="rgba(128,128,128,1)",
        plot_bargap=0.15,
        plot_line_color="rgba(255,0,0,1)",
        plot_line_width=3,
        plot_renderer: str | None = None,
        plot_engine: typing.Literal["plotly", "matplotlib"] = "plotly",
    ):
        if self.fit_type == "continuous":
            if plot_engine == "matplotlib":
                self.phitter_continuous.plot_distribution_pdf_matplotlib(
                    distribution_name=distribution_name,
                    plot_title=plot_title,
                    plot_xaxis_title=plot_xaxis_title,
                    plot_yaxis_title=plot_yaxis_title,
                    plot_legend_title=plot_legend_title,
                    plot_height=plot_height,
                    plot_width=plot_width,
                    plot_bar_color=plot_bar_color,
                    plot_bargap=plot_bargap,
                    plot_line_color=plot_line_color,
                    plot_line_width=plot_line_width,
                    plot_renderer=plot_renderer,
                )
            else:
                self.phitter_continuous.plot_distribution_pdf_plotly(
                    distribution_name=distribution_name,
                    plot_title=plot_title,
                    plot_xaxis_title=plot_xaxis_title,
                    plot_yaxis_title=plot_yaxis_title,
                    plot_legend_title=plot_legend_title,
                    plot_height=plot_height,
                    plot_width=plot_width,
                    plot_bar_color=plot_bar_color,
                    plot_bargap=plot_bargap,
                    plot_line_color=plot_line_color,
                    plot_line_width=plot_line_width,
                    plot_renderer=plot_renderer,
                )

        if self.fit_type == "discrete":
            if plot_engine == "matplotlib":
                self.phitter_discrete.plot_distribution_pmf_matplotlib(
                    distribution_name=distribution_name,
                    plot_title=plot_title,
                    plot_xaxis_title=plot_xaxis_title,
                    plot_yaxis_title=plot_yaxis_title,
                    plot_legend_title=plot_legend_title,
                    plot_height=plot_height,
                    plot_width=plot_width,
                    plot_bar_color=plot_bar_color,
                    plot_bargap=plot_bargap,
                    plot_line_color=plot_line_color,
                    plot_line_width=plot_line_width,
                    plot_renderer=plot_renderer,
                )
            else:
                self.phitter_discrete.plot_distribution_pmf_plotly(
                    distribution_name=distribution_name,
                    plot_title=plot_title,
                    plot_xaxis_title=plot_xaxis_title,
                    plot_yaxis_title=plot_yaxis_title,
                    plot_legend_title=plot_legend_title,
                    plot_height=plot_height,
                    plot_width=plot_width,
                    plot_bar_color=plot_bar_color,
                    plot_bargap=plot_bargap,
                    plot_line_color=plot_line_color,
                    plot_line_width=plot_line_width,
                    plot_renderer=plot_renderer,
                )

    def plot_ecdf(
        self,
        plot_title="ECDF",
        plot_xaxis_title="Domain",
        plot_yaxis_title="Cumulative Distribution Function",
        plot_xaxis_min_offset=0.3,
        plot_xaxis_max_offset=0.3,
        plot_legend_title: str | None = None,
        plot_height=400,
        plot_width=600,
        plot_line_color="rgba(255,0,0,1)",
        plot_line_width=2,
        plot_line_name="Empirical Distribution",
        plot_bar_color="rgba(128,128,128,1)",
        plot_bargap=0.15,
        plot_renderer: str | None = None,
        plot_engine: typing.Literal["plotly", "matplotlib"] = "plotly",
    ):
        if self.fit_type == "continuous":
            if len(self.data) > 10000 or plot_engine == "matplotlib":
                self.phitter_continuous.plot_ecdf_matplotlib(
                    plot_title=plot_title,
                    plot_xaxis_title=plot_xaxis_title,
                    plot_xaxis_min_offset=plot_xaxis_min_offset,
                    plot_xaxis_max_offset=plot_xaxis_max_offset,
                    plot_yaxis_title=plot_yaxis_title,
                    plot_legend_title=plot_legend_title,
                    plot_height=plot_height,
                    plot_width=plot_width,
                    plot_line_color=plot_line_color,
                    plot_line_width=plot_line_width,
                    plot_line_name=plot_line_name,
                    plot_renderer=plot_renderer if len(self.data) <= 10000 else "png",
                )
            else:
                self.phitter_continuous.plot_ecdf_plotly(
                    plot_title=plot_title,
                    plot_xaxis_title=plot_xaxis_title,
                    plot_xaxis_min_offset=plot_xaxis_min_offset,
                    plot_xaxis_max_offset=plot_xaxis_max_offset,
                    plot_yaxis_title=plot_yaxis_title,
                    plot_legend_title=plot_legend_title,
                    plot_height=plot_height,
                    plot_width=plot_width,
                    plot_line_color=plot_line_color,
                    plot_line_width=plot_line_width,
                    plot_line_name=plot_line_name,
                    plot_renderer=plot_renderer if len(self.data) <= 10000 else "png",
                )

        if self.fit_type == "discrete":
            if len(self.data) > 10000 or plot_engine == "matplotlib":
                self.phitter_discrete.plot_ecdf_matplotlib(
                    plot_title=plot_title,
                    plot_xaxis_title=plot_xaxis_title,
                    plot_yaxis_title=plot_yaxis_title,
                    plot_legend_title=plot_legend_title,
                    plot_height=plot_height,
                    plot_width=plot_width,
                    plot_bar_color=plot_bar_color,
                    plot_bargap=plot_bargap,
                    plot_renderer=plot_renderer,
                )
            else:
                self.phitter_discrete.plot_ecdf_plotly(
                    plot_title=plot_title,
                    plot_xaxis_title=plot_xaxis_title,
                    plot_yaxis_title=plot_yaxis_title,
                    plot_legend_title=plot_legend_title,
                    plot_height=plot_height,
                    plot_width=plot_width,
                    plot_bar_color=plot_bar_color,
                    plot_bargap=plot_bargap,
                    plot_renderer=plot_renderer,
                )

    def plot_ecdf_distribution(
        self,
        distribution_name: str,
        plot_title="ECDF",
        plot_xaxis_title="Domain",
        plot_yaxis_title="Cumulative Distribution Function",
        plot_xaxis_min_offset=0.3,
        plot_xaxis_max_offset=0.3,
        plot_legend_title: str | None = None,
        plot_height=400,
        plot_width=600,
        plot_empirical_line_color="rgba(128,128,128,1)",
        plot_empirical_line_width=4,
        plot_empirical_line_name="Empirical Distribution",
        plot_empirical_bar_color="rgba(128,128,128,1)",
        plot_empirical_bargap=0.15,
        plot_distribution_line_color="rgba(255,0,0,1)",
        plot_distribution_line_width=2,
        plot_renderer: str | None = None,
        plot_engine: typing.Literal["plotly", "matplotlib"] = "plotly",
    ):
        if self.fit_type == "continuous":
            if len(self.data) > 10000 or plot_engine == "matplotlib":
                self.phitter_continuous.plot_ecdf_distribution_matplotlib(
                    distribution_name=distribution_name,
                    plot_title=plot_title,
                    plot_xaxis_title=plot_xaxis_title,
                    plot_xaxis_min_offset=plot_xaxis_min_offset,
                    plot_xaxis_max_offset=plot_xaxis_max_offset,
                    plot_yaxis_title=plot_yaxis_title,
                    plot_legend_title=plot_legend_title,
                    plot_height=plot_height,
                    plot_width=plot_width,
                    plot_empirical_line_color=plot_empirical_line_color,
                    plot_empirical_line_width=plot_empirical_line_width,
                    plot_empirical_line_name=plot_empirical_line_name,
                    plot_distribution_line_color=plot_distribution_line_color,
                    plot_distribution_line_width=plot_distribution_line_width,
                    plot_renderer=plot_renderer if len(self.data) <= 10000 else "png",
                )
            else:
                self.phitter_continuous.plot_ecdf_distribution_plotly(
                    distribution_name=distribution_name,
                    plot_title=plot_title,
                    plot_xaxis_title=plot_xaxis_title,
                    plot_xaxis_min_offset=plot_xaxis_min_offset,
                    plot_xaxis_max_offset=plot_xaxis_max_offset,
                    plot_yaxis_title=plot_yaxis_title,
                    plot_legend_title=plot_legend_title,
                    plot_height=plot_height,
                    plot_width=plot_width,
                    plot_empirical_line_color=plot_empirical_line_color,
                    plot_empirical_line_width=plot_empirical_line_width,
                    plot_empirical_line_name=plot_empirical_line_name,
                    plot_distribution_line_color=plot_distribution_line_color,
                    plot_distribution_line_width=plot_distribution_line_width,
                    plot_renderer=plot_renderer if len(self.data) <= 10000 else "png",
                )

        if self.fit_type == "discrete":
            if len(self.data) > 10000 or plot_engine == "matplotlib":
                self.phitter_discrete.plot_ecdf_distribution_matplotlib(
                    distribution_name=distribution_name,
                    plot_title=plot_title,
                    plot_xaxis_title=plot_xaxis_title,
                    plot_yaxis_title=plot_yaxis_title,
                    plot_legend_title=plot_legend_title,
                    plot_height=plot_height,
                    plot_width=plot_width,
                    plot_empirical_bar_color=plot_empirical_bar_color,
                    plot_empirical_bargap=plot_empirical_bargap,
                    plot_distribution_line_color=plot_distribution_line_color,
                    plot_distribution_line_width=plot_distribution_line_width,
                    plot_renderer=plot_renderer,
                )
            else:
                self.phitter_discrete.plot_ecdf_distribution_plotly(
                    distribution_name=distribution_name,
                    plot_title=plot_title,
                    plot_xaxis_title=plot_xaxis_title,
                    plot_yaxis_title=plot_yaxis_title,
                    plot_legend_title=plot_legend_title,
                    plot_height=plot_height,
                    plot_width=plot_width,
                    plot_empirical_bar_color=plot_empirical_bar_color,
                    plot_empirical_bargap=plot_empirical_bargap,
                    plot_distribution_line_color=plot_distribution_line_color,
                    plot_distribution_line_width=plot_distribution_line_width,
                    plot_renderer=plot_renderer,
                )

    def qq_plot(
        self,
        distribution_name: str,
        plot_title="QQ PLOT",
        plot_xaxis_title="Theorical Quantiles",
        plot_yaxis_title="Sample Quantiles",
        plot_legend_title: str | None = None,
        plot_height=400,
        plot_width=600,
        qq_marker_name="Markers QQ",
        qq_marker_color="rgba(128,128,128,1)",
        qq_marker_size=6,
        plot_renderer: str | None = None,
        plot_engine: typing.Literal["plotly", "matplotlib"] = "plotly",
    ):
        if self.fit_type == "continuous":
            if len(self.data) > 10000 or plot_engine == "matplotlib":
                self.phitter_continuous.qq_plot_matplotlib(
                    distribution_name=distribution_name,
                    plot_title=plot_title,
                    plot_xaxis_title=plot_xaxis_title,
                    plot_yaxis_title=plot_yaxis_title,
                    plot_legend_title=plot_legend_title,
                    plot_height=plot_height,
                    plot_width=plot_width,
                    qq_marker_name=qq_marker_name,
                    qq_marker_color=qq_marker_color,
                    qq_marker_size=qq_marker_size,
                    plot_renderer=plot_renderer,
                )
            else:
                self.phitter_continuous.qq_plot_plotly(
                    distribution_name=distribution_name,
                    plot_title=plot_title,
                    plot_xaxis_title=plot_xaxis_title,
                    plot_yaxis_title=plot_yaxis_title,
                    plot_legend_title=plot_legend_title,
                    plot_height=plot_height,
                    plot_width=plot_width,
                    qq_marker_name=qq_marker_name,
                    qq_marker_color=qq_marker_color,
                    qq_marker_size=qq_marker_size,
                    plot_renderer=plot_renderer if len(self.data) <= 10000 else "png",
                )

        if self.fit_type == "discrete":
            if len(self.data) > 10000 or plot_engine == "matplotlib":
                self.phitter_discrete.qq_plot_matplotlib(
                    distribution_name=distribution_name,
                    plot_title=plot_title,
                    plot_xaxis_title=plot_xaxis_title,
                    plot_yaxis_title=plot_yaxis_title,
                    plot_legend_title=plot_legend_title,
                    plot_height=plot_height,
                    plot_width=plot_width,
                    qq_marker_name=qq_marker_name,
                    qq_marker_color=qq_marker_color,
                    qq_marker_size=qq_marker_size,
                    plot_renderer=plot_renderer,
                )
            else:
                self.phitter_discrete.qq_plot_plotly(
                    distribution_name=distribution_name,
                    plot_title=plot_title,
                    plot_xaxis_title=plot_xaxis_title,
                    plot_yaxis_title=plot_yaxis_title,
                    plot_legend_title=plot_legend_title,
                    plot_height=plot_height,
                    plot_width=plot_width,
                    qq_marker_name=qq_marker_name,
                    qq_marker_color=qq_marker_color,
                    qq_marker_size=qq_marker_size,
                    plot_renderer=plot_renderer,
                )

    def qq_plot_regression(
        self,
        distribution_name: str,
        plot_title="QQ PLOT",
        plot_xaxis_title="Theorical Quantiles",
        plot_yaxis_title="Sample Quantiles",
        plot_legend_title: str | None = None,
        plot_height=400,
        plot_width=600,
        qq_marker_name="Markers QQ",
        qq_marker_color="rgba(128,128,128,1)",
        qq_marker_size=6,
        regression_line_name="Regression",
        regression_line_color="rgba(255,0,0,1)",
        regression_line_width=2,
        plot_renderer: str | None = None,
        plot_engine: typing.Literal["plotly", "matplotlib"] = "plotly",
    ):
        if self.fit_type == "continuous":
            if len(self.data) > 10000 or plot_engine == "matplotlib":
                self.phitter_continuous.qq_plot_regression_matplotlib(
                    distribution_name=distribution_name,
                    plot_title=plot_title,
                    plot_xaxis_title=plot_xaxis_title,
                    plot_yaxis_title=plot_yaxis_title,
                    plot_legend_title=plot_legend_title,
                    plot_height=plot_height,
                    plot_width=plot_width,
                    qq_marker_name=qq_marker_name,
                    qq_marker_color=qq_marker_color,
                    qq_marker_size=qq_marker_size,
                    regression_line_name=regression_line_name,
                    regression_line_color=regression_line_color,
                    regression_line_width=regression_line_width,
                    plot_renderer=plot_renderer,
                )
            else:
                self.phitter_continuous.qq_plot_regression_plotly(
                    distribution_name=distribution_name,
                    plot_title=plot_title,
                    plot_xaxis_title=plot_xaxis_title,
                    plot_yaxis_title=plot_yaxis_title,
                    plot_legend_title=plot_legend_title,
                    plot_height=plot_height,
                    plot_width=plot_width,
                    qq_marker_name=qq_marker_name,
                    qq_marker_color=qq_marker_color,
                    qq_marker_size=qq_marker_size,
                    regression_line_name=regression_line_name,
                    regression_line_color=regression_line_color,
                    regression_line_width=regression_line_width,
                    plot_renderer=plot_renderer if len(self.data) <= 10000 else "png",
                )

        if self.fit_type == "discrete":
            if len(self.data) > 10000 or plot_engine == "matplotlib":
                self.phitter_discrete.qq_plot_regression_matplotlib(
                    distribution_name=distribution_name,
                    plot_title=plot_title,
                    plot_xaxis_title=plot_xaxis_title,
                    plot_yaxis_title=plot_yaxis_title,
                    plot_legend_title=plot_legend_title,
                    plot_height=plot_height,
                    plot_width=plot_width,
                    qq_marker_name=qq_marker_name,
                    qq_marker_color=qq_marker_color,
                    qq_marker_size=qq_marker_size,
                    regression_line_name=regression_line_name,
                    regression_line_color=regression_line_color,
                    regression_line_width=regression_line_width,
                    plot_renderer=plot_renderer if len(self.data) <= 10000 else "png",
                )
            else:
                self.phitter_discrete.qq_plot_regression_plotly(
                    distribution_name=distribution_name,
                    plot_title=plot_title,
                    plot_xaxis_title=plot_xaxis_title,
                    plot_yaxis_title=plot_yaxis_title,
                    plot_legend_title=plot_legend_title,
                    plot_height=plot_height,
                    plot_width=plot_width,
                    qq_marker_name=qq_marker_name,
                    qq_marker_color=qq_marker_color,
                    qq_marker_size=qq_marker_size,
                    regression_line_name=regression_line_name,
                    regression_line_color=regression_line_color,
                    regression_line_width=regression_line_width,
                    plot_renderer=plot_renderer if len(self.data) <= 10000 else "png",
                )


if __name__ == "__main__":
    # path = "../datasets_test/discrete/book2.txt"
    path = "../datasets_test/continuous/data.txt"
    sample_distribution_file = open(path, "r", encoding="utf-8-sig")
    data = [float(x.strip().replace(",", ".")) for x in sample_distribution_file.read().splitlines()]

    phitter = PHITTER(data, fit_type="continuous")
    phitter.fit(n_workers=6)

    print(f"Best Distribution: {phitter.best_distribution}")

    for distribution, results in phitter.sorted_distributions_sse.items():
        print(f"Distribution: {distribution}, SSE: {results['sse']}, Aprobados: {results['n_test_passed']}")
