"""Strategy-focused charts for tyre stints and race flow."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from fastf1 import plotting
from plotly.subplots import make_subplots

from utils.constants import COMPOUND_FALLBACKS
from utils.time_format import format_delta, format_duration, format_lap_time
from .common import apply_dark_theme, empty_figure


def _compound_color(compound: str, session) -> str:
    """Resolve official compound colours with a safe fallback."""
    normalized = str(compound).upper()
    try:
        return plotting.get_compound_color(normalized, session)
    except Exception:
        return COMPOUND_FALLBACKS.get(normalized, COMPOUND_FALLBACKS["UNKNOWN"])


def _time_ticks(values: pd.Series, formatter) -> tuple[list[float], list[str]]:
    """Create readable formatted time ticks while charts retain numeric seconds."""
    finite = pd.to_numeric(values, errors="coerce").dropna()
    if finite.empty:
        return [], []
    low, high = float(finite.min()), float(finite.max())
    ticks = [low] if low == high else list(pd.Series([low + (high - low) * point / 4 for point in range(5)]))
    return ticks, [formatter(tick) for tick in ticks]


def create_tyre_strategy_chart(
    stints: pd.DataFrame,
    session,
    driver_order: list[str],
    title: str | None = None,
) -> go.Figure:
    """Render Formula 1 style horizontal tyre stint bars."""
    if stints.empty:
        return empty_figure("Tyre strategy data will appear after the session loads.", height=380)

    ordered = stints.copy()
    existing_drivers = [d for d in ordered["Driver"].unique() if d not in driver_order]
    full_categories = driver_order[::-1] + existing_drivers
    ordered["Driver"] = pd.Categorical(ordered["Driver"], categories=full_categories, ordered=True)
    ordered = ordered.sort_values(["Driver", "StartLap"])

    figure = go.Figure()
    seen_compounds: set[str] = set()
    for _, stint in ordered.iterrows():
        compound = str(stint["Compound"]).upper()
        showlegend = compound not in seen_compounds
        seen_compounds.add(compound)

        figure.add_trace(
            go.Bar(
                x=[stint["StintLength"]],
                y=[stint["Driver"]],
                base=[max(stint["StartLap"] - 1, 0)],
                orientation="h",
                marker=dict(color=_compound_color(compound, session)),
                name=compound.title(),
                legendgroup=compound,
                showlegend=showlegend,
                text=[f"{compound} | {int(stint['StintLength'])} laps"],
                textposition="inside",
                insidetextanchor="middle",
                hovertemplate=(
                    f"<b>{stint['Driver']}</b><br>"
                    f"{compound.title()} tyres<br>"
                    f"Laps {int(stint['StartLap'])}-{int(stint['EndLap'])}<br>"
                    f"Stint length {int(stint['StintLength'])}<extra></extra>"
                ),
            )
        )

    calc_height = max(420, len(ordered["Driver"].unique()) * 26 + 90)
    figure.update_layout(
        barmode="stack",
        xaxis_title="Lap Number",
        yaxis_title="",
    )
    return apply_dark_theme(figure, title=title, height=calc_height, legend_orientation="h")


def create_stint_analysis_chart(
    stints: pd.DataFrame,
    driver_colors: dict[str, str],
    title: str | None = None,
) -> go.Figure:
    """Visualise mean stint pace against stint start lap."""
    if stints.empty:
        return empty_figure("Stint analysis is unavailable for this session.", height=360)

    fig = go.Figure()
    for driver in sorted(stints["Driver"].unique()):
        driver_stints = stints.loc[stints["Driver"] == driver]
        fig.add_trace(
            go.Scatter(
                x=driver_stints["StartLap"],
                y=driver_stints["MeanLapSeconds"],
                mode="markers+lines",
                name=driver,
                line=dict(color=driver_colors.get(driver, "#E10600"), width=2),
                marker=dict(
                    size=driver_stints["StintLength"].fillna(0).clip(lower=6) + 4,
                    color=driver_colors.get(driver, "#E10600"),
                ),
                customdata=driver_stints[["Compound", "StintLength"]].to_numpy(),
                hovertemplate=(
                    "<b>%{fullData.name}</b><br>"
                    "Start lap %{x}<br>"
                    "Mean pace %{y:.3f}s<br>"
                    "Compound %{customdata[0]}<br>"
                    "Length %{customdata[1]} laps<extra></extra>"
                ),
            )
        )

    fig.update_layout(xaxis_title="Stint Start Lap", yaxis_title="Mean Stint Pace (s)")
    return apply_dark_theme(fig, title=title, height=410, legend_orientation="h")


def create_position_changes_chart(
    positions: pd.DataFrame,
    driver_colors: dict[str, str],
    title: str | None = None,
) -> go.Figure:
    """Plot race position changes across laps with a clean vertical right-hand legend.

    Positions the legend vertically on the right side to prevent overlap with the
    chart title, page header, and plotting area when displaying all 20 drivers.
    """
    if positions.empty:
        return empty_figure(
            "Position changes are available for race-like sessions with classification data.",
            height=380,
        )

    figure = go.Figure()
    drivers = sorted(positions["Driver"].unique())
    for driver in drivers:
        driver_positions = positions.loc[positions["Driver"] == driver].sort_values("LapNumber")
        figure.add_trace(
            go.Scatter(
                x=driver_positions["LapNumber"],
                y=driver_positions["Position"],
                mode="lines+markers",
                name=driver,
                line=dict(color=driver_colors.get(driver, "#E10600"), width=2.2),
                marker=dict(size=5),
                hovertemplate=(
                    "<b>%{fullData.name}</b><br>"
                    "Lap %{x}<br>"
                    "Position P%{y}<extra></extra>"
                ),
            )
        )

    figure.update_layout(
        xaxis_title="Lap Number",
        yaxis_title="Track Position",
    )
    figure.update_yaxes(autorange="reversed", dtick=2)
    return apply_dark_theme(
        figure,
        title=title,
        height=450,
        legend_orientation="v",
        margin=dict(l=45, r=85, t=25 if not title else 50, b=40),
    )


def create_compound_usage_chart(
    stints: pd.DataFrame,
    session,
    title: str | None = None,
) -> go.Figure:
    """Render compound usage as a clean donut chart weighted by stint length."""
    if stints.empty:
        return empty_figure("Compound usage will populate once stint data is available.", height=320)

    compound_totals = (
        stints.groupby("Compound", as_index=False)["StintLength"]
        .sum()
        .sort_values("StintLength", ascending=False)
    )
    colors = [_compound_color(compound, session) for compound in compound_totals["Compound"]]
    fig = go.Figure(
        go.Pie(
            labels=compound_totals["Compound"],
            values=compound_totals["StintLength"],
            hole=0.6,
            marker=dict(colors=colors, line=dict(color="#1C1C1C", width=2)),
            hovertemplate="<b>%{label}</b><br>%{value} laps<extra></extra>",
            textinfo="label+percent",
            textfont=dict(color="#F5F5F5", size=11),
        )
    )
    return apply_dark_theme(
        fig,
        title=title,
        height=350,
        show_legend=True,
        legend_orientation="h",
        margin=dict(l=20, r=20, t=25 if not title else 45, b=25),
    )


def create_simulation_strategy_chart(
    actual_stints: list[dict[str, object]],
    simulated_stints: list[dict[str, object]],
    session,
) -> go.Figure:
    """Compare recorded and hypothetical tyre plans on a shared lap timeline."""
    figure = go.Figure()
    seen_compounds: set[str] = set()

    for strategy_name, stints in (("Actual", actual_stints), ("Simulated", simulated_stints)):
        for stint in stints:
            compound = str(stint["compound"]).upper()
            start_lap = int(stint["start_lap"])
            end_lap = int(stint["end_lap"])
            length = int(stint["stint_length"])
            show_legend = compound not in seen_compounds
            seen_compounds.add(compound)
            figure.add_trace(
                go.Bar(
                    x=[length],
                    y=[strategy_name],
                    base=[start_lap - 1],
                    orientation="h",
                    marker=dict(color=_compound_color(compound, session)),
                    name=compound.title(),
                    legendgroup=compound,
                    showlegend=show_legend,
                    text=[f"{compound} | {length} laps"],
                    textposition="inside",
                    insidetextanchor="middle",
                    hovertemplate=(
                        f"<b>{strategy_name}</b><br>{compound.title()} tyres<br>"
                        f"Laps {start_lap}-{end_lap}<br>{length} laps<extra></extra>"
                    ),
                )
            )

    figure.update_layout(barmode="stack", xaxis_title="Lap Number", yaxis_title="")
    return apply_dark_theme(figure, title="Tyre Plan: Actual vs What-if", height=285)


def create_simulation_delta_chart(simulated_laps: pd.DataFrame) -> go.Figure:
    """Show predicted time saved or lost relative to the recorded strategy."""
    if simulated_laps.empty:
        return empty_figure("Run a scenario to view its pace delta.", height=340)

    figure = go.Figure()
    valid = simulated_laps.dropna(subset=["CumulativeDelta"])
    if valid.empty:
        return empty_figure("Recorded timing is unavailable for the selected comparison laps.", height=340)
    figure.add_trace(
        go.Scatter(
            x=valid["LapNumber"],
            y=valid["CumulativeDelta"],
            mode="lines",
            name="Time saved",
            customdata=[[format_delta(value)] for value in valid["CumulativeDelta"]],
            line=dict(color="#E10600", width=3),
            fill="tozeroy",
            fillcolor="rgba(225, 6, 0, 0.16)",
            hovertemplate="Lap %{x}<br>%{customdata[0]} vs actual<extra></extra>",
        )
    )
    pit_laps = valid.loc[valid["IsPitLap"], ["LapNumber", "CumulativeDelta"]]
    if not pit_laps.empty:
        figure.add_trace(
            go.Scatter(
                x=pit_laps["LapNumber"],
                y=pit_laps["CumulativeDelta"],
                mode="markers",
                name="Pit lap",
                marker=dict(color="#F5F5F5", size=9, symbol="diamond"),
            customdata=[[format_delta(value)] for value in pit_laps["CumulativeDelta"]],
            hovertemplate="Pit on lap %{x}<br>%{customdata[0]} vs actual<extra></extra>",
            )
        )

    figure.add_hline(y=0, line_color="#B3B3B3", line_width=1)
    tickvals, signed_ticktext = _time_ticks(valid["CumulativeDelta"], format_delta)
    figure.update_layout(xaxis_title="Lap Number", yaxis_title="Cumulative time difference")
    figure.update_yaxes(tickvals=tickvals, ticktext=signed_ticktext)
    return apply_dark_theme(figure, title="Predicted Strategy Delta", height=350)


def create_simulation_time_chart(simulated_laps: pd.DataFrame) -> go.Figure:
    """Align recorded and what-if pace on the same valid lap timeline."""
    valid = simulated_laps.dropna(subset=["ActualLapTime", "SimulatedLapTime"])
    if valid.empty:
        return empty_figure("No equivalent recorded laps are available for this scenario.", height=500)

    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.13,
        subplot_titles=("Lap time", "Cumulative elapsed time"),
    )
    figure.add_trace(
        go.Scatter(x=valid["LapNumber"], y=valid["ActualLapTime"], mode="lines", name="Actual (recorded)", customdata=[[format_lap_time(value)] for value in valid["ActualLapTime"]], hovertemplate="Lap %{x}<br>Recorded %{customdata[0]}<extra></extra>", line=dict(color="#B3B3B3", width=2)),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(x=valid["LapNumber"], y=valid["SimulatedLapTime"], mode="lines", name="What-if (simulated)", customdata=[[format_lap_time(value)] for value in valid["SimulatedLapTime"]], hovertemplate="Lap %{x}<br>Simulated %{customdata[0]}<extra></extra>", line=dict(color="#E10600", width=2.5)),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(x=valid["LapNumber"], y=valid["ActualCumulativeTime"], mode="lines", name="Actual cumulative", customdata=[[format_duration(value)] for value in valid["ActualCumulativeTime"]], hovertemplate="Lap %{x}<br>Recorded %{customdata[0]}<extra></extra>", line=dict(color="#B3B3B3", width=2, dash="dot"), showlegend=False),
        row=2,
        col=1,
    )
    figure.add_trace(
        go.Scatter(x=valid["LapNumber"], y=valid["SimulatedCumulativeTime"], mode="lines", name="What-if cumulative", customdata=[[format_duration(value)] for value in valid["SimulatedCumulativeTime"]], hovertemplate="Lap %{x}<br>Simulated %{customdata[0]}<extra></extra>", line=dict(color="#E10600", width=2.5), showlegend=False),
        row=2,
        col=1,
    )
    pit_laps = valid.loc[valid["IsPitLap"]]
    if not pit_laps.empty:
        figure.add_trace(
            go.Scatter(x=pit_laps["LapNumber"], y=pit_laps["SimulatedLapTime"], mode="markers", name="What-if pit lap", marker=dict(color="#F5F5F5", size=8, symbol="diamond")),
            row=1,
            col=1,
        )
    figure.update_xaxes(title_text="Lap number", row=2, col=1)
    lap_ticks, lap_labels = _time_ticks(pd.concat([valid["ActualLapTime"], valid["SimulatedLapTime"]]), format_lap_time)
    duration_ticks, duration_labels = _time_ticks(pd.concat([valid["ActualCumulativeTime"], valid["SimulatedCumulativeTime"]]), format_duration)
    figure.update_yaxes(title_text="Lap time", tickvals=lap_ticks, ticktext=lap_labels, row=1, col=1)
    figure.update_yaxes(title_text="Elapsed time", tickvals=duration_ticks, ticktext=duration_labels, row=2, col=1)
    return apply_dark_theme(figure, title="Recorded vs What-if Race Time", height=540, legend_orientation="h")


def create_recorded_lap_time_chart(driver_laps: pd.DataFrame) -> go.Figure:
    """Render the recorded lap-time progression before a what-if run."""
    valid = driver_laps.dropna(subset=["LapNumber", "LapTimeSeconds"]).copy()
    valid = valid.loc[pd.to_numeric(valid["LapTimeSeconds"], errors="coerce") > 0]
    if valid.empty:
        return empty_figure("Recorded lap timing is unavailable for this driver.", height=330)
    figure = go.Figure(
        go.Scatter(
            x=valid["LapNumber"], y=valid["LapTimeSeconds"], mode="lines+markers", name="Recorded laps",
            customdata=[[format_lap_time(value)] for value in valid["LapTimeSeconds"]],
            hovertemplate="Lap %{x}<br>Recorded %{customdata[0]}<extra></extra>",
            line=dict(color="#B3B3B3", width=2.5), marker=dict(size=4),
        )
    )
    tickvals, ticktext = _time_ticks(valid["LapTimeSeconds"], format_lap_time)
    figure.update_layout(xaxis_title="Lap number", yaxis_title="Lap time")
    figure.update_yaxes(tickvals=tickvals, ticktext=ticktext)
    return apply_dark_theme(figure, title="Recorded Lap-Time Progression", height=350)


def create_recorded_degradation_chart(driver_laps: pd.DataFrame) -> go.Figure:
    """Render recorded tyre-age pace trends by stored stint."""
    valid = driver_laps.dropna(subset=["Stint", "TyreLife", "LapTimeSeconds"]).copy()
    valid = valid.loc[pd.to_numeric(valid["LapTimeSeconds"], errors="coerce") > 0]
    if valid.empty:
        return empty_figure("Recorded stint timing is unavailable for tyre-age analysis.", height=330)
    figure = go.Figure()
    for stint, group in valid.groupby("Stint", sort=True):
        compound = str(group["Compound"].iloc[0]).upper()
        figure.add_trace(
            go.Scatter(
                x=group["TyreLife"], y=group["LapTimeSeconds"], mode="lines+markers",
                name=f"Stint {int(stint)} {compound.title()}",
                customdata=[[format_lap_time(value)] for value in group["LapTimeSeconds"]],
                hovertemplate="Tyre age %{x}<br>Recorded %{customdata[0]}<extra></extra>",
                line=dict(color=_compound_color(compound, None), width=2.5), marker=dict(size=5),
            )
        )
    tickvals, ticktext = _time_ticks(valid["LapTimeSeconds"], format_lap_time)
    figure.update_layout(xaxis_title="Tyre age (laps)", yaxis_title="Lap time")
    figure.update_yaxes(tickvals=tickvals, ticktext=ticktext)
    return apply_dark_theme(figure, title="Recorded Stint and Tyre-Age Pace", height=350, legend_orientation="h")


def create_simulation_degradation_chart(simulated_laps: pd.DataFrame) -> go.Figure:
    """Show actual and simulated tyre-age pace trends by stint."""
    valid = simulated_laps.dropna(subset=["ActualLapTime", "SimulatedLapTime"])
    if valid.empty:
        return empty_figure("No equivalent lap timing is available for tyre degradation analysis.", height=360)

    figure = go.Figure()
    for stint, group in valid.groupby("ActualStint", sort=True):
        figure.add_trace(
            go.Scatter(
                x=group["ActualTyreLife"], y=group["ActualLapTime"], mode="lines+markers", customdata=[[format_lap_time(value)] for value in group["ActualLapTime"]], hovertemplate="Tyre age %{x}<br>Recorded %{customdata[0]}<extra></extra>",
                name=f"Actual stint {int(stint)}", line=dict(color="#8C8C8C", width=2, dash="dot"), marker=dict(size=5),
            )
        )
    for stint, group in valid.groupby("Stint", sort=True):
        figure.add_trace(
            go.Scatter(
                x=group["TyreLife"], y=group["SimulatedLapTime"], mode="lines+markers", customdata=[[format_lap_time(value)] for value in group["SimulatedLapTime"]], hovertemplate="Tyre age %{x}<br>Simulated %{customdata[0]}<extra></extra>",
                name=f"What-if stint {int(stint)}", line=dict(color="#E10600", width=2.5), marker=dict(size=5),
            )
        )
    tickvals, ticktext = _time_ticks(pd.concat([valid["ActualLapTime"], valid["SimulatedLapTime"]]), format_lap_time)
    figure.update_layout(xaxis_title="Tyre age (laps)", yaxis_title="Lap time")
    figure.update_yaxes(tickvals=tickvals, ticktext=ticktext)
    return apply_dark_theme(figure, title="Stint-by-Stint Tyre Degradation", height=380, legend_orientation="h")
