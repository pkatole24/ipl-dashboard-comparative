from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data" / "current"

PLAYER_COLORS = ["#0B6E69", "#C2410C"]
BACKGROUND_COLOR = "#D8DEE9"
PHASE_ORDER = ["Powerplay", "Death last 5"]
OWN_PHASE_ORDER = list(range(1, 7))
OWN_PHASE_LABELS = [f"{(phase - 1) * 15 + 1}-{phase * 15}" for phase in OWN_PHASE_ORDER]


st.set_page_config(
    page_title="IPL Batsman Comparison",
    page_icon=None,
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_csv(name: str, parse_dates: tuple[str, ...] = ()) -> pd.DataFrame:
    path = DATA_DIR / name
    df = pd.read_csv(path)
    for column in parse_dates:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def load_metadata() -> dict:
    path = DATA_DIR / "last_updated.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def default_player(players: list[str], preferred: list[str], fallback_contains: str, fallback_index: int) -> str:
    lowered = {player.lower(): player for player in players}
    for name in preferred:
        if name.lower() in lowered:
            return lowered[name.lower()]
    matches = [player for player in players if fallback_contains.lower() in player.lower()]
    if matches:
        return matches[0]
    return players[min(fallback_index, len(players) - 1)]


def selected_players_frame(df: pd.DataFrame, player_a: str, player_b: str, column: str = "player") -> pd.DataFrame:
    return df[df[column].isin([player_a, player_b])].copy()


def format_number(value: float | int | None, digits: int = 2) -> str:
    if pd.isna(value):
        return "-"
    return f"{float(value):,.{digits}f}"


def phase_label(phase: int) -> str:
    return f"{(int(phase) - 1) * 15 + 1}-{int(phase) * 15}"


def complete_phase_grid(
    df: pd.DataFrame,
    players: list[str],
    value_columns: list[str],
    zero_columns: list[str] | None = None,
) -> pd.DataFrame:
    zero_columns = zero_columns or []
    grid = pd.MultiIndex.from_product(
        [players, OWN_PHASE_ORDER],
        names=["player", "own_15_ball_phase"],
    ).to_frame(index=False)
    grid["phase_label"] = grid["own_15_ball_phase"].map(phase_label)

    completed = grid.merge(df, on=["player", "own_15_ball_phase", "phase_label"], how="left")
    for column in value_columns:
        if column not in completed.columns:
            completed[column] = pd.NA
    for column in zero_columns:
        if column in completed.columns:
            completed[column] = completed[column].fillna(0)
    return completed


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    valid = values.notna() & weights.notna() & weights.gt(0)
    if not valid.any():
        return pd.NA
    return float((values[valid] * weights[valid]).sum() / weights[valid].sum())


def build_phase_summary(selected_phase: pd.DataFrame, players: list[str], min_balls: int) -> pd.DataFrame:
    summary = (
        selected_phase.groupby(["player", "own_15_ball_phase", "phase_label"], as_index=False)
        .agg(
            runs=("runs", "sum"),
            balls=("balls", "sum"),
            expected_runs=("expected_runs", "sum"),
            runs_above_expected=("runs_above_expected", "sum"),
        )
    )
    summary = summary[summary["own_15_ball_phase"].isin(OWN_PHASE_ORDER)]
    summary = complete_phase_grid(
        summary,
        players,
        value_columns=["runs", "balls", "expected_runs", "runs_above_expected"],
        zero_columns=["runs", "balls", "expected_runs", "runs_above_expected"],
    )

    has_enough_balls = summary["balls"].gt(0) & summary["balls"].ge(min_balls)
    summary["strike_rate"] = summary["runs"] * 100 / summary["balls"].clip(lower=1)
    summary["expected_sr"] = summary["expected_runs"] * 100 / summary["balls"].clip(lower=1)
    summary["runs_above_expected_per_100"] = summary["runs_above_expected"] * 100 / summary["balls"].clip(lower=1)
    for column in [
        "runs",
        "expected_runs",
        "runs_above_expected",
        "strike_rate",
        "expected_sr",
        "runs_above_expected_per_100",
    ]:
        summary.loc[~has_enough_balls, column] = pd.NA
    return summary


def build_context_phase_summary(selected_context: pd.DataFrame, players: list[str], min_balls: int) -> pd.DataFrame:
    rows = []
    context = selected_context[selected_context["own_15_ball_phase"].isin(OWN_PHASE_ORDER)].copy()
    for (player, phase, label), group in context.groupby(["player", "own_15_ball_phase", "phase_label"], sort=False):
        weights = group["balls"].astype("float64")
        rows.append(
            {
                "player": player,
                "own_15_ball_phase": phase,
                "phase_label": label,
                "runs": group["runs"].sum(),
                "balls": group["balls"].sum(),
                "sr_points_vs_match": weighted_mean(group["sr_points_vs_match"], weights),
                "sr_points_vs_teammates": weighted_mean(group["sr_points_vs_teammates"], weights),
                "sr_points_vs_league_phase": weighted_mean(group["sr_points_vs_league_phase"], weights),
                "difficulty_adjusted_sr": weighted_mean(group["difficulty_adjusted_sr"], weights),
                "difficulty_adjustment_delta": weighted_mean(group.get("difficulty_adjustment_delta", pd.Series(dtype="float64")), weights),
            }
        )

    summary = pd.DataFrame(
        rows,
        columns=[
            "player",
            "own_15_ball_phase",
            "phase_label",
            "runs",
            "balls",
            "sr_points_vs_match",
            "sr_points_vs_teammates",
            "sr_points_vs_league_phase",
            "difficulty_adjusted_sr",
            "difficulty_adjustment_delta",
        ],
    )
    summary = complete_phase_grid(
        summary,
        players,
        value_columns=[
            "runs",
            "balls",
            "sr_points_vs_match",
            "sr_points_vs_teammates",
            "sr_points_vs_league_phase",
            "difficulty_adjusted_sr",
            "difficulty_adjustment_delta",
        ],
        zero_columns=["runs", "balls"],
    )
    has_enough_balls = summary["balls"].gt(0) & summary["balls"].ge(min_balls)
    for column in [
        "sr_points_vs_match",
        "sr_points_vs_teammates",
        "sr_points_vs_league_phase",
        "difficulty_adjusted_sr",
        "difficulty_adjustment_delta",
    ]:
        summary.loc[~has_enough_balls, column] = pd.NA
    return summary


def powerplay_death_chart(pd_filtered: pd.DataFrame, selected: list[str]) -> go.Figure:
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=PHASE_ORDER,
        shared_yaxes=True,
        horizontal_spacing=0.08,
    )
    phase_to_col = {"Powerplay": 1, "Death last 5": 2}

    for phase_name in PHASE_ORDER:
        phase_rows = pd_filtered[pd_filtered["phase_group"].eq(phase_name)]
        col = phase_to_col[phase_name]

        background = phase_rows[~phase_rows["batter"].isin(selected)]
        fig.add_trace(
            go.Scatter(
                x=background["balls"],
                y=background["sr_points_above_league_phase"],
                mode="markers",
                marker={"size": 7, "color": "#CBD5E1", "opacity": 0.65},
                text=background["batter"],
                customdata=background[["runs", "raw_sr", "leave_one_out_league_sr", "runs_above_league_phase_rate"]],
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Balls: %{x}<br>"
                    "SR points vs phase: %{y:.2f}<br>"
                    "Runs: %{customdata[0]}<br>"
                    "Raw SR: %{customdata[1]:.2f}<br>"
                    "League phase SR: %{customdata[2]:.2f}<br>"
                    "Runs above phase rate: %{customdata[3]:.2f}<extra></extra>"
                ),
                showlegend=False,
            ),
            row=1,
            col=col,
        )

        for player, color in zip(selected, PLAYER_COLORS):
            player_rows = phase_rows[phase_rows["batter"].eq(player)]
            if player_rows.empty:
                continue
            fig.add_trace(
                go.Scatter(
                    x=player_rows["balls"],
                    y=player_rows["sr_points_above_league_phase"],
                    mode="markers+text",
                    text=player_rows["batter"],
                    textposition="top center",
                    marker={"size": 16, "color": color, "line": {"width": 2, "color": "#111111"}},
                    customdata=player_rows[["runs", "raw_sr", "leave_one_out_league_sr", "runs_above_league_phase_rate"]],
                    hovertemplate=(
                        "<b>%{text}</b><br>"
                        "Balls: %{x}<br>"
                        "SR points vs phase: %{y:.2f}<br>"
                        "Runs: %{customdata[0]}<br>"
                        "Raw SR: %{customdata[1]:.2f}<br>"
                        "League phase SR: %{customdata[2]:.2f}<br>"
                        "Runs above phase rate: %{customdata[3]:.2f}<extra></extra>"
                    ),
                    name=f"{player} - {phase_name}",
                    showlegend=False,
                ),
                row=1,
                col=col,
            )

    fig.update_layout(
        title="Balls Faced vs SR Points Above Leave-One-Out League Phase Rate",
        height=520,
    )
    fig.update_xaxes(title_text="Balls faced in phase", row=1, col=1)
    fig.update_xaxes(title_text="Balls faced in phase", row=1, col=2)
    fig.update_yaxes(title_text="SR points above league phase", row=1, col=1)
    return fig


def add_live_quartiles(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        out = summary.copy()
        out["quartile"] = pd.Series(dtype="object")
        return out

    out = summary.copy()
    out["quartile"] = pd.NA
    out["quartile_number"] = pd.NA
    for phase_name, idx in out.groupby("phase_group").groups.items():
        phase_values = out.loc[idx, "sr_points_above_league_phase"]
        if len(phase_values) < 4:
            ranks = phase_values.rank(method="first", ascending=True)
            denom = max(len(phase_values), 1)
            quartile_number = ((ranks - 1) * 4 / denom).astype("int64") + 1
        else:
            quartile_number = pd.qcut(
                phase_values.rank(method="first"),
                4,
                labels=[1, 2, 3, 4],
            ).astype("int64")
        out.loc[idx, "quartile_number"] = quartile_number

    out["quartile"] = out["quartile_number"].map(
        {1: "Q1 lowest", 2: "Q2", 3: "Q3", 4: "Q4 highest"}
    )
    return out


def latest_match_from_matches(player_match: pd.DataFrame) -> dict[str, str | None]:
    if player_match.empty:
        return {"date": None, "teams": None, "match_id": None}

    matches = player_match.copy()
    matches["start_date"] = pd.to_datetime(matches["start_date"], errors="coerce")
    matches["match_id_sort"] = pd.to_numeric(matches["match_id"], errors="coerce")
    latest = matches.sort_values(["start_date", "match_id_sort"], kind="stable").iloc[-1]
    latest_rows = matches[matches["match_id"].astype("string").eq(str(latest["match_id"]))]
    teams = latest_rows.sort_values("innings", kind="stable")["batting_team"].dropna().astype(str).drop_duplicates().tolist()
    if len(teams) < 2 and "opposition" in latest_rows.columns:
        teams.extend(latest_rows["opposition"].dropna().astype(str).drop_duplicates().tolist())
        teams = list(dict.fromkeys(teams))

    return {
        "date": latest["start_date"].date().isoformat(),
        "teams": " vs ".join(teams[:2]) if teams else None,
        "match_id": str(latest["match_id"]),
    }


def metric_cards(player_row: pd.Series, label: str) -> None:
    st.markdown(f"#### {label}")
    cols = st.columns(4)
    cols[0].metric("Runs", format_number(player_row.get("actual_runs"), 0))
    cols[1].metric("Balls", format_number(player_row.get("balls_faced"), 0))
    cols[2].metric("Strike rate", format_number(player_row.get("strike_rate"), 2))
    cols[3].metric("Expected runs", format_number(player_row.get("expected_runs"), 2))

    cols = st.columns(3)
    cols[0].metric("Runs Above Expected", format_number(player_row.get("runs_above_expected"), 2))
    cols[1].metric("RAE per 30 balls", format_number(player_row.get("runs_above_expected_per_30_balls"), 2))
    cols[2].metric("RAE per 100 balls", format_number(player_row.get("runs_above_expected_per_100_balls"), 2))


def line_metric_chart(df: pd.DataFrame, y: str, title: str, y_title: str) -> go.Figure:
    plot_df = df[df["own_15_ball_phase"].isin(OWN_PHASE_ORDER)].copy()
    plot_df["own_innings_phase"] = plot_df["phase_label"]
    fig = px.line(
        plot_df.sort_values(["player", "own_15_ball_phase"]),
        x="own_15_ball_phase",
        y=y,
        color="player",
        markers=True,
        color_discrete_sequence=PLAYER_COLORS,
        title=title,
        custom_data=[
            "own_innings_phase",
            "balls",
            "runs",
            "strike_rate" if "strike_rate" in plot_df.columns else "balls",
            "expected_runs" if "expected_runs" in plot_df.columns else "balls",
            "runs_above_expected" if "runs_above_expected" in plot_df.columns else "balls",
        ],
    )
    fig.update_traces(
        hovertemplate=(
            "<b>%{fullData.name}</b><br>"
            "Own innings phase: %{customdata[0]}<br>"
            "Total balls in this phase: %{customdata[1]}<br>"
            "Runs: %{customdata[2]}<br>"
            "Strike rate: %{customdata[3]:.2f}<br>"
            "Expected runs: %{customdata[4]:.2f}<br>"
            "Runs Above Expected: %{customdata[5]:.2f}<br>"
            f"{y_title}: " + "%{y:.2f}<extra></extra>"
        )
    )
    fig.update_layout(xaxis_title="Own-ball phase across innings", yaxis_title=y_title, legend_title=None)
    fig.update_xaxes(
        tickmode="array",
        tickvals=OWN_PHASE_ORDER,
        ticktext=OWN_PHASE_LABELS,
        range=[0.8, 6.2],
    )
    fig.update_traces(connectgaps=False)
    return fig


def explain(text: str) -> None:
    st.markdown(f"<div class='chart-note'>{text}</div>", unsafe_allow_html=True)


def phase_coverage_note(source_df: pd.DataFrame, filtered_df: pd.DataFrame, min_balls: int, players: list[str]) -> None:
    if source_df.empty:
        explain("No own 15-ball phase data is available for the selected batters.")
        return

    lines = [f"The x-axis always shows all own 15-ball phases through `76-90`. Minimum balls is currently `{min_balls}`."]
    for player in players:
        player_grid = filtered_df[filtered_df["player"].eq(player)].sort_values("own_15_ball_phase")
        faced = player_grid[player_grid["balls"].gt(0)]
        plotted = player_grid[player_grid["balls"].gt(0) & player_grid["balls"].ge(min_balls)]
        low_sample = player_grid[player_grid["balls"].gt(0) & player_grid["balls"].lt(min_balls)]

        if faced.empty:
            lines.append(f"**{player}** has no balls in these phases.")
            continue

        faced_labels = ", ".join(f"`{phase_label(phase)}`" for phase in faced["own_15_ball_phase"])
        plotted_text = (
            ", ".join(f"`{phase_label(phase)}`" for phase in plotted["own_15_ball_phase"])
            if not plotted.empty
            else "none"
        )
        line = f"**{player}** has balls in {faced_labels}; plotted phases: {plotted_text}."
        if not low_sample.empty:
            hidden = ", ".join(
                f"`{row.phase_label}` ({int(row.balls)} balls)" for row in low_sample.itertuples()
            )
            line += f" Hidden by the minimum-balls filter: {hidden}."
        lines.append(line)

    explain(" ".join(lines))


def main() -> None:
    season = load_csv("player_season.csv")
    phase = load_csv("phase_15_ball.csv", parse_dates=("start_date",))
    context = load_csv("context_15_ball.csv", parse_dates=("start_date",))
    player_match = load_csv("player_match.csv", parse_dates=("start_date",))
    pd_summary = load_csv("powerplay_death_summary.csv", parse_dates=("first_date", "last_date"))
    worms = load_csv("powerplay_death_worms.csv", parse_dates=("start_date",))
    metadata = load_metadata()

    players = sorted(season["player"].dropna().unique().tolist())
    default_a = default_player(players, ["V Kohli", "Virat Kohli"], "Kohli", 0)
    default_b = default_player(players, ["RG Sharma", "Rohit Sharma"], "Sharma", 1)

    st.title("IPL Batsman Comparison")
    st.markdown(
        """
        <style>
        .chart-note {
            margin: -0.4rem 0 1.25rem 0;
            padding: 0.8rem 1rem;
            border-left: 4px solid #0B6E69;
            background: #F8FAFC;
            color: #334155;
            font-size: 0.94rem;
            line-height: 1.45;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    generated = metadata.get("generated_at_utc")
    latest_match = {
        "date": metadata.get("latest_match_date"),
        "teams": metadata.get("latest_match_teams"),
        "match_id": metadata.get("latest_match_id"),
    }
    if not latest_match["date"]:
        latest_match = latest_match_from_matches(player_match)

    indicator_parts = [f"Current season: {metadata.get('season', 'unknown')}"]
    if latest_match["date"]:
        match_text = latest_match["date"]
        if latest_match["teams"]:
            match_text = f"{latest_match['teams']} ({match_text})"
        indicator_parts.append(f"Data through: {match_text}")
    if generated:
        indicator_parts.append(f"Last refreshed: {generated}")
    st.caption(" | ".join(indicator_parts))

    with st.sidebar:
        st.header("Compare")
        player_a = st.selectbox("Batter A", players, index=players.index(default_a))
        player_b_options = [player for player in players if player != player_a]
        player_b_default = default_b if default_b in player_b_options else player_b_options[0]
        player_b = st.selectbox("Batter B", player_b_options, index=player_b_options.index(player_b_default))

        min_balls = st.slider("Minimum balls", min_value=0, max_value=100, value=6, step=1)
        metric_choice = st.selectbox(
            "Own 15-ball chart metric",
            [
                "Runs Above Expected",
                "Runs Above Expected per 100",
                "Strike rate",
                "SR points vs match",
                "SR points vs teammates",
                "SR points vs league phase",
                "Difficulty-adjusted strike rate",
            ],
        )
        show_adjusted = st.toggle("Adjusted views", value=True)

    selected = [player_a, player_b]
    selected_season = selected_players_frame(season, player_a, player_b)
    selected_phase = selected_players_frame(phase, player_a, player_b)
    selected_context = selected_players_frame(context, player_a, player_b)

    st.subheader("Season Snapshot")
    card_cols = st.columns(2)
    for col, player in zip(card_cols, selected):
        row = selected_season[selected_season["player"].eq(player)]
        with col:
            if row.empty:
                st.warning(f"No season row for {player}.")
            else:
                metric_cards(row.iloc[0], player)

    st.divider()
    tab_phase, tab_adjusted, tab_power, tab_worm, tab_table = st.tabs(
        [
            "Own 15-Ball Phases",
            "Context Adjustments",
            "Powerplay and Death",
            "Worm Chart",
            "Match Details",
        ]
    )

    with tab_phase:
        phase_summary = build_phase_summary(selected_phase, selected, min_balls)
        context_phase_summary = build_context_phase_summary(selected_context, selected, min_balls)

        phase_metric_options = {
            "Runs Above Expected": (
                phase_summary,
                "runs_above_expected",
                "Runs Above Expected",
            ),
            "Runs Above Expected per 100": (
                phase_summary,
                "runs_above_expected_per_100",
                "Runs Above Expected per 100 balls",
            ),
            "Strike rate": (
                phase_summary,
                "strike_rate",
                "Strike rate",
            ),
            "SR points vs match": (
                context_phase_summary,
                "sr_points_vs_match",
                "SR points vs match",
            ),
            "SR points vs teammates": (
                context_phase_summary,
                "sr_points_vs_teammates",
                "SR points vs teammates",
            ),
            "SR points vs league phase": (
                context_phase_summary,
                "sr_points_vs_league_phase",
                "SR points vs league phase",
            ),
            "Difficulty-adjusted strike rate": (
                context_phase_summary,
                "difficulty_adjusted_sr",
                "Difficulty-adjusted strike rate",
            ),
        }
        chart_df, y_col, y_title = phase_metric_options[metric_choice]
        st.plotly_chart(
            line_metric_chart(chart_df, y_col, f"{y_title} by Own 15-Ball Phase", y_title),
            use_container_width=True,
        )
        explain(
            "Each point is summed across innings. Phase `1-15` means the batter's first 15 balls in each innings, "
            "not his first 15 balls of the season. So `49` balls in phase `1-15` means 49 total balls across all innings "
            "where those balls were within balls `1-15` of that innings."
        )
        phase_coverage_note(selected_phase, chart_df, min_balls, selected)
        if metric_choice == "Runs Above Expected":
            explain(
                "**Runs Above Expected** compares what a batter actually scored with what the model expected an average IPL batter "
                "to score from the same balls. The model looks only at the situation before each ball: balls already faced by the batter, "
                "balls left in the innings, wickets in hand, current scoring rate, and whether the innings is in the powerplay, middle overs, "
                "or death overs. It does not use the batter's name, team, venue, or reputation. "
                "Example: if a batter scores 24 from balls `16-30`, and the model says those balls were worth 18.5 expected runs, "
                "that phase is `+5.5` Runs Above Expected. Positive means the batter scored more than expected for the situations he faced; "
                "negative means he scored less. This is designed to ask: given the same context, did this batter add runs above what would normally be expected?"
            )
        elif metric_choice == "Runs Above Expected per 100":
            explain(
                "**Runs Above Expected per 100** puts the same idea on a rate scale. "
                "Example: `+6` Runs Above Expected from 30 balls becomes `+20 per 100 balls`, "
                "which makes unequal phase samples easier to compare."
            )
        elif metric_choice == "Strike rate":
            explain(
                "**Strike rate** is runs per 100 balls in each batter's own innings phase. "
                "Example: 28 runs from balls 1-15 is `28 / 15 * 100 = 186.7`. "
                "The phase labels are the batter's own balls, not team overs."
            )
        elif metric_choice == "SR points vs match":
            explain(
                "**SR points vs match** compares the batter's phase strike rate with the overall match scoring rate. "
                "Example: a 150 phase SR in a match where everyone scored at 130 is `+20`; "
                "the same 150 in a 180-SR match is `-30`."
            )
        elif metric_choice == "SR points vs teammates":
            explain(
                "**SR points vs teammates** compares the batter's phase strike rate with his teammates in the same innings, "
                "excluding that batter. Example: if the batter's phase SR is 170 and the rest of the team scored at 145, "
                "the phase is `+25`."
            )
        elif metric_choice == "SR points vs league phase":
            explain(
                "**SR points vs league phase** compares a batter with the league average for the same own-ball phase. "
                "Example: balls 31-45 are compared with other batters' balls 31-45, not with powerplay or death-over balls."
            )
        elif metric_choice == "Difficulty-adjusted strike rate":
            explain(
                "**Difficulty-adjusted strike rate** gives extra credit for scoring in slower matches and reduces credit in very fast-scoring matches. "
                "Example: 150 SR in a low-scoring match can adjust upward, while 150 in a run-fest can adjust downward."
            )

        scatter_df = selected_phase[selected_phase["balls"].ge(max(1, min_balls))].copy()
        scatter_df = scatter_df[scatter_df["own_15_ball_phase"].isin(OWN_PHASE_ORDER)]
        scatter_df["own_innings_phase"] = scatter_df["phase_label"]
        fig = px.scatter(
            scatter_df,
            x="own_15_ball_phase",
            y="strike_rate",
            color="player",
            size="balls",
            color_discrete_sequence=PLAYER_COLORS,
            custom_data=[
                "own_innings_phase",
                "balls",
                "runs",
                "expected_runs",
                "runs_above_expected",
                "start_date",
                "opposition",
                "venue",
            ],
            title="Individual Innings Phase Dots",
        )
        fig.update_traces(
            hovertemplate=(
                "<b>%{fullData.name}</b><br>"
                "Own innings phase: %{customdata[0]}<br>"
                "Balls in this innings phase: %{customdata[1]}<br>"
                "Runs: %{customdata[2]}<br>"
                "Strike rate: %{y:.2f}<br>"
                "Expected runs: %{customdata[3]:.2f}<br>"
                "Runs Above Expected: %{customdata[4]:.2f}<br>"
                "Date: %{customdata[5]}<br>"
                "Opposition: %{customdata[6]}<br>"
                "Venue: %{customdata[7]}<extra></extra>"
            )
        )
        fig.update_layout(xaxis_title="Own-ball phase within innings", yaxis_title="Strike rate", legend_title=None)
        fig.update_xaxes(
            tickmode="array",
            tickvals=OWN_PHASE_ORDER,
            ticktext=OWN_PHASE_LABELS,
            range=[0.8, 6.2],
        )
        st.plotly_chart(fig, use_container_width=True)
        explain(
            "**Individual innings phase dots** show every qualifying innings chunk. "
            "Bigger dots mean more balls in that chunk. Example: a short `31-38` phase is kept if the innings ended there, "
            "but it will appear smaller than a full 15-ball phase."
        )

    with tab_adjusted:
        if show_adjusted:
            context_summary = build_context_phase_summary(selected_context, selected, min_balls)
            metrics = [
                ("sr_points_vs_match", "SR Points vs Match"),
                ("sr_points_vs_teammates", "SR Points vs Teammates"),
                ("sr_points_vs_league_phase", "SR Points vs League Phase"),
                ("difficulty_adjusted_sr", "Difficulty-Adjusted Strike Rate"),
            ]
            for metric_col, title in metrics:
                st.plotly_chart(line_metric_chart(context_summary, metric_col, title, title), use_container_width=True)
                if metric_col == "sr_points_vs_match":
                    explain(
                        "This adjusts for match scoring environment. Example: `+15` means the batter's phase SR was "
                        "15 points faster than the match as a whole."
                    )
                elif metric_col == "sr_points_vs_teammates":
                    explain(
                        "This adjusts for team innings context. Example: `-10` means the batter's phase SR was "
                        "10 points slower than teammates in the same innings after removing his own balls."
                    )
                elif metric_col == "sr_points_vs_league_phase":
                    explain(
                        "This adjusts for where the batter was in his innings. Example: balls `46-60` are compared with "
                        "the league's balls `46-60`, so finishers and openers are judged against the right phase."
                    )
                elif metric_col == "difficulty_adjusted_sr":
                    explain(
                        "This converts raw scoring through a match-difficulty index. Example: a phase in a slow match "
                        "gets boosted relative to the same raw SR in a high-scoring match."
                    )
        else:
            st.info("Turn on adjusted views in the sidebar to show context adjustment charts.")

    with tab_power:
        pd_filtered = pd_summary[pd_summary["balls"].ge(min_balls)].copy()
        pd_filtered = add_live_quartiles(pd_filtered)
        st.plotly_chart(powerplay_death_chart(pd_filtered, selected), use_container_width=True)
        explain(
            "**SR points above leave-one-out league phase rate** compares each batter with everyone else in that phase. "
            "The batter's own balls are removed from the league baseline. Example: if the rest of the league scores at 140 "
            "in the powerplay and the batter scores at 165, he is `+25`."
        )

        q_selected = pd_filtered[pd_filtered["batter"].isin(selected)].copy()
        st.dataframe(
            q_selected[
                [
                    "phase_group",
                    "batter",
                    "runs",
                    "balls",
                    "raw_sr",
                    "leave_one_out_league_sr",
                    "sr_points_above_league_phase",
                    "runs_above_league_phase_rate",
                    "quartile",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

    with tab_worm:
        worm_phase = st.radio("Worm phase", ["Powerplay", "Death last 5"], horizontal=True)

        phase_worms = worms[worms["phase_group"].eq(worm_phase)].copy()
        qualified = phase_worms[phase_worms["quartile"].notna()]
        background = qualified[~qualified["batter"].isin(selected)].copy()
        fig = go.Figure()
        for batter, group in background.groupby("batter"):
            fig.add_trace(
                go.Scatter(
                    x=group["own_phase_ball_no"],
                    y=group["cum_runs_above_league_phase_rate"],
                    mode="lines",
                    line={"color": BACKGROUND_COLOR, "width": 1},
                    opacity=0.35,
                    customdata=group[["start_date", "runs_off_bat"]],
                    hovertemplate=(
                        "<b>%{fullData.name}</b><br>"
                        "Season phase ball number: %{x}<br>"
                        "Date: %{customdata[0]}<br>"
                        "Runs on ball: %{customdata[1]}<br>"
                        "Cumulative runs above phase rate: %{y:.2f}<extra></extra>"
                    ),
                    name=batter,
                    showlegend=False,
                )
            )
        for player, color in zip(selected, PLAYER_COLORS):
            group = phase_worms[phase_worms["batter"].eq(player)]
            if not group.empty:
                fig.add_trace(
                    go.Scatter(
                        x=group["own_phase_ball_no"],
                        y=group["cum_runs_above_league_phase_rate"],
                        mode="lines+markers",
                        line={"color": color, "width": 3},
                        marker={"size": 5},
                        name=player,
                        customdata=group[
                            [
                                "start_date",
                                "batting_team",
                                "bowling_team",
                                "runs_off_bat",
                                "cum_runs",
                                "leave_one_out_league_sr",
                            ]
                        ],
                        hovertemplate=(
                            "<b>%{fullData.name}</b><br>"
                            "Season phase ball number: %{x}<br>"
                            "Date: %{customdata[0]}<br>"
                            "Match: %{customdata[1]} vs %{customdata[2]}<br>"
                            "Runs on ball: %{customdata[3]}<br>"
                            "Cumulative runs: %{customdata[4]}<br>"
                            "League phase SR: %{customdata[5]:.2f}<br>"
                            "Cumulative runs above phase rate: %{y:.2f}<extra></extra>"
                        ),
                    )
                )
        fig.update_layout(
            title=f"{worm_phase} Worm: Cumulative Runs Above League Phase Rate",
            xaxis_title="Cumulative balls faced in phase this season",
            yaxis_title="Cumulative runs above league phase rate",
        )
        st.plotly_chart(fig, use_container_width=True)
        explain(
            "**Worm chart** is season-cumulative within the selected phase. The x-axis does not reset each match. "
            "If a batter faces one powerplay ball in match 1 and another in match 2, the second point is `x = 2`. "
            "A value like `(1, -1.6)` means that after one ball in that phase, the batter was 1.6 runs below the league phase rate. "
            "A rising line means he is adding runs faster than the phase baseline; a falling line means he is losing ground."
        )

    with tab_table:
        table = selected_phase[selected_phase["balls"].ge(max(1, min_balls))].copy()
        table = table[table["own_15_ball_phase"].isin(OWN_PHASE_ORDER)]
        table = table.sort_values(["start_date", "player", "match_id", "innings", "own_15_ball_phase"])
        st.dataframe(
            table[
                [
                    "start_date",
                    "player",
                    "batting_team",
                    "opposition",
                    "venue",
                    "innings",
                    "phase_label",
                    "runs",
                    "balls",
                    "strike_rate",
                    "expected_runs",
                    "runs_above_expected",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Download match detail CSV",
            data=table.to_csv(index=False).encode("utf-8"),
            file_name="batsman_comparison_match_details.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
