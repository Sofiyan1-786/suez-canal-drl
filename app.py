"""
Streamlit app — Vessel Scheduling and Inventory Optimization using RL
Thesis demo for Tarek Ammar, WSB University 2026
"""
from __future__ import annotations

import sys
import pathlib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Path setup
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "suez_drl"))

from suez_env import SuezCanalEnv
from train_fcfs import fcfs_action

# ============================================================================
# Page config
# ============================================================================
st.set_page_config(
    page_title="Suez Canal DRL Scheduling",
    page_icon="",
    layout="wide",
)

# ============================================================================
# Load cached data
# ============================================================================
@st.cache_data
def load_results():
    import json
    with open("suez_drl/results/summary.json") as f:
        summary = json.load(f)
    per_scenario = pd.read_csv("suez_drl/results/per_scenario.csv")
    return summary, per_scenario


summary_data, per_scenario_df = load_results()

# ============================================================================
# Sidebar navigation
# ============================================================================
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Home", "Digital Twin Simulator", "Results Dashboard",
     "Method Comparison", "About the Thesis"],
    index=0,
)

# ============================================================================
# PAGE: Home
# ============================================================================
if page == "Home":
    st.title("Vessel Scheduling and Inventory Optimization using Reinforcement Learning")
    st.markdown("**Tarek Ammar** | WSB University, Faculty of Economics and Political Science | 2026")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("The Problem")
        st.markdown(
            "The Suez Canal handles **12% of global trade** but still schedules ships "
            "**first-come, first-served**. A ship carrying \$500M of semiconductors "
            "waits as long as one carrying \$5M of iron ore. Every hour of delay costs "
            "the high-value ship about **\$96 in interest**."
        )

        st.subheader("Our Approach")
        st.markdown(
            "We built a **digital twin** of the Suez Canal and trained a Deep "
            "Reinforcement Learning (DRL) agent using **Proximal Policy Optimization (PPO)** "
            "to schedule ships based on cargo value."
        )

    with col2:
        st.subheader("Key Results")
        metrics = summary_data["summaries"]
        fcfs_cap = next(s for s in metrics if s["method"] == "FCFS")["mean_capital_usd"]
        milp_cap = next(s for s in metrics if s["method"] == "MILP-CBC")["mean_capital_usd"]
        savings = summary_data["savings_vs_fcfs"]["MILP-CBC"]

        st.metric("FCFS Baseline (Current Rule)", f"${fcfs_cap:,.0f}", help="Mean total capital cost per week")
        st.metric("MILP Optimal (Perfect Info)", f"${milp_cap:,.0f}", delta=f"-{savings:.1f}% vs FCFS")
        st.metric("MILP Savings vs FCFS", f"{savings:.1f}%",
                   help="Proves value-aware scheduling can recover 92.5% of capital cost")

        st.metric("PPO at 200k Steps", f"${next(s for s in metrics if s['method'] == 'PPO')['mean_capital_usd']:,.0f}",
                   delta=f"{summary_data['savings_vs_fcfs']['PPO']:+.1f}% vs FCFS",
                   help="Matches FCFS; needs more training to improve")

    st.markdown("---")
    st.subheader("Four Scheduling Methods Compared")

    method_data = []
    for s in metrics:
        method_data.append({
            "Method": s["method"],
            "Mean Capital Cost ($)": f"${s['mean_capital_usd']:,.0f}",
            "Mean Delay (h)": f"{s['mean_delay_h']:.1f}",
            "Ships/Episode": f"{s['mean_ships_served']:.1f}",
            "Savings vs FCFS": f"{summary_data['savings_vs_fcfs'][s['method']]:+.1f}%",
        })
    st.dataframe(pd.DataFrame(method_data), use_container_width=True, hide_index=True)

# ============================================================================
# PAGE: Digital Twin Simulator
# ============================================================================
elif page == "Digital Twin Simulator":
    st.title("Digital Twin Simulator")
    st.markdown("Run a single week of Suez Canal operations with different scheduling policies.")

    col_ctrl, col_viz = st.columns([1, 2])

    with col_ctrl:
        st.subheader("Settings")
        seed = st.number_input("Random Seed", min_value=1, max_value=99999, value=42)
        policy = st.selectbox("Scheduling Policy", ["FCFS", "Random", "Value-First (Greedy)"])

        st.markdown("**Environment Parameters**")
        n_days = st.slider("Episode Length (days)", 3, 14, 7)
        convoy_capacity = st.slider("Convoy Capacity", 4, 12, 8)
        arrival_rate = st.slider("Arrival Rate (ships/day)", 2.0, 10.0, 5.0, 0.5)
        disruption = st.checkbox("Enable Disruptions (50% slowdown every other convoy)")

        run_button = st.button("Run Simulation", type="primary")

    if run_button:
        env = SuezCanalEnv(
            n_days=n_days,
            convoy_capacity=convoy_capacity,
            arrival_rate_per_day=arrival_rate,
            seed=seed,
            disruption_active=disruption,
        )

        obs, info = env.reset(seed=seed)
        done = False
        step_log = []

        while not done:
            mask = info["action_mask"]

            if policy == "FCFS":
                action = fcfs_action(env)
            elif policy == "Random":
                valid_idx = np.where(mask)[0]
                action = np.zeros(env.max_waiting, dtype=np.int8)
                k = min(env.convoy_capacity, len(valid_idx))
                if k > 0:
                    chosen = np.random.choice(valid_idx, size=k, replace=False)
                    action[chosen] = 1
            elif policy == "Value-First (Greedy)":
                action = np.zeros(env.max_waiting, dtype=np.int8)
                n_queue = min(len(env.waiting_queue), env.max_waiting)
                if n_queue > 0:
                    values = [env.waiting_queue[i].cargo_value_usd for i in range(n_queue)]
                    ranked = np.argsort(values)[::-1]
                    for idx in ranked[:env.convoy_capacity]:
                        action[idx] = 1

            obs, reward, term, trunc, info = env.step(action)
            done = term or trunc
            step_log.append({
                "Step": env.current_step,
                "Time (h)": env.current_time_h,
                "Ships Served": info["ships_served_this_step"],
                "Queue Remaining": info["queue_length"],
                "Delay (h)": info["delay_this_step_h"],
                "Capital Cost ($)": info["capital_cost_this_step_usd"],
                "Reward": reward,
            })

        summary = env.episode_summary()
        step_df = pd.DataFrame(step_log)

        with col_viz:
            st.subheader("Simulation Results")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Delay", f"{summary['total_delay_h']:.0f} h")
            m2.metric("Capital Cost", f"${summary['total_capital_cost_usd']:,.0f}")
            m3.metric("Ships Served", f"{summary['total_ships_served']}")
            m4.metric("Cargo Value", f"${summary['total_cargo_value_usd']/1e6:.0f}M")

            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=("Ships Served per Convoy", "Queue Over Time",
                                "Capital Cost per Convoy", "Cumulative Cost"),
            )
            fig.add_trace(go.Bar(x=step_df["Step"], y=step_df["Ships Served"],
                                 name="Ships Served", marker_color="#1f77b4"), row=1, col=1)
            fig.add_trace(go.Scatter(x=step_df["Step"], y=step_df["Queue Remaining"],
                                     mode="lines+markers", name="Queue",
                                     line=dict(color="#ff7f0e")), row=1, col=2)
            fig.add_trace(go.Bar(x=step_df["Step"], y=step_df["Capital Cost ($)"],
                                 name="Capital Cost", marker_color="#2ca02c"), row=2, col=1)
            fig.add_trace(go.Scatter(x=step_df["Step"],
                                     y=step_df["Capital Cost ($)"].cumsum(),
                                     mode="lines+markers", name="Cumulative Cost",
                                     line=dict(color="#d62728")), row=2, col=2)

            fig.update_layout(height=500, showlegend=False)
            fig.update_xaxes(title_text="Convoy Step", row=2, col=1)
            fig.update_xaxes(title_text="Convoy Step", row=2, col=2)
            fig.update_yaxes(title_text="Count", row=1, col=1)
            fig.update_yaxes(title_text="Ships", row=1, col=2)
            fig.update_yaxes(title_text="USD", row=2, col=1)
            fig.update_yaxes(title_text="USD", row=2, col=2)
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Convoy-by-Convoy Log")
        st.dataframe(step_df, use_container_width=True, hide_index=True)

# ============================================================================
# PAGE: Results Dashboard
# ============================================================================
elif page == "Results Dashboard":
    st.title("Results Dashboard")
    st.markdown("50-scenario test campaign results (seeds 5000 to 5049).")

    tab1, tab2, tab3 = st.tabs(["Overview", "Per-Scenario Analysis", "Heatmap"])

    with tab1:
        st.subheader("Summary Statistics")
        metrics = summary_data["summaries"]

        cols = st.columns(4)
        for i, m in enumerate(metrics):
            with cols[i]:
                st.metric(
                    label=m["method"],
                    value=f"${m['mean_capital_usd']:,.0f}",
                    help=f"Mean delay: {m['mean_delay_h']:.1f}h | Ships: {m['mean_ships_served']:.1f}",
                )

        st.markdown("---")

        fig = go.Figure()
        colors = {"FCFS": "#d62728", "MILP-CBC": "#2ca02c", "GA": "#ff7f0e", "PPO": "#1f77b4"}
        for m in metrics:
            fig.add_trace(go.Bar(
                name=m["method"],
                x=[m["method"]],
                y=[m["mean_capital_usd"]],
                error_y=dict(type="data", array=[m["std_capital_usd"]], visible=True),
                marker_color=colors.get(m["method"], "gray"),
                text=[f"${m['mean_capital_usd']:,.0f}"],
                textposition="outside",
            ))
        fig.update_layout(
            title="Mean Total Cost of Capital per Week (lower is better)",
            yaxis_title="Mean TCC (USD)",
            height=450,
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

        fig2 = go.Figure()
        fcfs_cap = next(s for s in metrics if s["method"] == "FCFS")["mean_capital_usd"]
        for m in metrics:
            if m["method"] == "FCFS":
                continue
            pct = 100 * (fcfs_cap - m["mean_capital_usd"]) / fcfs_cap
            fig2.add_trace(go.Bar(
                name=m["method"],
                x=[m["method"]],
                y=[pct],
                marker_color=colors.get(m["method"], "gray"),
                text=[f"{pct:+.1f}%"],
                textposition="outside",
            ))
        fig2.update_layout(
            title="Savings vs FCFS (positive = better)",
            yaxis_title="Savings (%)",
            height=400,
            showlegend=False,
        )
        fig2.add_hline(y=0, line_dash="dash", line_color="black")
        st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        st.subheader("Per-Scenario Distribution")

        selected_methods = st.multiselect(
            "Select Methods",
            ["FCFS", "MILP-CBC", "GA", "PPO"],
            default=["FCFS", "MILP-CBC", "PPO"],
        )

        if selected_methods:
            filtered = per_scenario_df[per_scenario_df["method"].isin(selected_methods)]

            fig_box = px.box(
                filtered, x="method", y="total_capital_cost_usd",
                color="method",
                color_discrete_map=colors,
                title="Capital Cost Distribution by Method",
                labels={"total_capital_cost_usd": "Capital Cost (USD)", "method": "Method"},
            )
            fig_box.update_layout(height=450)
            st.plotly_chart(fig_box, use_container_width=True)

            fig_scatter = px.scatter(
                filtered, x="total_delay_h", y="total_capital_cost_usd",
                color="method", symbol="method",
                color_discrete_map=colors,
                title="Delay vs Capital Cost",
                labels={"total_delay_h": "Total Delay (hours)",
                        "total_capital_cost_usd": "Capital Cost (USD)"},
                opacity=0.7,
            )
            fig_scatter.update_layout(height=450)
            st.plotly_chart(fig_scatter, use_container_width=True)

    with tab3:
        st.subheader("Scenario Heatmap")
        pivot = per_scenario_df.pivot_table(
            index="scenario_seed", columns="method",
            values="total_capital_cost_usd",
        )
        fig_heat = px.imshow(
            pivot, aspect="auto",
            color_continuous_scale="RdYlGn_r",
            title="Capital Cost per Scenario (green = low cost)",
            labels=dict(x="Method", y="Scenario Seed", color="Cost (USD)"),
        )
        fig_heat.update_layout(height=600)
        st.plotly_chart(fig_heat, use_container_width=True)

# ============================================================================
# PAGE: Method Comparison
# ============================================================================
elif page == "Method Comparison":
    st.title("Method Comparison")
    st.markdown("Interactive comparison of the four scheduling approaches.")

    st.subheader("How Each Method Works")

    with st.expander("FCFS (First-Come-First-Served)", expanded=True):
        st.markdown(
            "**Current Suez Canal rule.** Each convoy takes the ships that have been "
            "waiting the longest, in arrival-time order, up to convoy capacity (8 ships). "
            "Ignores cargo value entirely."
        )

    with st.expander("MILP-CBC (Mixed-Integer Linear Programming)"):
        st.markdown(
            "**Offline optimum with perfect information.** Knows all ship arrivals "
            "in advance and solves a mathematical optimization to minimize total "
            "capital cost. Provides the theoretical lower bound."
        )

    with st.expander("GA (Genetic Algorithm)"):
        st.markdown(
            "**Evolutionary heuristic.** Encodes a 4D weight vector "
            "(cargo value, WACC, wait time, draught) and uses tournament selection "
            "with crossover and mutation to evolve better scheduling weights."
        )

    with st.expander("PPO (Proximal Policy Optimization)"):
        st.markdown(
            "**Deep Reinforcement Learning agent.** Uses a neural network "
            "to learn a scheduling policy from interaction with the digital twin. "
            "Trained for 200,000 timesteps with action masking."
        )

    st.markdown("---")

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Metric Comparison")
        metric_choice = st.radio(
            "Compare by",
            ["Capital Cost", "Delay", "Ships Served"],
            horizontal=True,
        )

        metrics = summary_data["summaries"]
        if metric_choice == "Capital Cost":
            vals = {m["method"]: m["mean_capital_usd"] for m in metrics}
            y_label = "Mean Capital Cost (USD)"
        elif metric_choice == "Delay":
            vals = {m["method"]: m["mean_delay_h"] for m in metrics}
            y_label = "Mean Delay (hours)"
        else:
            vals = {m["method"]: m["mean_ships_served"] for m in metrics}
            y_label = "Mean Ships Served per Episode"

        fig = go.Figure(go.Bar(
            x=list(vals.keys()), y=list(vals.values()),
            marker_color=[colors.get(k, "gray") for k in vals.keys()],
            text=[f"{v:,.1f}" for v in vals.values()],
            textposition="outside",
        ))
        fig.update_layout(yaxis_title=y_label, height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("Performance vs FCFS")
        fcfs_cap = next(s for s in metrics if s["method"] == "FCFS")["mean_capital_usd"]
        comparison_data = []
        for m in metrics:
            if m["method"] == "FCFS":
                continue
            pct = 100 * (fcfs_cap - m["mean_capital_usd"]) / fcfs_cap
            comparison_data.append({
                "Method": m["method"],
                "Savings (%)": pct,
                "Delay Reduction (%)": 100 * (next(s for s in metrics if s["method"] == "FCFS")["mean_delay_h"] - m["mean_delay_h"]) /
                                       next(s for s in metrics if s["method"] == "FCFS")["mean_delay_h"],
            })

        comp_df = pd.DataFrame(comparison_data)
        fig2 = go.Figure()
        for _, row in comp_df.iterrows():
            fig2.add_trace(go.Bar(
                name=row["Method"],
                x=["Capital Cost Savings", "Delay Reduction"],
                y=[row["Savings (%)"], row["Delay Reduction (%)"]],
                marker_color=colors.get(row["Method"], "gray"),
                text=[f"{row['Savings (%)']:+.1f}%", f"{row['Delay Reduction (%)']:+.1f}%"],
                textposition="outside",
            ))
        fig2.update_layout(
            title="Savings Relative to FCFS",
            yaxis_title="Improvement (%)",
            barmode="group",
            height=400,
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.subheader("Scenario-Level Comparison")
    st.markdown("Click a scenario seed to see detailed convoy-by-convoy breakdown.")

    selected_seed = st.selectbox(
        "Select Scenario Seed",
        sorted(per_scenario_df["scenario_seed"].unique()),
        index=0,
    )

    seed_data = per_scenario_df[per_scenario_df["scenario_seed"] == selected_seed]
    st.dataframe(seed_data[["method", "total_delay_h", "total_capital_cost_usd",
                             "total_ships_served", "total_cargo_value_usd"]].reset_index(drop=True),
                 use_container_width=True, hide_index=True)

# ============================================================================
# PAGE: About the Thesis
# ============================================================================
elif page == "About the Thesis":
    st.title("About the Thesis")

    st.markdown("""
    ### Vessel Scheduling and Inventory Optimization using Reinforcement Learning

    **Author:** Tarek Ammar (ID 60791)
    **Supervisor:** Adrian Kapczynski
    **University:** WSB University, Faculty of Economics and Political Science
    **Year:** 2026
    """)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Thesis Structure")
        chapters = [
            ("Chapter 1", "Introduction", "Background, research gap, objectives"),
            ("Chapter 2", "Literature Review", "SCF, AIS prediction, DRL for optimization"),
            ("Chapter 3", "Theoretical Framework", "MDP, PPO, action masking, IaR/TCC metrics"),
            ("Chapter 4", "Methodology", "Two-layer architecture: TFT + PPO"),
            ("Chapter 5", "Implementation", "Digital twin, baselines, training setup"),
            ("Chapter 6", "Results", "50-scenario benchmark, key findings"),
            ("Chapter 7", "Discussion", "Implications, limitations, future work"),
            ("Chapter 8", "Conclusion", "Summary, contributions"),
        ]
        for ch, title, desc in chapters:
            with st.container():
                st.markdown(f"**{ch}: {title}**")
                st.caption(desc)
                st.markdown("")

    with col2:
        st.subheader("Key Contributions")
        st.markdown("""
        1. **Value-based scheduling framework** with equity-constrained reward
        2. **Digital twin** of the Suez Canal (Gymnasium environment)
        3. **92.5% MILP-optimal savings** over FCFS baseline
        4. **PPO agent** matching FCFS at 200k steps; value-first heuristic at 6.5%
        """)

        st.subheader("Technology Stack")
        st.markdown("""
        - **Python 3.11** with Gymnasium, Stable-Baselines3, PuLP
        - **PPO** with action masking (sb3-contrib)
        - **MILP** with CBC solver (PuLP)
        - **TFT** for arrival prediction (PyTorch)
        """)

    st.markdown("---")
    st.subheader("Data Sources")
    st.markdown("""
    - Cargo mix calibrated to 2024 to 2025 Suez Canal traffic statistics
    - WACC sampled from N(7%, 1.5%), clipped to [3%, 12%]
    - 50 test scenarios with seeds 5000 to 5049
    - Hydrodynamic constraints (squat effect, bank suction)
    """)
