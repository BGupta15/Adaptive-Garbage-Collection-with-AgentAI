import streamlit as st
import pandas as pd
import json
import os
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="GC Agent Monitor", layout="wide")

# Available log files mapping
LOG_FILES = {
    "Cycle Generator": "cycle_generator_log.jsonl",
    "Unbounded Cache": "unbounded_cache_log.jsonl",
    "Normal Churn": "normal_churn_log.jsonl"
}

st.sidebar.title("⚙️ GC Dashboard Controls")
selected_workload = st.sidebar.radio("Select Workload to Analyze", list(LOG_FILES.keys()))
file_path = LOG_FILES[selected_workload]

# Function to parse the .jsonl logs
def load_data(path):
    if not os.path.exists(path):
        return pd.DataFrame(), pd.DataFrame(), None

    raw_logs = []
    timeline_data = []
    run_summary = None

    with open(path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)

            # Our actuator writes one run_summary line per run (always,
            # even with zero interventions) alongside zero-or-more action
            # records. It has no "pre"/"post"/"decision" keys, so it must
            # be handled separately rather than assumed to be an action.
            if record.get("type") == "run_summary":
                run_summary = record
                continue

            # Different actions return different result shapes:
            # collect -> {"objects_collected": N}
            # tune_threshold -> {"requested_threshold": N, "applied_threshold": N}
            # flag_cycle -> {"flagged": True}
            # Use .get() with a default so the table doesn't crash on
            # whichever action shape isn't "collect".
            objects_collected = record["result"].get("objects_collected", 0)

            raw_logs.append({
                "Timestamp": datetime.fromtimestamp(record["pre"]["timestamp"]).strftime('%H:%M:%S'),
                "Action": record["decision"]["action"],
                "Reason": record["decision"]["reason"],
                "Objects Collected": objects_collected,
                "Mem Drop (Bytes)": record["pre"]["traced_current_bytes"] - record["post"]["traced_current_bytes"]
            })

            timeline_data.append({
                "Time": datetime.fromtimestamp(record["pre"]["timestamp"]),
                "Memory": record["pre"]["traced_current_bytes"],
                "RSS": record["pre"]["rss_bytes"],
                "State": "Pre-Action",
                "Hover": record["decision"]["reason"]
            })
            timeline_data.append({
                "Time": datetime.fromtimestamp(record["post"]["timestamp"]),
                "Memory": record["post"]["traced_current_bytes"],
                "RSS": record["post"]["rss_bytes"],
                "State": "Post-Action",
                "Hover": f"Result: {record['result']}"
            })

    return pd.DataFrame(raw_logs), pd.DataFrame(timeline_data), run_summary

st.title(f"📊 Memory Analysis: {selected_workload}")

df_logs, df_timeline, run_summary = load_data(file_path)

if run_summary is None and df_logs.empty:
    # No file / no run_summary at all -- genuinely hasn't been run yet.
    st.warning(f"No log data found for {selected_workload}. Run `python main.py` first.")
elif df_logs.empty:
    # A run_summary exists but zero actions were logged. This is the
    # correct, healthy outcome for a workload like normal_churn -- it
    # should be shown as a clean result, not treated as missing data.
    st.success(
        f"Run completed with **zero interventions** -- "
        f"{run_summary['total_observations']} observations, no actions needed. "
        f"This is the expected result for a healthy workload."
    )
else:
    # --- KPIs ---
    total_actions = len(df_logs)
    total_collected = df_logs["Objects Collected"].sum()
    max_memory = df_timeline["Memory"].max() / 1024  # in KB
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Interventions", total_actions)
    col2.metric("Total Objects Collected", f"{total_collected:,}")
    col3.metric("Peak Traced Memory", f"{max_memory:.2f} KB")
    
    st.markdown("---")
    
    # --- Graph: Memory Spikes and Drops ---
    st.subheader("Memory Trace & GC Actions (Timeline)")
    
    fig = go.Figure()
    
    # Traced Memory Line
    fig.add_trace(go.Scatter(
        x=df_timeline["Time"], 
        y=df_timeline["Memory"],
        mode='lines+markers',
        name='Traced Memory (Bytes)',
        line=dict(color='cyan', width=2),
        marker=dict(
            color=['red' if s == 'Pre-Action' else 'green' for s in df_timeline['State']],
            size=8
        ),
        text=df_timeline['State'] + "<br>" + df_timeline['Hover'],
        hovertemplate='%{text}<br>Memory: %{y} bytes<extra></extra>'
    ))
    
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Bytes",
        template="plotly_dark",
        height=400,
        margin=dict(l=0, r=0, t=30, b=0)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # --- Logs Table ---
    st.subheader("Agent Decision Logs")
    st.dataframe(
        df_logs.style.map(
            lambda x: 'color: #ff4b4b' if x == 0 else 'color: #00ff00', 
            subset=['Objects Collected']
        ),
        use_container_width=True
    )
    
    st.caption("Pro Tip: Red markers on the graph indicate when the system detected a problem, and green markers show memory immediately after the GC action.")