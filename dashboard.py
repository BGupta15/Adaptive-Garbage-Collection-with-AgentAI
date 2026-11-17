import streamlit as st
import pandas as pd
import json
import os
import time
import plotly.graph_objects as go
from datetime import datetime

# Import backend modules for live interactive runs
from workloads import cycle_generator, unbounded_cache, normal_churn
from workloads import cycle_generator_custom, unbounded_cache_custom, normal_churn_custom
import custom_gc
from monitor.monitor import Monitor
from analysis.analysis import analyze
from agent.agentic_gc import AgenticGCAgent
from actuator.actuator import Actuator

st.set_page_config(
    page_title="Agentic GC: Adaptive Memory Management",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 Adaptive Garbage Collection with Agent AI")
st.caption("Autonomous memory profiling, cyclic GC optimization, and Human-in-the-Loop (HITL) cache governance.")

# Sidebar navigation
st.sidebar.title("⚙️ Control Panel")
mode = st.sidebar.radio("Dashboard Mode", ["🎮 Live Interactive Experiment", "📁 View Past Audit Logs"])

# -----------------------------------------------------------------------------
# MODE 1: LIVE INTERACTIVE RUN (WITH REAL HITL BUTTONS)
# -----------------------------------------------------------------------------
if mode == "🎮 Live Interactive Experiment":
    st.subheader("🎮 Live Runtime with Real-Time Human-in-the-Loop Governance")
    st.markdown(
        """
        Run the memory loop live in the browser. When the **Agentic AI** detects an **Unbounded Cache Leak**,
        it will halt automated actions and request **YOUR manual permission** via on-screen buttons!
        """
    )

    colA, colB, colC = st.columns(3)
    with colA:
        selected_workload = st.selectbox(
            "Select Target Workload",
            [
                "Unbounded Cache (Live Leak)",
                "Cycle Generator (Circular References)",
                "Normal Churn (Healthy Baseline)",
            ],
        )
    with colB:
        backend_choice = st.selectbox("GC Runtime Engine", ["Python Built-in GC", "From-Scratch Custom Mark-Sweep"])
    with colC:
        hitl_behavior = st.selectbox(
            "Human-in-the-Loop (HITL) Policy",
            [
                "Manual (Ask me on screen via buttons)",
                "Simulated Auto-Approve",
                "Simulated Auto-Reject",
            ],
        )

    # Initialize session state for interactive flow
    if "sim_status" not in st.session_state:
        st.session_state.sim_status = "idle"  # 'idle', 'running', 'paused_hitl', 'completed'
    if "live_samples" not in st.session_state:
        st.session_state.live_samples = []
    if "live_logs" not in st.session_state:
        st.session_state.live_logs = []
    if "pending_hitl" not in st.session_state:
        st.session_state.pending_hitl = None
    if "workload_gen" not in st.session_state:
        st.session_state.workload_gen = None
    if "live_agent" not in st.session_state:
        st.session_state.live_agent = None
    if "live_monitor" not in st.session_state:
        st.session_state.live_monitor = None
    if "live_actuator" not in st.session_state:
        st.session_state.live_actuator = None
    if "live_workload_module" not in st.session_state:
        st.session_state.live_workload_module = None

    def start_simulation():
        is_custom = "Custom" in backend_choice
        if "Unbounded" in selected_workload:
            mod = unbounded_cache_custom if is_custom else unbounded_cache
            label = "unbounded_cache_custom" if is_custom else "unbounded_cache"
        elif "Cycle" in selected_workload:
            mod = cycle_generator_custom if is_custom else cycle_generator
            label = "cycle_generator_custom" if is_custom else "cycle_generator"
        else:
            mod = normal_churn_custom if is_custom else normal_churn
            label = "normal_churn_custom" if is_custom else "normal_churn"

        gc_mod = custom_gc if is_custom else None
        if is_custom:
            custom_gc.get_heap().reset()

        agent = AgenticGCAgent(hitl_mode="interactive")
        agent.reset()
        monitor = Monitor(sample_interval=0.01, gc_module=gc_mod)
        actuator = Actuator(log_path=f"live_{label}_log.jsonl", gc_module=gc_mod, workload_module=mod)

        st.session_state.live_agent = agent
        st.session_state.live_monitor = monitor
        st.session_state.live_actuator = actuator
        st.session_state.live_workload_module = mod
        st.session_state.workload_label = label
        st.session_state.workload_gen = mod.run(iterations=70)
        st.session_state.live_samples = []
        st.session_state.live_logs = []
        st.session_state.pending_hitl = None
        st.session_state.sim_status = "running"

    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        if st.button("▶ Start Live Simulation", type="primary"):
            start_simulation()
    with col_btn2:
        if st.session_state.sim_status != "idle":
            if st.button("⏹ Reset"):
                st.session_state.sim_status = "idle"
                st.session_state.pending_hitl = None
                st.rerun()

    # Chart & Metrics Containers
    metrics_placeholder = st.empty()
    chart_placeholder = st.empty()
    hitl_placeholder = st.empty()
    table_placeholder = st.empty()

    def render_live_dashboard():
        logs = st.session_state.live_logs
        samples = st.session_state.live_samples

        if samples:
            latest_mem = samples[-1]["traced_current_bytes"] / 1024
            peak_mem = max(s["traced_current_bytes"] for s in samples) / 1024
            reclaimed = sum(l.get("items_reclaimed", 0) for l in logs)
            hitl_actions = len([l for l in logs if "Human" in l.get("HITL Governance", "")])

            with metrics_placeholder.container():
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Current Heap Memory", f"{latest_mem:.1f} KB")
                m2.metric("Peak Memory", f"{peak_mem:.1f} KB")
                m3.metric("Reclaimed Objects", f"{reclaimed:,}")
                m4.metric("HITL Approvals", hitl_actions)

            # Build timeline chart
            times = [datetime.fromtimestamp(s["timestamp"]) for s in samples]
            mems = [s["traced_current_bytes"] / 1024 for s in samples]

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=times,
                y=mems,
                mode="lines",
                name="Traced Memory (KB)",
                line=dict(color="#00e5ff", width=2.5),
            ))

            # Mark intervention points
            for l in logs:
                action_color = "#34c759" if "Approved" in l["HITL Governance"] else "#ff3b30"
                fig.add_trace(go.Scatter(
                    x=[l["Time_dt"]],
                    y=[l["Post_Mem_KB"]],
                    mode="markers",
                    name=l["Action"],
                    marker=dict(size=12, color=action_color, symbol="star" if "Approved" in l["HITL Governance"] else "circle"),
                    hovertext=f"<b>Action:</b> {l['Action']}<br><b>Drop:</b> {l['Memory Reclaimed (KB)']} KB",
                ))

            fig.update_layout(
                title="Real-Time Heap Trajectory",
                xaxis_title="Timeline",
                yaxis_title="Memory (KB)",
                template="plotly_dark",
                height=380,
                margin=dict(l=20, r=20, t=40, b=20),
            )
            chart_placeholder.plotly_chart(fig, use_container_width=True)

        if logs:
            df = pd.DataFrame(logs)[["Timestamp", "Diagnosis", "Action", "HITL Governance", "Items Reclaimed", "Memory Reclaimed (KB)", "Agent Reason"]]
            table_placeholder.dataframe(df, use_container_width=True)

    # Execution Loop
    if st.session_state.sim_status == "running":
        monitor = st.session_state.live_monitor
        agent = st.session_state.live_agent
        actuator = st.session_state.live_actuator
        gen = st.session_state.workload_gen
        mod = st.session_state.live_workload_module
        label = st.session_state.workload_label

        try:
            for _ in range(4):  # run 4 steps per frame for smooth animation
                _step_info = next(gen)
                snap = monitor.sample(tag="observation")
                st.session_state.live_samples.append(snap)

                decision = agent.diagnose_and_decide(
                    monitor.samples,
                    analyze,
                    workload_name=label,
                )

                if decision.get("action") == "evict_cache" and hitl_behavior == "Manual (Ask me on screen via buttons)":
                    # PAUSE FOR HUMAN INTERACTION!
                    st.session_state.pending_hitl = {
                        "decision": decision,
                        "diagnosis": decision.get("diagnosis", "UNBOUNDED_RETENTION_LEAK"),
                        "reason": decision.get("reason", ""),
                    }
                    st.session_state.sim_status = "paused_hitl"
                    render_live_dashboard()
                    st.rerun()

                elif decision.get("action") != "noop":
                    # Execute automated or pre-configured action
                    if decision.get("action") == "evict_cache":
                        if hitl_behavior == "Simulated Auto-Reject":
                            decision["action"] = "flag_unbounded_cache"
                            decision["hitl_status"] = "human_rejected"
                            decision["reason"] += " [Simulated Human Rejection]"
                        else:
                            decision["hitl_status"] = "human_approved"

                    pre = monitor.sample(tag="action_pre")
                    result = actuator.apply(decision, workload_module=mod)
                    post = monitor.sample(tag="action_post")

                    mem_drop = max(0, pre["traced_current_bytes"] - post["traced_current_bytes"])
                    reclaimed = max(result.get("objects_collected", 0), result.get("evicted_items", 0))

                    st.session_state.live_logs.append({
                        "Timestamp": datetime.fromtimestamp(pre["timestamp"]).strftime("%H:%M:%S"),
                        "Time_dt": datetime.fromtimestamp(pre["timestamp"]),
                        "Post_Mem_KB": post["traced_current_bytes"] / 1024,
                        "Diagnosis": decision.get("diagnosis", "GENERAL_GC"),
                        "Action": decision.get("action", ""),
                        "HITL Governance": decision.get("hitl_status", "automated").replace("_", " ").title(),
                        "Items Reclaimed": reclaimed,
                        "Memory Reclaimed (KB)": round(mem_drop / 1024, 2),
                        "Agent Reason": decision.get("reason", ""),
                    })

            render_live_dashboard()
            time.sleep(0.04)
            st.rerun()

        except StopIteration:
            st.session_state.sim_status = "completed"
            render_live_dashboard()
            st.success("🎉 Simulation run completed! All observation steps processed.")

    # Render HITL Decision Box when paused
    if st.session_state.sim_status == "paused_hitl":
        render_live_dashboard()
        info = st.session_state.pending_hitl

        with hitl_placeholder.container():
            st.error("🚨 **HUMAN-IN-THE-LOOP APPROVAL REQUIRED**")
            st.warning(
                f"**Agent Diagnosis:** `{info['diagnosis']}`\n\n"
                f"**Agent Rationale:** {info['reason']}\n\n"
                f"**Proposed Destructive Action:** Purge 50% oldest cached entries to halt memory growth."
            )
            col_yes, col_no = st.columns(2)
            with col_yes:
                if st.button("✅ APPROVE CACHE EVICTION (Purge 50%)", type="primary", use_container_width=True):
                    # Operator Approved
                    monitor = st.session_state.live_monitor
                    actuator = st.session_state.live_actuator
                    mod = st.session_state.live_workload_module
                    decision = info["decision"]
                    decision["hitl_status"] = "human_approved"
                    decision["reason"] += " [Authorized Live via UI]"

                    pre = monitor.sample(tag="action_pre")
                    result = actuator.apply(decision, workload_module=mod)
                    post = monitor.sample(tag="action_post")

                    mem_drop = max(0, pre["traced_current_bytes"] - post["traced_current_bytes"])
                    reclaimed = max(result.get("objects_collected", 0), result.get("evicted_items", 0))

                    st.session_state.live_logs.append({
                        "Timestamp": datetime.fromtimestamp(pre["timestamp"]).strftime("%H:%M:%S"),
                        "Time_dt": datetime.fromtimestamp(pre["timestamp"]),
                        "Post_Mem_KB": post["traced_current_bytes"] / 1024,
                        "Diagnosis": decision["diagnosis"],
                        "Action": "evict_cache",
                        "HITL Governance": "Human Approved (Live UI)",
                        "Items Reclaimed": reclaimed,
                        "Memory Reclaimed (KB)": round(mem_drop / 1024, 2),
                        "Agent Reason": decision["reason"],
                    })

                    st.session_state.pending_hitl = None
                    st.session_state.sim_status = "running"
                    st.rerun()

            with col_no:
                if st.button("❌ REJECT INTERVENTION (Do Not Evict)", use_container_width=True):
                    # Operator Rejected
                    decision = info["decision"]
                    decision["action"] = "flag_unbounded_cache"
                    decision["hitl_status"] = "human_rejected"
                    decision["reason"] += " [Rejected Live via UI - Backing off]"

                    st.session_state.live_logs.append({
                        "Timestamp": datetime.now().strftime("%H:%M:%S"),
                        "Time_dt": datetime.now(),
                        "Post_Mem_KB": st.session_state.live_samples[-1]["traced_current_bytes"] / 1024,
                        "Diagnosis": decision["diagnosis"],
                        "Action": "flag_unbounded_cache",
                        "HITL Governance": "Human Rejected (Live UI)",
                        "Items Reclaimed": 0,
                        "Memory Reclaimed (KB)": 0.0,
                        "Agent Reason": decision["reason"],
                    })

                    st.session_state.pending_hitl = None
                    st.session_state.sim_status = "running"
                    st.rerun()

    elif st.session_state.sim_status in ["completed", "idle"]:
        render_live_dashboard()


# -----------------------------------------------------------------------------
# MODE 2: VIEW PAST AUDIT LOGS
# -----------------------------------------------------------------------------
else:
    st.subheader("📁 Historical Audit Log Inspector")
    LOG_FILES = {
        "Unbounded Cache (Real GC)": "unbounded_cache_log.jsonl",
        "Cycle Generator (Real GC)": "cycle_generator_log.jsonl",
        "Normal Churn (Real GC)": "normal_churn_log.jsonl",
        "Unbounded Cache (Custom GC)": "unbounded_cache_custom_log.jsonl",
        "Cycle Generator (Custom GC)": "cycle_generator_custom_log.jsonl",
        "Normal Churn (Custom GC)": "normal_churn_custom_log.jsonl",
    }

    selected_log = st.selectbox("Select Log File", list(LOG_FILES.keys()))
    path = LOG_FILES[selected_log]

    if not os.path.exists(path):
        st.warning(f"File `{path}` not found. Run `python main.py` or execute a live simulation first.")
    else:
        raw_logs = []
        timeline = []
        run_summary = None

        with open(path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                if rec.get("type") == "run_summary":
                    run_summary = rec
                    continue

                res = rec.get("result", {})
                reclaimed = max(res.get("objects_collected", 0), res.get("evicted_items", 0))
                pre_b = rec["pre"]["traced_current_bytes"]
                post_b = rec["post"]["traced_current_bytes"]
                drop_b = max(0, pre_b - post_b)
                dec = rec.get("decision", {})

                raw_logs.append({
                    "Timestamp": datetime.fromtimestamp(rec["pre"]["timestamp"]).strftime("%H:%M:%S"),
                    "Diagnosis": dec.get("diagnosis", "GENERAL_GC"),
                    "Action": dec.get("action", ""),
                    "HITL Governance": dec.get("hitl_status", "automated").replace("_", " ").title(),
                    "Items Reclaimed": reclaimed,
                    "Memory Reclaimed (KB)": round(drop_b / 1024, 2),
                    "Agent Reason": dec.get("reason", ""),
                })

                timeline.append({
                    "Time": datetime.fromtimestamp(rec["pre"]["timestamp"]),
                    "Memory": pre_b / 1024,
                    "Type": "Trigger",
                })
                timeline.append({
                    "Time": datetime.fromtimestamp(rec["post"]["timestamp"]),
                    "Memory": post_b / 1024,
                    "Type": "Result",
                })

        df_logs = pd.DataFrame(raw_logs)
        df_time = pd.DataFrame(timeline)

        if df_logs.empty and run_summary:
            st.success(f"✅ Clean Baseline Run: {run_summary['total_observations']} observations, 0 interventions needed.")
        elif not df_logs.empty:
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Interventions", len(df_logs))
            c2.metric("Total Reclaimed", f"{df_logs['Items Reclaimed'].sum():,}")
            c3.metric("Peak Memory", f"{df_time['Memory'].max():.1f} KB")

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_time["Time"],
                y=df_time["Memory"],
                mode="lines+markers",
                name="Heap (KB)",
                line=dict(color="#00e5ff"),
                marker=dict(color=["#ff3b30" if t == "Trigger" else "#34c759" for t in df_time["Type"]], size=8),
            ))
            fig.update_layout(template="plotly_dark", height=350, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(df_logs, use_container_width=True)
