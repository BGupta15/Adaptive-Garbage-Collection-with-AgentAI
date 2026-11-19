"""
Phase VI: read-only FastAPI service over the run logs + evaluator.

    pip install fastapi uvicorn  (already in requirements.txt)
    uvicorn api.server:app --reload       # docs at http://127.0.0.1:8000/docs
"""

from fastapi import FastAPI, HTTPException

from evaluator.evaluate import GROUND_TRUTH, compare, narrate, summarize_run

app = FastAPI(title="gc-agent API", description="Adaptive GC Agent REST API")


@app.get("/runs")
def list_runs():
    """Every (workload, agent) run that has a log on disk."""
    return [r for r in compare() if r]


@app.get("/runs/{label}")
def get_run(label: str, agent: str = "rule"):
    if label not in GROUND_TRUTH:
        raise HTTPException(404, f"unknown workload {label!r}")
    r = summarize_run(label, agent)
    if r is None:
        raise HTTPException(404, f"no log for {label} / {agent}; run main.py first")
    return r


@app.get("/compare")
def compare_agents():
    return [r for r in compare() if r]


@app.post("/report")
def llm_report():
    """LLM-written analysis of the comparison table (external API call)."""
    rows = [r for r in compare() if r]
    if not rows:
        raise HTTPException(404, "no logs yet")
    from agent.llm_agent import make_groq_chat
    try:
        return {"report": narrate(rows, make_groq_chat())}
    except Exception as e:
        raise HTTPException(502, f"LLM call failed: {type(e).__name__}")
