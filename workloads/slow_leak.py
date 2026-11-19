"""
A leak too slow for a fixed B/s threshold: ~1.5 KB retained per step. The
rule-based agent (50 KB/s threshold) never fires on this; a trend-based
detector (r2 ~ 1, persistent floor) does. This is the workload that
justifies the stats layer.
"""


def run(iterations=2000, leak_bytes=1500):
    retained = []
    for i in range(iterations):
        retained.append("x" * leak_bytes)   # never released
        yield {"step": i, "retained_items": len(retained)}
