import requests
from typing import Tuple

from fastapi import FastAPI, BackgroundTasks
from cohort_analysis import summarize_cohort, compare_cohorts

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


def post_summary(cohort_prefix: str, timespan: Tuple[str], post_to: str):
    summary = summarize_cohort(cohort_prefix, timespan)
    requests.post(post_to, json=summary)


def post_comparison(cohort_prefix_a: str, timespan_a: Tuple[str],
                    cohort_prefix_b: str, timespan_b: Tuple[str],
                    post_to: str):
    comparison = compare_cohorts(cohort_prefix_a, timespan_a,
                                 cohort_prefix_b, timespan_b)
    requests.post(post_to, json=comparison)


@app.get("/cohort_summary")
async def cohort_summary(cohort_prefix: str,
                         timespan_start: str, timespan_end: str,
                         post_to: str,
                         background_tasks: BackgroundTasks):
    timespan = (timespan_start, timespan_end)
    background_tasks.add_task(post_summary, cohort_prefix, timespan, post_to)
    return {'message': 'running'}


@app.get("/cohort_comparison")
async def cohort_comparison(cohort_prefix_a: str,
                            timespan_a_start: str, timespan_a_end: str,
                            cohort_prefix_b: str,
                            timespan_b_start: str, timespan_b_end: str,
                            post_to: str,
                            background_tasks: BackgroundTasks):
    timespan_a = (timespan_a_start, timespan_a_end)
    timespan_b = (timespan_b_start, timespan_b_end)
    background_tasks.add_task(post_comparison,
                              cohort_prefix_a, timespan_a,
                              cohort_prefix_b, timespan_b,
                              post_to)
    return {'message': 'running'}
