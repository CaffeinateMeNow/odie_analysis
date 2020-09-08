import logging
import requests
from typing import Tuple

from fastapi import FastAPI, BackgroundTasks
from cohort_analysis import summarize_cohort, compare_cohorts
import config

app = FastAPI()

logging.basicConfig(level=logging.INFO)


@app.get("/")
def read_root():
    return {"Hello": "World"}


def auth_to_backend():
    return
    creds = {'api_user': {'email': config.BACKEND_EMAIL,
                          'password': config.BACKEND_PASSWORD}}
    url = config.BACKEND_HOST + 'login.json'
    try:
        resp = requests.post(url, json=creds)
        resp.raise_for_status()
    except Exception as e:
        logging.error(e)


def post_summary(cohort_prefix: str, timespan: Tuple[str], post_to: str):
    summary = summarize_cohort(cohort_prefix, timespan)
    auth_to_backend()
    requests.put(post_to, json={"results": summary})


def post_comparison(cohort_prefix_a: str, timespan_a: Tuple[str],
                    cohort_prefix_b: str, timespan_b: Tuple[str],
                    post_to: str):
    comparison = compare_cohorts(cohort_prefix_a, timespan_a,
                                 cohort_prefix_b, timespan_b)
    auth_to_backend()
    requests.put(post_to, json={"results": comparison})


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
