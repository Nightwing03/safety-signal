"""HTTP API. Everything it depends on is passed in, so tests use fakes and never touch the network."""
import dataclasses
import json
import re
from typing import Any, Callable

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from safety_signal.errors import SafetySignalError
from safety_signal.pipeline import analyse
from safety_signal.summary import REFUSAL, SummaryResult, answer_question, out_of_scope, summarize

DRUG_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 \-]{0,60}$")


class AskBody(BaseModel):
    drug: str
    question: str = Field(min_length=1, max_length=500)
    top: int = Field(15, ge=1, le=30)


class _NoModel:
    """Stands in when the model client cannot be built; summarize() then falls back to the template."""

    def complete(self, **_kwargs):
        raise RuntimeError("model unavailable")


def _result_json(drug: str, result: SummaryResult) -> dict:
    return {"drug": drug, **dataclasses.asdict(result)}


def create_app(source: Any, label_source: Any, llm_factory: Callable[[], Any]) -> FastAPI:
    app = FastAPI(title="safety-signal", version="0.1.0",
                  description="Screening of FDA adverse-event reports. Informational only, not medical advice.")
    holder: dict = {}

    def llm():
        if "llm" not in holder:
            try:
                holder["llm"] = llm_factory()
            except Exception:  # noqa: BLE001 - no model means template output, not a failed request
                return _NoModel()
        return holder["llm"]

    def facts_for(drug: str, top: int):
        if not DRUG_NAME.match(drug):
            raise HTTPException(422, "invalid drug name")
        try:
            facts = analyse(source, label_source, drug, top)
        except SafetySignalError as exc:
            raise HTTPException(502, "the upstream data source failed or returned unexpected data") from exc
        if facts is None:
            raise HTTPException(404, "no adverse event reports found for this drug name")
        return facts

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/signals/{drug}")
    def signals(drug: str, top: int = Query(15, ge=1, le=30)):
        return json.loads(facts_for(drug, top).to_json())

    @app.get("/summary/{drug}")
    def summary(drug: str, top: int = Query(15, ge=1, le=30)):
        facts = facts_for(drug, top)
        return _result_json(drug, summarize(facts, llm()))

    @app.post("/ask")
    def ask(body: AskBody):
        if out_of_scope(body.question):  # refuse before spending any upstream calls
            return _result_json(body.drug, SummaryResult(text=REFUSAL, source="refusal"))
        facts = facts_for(body.drug, body.top)
        return _result_json(body.drug, answer_question(body.question, facts, llm()))

    return app


def default_app() -> FastAPI:
    """uvicorn safety_signal.api:default_app --factory"""
    from safety_signal.cache import SqliteCache
    from safety_signal.client import OpenFDAClient
    from safety_signal.config import load_api_key
    from safety_signal.events import OpenFDAEvents
    from safety_signal.labels import OpenFDALabels
    from safety_signal.llm import build_llm

    client = OpenFDAClient(api_key=load_api_key(), cache=SqliteCache(".cache/openfda.sqlite"))
    return create_app(OpenFDAEvents(client), OpenFDALabels(client), build_llm)
