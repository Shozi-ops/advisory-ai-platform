import asyncio
import json
from typing import AsyncGenerator, List, Literal
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

# Modern Google GenAI SDK
from google import genai
from google.genai import types

# ---------------------------------------------------------------------------
# 1. System Prompts & Worker Directory
# ---------------------------------------------------------------------------

ROUTER_SYSTEM_PROMPT = """You are the Executive Orchestrator for the Advisory AI Platform. Your job is to classify user queries, break down complex requests into sub-tasks, select the optimal combination of specialized worker agents, and aggregate their outputs into a cohesive plan.

WORKER AGENT DIRECTORY:
1. FINANCIAL_STRATEGY: Handles cash flow, tax optimization, burn rates, revenue modeling, and subscription audits.
2. LEGAL_SENTINEL: Parses contracts, highlights liability risks, audits regulatory compliance (POPIA, GDPR), and checks terms.
3. OPERATIONS_COO: Generates SOPs, workflow automation schedules, task breakdown matrices, and team allocation rules.
4. MARKET_INTELLIGENCE: Performs competitive analysis, pricing benchmarks, target persona profiling, and GTM strategy.
5. RISK_RED_TEAMER: Conducts pre-mortem stress testing, downside scenario modeling, security audits, and counter-arguments.
6. BRAND_COPYWRITER: Drafts marketing copy, positioning statements, outreach emails, and pitch deck messaging.

ORCHESTRATION RULES:
1. Single Domain Request: Route to exactly 1 specialized agent.
2. Multi-Domain Request: Select 2 to 3 target agents. Specify execution order (Parallel vs. Sequential).
3. If information is ambiguous, assign OPERATIONS_COO to request clarification while other agents perform preliminary analysis.
"""

WORKER_SYSTEM_PROMPTS = {
    "FINANCIAL_STRATEGY": """You are the Financial Strategy & Tax Optimization Agent. You function as a virtual CFO and wealth advisor.
CAPABILITIES: Analyze cash flows, burn rates, tax deductions, and model runway scenarios.
MANDATES:
1. Categorize findings into "Immediate Savings", "Tax Risk Exposures", and "Long-term Runway Impact".
2. Present financial estimates in clean markdown comparison tables.
3. Always terminate outputs with: "Disclaimer: Output generated for advisory modeling only. Consult a certified financial planner or registered accountant for tax filings."
""",
    "LEGAL_SENTINEL": """You are the Legal & Regulatory Compliance Sentinel Agent. You specialize in contract analysis, risk disclosure, and statutory compliance frameworks (POPIA, GDPR, CCPA).
MANDATES:
1. Use threat scale: [CRITICAL RISK / HIGH RISK / MODERATE RISK / SAFE].
2. Quote exact clauses before providing the revised mitigation wording.
3. Always terminate outputs with: "Disclaimer: This output is an automated regulatory review, not formal legal counsel."
""",
    "OPERATIONS_COO": """You are the Operations & Workflow Automation Agent (Virtual COO). You convert high-level goals into executable operational workflows.
MANDATES:
1. Deliver workflows in step-by-step sequential lists with RACI tagging.
2. Clearly distinguish automated machine steps from manual human-in-the-loop steps.
""",
    "MARKET_INTELLIGENCE": """You are the Market Research & Competitive Intelligence Agent. You analyze industry dynamics, positioning, and growth strategies.
MANDATES:
1. Structure competitive positioning using 2x2 matrix text layouts or side-by-side spec tables.
2. Clearly distinguish validated empirical market facts from projected industry trends.
""",
    "RISK_RED_TEAMER": """You are the Risk & Crisis Management Agent ("Red Teamer"). Your purpose is to stress-test ideas, uncover logical fallacies, and identify failure modes.
MANDATES:
1. Adopt an objective, skeptical tone.
2. For every risk identified, present at least one concrete countermeasure or contingency plan.
""",
    "BRAND_COPYWRITER": """You are the Brand Communications & Growth Marketing Agent. You specialize in strategic messaging and conversion copywriting.
MANDATES:
1. Provide 3 variation options for key copy (Aggressive/Direct, Value-First, Narrative-Driven).
2. Highlight psychological triggers (Social Proof, Scarcity, Authority, Reciprocity) used.
""",
}

# ---------------------------------------------------------------------------
# 2. Pydantic Schemas for Router Enforcement
# ---------------------------------------------------------------------------

AgentID = Literal[
    "FINANCIAL_STRATEGY",
    "LEGAL_SENTINEL",
    "OPERATIONS_COO",
    "MARKET_INTELLIGENCE",
    "RISK_RED_TEAMER",
    "BRAND_COPYWRITER",
]

class TaskAssignment(BaseModel):
    agent_id: AgentID = Field(description="The assigned specialized worker agent.")
    task_description: str = Field(description="Precise operational prompt for this agent.")
    priority: int = Field(default=1, description="Execution priority order.")

class RouterPlan(BaseModel):
    intent_summary: str = Field(description="Concise statement of parsed user intent.")
    execution_plan: List[TaskAssignment] = Field(description="List of tasks assigned to worker agents.")
    synthesis_strategy: Literal["PARALLEL", "SEQUENTIAL"]
    final_response_format: Literal["EXECUTIVE_SUMMARY", "STEP_BY_STEP", "MATRIX_TABLE"]

class AdvisoryRequest(BaseModel):
    query: str
    model: str = "gemini-2.5-flash"

# ---------------------------------------------------------------------------
# 3. FastAPI Initialization & Gemini Client
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.genai_client = genai.Client()
    yield

app = FastAPI(title="Advisory AI Platform Orchestrator", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# 4. Agent Execution Helpers
# ---------------------------------------------------------------------------

async def run_router(client: genai.Client, user_query: str, model_name: str) -> RouterPlan:
    config = types.GenerateContentConfig(
        system_instruction=ROUTER_SYSTEM_PROMPT,
        response_mime_type="application/json",
        response_schema=RouterPlan,
        temperature=0.1,
    )

    response = await client.aio.models.generate_content(
        model=model_name,
        contents=user_query,
        config=config,
    )
    return RouterPlan.model_validate_json(response.text)


async def execute_worker_agent(
    client: genai.Client,
    agent_id: str,
    task_description: str,
    original_context: str,
    model_name: str,
) -> dict:
    system_prompt = WORKER_SYSTEM_PROMPTS.get(agent_id, "")
    worker_prompt = (
        f"USER CONTEXT:\n{original_context}\n\n"
        f"YOUR SPECIFIC TASK:\n{task_description}"
    )

    response = await client.aio.models.generate_content(
        model=model_name,
        contents=worker_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.3,
        ),
    )
    return {
        "agent_id": agent_id,
        "task": task_description,
        "output": response.text,
    }


async def synthesize_outputs(
    client: genai.Client,
    original_query: str,
    plan: RouterPlan,
    worker_results: List[dict],
    model_name: str,
) -> str:
    synthesis_prompt = f"""
ORIGINAL CLIENT QUERY:
{original_query}

INTENT:
{plan.intent_summary}

SPECIALIZED AGENT CONTRIBUTIONS:
{json.dumps(worker_results, indent=2)}

REQUESTED OUTPUT FORMAT: {plan.final_response_format}

Synthesize the findings into a cohesive, high-impact deliverable strictly adhering to the {plan.final_response_format} layout. Remove redundant agent meta-talk.
"""
    response = await client.aio.models.generate_content(
        model=model_name,
        contents=synthesis_prompt,
        config=types.GenerateContentConfig(temperature=0.4),
    )
    return response.text

# ---------------------------------------------------------------------------
# 5. Streaming Event Pipeline (SSE Endpoint)
# ---------------------------------------------------------------------------

async def orchestration_event_generator(
    client: genai.Client, user_query: str, model_name: str
) -> AsyncGenerator[dict, None]:
    yield {
        "event": "status",
        "data": json.dumps({"step": "routing", "message": "Analyzing query and orchestrating agents..."}),
    }

    try:
        plan = await run_router(client, user_query, model_name)
    except Exception as e:
        yield {
            "event": "error",
            "data": json.dumps({"error": f"Router failed: {str(e)}"}),
        }
        return

    yield {
        "event": "router_plan",
        "data": plan.model_dump_json(),
    }

    worker_results = []

    if plan.synthesis_strategy == "PARALLEL":
        yield {
            "event": "status",
            "data": json.dumps({"step": "dispatch", "strategy": "PARALLEL", "count": len(plan.execution_plan)}),
        }

        tasks = [
            execute_worker_agent(client, task.agent_id, task.task_description, user_query, model_name)
            for task in plan.execution_plan
        ]

        for future in asyncio.as_completed(tasks):
            result = await future
            worker_results.append(result)
            yield {
                "event": "agent_completed",
                "data": json.dumps(result),
            }

    else:
        sorted_tasks = sorted(plan.execution_plan, key=lambda x: x.priority)
        yield {
            "event": "status",
            "data": json.dumps({"step": "dispatch", "strategy": "SEQUENTIAL", "count": len(sorted_tasks)}),
        }

        for task in sorted_tasks:
            result = await execute_worker_agent(client, task.agent_id, task.task_description, user_query, model_name)
            worker_results.append(result)
            yield {
                "event": "agent_completed",
                "data": json.dumps(result),
            }

    yield {
        "event": "status",
        "data": json.dumps({"step": "synthesizing", "format": plan.final_response_format}),
    }

    final_delivery = await synthesize_outputs(client, user_query, plan, worker_results, model_name)

    yield {
        "event": "final_synthesis",
        "data": json.dumps({
            "format": plan.final_response_format,
            "content": final_delivery,
        }),
    }


@app.post("/api/v1/advisory/stream")
async def stream_advisory_pipeline(request: AdvisoryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    client = app.state.genai_client
    return EventSourceResponse(
        orchestration_event_generator(client, request.query, request.model)
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
