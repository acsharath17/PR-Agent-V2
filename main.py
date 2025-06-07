import os
import requests
import openai
import re
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

openai.api_key = os.getenv("OPENAI_API_KEY")  # Load API key from environment

# --- Define the shared state schema ---
class FileResult(TypedDict):
    filename: str
    diff: str
    is_apex: bool
    explanation: str
    review_comments: str
    business_summary: str

class PRState(TypedDict):
    pr_url: str
    diff: str
    files: List[FileResult]
    dev_summary: str
    business_summary: str

# FastAPI app setup
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Helper: Detect Apex file ---
def is_apex_file(filename: str) -> bool:
    return filename.endswith(".cls") or filename.endswith(".trigger")

# --- LangGraph Nodes ---
def fetch_pr_diff(state: PRState) -> PRState:
    pr_url = state["pr_url"]
    diff_url = pr_url + ".diff"
    diff = requests.get(diff_url).text
    return {**state, "diff": diff}

def split_by_file(state: PRState) -> PRState:
    pattern = re.compile(r'diff --git a/(.*?) b/.*?\n(.*?)(?=\ndiff --git a/|\Z)', re.DOTALL)
    matches = pattern.findall(state["diff"])
    files: List[FileResult] = []
    for filename, file_diff in matches:
        files.append({
            "filename": filename.strip(),
            "diff": file_diff.strip(),
            "is_apex": is_apex_file(filename),
            "explanation": "",
            "review_comments": "",
            "business_summary": "",
        })
    return {**state, "files": files}

def process_files(state: PRState) -> PRState:
    updated_files = []
    for file in state["files"]:
        filename = file["filename"]
        diff = file["diff"]
        explanation = openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": f"Explain in detail the changes made in this file and why they might have been made.\n\n{diff}"}],
            max_tokens=500
        ).choices[0].message.content.strip()
        review_comments = ""
        if file["is_apex"]:
            review_comments = openai.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": f"Review the following Apex code diff for adherence to Salesforce best practices.\n\n{diff}"}],
                max_tokens=500
            ).choices[0].message.content.strip()
        business_summary = openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": f"Explain what was changed in this file in plain language for business stakeholders. Avoid technical jargon.\n\n{diff}"}],
            max_tokens=300
        ).choices[0].message.content.strip()
        updated_files.append({
            **file,
            "explanation": explanation,
            "review_comments": review_comments,
            "business_summary": business_summary,
        })
    return {**state, "files": updated_files}

def aggregate_summaries(state: PRState) -> PRState:
    explanations = [f"Changes in {f['filename']}:\n{f['explanation']}" for f in state["files"]]
    business_summaries = [f["business_summary"] for f in state["files"]]
    dev_summary_prompt = "\n\n".join(explanations)
    business_prompt = "\n\n".join(business_summaries)
    dev_summary = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": f"Summarize this PR for developers:\n\n{dev_summary_prompt}"}],
        max_tokens=400
    ).choices[0].message.content.strip()
    business_summary = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": f"Summarize this PR for business stakeholders in non-technical language:\n\n{business_prompt}"}],
        max_tokens=300
    ).choices[0].message.content.strip()
    return {
        **state,
        "dev_summary": dev_summary,
        "business_summary": business_summary,
    }

# --- LangGraph Setup ---
builder = StateGraph(PRState)
builder.add_node("fetch_diff", fetch_pr_diff)
builder.add_node("split_by_file", split_by_file)
builder.add_node("process_files", process_files)
builder.add_node("aggregate_summaries", aggregate_summaries)
builder.set_entry_point("fetch_diff")
builder.add_edge("fetch_diff", "split_by_file")
builder.add_edge("split_by_file", "process_files")
builder.add_edge("process_files", "aggregate_summaries")
builder.add_edge("aggregate_summaries", END)
graph = builder.compile()

# --- FastAPI Routes ---
class PRRequest(BaseModel):
    pr_url: str

@app.post("/analyze_pr")
async def analyze_pr(request: PRRequest):
    result = graph.invoke({"pr_url": request.pr_url})
    return JSONResponse(content=result)

@app.post("/approve_pr")
async def approve_pr(pr: PRRequest):
    token = os.getenv("GITHUB_TOKEN")
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json"
    }
    match = re.search(r"github.com/(.*?)/(.*?)/pull/(\d+)", pr.pr_url)
    if not match:
        return {"error": "Invalid GitHub PR URL"}
    owner, repo, pr_number = match.groups()
    response = requests.post(
        f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}/reviews",
        headers=headers,
        json={"event": "APPROVE"}
    )
    if response.status_code in [200, 201]:
        return {"status": "approved"}
    return {"error": response.json()}
