import os
import re
import json
import time
import sqlite3
import threading
import socket
from datetime import datetime
from collections import Counter

import requests
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from dotenv import load_dotenv
from kubernetes import client, config
from kubernetes.client import Configuration
from prometheus_client import start_http_server, Gauge
import boto3
from botocore.config import Config as BotoConfig

# =========================
# Environment & Constants
# =========================
load_dotenv()

ANOMALY_THRESHOLD = int(os.getenv("ANOMALY_THRESHOLD", 3))
SLACK_ENABLED = os.getenv("SLACK_ENABLED", "False").lower() == "true"
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "")
METRICS_SERVER = os.getenv("METRICS_SERVER", "False").lower() == "true"

BEDROCK_REGION = os.getenv("BEDROCK_REGION", "us-east-1")
BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-haiku-20240307-v1:0")

# If you're hitting a self-signed apiserver, set K8S_INSECURE=true to skip TLS verification (dev/test only!)
K8S_INSECURE = os.getenv("K8S_INSECURE", "False").lower() == "true"

# Optional explicit kubeconfig path (e.g., Rancher k3s.yaml)
KUBECONFIG_PATH = os.getenv("KUBECONFIG_PATH", "~/.kube/config")

DB_FILE = "k8s_logs.db"


# =========================
# Kubernetes Config & Client
# =========================
config.load_kube_config(config_file="~/.kube/config", persist_config=False)

v1 = client.CoreV1Api()

# =========================
# SQLite Storage
# =========================
def init_db():
    conn = sqlite3.connect(DB_FILE)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pod TEXT,
            namespace TEXT,
            log TEXT,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

def save_logs_to_db(logs):
    if not logs:
        return
    conn = sqlite3.connect(DB_FILE)
    conn.executemany(
        "INSERT INTO logs (pod, namespace, log, timestamp) VALUES (?, ?, ?, ?)",
        [(l["pod"], l["namespace"], l["log"], l["timestamp"]) for l in logs]
    )
    conn.commit()
    conn.close()

def get_recent_logs(limit=100):
    conn = sqlite3.connect(DB_FILE)
    rows = conn.execute(
        "SELECT pod, namespace, log, timestamp FROM logs ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [{"pod": pod, "namespace": ns, "log": lg, "timestamp": ts} for pod, ns, lg, ts in rows]

# =========================
# Slack
# =========================
def send_slack_notification(message: str):
    if SLACK_ENABLED and SLACK_WEBHOOK_URL:
        try:
            resp = requests.post(SLACK_WEBHOOK_URL, json={"text": message}, timeout=10)
            if resp.status_code >= 400:
                st.warning(f"Slack error {resp.status_code}: {resp.text}")
        except Exception as e:
            st.warning(f"Slack notify failed: {e}")
    else:
        st.info("Slack disabled. Set SLACK_ENABLED=True and SLACK_WEBHOOK_URL to enable alerts.")

# =========================
# Log Fetch & Parsing
# =========================
def fetch_live_k8s_logs():
    logs = []
    try:
        pods = v1.list_pod_for_all_namespaces(watch=False)
    except Exception as e:
        st.error(f"Failed to list pods: {e}")
        return logs

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for p in pods.items:
        try:
            content = v1.read_namespaced_pod_log(name=p.metadata.name, namespace=p.metadata.namespace)
            logs.append({
                "pod": p.metadata.name,
                "namespace": p.metadata.namespace,
                "log": content[:1000],
                "timestamp": now
            })
        except Exception:
            # Skip pods with no readable logs
            pass

    save_logs_to_db(logs)
    return logs

def extract_errors_warnings(logs):
    patterns = [
        r'(?i)\b(error|failed|exception|crash|critical)\b',
        r'(?i)\b(timeout|unavailable|unreachable|rejected|connection refused)\b',
        r'(?i)\b(unauthorized|forbidden|access denied)\b',
        r'(?i)\b(pending|node not ready|pod not scheduled|evicted)\b',
        r'(?i)\b(back-off restarting failed container|crashloopbackoff|oomkilled)\b',
        r'(?i)\b(image pull error|failed to start container|container terminated)\b',
    ]
    compiled = [re.compile(p) for p in patterns]
    return [l for l in logs if any(c.search(l["log"]) for c in compiled)]

# =========================
# Bedrock (Claude Messages)
# =========================
def _bedrock_client():
    return boto3.client(
        "bedrock-runtime",
        region_name=BEDROCK_REGION,
        config=BotoConfig(retries={"max_attempts": 3, "mode": "standard"})
    )

def _invoke_claude_messages(client_bedrock, model_id, system_prompt, user_text, max_tokens=700, temperature=0.1):
    """
    Correct Claude 3 Messages schema:
      - Top-level "system" (string)
      - messages: role in {"user","assistant"}
      - content: list of {type: "text", text: "..."}
    """
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "system": system_prompt,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": user_text}]
            }
        ],
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    resp = client_bedrock.invoke_model(
        modelId=model_id,
        body=json.dumps(body),
        accept="application/json",
        contentType="application/json"
    )
    payload = json.loads(resp["body"].read())
    return "\n".join(
        c.get("text", "") for c in payload.get("content", []) if c.get("type") == "text"
    ).strip()

def detect_anomalies_bedrock(logs):
    if not logs:
        return "No logs to analyze."
    client_b = _bedrock_client()
    chunk = 12
    groups = [logs[i:i+chunk] for i in range(0, len(logs), chunk)]
    sys_prompt = (
        "You are a Kubernetes SRE assistant. From raw pod logs, extract anomalies, "
        "group by namespace/pod, infer likely root causes, and give actionable remediations. "
        "Be concise and operational."
    )
    outputs = []
    for g in groups:
        logs_text = "\n".join(
            f"[{l['timestamp']}] {l['namespace']}/{l['pod']}: {l['log'][:500]}" for l in g
        )
        user_text = (
            "Analyze and return:\n"
            "- Top anomalies (bullets)\n- Likely root causes\n- Immediate runbooks\n\n"
            f"Logs:\n{logs_text}"
        )
        try:
            summary = _invoke_claude_messages(
                client_b, BEDROCK_MODEL_ID, sys_prompt, user_text, max_tokens=700, temperature=0.1
            )
            outputs.append(summary or "(no content)")
        except Exception as e:
            outputs.append(f"(Bedrock error) {e}")
    return "\n\n---\n\n".join(outputs)

# =========================
# Charts
# =========================
def plot_logs_pie(logs):
    namespaces = [l.get("namespace") for l in logs if l.get("namespace")]
    if not namespaces:
        st.info("No namespaces found in logs.")
        return
    counts = Counter(namespaces)
    plt.figure(figsize=(8, 6))
    # (No explicit colors to keep defaults clean)
    plt.pie(counts.values(), labels=counts.keys(), autopct="%1.1f%%", startangle=140)
    plt.title("Log Distribution by Namespace")
    st.pyplot(plt)

def plot_error_distribution(errors):
    if not errors:
        st.info("No errors to chart.")
        return
    kinds = [e["log"].split(" ", 1)[0] for e in errors]
    counts = Counter(kinds)
    plt.figure(figsize=(8, 6))
    plt.pie(counts.values(), labels=counts.keys(), autopct="%1.1f%%", startangle=140)
    plt.title("Error Type Distribution")
    st.pyplot(plt)

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="Kubernetes Anomaly Alerts", layout="wide")
st.sidebar.image("./images/logo.png", use_container_width=True)
st.title("🚨 Kubernetes Anomaly Detection Dashboard")

st.markdown("Real-time monitoring with **AWS Bedrock** (Claude 3 Messages).")

refresh_interval = st.sidebar.slider("⏳ Refresh Interval (seconds)", 5, 60, 10)

# ChatBedrock helper
st.sidebar.markdown("---")
st.sidebar.subheader("💬 ChatBedrock: Ask about your logs")
chat_q = st.sidebar.text_area(
    "Question (recent logs used as context):",
    height=100,
    placeholder="e.g., Which services look unhealthy and why?"
)
if st.sidebar.button("Ask Bedrock"):
    try:
        client_b = _bedrock_client()
        recent = get_recent_logs(limit=40)
        ctx = "\n".join(
            f"{r['timestamp']} {r['namespace']}/{r['pod']}: {r['log'][:300]}" for r in recent
        )
        sys_prompt = "You are a K8s SRE copilot. Be concise and operational."
        user_text = f"Logs:\n{ctx}\n\nQuestion:\n{chat_q}"
        answer = _invoke_claude_messages(
            client_b, BEDROCK_MODEL_ID, sys_prompt, user_text, max_tokens=700, temperature=0.2
        )
        st.sidebar.success("Response:")
        st.sidebar.write(answer or "(no content)")
    except Exception as e:
        st.sidebar.error(f"Bedrock error: {e}")

# Fetch + show
_ = fetch_live_k8s_logs()
stored_logs = get_recent_logs()
error_logs = extract_errors_warnings(stored_logs)

if st.sidebar.button("🚀 Run Anomaly Detection"):
    st.subheader("🚨 Anomaly Detection Results")
    summary = detect_anomalies_bedrock(stored_logs)
    st.write(summary)
    anomaly_count = summary.count("- ")
    if anomaly_count >= ANOMALY_THRESHOLD:
        st.error(f"⚠️ High Anomaly Alert! ~{anomaly_count} anomalies found.")
        send_slack_notification(f"🚨 High Anomaly Alert! ~{anomaly_count} anomalies found:\n\n{summary}")

    st.subheader("📈 Log Trend Analysis")
    plot_logs_pie(stored_logs)

    st.subheader("📊 Error Type Distribution")
    plot_error_distribution(error_logs)
else:
    st.subheader("📄 Recent Kubernetes Logs")
    st.dataframe(pd.DataFrame(stored_logs))
    st.metric("Total Errors & Warnings", len(error_logs))
    st.subheader("📈 Log Trend Analysis")
    plot_logs_pie(stored_logs)
    st.subheader("📊 Error Type Distribution")
    plot_error_distribution(error_logs)



# Simple auto-refresh loop
time.sleep(refresh_interval)
st.rerun()
