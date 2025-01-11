import streamlit as st
import re
import faiss
import json
import requests
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from kubernetes import client, config
import smtplib
from email.mime.text import MIMEText
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set API keys & configurations
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_ENDPOINT = os.getenv("GROQ_ENDPOINT")
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")
EMAIL_ALERTS_ENABLED = os.getenv("EMAIL_ALERTS_ENABLED") == "True"
SMTP_SERVER = os.getenv("SMTP_SERVER")
SMTP_PORT = os.getenv("SMTP_PORT")
EMAIL_SENDER = os.getenv("EMAIL_SENDER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER")

# Load Kubernetes configuration
config.load_kube_config()
v1 = client.CoreV1Api()

# Load Sentence Transformer Model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# FAISS Index (for Storing & Retrieving Logs)
vector_dim = 384
index = faiss.IndexFlatL2(vector_dim)
log_texts = []
log_data = []
anomaly_logs = []

# Streamlit UI Setup
st.set_page_config(page_title="Kubernetes Anomaly Alerts", layout="wide")

st.title("🚨 Kubernetes Anomaly Detection Dashboard")
st.markdown("Real-time monitoring and AI-based anomaly detection with **Slack Alerts**.")

# 📌 Fetch Logs from Kubernetes API
def fetch_live_k8s_logs():
    logs = []
    pods = v1.list_pod_for_all_namespaces(watch=False)
    
    for pod in pods.items:
        pod_name = pod.metadata.name
        namespace = pod.metadata.namespace

        try:
            log = v1.read_namespaced_pod_log(name=pod_name, namespace=namespace)
            logs.append({"pod": pod_name, "namespace": namespace, "log": log[:500], "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
        except Exception:
            pass  # Skip pods with no logs

    return logs

# 📌 Extract Errors & Warnings
def extract_errors_warnings(logs):
    error_patterns = [
        r'(?i)\b(error|failed|exception|crash|critical)\b',
        r'(?i)\b(timeout|unavailable|unreachable|rejected|connection refused)\b',
        r'(?i)\b(unauthorized|forbidden|access denied)\b'
    ]
    
    return [log for log in logs if any(re.search(pattern, log["log"]) for pattern in error_patterns)]

# 📌 Store Logs in FAISS for Search
def store_logs_as_vectors(logs):
    global log_texts, log_data

    if not logs:  # Ensure logs are available before processing
        return

    log_texts = [log["log"] for log in logs]
    log_data = logs  # Store structured logs

    embeddings = embedding_model.encode(log_texts, convert_to_numpy=True)
    
    # Clear FAISS index before re-adding
    index.reset()  
    index.add(embeddings)  

# 📌 Retrieve Relevant Logs Safely
def retrieve_relevant_logs(query, top_k=5):
    if index.ntotal == 0:  # Check if FAISS index is empty
        return []  # Return empty list if no logs are indexed

    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)

    # Ensure indices are valid before retrieving logs
    valid_results = [log_data[i] for i in indices[0] if i < len(log_data)]
    return valid_results

# 📌 AI-Based Anomaly Detection
def detect_anomalies(logs):
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}

    if not logs:
        return "No logs available for anomaly detection."

    max_logs = 50
    trimmed_logs = logs[:max_logs]
    logs_text = "\n".join([log["log"][:500] for log in trimmed_logs])

    payload = {
        "model": "mixtral-8x7b-32768",
        "messages": [
            {"role": "system", "content": "You are an AI that detects anomalies in Kubernetes logs."},
            {"role": "user", "content": f"Identify unusual patterns in these logs (showing {max_logs} logs):\n\n{logs_text}"}
        ],
        "max_tokens": 500
    }

    response = requests.post(GROQ_ENDPOINT, headers=headers, data=json.dumps(payload))
    
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"Error: {response.json().get('error', {}).get('message', 'Unknown error')}"

# 📌 AI-Powered Log Analysis
def generate_ai_response(query, relevant_logs):
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}

    if not relevant_logs:
        return "No relevant logs found for analysis."

    max_logs = 10
    trimmed_logs = relevant_logs[:max_logs]
    logs_text = "\n".join([log["log"][:500] for log in trimmed_logs])

    payload = {
        "model": "mixtral-8x7b-32768",
        "messages": [
            {"role": "system", "content": "You are an AI that analyzes Kubernetes logs and provides insights."},
            {"role": "user", "content": f"Analyze these logs (showing {max_logs} logs):\n\n{logs_text}"}
        ],
        "max_tokens": 500
    }

    response = requests.post(GROQ_ENDPOINT, headers=headers, data=json.dumps(payload))

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"Error: {response.json().get('error', {}).get('message', 'Unknown error')}"

# Sidebar Filters
st.sidebar.header("📊 Log Filters")
refresh_interval = st.sidebar.slider("⏳ Refresh Interval (seconds)", min_value=5, max_value=60, value=10)

# 📌 Search Query Box
query = st.text_input("Enter a log query (e.g., 'pod failed')", placeholder="Search logs...")

if query:
    if index.ntotal == 0:
        st.warning("⚠️ No logs available in FAISS index. Try fetching logs first.")
    else:
        search_results = retrieve_relevant_logs(query)
        if search_results:
            st.subheader("🔍 Relevant Logs")
            st.code("\n".join([log["log"] for log in search_results]), language="plaintext")

            # AI Insights
            st.subheader("🤖 AI Log Analysis")
            ai_response = generate_ai_response(query, search_results)
            st.text(ai_response)
        else:
            st.info("No relevant logs found for the query.")

# Fetch Logs & Store in FAISS
logs = fetch_live_k8s_logs()
store_logs_as_vectors(logs)

# Detect Anomalies
anomaly_logs = detect_anomalies(logs)

# Display Logs
st.subheader("📄 Live Kubernetes Logs")
df = pd.DataFrame(logs)
st.dataframe(df)

# Error Count
error_count = len(extract_errors_warnings(logs))
st.metric(label="Total Errors & Warnings", value=error_count)

# Anomaly Detection Results
st.subheader("🚨 Anomaly Detection")
st.text(anomaly_logs)

# Auto-refresh data
time.sleep(refresh_interval)
st.rerun()
