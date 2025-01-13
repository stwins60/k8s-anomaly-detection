import streamlit as st
import re
import json
import numpy as np
import pandas as pd
import time
import sqlite3
from datetime import datetime
from sentence_transformers import SentenceTransformer
from kubernetes import client, config
import os
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from collections import Counter
import requests
import matplotlib.pyplot as plt
from prometheus_client import start_http_server, Gauge, REGISTRY, CollectorRegistry
import threading
import socket

# Load environment variables
load_dotenv()

# Set API keys & configurations
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_ENDPOINT = os.getenv("GROQ_ENDPOINT")
USE_GROQ = os.getenv("USE_GROQ", "False").lower() == "true"  # Toggle AI calls
ANOMALY_THRESHOLD = int(os.getenv("ANOMALY_THRESHOLD", 3))  # Threshold for anomaly alerts
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")

# Global registry for Prometheus metrics
registry = CollectorRegistry()

# Check if port 8090 is already in use
def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0  # Returns True if port is in use

# Start Prometheus server **only if not already running**
def start_metrics_server():
    if os.getenv("METRICS_SERVER", "False").lower() == "true":
        try:
            if is_port_in_use(8090):
                print("✅ Prometheus server is already running on port 8090. Skipping restart.")
            else:
                print("🚀 Starting Prometheus server on port 8090...")
                start_http_server(8090)  # Start the server only if it's not already running
        except BrokenPipeError:
            print("⚠️ Warning: Broken pipe detected. Restarting Prometheus server...")
            os.system("fuser -k 8090/tcp")  # Kill any process using port 8090
            start_http_server(8090)  # Restart Prometheus server

# Run Prometheus server in a background thread
if "prometheus_thread" not in globals():
    prometheus_thread = threading.Thread(target=start_metrics_server, daemon=True)
    prometheus_thread.start()

# Register metrics **only if not already registered**
if "log_count_metric" not in globals():
    anomaly_metric = Gauge("kubernetes_anomalies", "Number of detected anomalies", registry=registry)
    error_log_metric = Gauge("kubernetes_error_logs", "Total number of error logs", registry=registry)
    log_count_metric = Gauge("kubernetes_log_count", "Total number of logs processed", registry=registry)


# Load Kubernetes configuration
config.load_kube_config()
v1 = client.CoreV1Api()

# Load Sentence Transformer Model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# SQLite Database Setup
DB_FILE = "k8s_logs.db"

def init_db():
    """Initialize the SQLite database."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        pod TEXT,
                        namespace TEXT,
                        log TEXT,
                        timestamp TEXT
                      )''')
    conn.commit()
    conn.close()

init_db()

# Function to Insert Logs into SQLite
def save_logs_to_db(logs):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    for log in logs:
        cursor.execute("INSERT INTO logs (pod, namespace, log, timestamp) VALUES (?, ?, ?, ?)", 
                       (log["pod"], log["namespace"], log["log"], log["timestamp"]))
    conn.commit()
    conn.close()

# Function to Fetch Logs from SQLite
def get_recent_logs(limit=100):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM logs ORDER BY id DESC LIMIT ?", (limit,))
    logs = [{"pod": row[1], "namespace": row[2], "log": row[3], "timestamp": row[4]} for row in cursor.fetchall()]
    conn.close()
    return logs

# 📌 Slack Notification
def send_slack_notification(message):
    if SLACK_WEBHOOK_URL:
        payload = {"text": message}
        requests.post(SLACK_WEBHOOK_URL, json=payload)

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

    save_logs_to_db(logs)  # Save logs to SQLite
    return logs

# 📌 Extract Errors & Warnings Locally
def extract_errors_warnings(logs):
    error_patterns = [
        r'(?i)\b(error|failed|exception|crash|critical)\b',
        r'(?i)\b(timeout|unavailable|unreachable|rejected|connection refused)\b',
        r'(?i)\b(unauthorized|forbidden|access denied)\b'
    ]
    
    return [log for log in logs if any(re.search(pattern, log["log"]) for pattern in error_patterns)]

# 📌 Local Anomaly Detection using TF-IDF and K-Means Clustering
def detect_anomalies_locally(logs, n_clusters=3):
    if not logs:
        return "No logs available for anomaly detection."

    log_texts = [log["log"] for log in logs]
    
    # Vectorize logs using TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
    log_vectors = vectorizer.fit_transform(log_texts)

    # Apply K-Means Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(log_vectors)

    # Find the smallest cluster (potential anomalies)
    cluster_counts = Counter(clusters)
    anomaly_cluster = min(cluster_counts, key=cluster_counts.get)

    anomaly_logs = [logs[i] for i in range(len(logs)) if clusters[i] == anomaly_cluster]
    
    return anomaly_logs if anomaly_logs else "No anomalies detected."

# 📌 AI-Based Anomaly Detection (Only Sends Logs to Groq if Necessary)
def detect_anomalies_ai(logs):
    if not USE_GROQ:
        return "Groq API calls are disabled. Using local anomaly detection."

    if not logs:
        return "No logs available for anomaly detection."

    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    
    max_logs = min(50, len(logs))
    logs_text = "\n".join([log["log"][:500] for log in logs[:max_logs]])

    payload = {
        "model": "mixtral-8x7b-32768",
        "messages": [
            {"role": "system", "content": "You are an AI that detects anomalies in Kubernetes logs."},
            {"role": "user", "content": f"Identify unusual patterns in these logs:\n\n{logs_text}"}
        ],
        "max_tokens": 500
    }

    response = requests.post(GROQ_ENDPOINT, headers=headers, data=json.dumps(payload))
    
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"Error: {response.json().get('error', {}).get('message', 'Unknown error')}"
    
# 📊 Visualize Log Errors Over Time
def plot_logs(logs):
    timestamps = [datetime.strptime(log["timestamp"], "%Y-%m-%d %H:%M:%S") for log in logs]
    error_counts = list(range(1, len(timestamps) + 1))

    plt.figure(figsize=(10, 4))
    plt.plot(timestamps, error_counts, marker="o", linestyle="-", label="Log Count", color="red")
    plt.xlabel("Time")
    plt.ylabel("Logs Count")
    plt.title("Log Entries Over Time")
    plt.xticks(rotation=45)
    plt.legend()
    st.pyplot(plt)

# 📊 Visualize Log Distribution in Pie Chart
def plot_logs_pie(logs):
    if not logs:
        st.warning("No logs available to display in the pie chart.")
        return

    # Aggregate logs by namespace or pod
    namespace_counts = Counter(log["namespace"] for log in logs)

    plt.figure(figsize=(8, 6))
    plt.pie(
        namespace_counts.values(), 
        labels=namespace_counts.keys(), 
        autopct="%1.1f%%", 
        startangle=140, 
        colors=plt.cm.Paired.colors
    )
    plt.title("Log Distribution by Namespace")
    st.pyplot(plt)

# 📊 Error Type Distribution
def plot_error_distribution(errors):
    if not errors:
        st.warning("No errors to display in the pie chart.")
        return

    error_types = [log["log"].split(" ")[0] for log in errors]  # Extract first word of log
    error_counts = Counter(error_types)

    plt.figure(figsize=(8, 6))
    plt.pie(
        error_counts.values(), 
        labels=error_counts.keys(), 
        autopct="%1.1f%%", 
        startangle=140, 
        colors=plt.cm.Paired.colors
    )
    plt.title("Error Type Distribution")
    st.pyplot(plt)

# 🖥️ Streamlit UI Setup
st.set_page_config(page_title="Kubernetes Anomaly Alerts", layout="wide")
# st.sidebar.markdown("![Kubernetes Logo](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQHyAZraIZcf_GMLOnsw9p-KWVc0LmDFZfvWhLUSjhWt9J-YYraBGSHsqIRDQnjNlAbS3s&usqp=CAU)")
st.sidebar.image("./images/logo.png", use_container_width=True)
st.title("🚨 Kubernetes Anomaly Detection Dashboard")
st.markdown("Real-time monitoring and AI-based anomaly detection.")

# Sidebar Filters
st.sidebar.header("📊 Log Filters")
refresh_interval = st.sidebar.slider("⏳ Refresh Interval (seconds)", min_value=5, max_value=60, value=10)

# Dropdown for Anomaly Detection Type
anomaly_detection_method = st.sidebar.selectbox(
    "🔍 Choose Anomaly Detection Method",
    ["Local (TF-IDF & K-Means)", "AI (Groq API)"]
)

# Fetch Logs (only once)
logs = fetch_live_k8s_logs()
# log_count_metric.set(len(logs))

# Retrieve Recent Logs from SQLite
stored_logs = get_recent_logs()

# Extract Warnings & Errors
error_logs = extract_errors_warnings(stored_logs)
# error_log_metric.set(len(error_logs))

# Detect Anomalies Based on Selected Method
if st.sidebar.button("🚀 Run Anomaly Detection"):
    st.subheader("🚨 Anomaly Detection Results")
    anomaly_logs = detect_anomalies_locally(stored_logs) if anomaly_detection_method == "Local (TF-IDF & K-Means)" else detect_anomalies_ai(stored_logs)

    # anomaly_metric.set(len(anomaly_logs))

    st.write(pd.DataFrame(anomaly_logs))

    # Alert if anomalies exceed threshold
    if len(anomaly_logs) >= ANOMALY_THRESHOLD:
        alert_msg = f"⚠️ High Anomaly Alert! {len(anomaly_logs)} anomalies found."
        st.error(alert_msg)
        send_slack_notification(alert_msg)  # Send Slack Alert
    
    st.subheader("📈 Log Trend Analysis")
    # plot_logs(anomaly_logs)
    plot_logs_pie(anomaly_logs)

    st.subheader("📊 Error Type Distribution")
    plot_error_distribution(error_logs)
else:
    # if the button is not clicked, only show the recent logs
    st.subheader("📄 Recent Kubernetes Logs")
    df = pd.DataFrame(stored_logs)
    st.dataframe(df)
    st.metric(label="Total Errors & Warnings", value=len(error_logs))
    # 📊 Visualization
    st.subheader("📈 Log Trend Analysis")
    # plot_logs(stored_logs)
    plot_logs_pie(stored_logs)

    st.subheader("📊 Error Type Distribution")
    plot_error_distribution(error_logs)

# Use Metrics
try:
    logs = fetch_live_k8s_logs()
    log_count_metric.set(len(logs))  # ✅ Ensures metrics are defined before use

    error_logs = extract_errors_warnings(logs)
    error_log_metric.set(len(error_logs))  # ✅ Ensures metrics are defined before use

    anomaly_logs = detect_anomalies_locally(logs) if anomaly_detection_method == "Local (TF-IDF & K-Means)" else detect_anomalies_ai(logs)
    anomaly_metric.set(len(anomaly_logs))  # ✅ Ensures metrics are defined before use
except NameError:
    st.error("Prometheus metrics not initialized properly. Restart the app.")

# Auto-refresh data
time.sleep(refresh_interval)
st.rerun()
st.cache.clear()
st.code("© 2025 - Built with Streamlit & Kubernetes")
# st.sidebar.markdown("![Kubernetes Logo](https://kubernetes.io/images/favicon.png)")