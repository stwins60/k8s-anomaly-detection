from kubernetes import client, config

# Load Kubernetes configuration
config.load_kube_config()

# Initialize Kubernetes API clients
v1 = client.CoreV1Api()

# Output file for storing logs
log_file = "kubernetes_pod_logs.txt"


def get_pod_logs():
    # Open file in write mode
    with open(log_file, "w", encoding="utf-8") as f:
        # List all pods in all namespaces
        pods = v1.list_pod_for_all_namespaces(watch=False)

        # Iterate through each pod and retrieve logs
        for pod in pods.items:
            pod_name = pod.metadata.name
            namespace = pod.metadata.namespace

            f.write(f"Logs for Pod: {pod_name} in Namespace: {namespace}\n")
            f.write("=" * 80 + "\n")

            try:
                log = v1.read_namespaced_pod_log(name=pod_name, namespace=namespace)
                f.write(log + "\n")
            except Exception as e:
                f.write(f"Could not retrieve logs for {pod_name}: {e}\n")

            f.write("\n" + "=" * 80 + "\n")

    print(f"Logs saved to {log_file}")
