# Deploying Feather DB on Azure

This guide walks you through deploying the Feather DB Cloud API and Dashboard on an Azure Virtual Machine using Docker Compose. The entire stack runs on a single cheap VM and persists all vector data to the VM's disk.

---

## What Gets Deployed

| Service | Port | Description |
|---------|------|-------------|
| `feather-api` | `8000` | FastAPI REST backend — vector add/search/delete |
| Dashboard UI | `8000/dashboard` | Gradio management UI (mounted inside the API) |

---

## 1. Create the Azure VM

From the [Azure Portal](https://portal.azure.com) or Azure CLI:

```bash
# Create resource group
az group create --name feather-rg --location eastus

# Create VM (Standard_B2s = 2 vCPU, 4 GB RAM — ~$37/month)
az vm create \
  --resource-group feather-rg \
  --name feather-vm \
  --image Ubuntu2204 \
  --size Standard_B2s \
  --admin-username featheradmin \
  --generate-ssh-keys \
  --public-ip-sku Standard

# Open port 8000 (API + Dashboard)
az vm open-port \
  --resource-group feather-rg \
  --name feather-vm \
  --port 8000 \
  --priority 300
```

Get your VM's public IP:
```bash
az vm show \
  --resource-group feather-rg \
  --name feather-vm \
  --show-details \
  --query publicIps \
  --output tsv
```

---

## 2. Install Docker on the VM

SSH into the VM:
```bash
ssh featheradmin@<YOUR_VM_IP>
```

Install Docker:
```bash
sudo apt-get update
sudo apt-get install -y docker.io docker-compose-plugin
sudo usermod -aG docker $USER
newgrp docker
docker --version
```

---

## 3. Clone and Deploy

```bash
git clone https://github.com/feather-store/feather.git
cd feather
```

Set your API key (choose any secret string):
```bash
export FEATHER_API_KEY="your-secret-api-key-here"
```

Or create a `.env` file:
```bash
echo "FEATHER_API_KEY=your-secret-api-key-here" > feather-api/.env
```

Build and start:
```bash
docker compose -f feather-api/docker-compose.yml up -d --build
```

Verify it's running:
```bash
curl http://localhost:8000/health
# {"status":"ok","version":"0.7.0","namespaces_loaded":0}
```

---

## 4. Access the Services

| URL | Description |
|-----|-------------|
| `http://<VM_IP>:8000/health` | Health check |
| `http://<VM_IP>:8000/docs` | Swagger / OpenAPI docs |
| `http://<VM_IP>:8000/dashboard` | Management Dashboard UI |

---

## 5. Connect from Python

Install the client:
```bash
pip install feather-db requests
```

Example — add and search vectors:
```python
import requests, numpy as np

BASE = "http://<VM_IP>:8000"
HEADERS = {"X-API-Key": "your-secret-api-key-here"}
NS = "my-namespace"

# Add a vector
vec = np.random.rand(768).astype(np.float32).tolist()
requests.post(f"{BASE}/v1/{NS}/vectors", headers=HEADERS, json={
    "id": 1,
    "vector": vec,
    "metadata": {
        "content": "Hello from Feather DB",
        "source": "my-app",
        "importance": 1.0,
        "type": 0,
        "namespace_id": NS,
        "entity_id": "doc_1",
        "tags_json": "[\"demo\"]",
        "timestamp": 0,
        "attributes": {}
    }
})

# Search
query = np.random.rand(768).astype(np.float32).tolist()
resp = requests.post(f"{BASE}/v1/{NS}/search", headers=HEADERS, json={
    "vector": query,
    "k": 5
})
for r in resp.json()["results"]:
    print(r["id"], r["score"], r["metadata"]["content"])
```

---

## 6. Where Data Lives

All vector data is stored in a Docker named volume:

```
feather-data  →  /data  (inside container)
                 *.feather files, one per namespace
```

The volume persists across:
- Container restarts
- `docker compose down` / `up`
- Image rebuilds

To find the volume on disk:
```bash
docker volume inspect feather-data
# Shows: "Mountpoint": "/var/lib/docker/volumes/feather-data/_data"
```

To back up your data:
```bash
sudo tar -czf feather-backup-$(date +%Y%m%d).tar.gz \
  /var/lib/docker/volumes/feather-data/_data
```

---

## 7. Managing the Deployment

```bash
# View logs
docker compose -f feather-api/docker-compose.yml logs -f

# Restart
docker compose -f feather-api/docker-compose.yml restart

# Stop (data preserved in volume)
docker compose -f feather-api/docker-compose.yml down

# Stop and wipe ALL data (destructive!)
docker compose -f feather-api/docker-compose.yml down -v

# Update to latest version
git pull origin master
docker compose -f feather-api/docker-compose.yml up -d --build
```

---

## 8. Securing the API

By default, if `FEATHER_API_KEY` is unset the API runs in **dev mode** (no auth). For production always set a key:

```bash
# Generate a strong key
openssl rand -hex 16

# Set it permanently
echo "FEATHER_API_KEY=$(openssl rand -hex 16)" >> ~/.bashrc
source ~/.bashrc
docker compose -f feather-api/docker-compose.yml up -d
```

Clients must pass the key as a header:
```
X-API-Key: your-secret-api-key-here
```

---

## 9. VM Sizing Guide

| Workload | Recommended VM | Monthly Cost (approx) |
|----------|---------------|----------------------|
| Dev / testing | Standard_B1s (1 vCPU, 1 GB) | ~$8 |
| Small production | Standard_B2s (2 vCPU, 4 GB) | ~$37 |
| Medium production | Standard_B4ms (4 vCPU, 16 GB) | ~$120 |
| Large / high QPS | Standard_D4s_v3 (4 vCPU, 16 GB SSD) | ~$175 |

> Feather DB is memory-mapped and CPU-bound for search. For >500k vectors or >100 QPS, choose a larger VM and increase disk size during VM creation (`--os-disk-size-gb 128`).

---

## 10. Useful API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health + version |
| `GET` | `/v1/namespaces` | List all namespaces |
| `GET` | `/v1/namespaces/{ns}/stats` | Namespace stats |
| `POST` | `/v1/{ns}/vectors` | Add a vector |
| `POST` | `/v1/{ns}/search` | Search vectors |
| `GET` | `/v1/{ns}/records` | Browse records |
| `GET` | `/v1/{ns}/records/{id}` | Get a record |
| `DELETE` | `/v1/{ns}/records/{id}` | Delete a record |
| `PUT` | `/v1/{ns}/records/{id}/importance` | Update importance |
| `POST` | `/v1/{ns}/save` | Flush to disk |

Full interactive docs: `http://<VM_IP>:8000/docs`
