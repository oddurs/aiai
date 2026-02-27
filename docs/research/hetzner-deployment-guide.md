# Deploying Autonomous AI Agent Infrastructure on Hetzner Cloud

**Research compiled: 2026-02-26**

---

## Table of Contents

1. [Hetzner Cloud Architecture for AI Agents](#1-hetzner-cloud-architecture-for-ai-agents)
2. [Cloud-Init Provisioning](#2-cloud-init-provisioning)
3. [Systemd Service Management](#3-systemd-service-management)
4. [Self-Hosted GitHub Actions Runners](#4-self-hosted-github-actions-runners)
5. [Monitoring Stack on Hetzner](#5-monitoring-stack-on-hetzner)
6. [Hetzner Object Storage (S3-Compatible)](#6-hetzner-object-storage-s3-compatible)
7. [Security Hardening](#7-security-hardening)
8. [Automated Deployment](#8-automated-deployment)
9. [Cost Optimization on Hetzner](#9-cost-optimization-on-hetzner)
10. [Disaster Recovery](#10-disaster-recovery)

---

## 1. Hetzner Cloud Architecture for AI Agents

### Why Hetzner for aiai

Hetzner Cloud offers the best price-to-performance ratio in European cloud hosting. For aiai -- a system where all inference happens via API calls to OpenRouter -- there is no need for GPU servers. The workloads are Python orchestration, metrics collection, CI runners, and coordination services. These are CPU-bound and memory-bound tasks that map perfectly to Hetzner's shared and dedicated vCPU offerings at a fraction of AWS/GCP pricing.

**References:**
- [Hetzner Cloud Products](https://www.hetzner.com/cloud)
- [Hetzner Server Comparison 2025 (Achromatic)](https://www.achromatic.dev/blog/hetzner-server-comparison)
- [Hetzner Cloud Pricing Calculator (CostGoat)](https://costgoat.com/pricing/hetzner)

### Server Type Comparison

Hetzner offers four server families. Here is how they map to aiai workloads:

| Series | CPU Type | vCPU Model | Best For | Availability |
|--------|----------|------------|----------|--------------|
| **CX** | Intel/AMD shared | Cost-optimized | Dev, staging, low-traffic services | EU only |
| **CAX** | Ampere Altra ARM | Shared ARM64 | Steady-state services, monitoring | EU only |
| **CPX** | AMD EPYC shared | Regular performance | Production, CI runners | Global |
| **CCX** | AMD EPYC dedicated | Guaranteed cores | Databases, heavy coordination | Global |

#### Detailed Pricing (EUR/month, Germany/Finland location)

**CX Series -- Cost-Optimized Shared vCPU (Intel/AMD)**

| Plan | vCPU | RAM | SSD | Traffic | Price/mo |
|------|------|-----|-----|---------|----------|
| CX23 | 2 | 4 GB | 40 GB | 20 TB | ~3.49 |
| CX33 | 4 | 8 GB | 80 GB | 20 TB | ~5.49 |
| CX43 | 8 | 16 GB | 160 GB | 20 TB | ~9.49 |
| CX53 | 16 | 32 GB | 320 GB | 20 TB | ~17.49 |

**CAX Series -- ARM64 Shared vCPU (Ampere Altra)**

| Plan | vCPU | RAM | SSD | Traffic | Price/mo |
|------|------|-----|-----|---------|----------|
| CAX11 | 2 | 4 GB | 40 GB | 20 TB | ~3.79 |
| CAX21 | 4 | 8 GB | 80 GB | 20 TB | ~6.49 |
| CAX31 | 8 | 16 GB | 160 GB | 20 TB | ~12.49 |
| CAX41 | 16 | 32 GB | 320 GB | 20 TB | ~24.49 |

**CPX Series -- AMD EPYC Shared vCPU (Regular Performance)**

| Plan | vCPU | RAM | SSD | Traffic | Price/mo |
|------|------|-----|-----|---------|----------|
| CPX11 | 2 | 2 GB | 40 GB | 20 TB | ~4.99 |
| CPX21 | 3 | 4 GB | 80 GB | 20 TB | ~9.49 |
| CPX31 | 4 | 8 GB | 160 GB | 20 TB | ~16.49 |
| CPX41 | 8 | 16 GB | 240 GB | 20 TB | ~30.49 |

**CCX Series -- Dedicated AMD EPYC vCPU**

| Plan | vCPU | RAM | SSD | Traffic | Price/mo |
|------|------|-----|-----|---------|----------|
| CCX13 | 2 | 8 GB | 80 GB | 20 TB | ~12.49 |
| CCX23 | 4 | 16 GB | 160 GB | 20 TB | ~24.49 |
| CCX33 | 8 | 32 GB | 240 GB | 20 TB | ~48.49 |
| CCX43 | 16 | 64 GB | 360 GB | 20 TB | ~96.49 |

Note: Prices in US locations are approximately 20% higher and include significantly less traffic (1-5 TB vs 20 TB). Singapore pricing is higher still. All prices are subject to the April 2026 increase of 30-37%.

**References:**
- [Hetzner Cloud Pricing](https://www.hetzner.com/cloud)
- [Hetzner Price Increase Announcement (Tom's Hardware)](https://www.tomshardware.com/tech-industry/hetzner-to-raise-prices-by-up-to-37-percent-from-april-1)

### Why GPU Servers Are Not Needed

aiai uses API-based inference through OpenRouter. All model calls -- whether to Claude, GPT-4, DeepSeek, or Gemini -- are HTTP requests to external endpoints. The servers never run local model inference. This means:

- No NVIDIA drivers, no CUDA, no GPU memory management
- No specialized GPU instance types (which Hetzner offers as dedicated servers, not cloud instances)
- CPU and RAM are the only compute dimensions that matter
- Network bandwidth matters for API calls but Hetzner includes 20 TB/month in EU

The workloads are:
- **Orchestration**: Python asyncio event loops managing agent coordination -- CPU + RAM
- **Metrics/Monitoring**: Prometheus scraping, Grafana rendering -- CPU + RAM + disk I/O
- **CI Runners**: Running tests, linting, building -- CPU + RAM + disk
- **Object Storage Client**: Reading/writing logs and artifacts -- network I/O

### Recommended Architecture

```
                     ┌─────────────────────────────────┐
                     │         Hetzner Cloud            │
                     │       Private Network            │
                     │        10.0.0.0/16               │
                     │                                  │
    Internet ──────► │  ┌──────────────────────┐       │
                     │  │   Load Balancer       │       │
                     │  │   (LB11 - 5.39/mo)   │       │
                     │  └──────┬───────────────┘       │
                     │         │                        │
                     │    ┌────┴────┐                   │
                     │    │         │                    │
                     │  ┌─┴──┐  ┌──┴──┐                │
                     │  │ A  │  │  B  │  Orchestration  │
                     │  │CX33│  │CX33 │  (Python)       │
                     │  └────┘  └─────┘                 │
                     │                                  │
                     │  ┌──────────────────────┐       │
                     │  │   Monitoring Server   │       │
                     │  │   CAX21 (6.49/mo)    │       │
                     │  │   Prometheus+Grafana  │       │
                     │  │   +Loki              │       │
                     │  └──────────────────────┘       │
                     │                                  │
                     │  ┌──────────────────────┐       │
                     │  │   CI Runner (on-demand)│       │
                     │  │   CPX31 (16.49/mo)    │       │
                     │  │   GitHub Actions       │       │
                     │  └──────────────────────┘       │
                     │                                  │
                     └─────────────────────────────────┘
                                    │
                                    ▼
                     ┌──────────────────────────┐
                     │  Hetzner Object Storage   │
                     │  (S3-compatible)           │
                     │  Logs, artifacts, backups  │
                     └──────────────────────────┘
```

### Server Role Assignment

| Role | Recommended Server | Monthly Cost | Justification |
|------|--------------------|--------------|---------------|
| Orchestrator (primary) | CX33 (4 vCPU, 8 GB) | ~5.49 | Python asyncio, agent coordination, API calls |
| Orchestrator (standby) | CX23 (2 vCPU, 4 GB) | ~3.49 | Warm standby for failover |
| Monitoring | CAX21 (4 ARM vCPU, 8 GB) | ~6.49 | Prometheus + Grafana + Loki, steady workload suits ARM |
| CI Runner | CPX31 (4 vCPU, 8 GB) | ~16.49 | On-demand, spin down when idle. AMD for broad compatibility |
| **Total (always-on)** | | **~15.47** | Orchestrator + standby + monitoring |
| **Total (with CI)** | | **~31.96** | All servers running |

### Network Topology

Create a private network for all servers to communicate without traversing the public internet:

```bash
# Create the private network
hcloud network create --name aiai-net --ip-range 10.0.0.0/16

# Create a subnet
hcloud network add-subnet aiai-net \
    --type cloud \
    --network-zone eu-central \
    --ip-range 10.0.1.0/24

# Attach servers to the network
hcloud server attach-to-network orchestrator-1 --network aiai-net --ip 10.0.1.10
hcloud server attach-to-network orchestrator-2 --network aiai-net --ip 10.0.1.11
hcloud server attach-to-network monitoring    --network aiai-net --ip 10.0.1.20
hcloud server attach-to-network ci-runner     --network aiai-net --ip 10.0.1.30
```

Private network traffic is free, unmetered, and does not count against the monthly traffic quota. All inter-service communication (metrics scraping, log shipping, health checks) should use private IPs.

### Load Balancer Configuration

For the orchestration layer, a load balancer distributes incoming webhook traffic and API requests:

```bash
# Create load balancer in the same network
hcloud load-balancer create \
    --name aiai-lb \
    --type lb11 \
    --location fsn1 \
    --network-zone eu-central

# Attach to private network
hcloud load-balancer attach-to-network aiai-lb --network aiai-net --ip 10.0.1.5

# Add targets
hcloud load-balancer add-target aiai-lb \
    --server orchestrator-1 \
    --use-private-ip

hcloud load-balancer add-target aiai-lb \
    --server orchestrator-2 \
    --use-private-ip

# Add HTTP service
hcloud load-balancer add-service aiai-lb \
    --protocol http \
    --listen-port 80 \
    --destination-port 8000 \
    --health-check-protocol http \
    --health-check-port 8000 \
    --health-check-path /health \
    --health-check-interval 10s \
    --health-check-timeout 5s \
    --health-check-retries 3
```

**References:**
- [Creating a Load Balancer (Hetzner Docs)](https://docs.hetzner.com/networking/load-balancers/getting-started/creating-a-load-balancer/)
- [Hetzner Load Balancer Pricing](https://www.hetzner.com/cloud/load-balancer/)

---

## 2. Cloud-Init Provisioning

### Overview

Cloud-init is the industry standard for early-stage server initialization. When you create a Hetzner Cloud server, you can pass a cloud-init configuration that executes on first boot. This means fully provisioned, production-ready servers with zero manual SSH sessions.

For aiai, the cloud-init configuration handles: system updates, Python 3.11 installation, user creation, SSH hardening, firewall setup, git clone, systemd service installation, and automatic security updates.

**References:**
- [Basic Cloud Config (Hetzner Community)](https://community.hetzner.com/tutorials/basic-cloud-config/)
- [Creating a Server (Hetzner Docs)](https://docs.hetzner.com/cloud/servers/getting-started/creating-a-server/)

### Complete Cloud-Init Configuration for Orchestration Server

```yaml
#cloud-config

# ============================================================
# aiai Orchestration Server - Cloud-Init Configuration
# Deploy with: hcloud server create --cloud-init cloud-init-orchestrator.yaml
# ============================================================

# --- System Configuration ---
hostname: aiai-orchestrator
timezone: UTC
locale: en_US.UTF-8

# --- Package Management ---
package_update: true
package_upgrade: true
packages:
  # Python 3.11 and build dependencies
  - python3.11
  - python3.11-venv
  - python3.11-dev
  - python3-pip
  # System essentials
  - git
  - curl
  - wget
  - jq
  - htop
  - tmux
  # Security
  - fail2ban
  - ufw
  - unattended-upgrades
  - apt-listchanges
  # Monitoring agent
  - prometheus-node-exporter

# --- Users ---
users:
  - name: aiai
    groups: [sudo, docker]
    shell: /bin/bash
    sudo: ['ALL=(ALL) NOPASSWD:ALL']
    ssh_authorized_keys:
      - ssh-ed25519 AAAA... deploy@aiai
    lock_passwd: true

# --- Write Configuration Files ---
write_files:
  # SSH hardening
  - path: /etc/ssh/sshd_config.d/99-aiai-hardening.conf
    content: |
      # aiai SSH hardening
      PermitRootLogin no
      PasswordAuthentication no
      PubkeyAuthentication yes
      AuthenticationMethods publickey
      X11Forwarding no
      AllowTcpForwarding no
      MaxAuthTries 3
      LoginGraceTime 30
      ClientAliveInterval 300
      ClientAliveCountMax 2
      AllowUsers aiai
    permissions: '0644'

  # UFW defaults -- deny incoming, allow outgoing
  - path: /etc/ufw/ufw.conf
    content: |
      ENABLED=yes
      LOGLEVEL=medium
    permissions: '0644'

  # Unattended upgrades configuration
  - path: /etc/apt/apt.conf.d/50unattended-upgrades
    content: |
      Unattended-Upgrade::Allowed-Origins {
          "${distro_id}:${distro_codename}";
          "${distro_id}:${distro_codename}-security";
          "${distro_id}ESMApps:${distro_codename}-apps-security";
          "${distro_id}ESM:${distro_codename}-infra-security";
      };
      Unattended-Upgrade::AutoFixInterruptedDpkg "true";
      Unattended-Upgrade::MinimalSteps "true";
      Unattended-Upgrade::Remove-Unused-Kernel-Packages "true";
      Unattended-Upgrade::Remove-New-Unused-Dependencies "true";
      Unattended-Upgrade::Remove-Unused-Dependencies "true";
      Unattended-Upgrade::Automatic-Reboot "false";
    permissions: '0644'

  # Periodic upgrades
  - path: /etc/apt/apt.conf.d/20auto-upgrades
    content: |
      APT::Periodic::Update-Package-Lists "1";
      APT::Periodic::Unattended-Upgrade "1";
      APT::Periodic::Download-Upgradeable-Packages "1";
      APT::Periodic::AutocleanInterval "7";
    permissions: '0644'

  # Systemd service for aiai orchestrator
  - path: /etc/systemd/system/aiai-orchestrator.service
    content: |
      [Unit]
      Description=aiai Orchestrator Service
      After=network-online.target
      Wants=network-online.target
      StartLimitIntervalSec=300
      StartLimitBurst=5

      [Service]
      Type=exec
      User=aiai
      Group=aiai
      WorkingDirectory=/opt/aiai
      ExecStart=/opt/aiai/.venv/bin/python -m src.orchestrator
      Restart=on-failure
      RestartSec=10
      WatchdogSec=60

      # Environment
      EnvironmentFile=/etc/aiai/env
      Environment=PYTHONUNBUFFERED=1
      Environment=PYTHONDONTWRITEBYTECODE=1

      # Resource limits
      MemoryMax=6G
      MemoryHigh=5G
      CPUQuota=350%
      TasksMax=256
      LimitNOFILE=65536

      # Security hardening
      NoNewPrivileges=true
      ProtectSystem=strict
      ProtectHome=true
      ReadWritePaths=/opt/aiai /var/log/aiai
      PrivateTmp=true
      ProtectKernelTunables=true
      ProtectKernelModules=true
      ProtectControlGroups=true

      # Logging
      StandardOutput=journal
      StandardError=journal
      SyslogIdentifier=aiai-orchestrator

      [Install]
      WantedBy=multi-user.target
    permissions: '0644'

  # Environment file template (secrets injected separately)
  - path: /etc/aiai/env
    content: |
      # Populated by deployment script -- never commit this file
      OPENROUTER_API_KEY=
      GITHUB_TOKEN=
      HETZNER_API_TOKEN=
      AIAI_ENVIRONMENT=production
      AIAI_LOG_LEVEL=INFO
      AIAI_METRICS_PORT=9090
    permissions: '0600'

  # Fail2ban jail for SSH
  - path: /etc/fail2ban/jail.d/sshd.conf
    content: |
      [sshd]
      enabled = true
      port = ssh
      filter = sshd
      logpath = /var/log/auth.log
      maxretry = 3
      bantime = 3600
      findtime = 600
    permissions: '0644'

# --- Run Commands (executed in order) ---
runcmd:
  # --- Firewall ---
  - ufw default deny incoming
  - ufw default allow outgoing
  - ufw allow 22/tcp comment 'SSH'
  - ufw allow from 10.0.0.0/16 to any port 8000 comment 'Orchestrator API (private)'
  - ufw allow from 10.0.0.0/16 to any port 9100 comment 'Node Exporter (private)'
  - ufw allow from 10.0.0.0/16 to any port 9090 comment 'App Metrics (private)'
  - ufw --force enable

  # --- Python setup ---
  - update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
  - update-alternatives --set python3 /usr/bin/python3.11

  # --- Clone and setup aiai ---
  - mkdir -p /opt/aiai /var/log/aiai
  - git clone https://github.com/your-org/aiai.git /opt/aiai
  - chown -R aiai:aiai /opt/aiai /var/log/aiai
  - su - aiai -c "cd /opt/aiai && python3 -m venv .venv"
  - su - aiai -c "cd /opt/aiai && .venv/bin/pip install --upgrade pip"
  - su - aiai -c "cd /opt/aiai && .venv/bin/pip install -r requirements.txt"

  # --- Start services ---
  - systemctl daemon-reload
  - systemctl enable --now fail2ban
  - systemctl enable --now prometheus-node-exporter
  - systemctl enable --now aiai-orchestrator
  - systemctl restart sshd

  # --- Log completion ---
  - echo "aiai cloud-init provisioning complete at $(date -u)" >> /var/log/aiai/provision.log
```

### Deploying with Cloud-Init via hcloud CLI

```bash
# Create the orchestration server with cloud-init
hcloud server create \
    --name aiai-orchestrator-1 \
    --type cx33 \
    --image ubuntu-24.04 \
    --location fsn1 \
    --ssh-key aiai-deploy \
    --network aiai-net \
    --user-data-from-file cloud-init-orchestrator.yaml \
    --label env=production \
    --label role=orchestrator

# Check cloud-init progress (after SSH is available)
ssh aiai@<server-ip> "cloud-init status --wait"

# View cloud-init logs for debugging
ssh aiai@<server-ip> "sudo cat /var/log/cloud-init-output.log"
```

### Cloud-Init for Monitoring Server

```yaml
#cloud-config

hostname: aiai-monitoring
timezone: UTC

package_update: true
package_upgrade: true
packages:
  - apt-transport-https
  - software-properties-common
  - wget
  - curl
  - jq
  - fail2ban
  - ufw
  - unattended-upgrades

users:
  - name: aiai
    groups: [sudo]
    shell: /bin/bash
    sudo: ['ALL=(ALL) NOPASSWD:ALL']
    ssh_authorized_keys:
      - ssh-ed25519 AAAA... deploy@aiai
    lock_passwd: true

write_files:
  - path: /etc/ssh/sshd_config.d/99-aiai-hardening.conf
    content: |
      PermitRootLogin no
      PasswordAuthentication no
      PubkeyAuthentication yes
      MaxAuthTries 3
      AllowUsers aiai
    permissions: '0644'

runcmd:
  # Firewall
  - ufw default deny incoming
  - ufw default allow outgoing
  - ufw allow 22/tcp comment 'SSH'
  - ufw allow from 10.0.0.0/16 to any port 3000 comment 'Grafana (private)'
  - ufw allow from 10.0.0.0/16 to any port 9090 comment 'Prometheus (private)'
  - ufw allow from 10.0.0.0/16 to any port 3100 comment 'Loki (private)'
  - ufw --force enable

  # Install Prometheus
  - useradd --no-create-home --shell /bin/false prometheus
  - mkdir -p /etc/prometheus /var/lib/prometheus
  - wget -q https://github.com/prometheus/prometheus/releases/download/v2.53.0/prometheus-2.53.0.linux-amd64.tar.gz -O /tmp/prometheus.tar.gz
  - tar xzf /tmp/prometheus.tar.gz -C /tmp/
  - cp /tmp/prometheus-*/prometheus /usr/local/bin/
  - cp /tmp/prometheus-*/promtool /usr/local/bin/
  - chown prometheus:prometheus /usr/local/bin/prometheus /usr/local/bin/promtool
  - chown -R prometheus:prometheus /etc/prometheus /var/lib/prometheus

  # Install Grafana
  - wget -q -O /usr/share/keyrings/grafana.key https://apt.grafana.com/gpg.key
  - echo "deb [signed-by=/usr/share/keyrings/grafana.key] https://apt.grafana.com stable main" > /etc/apt/sources.list.d/grafana.list
  - apt-get update
  - apt-get install -y grafana

  # Install Loki and Promtail
  - wget -q https://github.com/grafana/loki/releases/download/v3.1.0/loki-linux-amd64.zip -O /tmp/loki.zip
  - unzip /tmp/loki.zip -d /usr/local/bin/
  - chmod +x /usr/local/bin/loki-linux-amd64
  - mv /usr/local/bin/loki-linux-amd64 /usr/local/bin/loki

  # Start services
  - systemctl daemon-reload
  - systemctl enable --now prometheus
  - systemctl enable --now grafana-server
  - systemctl restart sshd
```

---

## 3. Systemd Service Management

### Design Principles for aiai Services

Every aiai Python process runs as a systemd service. This provides:
- Automatic restart on failure
- Structured logging via journald
- Resource limits (memory, CPU) to prevent runaway processes
- Dependency ordering (start after network, after database, etc.)
- Watchdog integration for hang detection
- Clean shutdown handling

### Primary Orchestrator Service

```ini
# /etc/systemd/system/aiai-orchestrator.service
[Unit]
Description=aiai Orchestrator - Agent Coordination Service
Documentation=https://github.com/your-org/aiai
After=network-online.target
Wants=network-online.target
StartLimitIntervalSec=300
StartLimitBurst=5

[Service]
Type=exec
User=aiai
Group=aiai
WorkingDirectory=/opt/aiai

# Main process
ExecStart=/opt/aiai/.venv/bin/python -m src.orchestrator
ExecReload=/bin/kill -HUP $MAINPID

# Restart policy: restart on failure, but not on clean exit
Restart=on-failure
RestartSec=10

# Watchdog: service must notify systemd every 60s or it gets killed
WatchdogSec=60
# Notify systemd of readiness
Type=notify

# Environment
EnvironmentFile=/etc/aiai/env
Environment=PYTHONUNBUFFERED=1
Environment=PYTHONDONTWRITEBYTECODE=1
Environment=PYTHONPATH=/opt/aiai

# Resource limits
MemoryMax=6G
MemoryHigh=5G
CPUQuota=350%
TasksMax=256
LimitNOFILE=65536

# Security sandbox
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/opt/aiai /var/log/aiai
PrivateTmp=true
ProtectKernelTunables=true
ProtectKernelModules=true
ProtectControlGroups=true
RestrictSUIDSGID=true
RestrictNamespaces=true

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=aiai-orchestrator

[Install]
WantedBy=multi-user.target
```

### Python Watchdog Integration

To use systemd watchdog from Python, the service must periodically notify systemd that it is still alive:

```python
"""systemd watchdog integration for aiai services."""

import os
import socket
import asyncio
from pathlib import Path


def notify_systemd(state: str) -> None:
    """Send a notification to systemd.

    Args:
        state: Notification string, e.g. "READY=1", "WATCHDOG=1", "STOPPING=1"
    """
    addr = os.environ.get("NOTIFY_SOCKET")
    if not addr:
        return

    if addr.startswith("@"):
        addr = "\0" + addr[1:]

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    try:
        sock.connect(addr)
        sock.sendall(state.encode())
    finally:
        sock.close()


async def watchdog_loop(interval_sec: float | None = None) -> None:
    """Continuously ping the systemd watchdog.

    If WATCHDOG_USEC is set, pings at half that interval.
    Otherwise uses the provided interval or does nothing.
    """
    usec = os.environ.get("WATCHDOG_USEC")
    if usec:
        interval_sec = int(usec) / 1_000_000 / 2  # ping at half the timeout
    elif interval_sec is None:
        return  # no watchdog configured

    while True:
        notify_systemd("WATCHDOG=1")
        await asyncio.sleep(interval_sec)


async def main() -> None:
    """Example main with watchdog integration."""
    # Signal readiness
    notify_systemd("READY=1")

    # Start watchdog pinger as background task
    watchdog_task = asyncio.create_task(watchdog_loop())

    try:
        # ... run your actual service logic here ...
        await run_orchestrator()
    finally:
        notify_systemd("STOPPING=1")
        watchdog_task.cancel()
```

### Metrics Exporter Service

```ini
# /etc/systemd/system/aiai-metrics.service
[Unit]
Description=aiai Metrics Exporter
After=network-online.target
Wants=network-online.target

[Service]
Type=exec
User=aiai
Group=aiai
WorkingDirectory=/opt/aiai
ExecStart=/opt/aiai/.venv/bin/python -m src.metrics.exporter
Restart=on-failure
RestartSec=5

EnvironmentFile=/etc/aiai/env
Environment=PYTHONUNBUFFERED=1
Environment=METRICS_PORT=9091

MemoryMax=512M
CPUQuota=50%

NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/opt/aiai

StandardOutput=journal
StandardError=journal
SyslogIdentifier=aiai-metrics

[Install]
WantedBy=multi-user.target
```

### Timer-Based Evolution Service

For periodic tasks like self-improvement analysis, use a systemd timer instead of cron:

```ini
# /etc/systemd/system/aiai-evolution.service
[Unit]
Description=aiai Evolution Analysis Run
After=network-online.target

[Service]
Type=oneshot
User=aiai
Group=aiai
WorkingDirectory=/opt/aiai
ExecStart=/opt/aiai/.venv/bin/python -m src.evolution.analyze

EnvironmentFile=/etc/aiai/env
Environment=PYTHONUNBUFFERED=1

MemoryMax=4G
CPUQuota=200%
TimeoutStartSec=600

NoNewPrivileges=true
ProtectSystem=strict
ReadWritePaths=/opt/aiai /var/log/aiai

StandardOutput=journal
StandardError=journal
SyslogIdentifier=aiai-evolution
```

```ini
# /etc/systemd/system/aiai-evolution.timer
[Unit]
Description=Run aiai evolution analysis every 6 hours

[Timer]
OnCalendar=*-*-* 00/6:00:00
RandomizedDelaySec=300
Persistent=true

[Install]
WantedBy=timers.target
```

### Service Management Commands

```bash
# Enable and start
sudo systemctl enable --now aiai-orchestrator.service
sudo systemctl enable --now aiai-metrics.service
sudo systemctl enable --now aiai-evolution.timer

# Check status
systemctl status aiai-orchestrator
systemctl list-timers aiai-*

# View logs
journalctl -u aiai-orchestrator -f                    # follow live
journalctl -u aiai-orchestrator --since "1 hour ago"  # recent logs
journalctl -u aiai-orchestrator -p err                # errors only
journalctl -u aiai-orchestrator --output json-pretty  # structured output

# Resource usage
systemctl show aiai-orchestrator -p MemoryCurrent,CPUUsageNSec,TasksCurrent

# Reload configuration without restart
sudo systemctl reload aiai-orchestrator

# Restart with rate limiting respected
sudo systemctl restart aiai-orchestrator
```

---

## 4. Self-Hosted GitHub Actions Runners

### Why Self-Hosted Runners on Hetzner

GitHub-hosted runners cost $0.008/minute for Linux (2-core). A CPX31 on Hetzner at ~16.49/month provides 4 vCPUs and 8 GB RAM -- equivalent to running a GitHub-hosted runner for ~34 hours/month. If your CI runs more than 34 hours/month, self-hosted on Hetzner is cheaper. For aiai with continuous autonomous development, CI usage will far exceed that.

**Cost comparison:**

| Scenario | GitHub-Hosted | Hetzner Self-Hosted |
|----------|---------------|---------------------|
| 50 hours/month CI | ~$24 | ~$18 (CPX31) |
| 100 hours/month CI | ~$48 | ~$18 (CPX31) |
| 200 hours/month CI | ~$96 | ~$18 (CPX31) |
| 24/7 runner | ~$346 | ~$18 (CPX31) |

**References:**
- [Self-Hosted GitHub Actions Runner on Hetzner Cloud (GitHub Marketplace)](https://github.com/marketplace/actions/self-hosted-github-actions-runner-on-hetzner-cloud)
- [Autoscaling Self-Hosted Runners (TestFlows)](https://github.com/testflows/TestFlows-GitHub-Hetzner-Runners)
- [Slash CI/CD Bills with Hetzner Cloud GitHub Runners (Altinity)](https://altinity.com/blog/slash-ci-cd-bills-part-2-using-hetzner-cloud-github-runners-for-your-repository)

### Persistent Runner Setup

For a dedicated CI runner that is always available:

```bash
#!/usr/bin/env bash
# scripts/setup-github-runner.sh
# Install and configure a persistent GitHub Actions runner on Hetzner
set -euo pipefail

RUNNER_VERSION="2.321.0"
RUNNER_USER="runner"
RUNNER_DIR="/opt/actions-runner"
REPO_URL="https://github.com/your-org/aiai"

# Get a registration token from GitHub API
RUNNER_TOKEN=$(curl -s \
    -X POST \
    -H "Authorization: token ${GITHUB_TOKEN}" \
    -H "Accept: application/vnd.github.v3+json" \
    "https://api.github.com/repos/your-org/aiai/actions/runners/registration-token" \
    | jq -r .token)

# Create runner user
useradd -m -s /bin/bash "${RUNNER_USER}"

# Download and extract runner
mkdir -p "${RUNNER_DIR}"
cd "${RUNNER_DIR}"
curl -sL "https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz" \
    | tar xz

chown -R "${RUNNER_USER}:${RUNNER_USER}" "${RUNNER_DIR}"

# Configure the runner
su - "${RUNNER_USER}" -c "cd ${RUNNER_DIR} && ./config.sh \
    --url '${REPO_URL}' \
    --token '${RUNNER_TOKEN}' \
    --name 'hetzner-cpx31-$(hostname)' \
    --labels 'self-hosted,linux,x64,hetzner,aiai' \
    --work '_work' \
    --unattended \
    --replace"

# Install as systemd service
cd "${RUNNER_DIR}"
./svc.sh install "${RUNNER_USER}"
./svc.sh start
```

### Runner Systemd Service (Manual)

If you prefer manual control over the systemd service:

```ini
# /etc/systemd/system/github-runner.service
[Unit]
Description=GitHub Actions Runner for aiai
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=runner
Group=runner
WorkingDirectory=/opt/actions-runner
ExecStart=/opt/actions-runner/run.sh
Restart=always
RestartSec=10

# Resource limits -- CI can be resource-hungry
MemoryMax=7G
CPUQuota=400%
TasksMax=512
LimitNOFILE=65536

# Runner needs more filesystem access than typical services
ProtectSystem=false
ProtectHome=false
NoNewPrivileges=false

StandardOutput=journal
StandardError=journal
SyslogIdentifier=github-runner

[Install]
WantedBy=multi-user.target
```

### On-Demand Ephemeral Runners

For cost optimization, create runners on-demand and destroy them after each job. Use the `hcloud-github-runner` approach:

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [main]
  pull_request:

jobs:
  # Step 1: Create a Hetzner server for the runner
  create-runner:
    runs-on: ubuntu-latest
    outputs:
      server-id: ${{ steps.create.outputs.server-id }}
      runner-label: ${{ steps.create.outputs.runner-label }}
    steps:
      - name: Create Hetzner runner
        id: create
        uses: Cyclenerd/hcloud-github-runner@v1
        with:
          hcloud-token: ${{ secrets.HETZNER_API_TOKEN }}
          github-token: ${{ secrets.GH_RUNNER_TOKEN }}
          server-type: cpx31
          image: ubuntu-24.04
          location: fsn1
          ssh-key: aiai-deploy

  # Step 2: Run tests on the ephemeral runner
  test:
    needs: create-runner
    runs-on: ${{ needs.create-runner.outputs.runner-label }}
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          python -m venv .venv
          .venv/bin/pip install -r requirements.txt
      - name: Run tests
        run: .venv/bin/pytest tests/ -v --tb=short

  # Step 3: Destroy the server
  cleanup-runner:
    needs: [create-runner, test]
    runs-on: ubuntu-latest
    if: always()
    steps:
      - name: Delete Hetzner server
        uses: Cyclenerd/hcloud-github-runner@v1
        with:
          hcloud-token: ${{ secrets.HETZNER_API_TOKEN }}
          github-token: ${{ secrets.GH_RUNNER_TOKEN }}
          server-id: ${{ needs.create-runner.outputs.server-id }}
          action: delete
```

### Docker-in-Docker for Job Isolation

If running a persistent runner and you need job isolation:

```bash
# Install Docker on the runner
apt-get update
apt-get install -y docker.io
usermod -aG docker runner
systemctl enable --now docker

# Configure Docker daemon for resource limits
cat > /etc/docker/daemon.json << 'EOF'
{
    "default-ulimits": {
        "nofile": { "Name": "nofile", "Hard": 65536, "Soft": 65536 }
    },
    "log-driver": "journald",
    "storage-driver": "overlay2",
    "default-address-pools": [
        { "base": "172.17.0.0/16", "size": 24 }
    ]
}
EOF
systemctl restart docker
```

### Security Considerations for Self-Hosted Runners

Self-hosted runners executing code from pull requests present a significant attack surface. For aiai in full-auto mode (no PRs, direct commits to main), the risk surface is smaller but still requires attention:

1. **Runner user has limited privileges** -- never run as root
2. **Ephemeral runners** are preferred -- destroy after each job, no state leaks
3. **Network isolation** -- runner only needs outbound internet access and private network to monitoring
4. **Secret isolation** -- use GitHub Actions secrets, never embed in runner config
5. **Docker isolation** -- each job runs in a container if Docker-in-Docker is configured
6. **Firewall rules** -- restrict runner to only necessary outbound ports (443 for GitHub, API endpoints)

---

## 5. Monitoring Stack on Hetzner

### Architecture Overview

The monitoring stack runs on a single CAX21 (ARM, 4 vCPU, 8 GB RAM) for approximately 6.49/month. This server hosts Prometheus (metrics), Grafana (visualization), Loki (logs), and Alertmanager (alerts). All components are lightweight enough for a single node at aiai's scale.

```
┌─────────────────────────────────────────────────┐
│              Monitoring Server (CAX21)            │
│                                                   │
│  ┌────────────┐  ┌──────────┐  ┌──────────────┐ │
│  │ Prometheus  │  │ Grafana   │  │ Alertmanager │ │
│  │ :9090       │  │ :3000     │  │ :9093        │ │
│  └──────┬─────┘  └────┬─────┘  └──────────────┘ │
│         │              │                          │
│  ┌──────┴──────────────┴────────────────────┐    │
│  │               Loki :3100                  │    │
│  └───────────────────────────────────────────┘    │
│                                                   │
│  Scrapes via private network (10.0.1.0/24):      │
│  - orchestrator-1:9100 (node_exporter)            │
│  - orchestrator-1:9091 (aiai metrics)             │
│  - orchestrator-2:9100 (node_exporter)            │
│  - ci-runner:9100 (node_exporter)                 │
│  - monitoring:9100 (self, node_exporter)          │
└─────────────────────────────────────────────────┘
```

### Prometheus Configuration

```yaml
# /etc/prometheus/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  scrape_timeout: 10s

  external_labels:
    cluster: 'aiai-production'
    environment: 'prod'

# Alerting configuration
alerting:
  alertmanagers:
    - static_configs:
        - targets:
            - 'localhost:9093'

# Rules files
rule_files:
  - '/etc/prometheus/rules/*.yml'

# Scrape targets
scrape_configs:
  # Prometheus self-monitoring
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  # Node exporters on all servers
  - job_name: 'node'
    static_configs:
      - targets:
          - '10.0.1.10:9100'  # orchestrator-1
          - '10.0.1.11:9100'  # orchestrator-2
          - '10.0.1.20:9100'  # monitoring (self)
          - '10.0.1.30:9100'  # ci-runner
        labels:
          env: 'production'

  # aiai application metrics
  - job_name: 'aiai-orchestrator'
    metrics_path: /metrics
    static_configs:
      - targets:
          - '10.0.1.10:9091'  # orchestrator-1
          - '10.0.1.11:9091'  # orchestrator-2
        labels:
          service: 'orchestrator'

  # aiai custom metrics exporter
  - job_name: 'aiai-metrics'
    metrics_path: /metrics
    static_configs:
      - targets:
          - '10.0.1.10:9092'
        labels:
          service: 'metrics-exporter'

  # GitHub runner metrics (when running)
  - job_name: 'github-runner'
    static_configs:
      - targets:
          - '10.0.1.30:9100'
        labels:
          service: 'ci-runner'
```

### Prometheus Alert Rules for aiai

```yaml
# /etc/prometheus/rules/aiai-alerts.yml
groups:
  - name: aiai-system
    interval: 30s
    rules:
      # Service down
      - alert: AiaiOrchestratorDown
        expr: up{job="aiai-orchestrator"} == 0
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "aiai orchestrator is down on {{ $labels.instance }}"
          description: "The orchestrator service has been unreachable for 2 minutes."

      # High memory usage
      - alert: HighMemoryUsage
        expr: (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes) < 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage on {{ $labels.instance }}"
          description: "Available memory is below 10% for 5 minutes."

      # Disk space low
      - alert: DiskSpaceLow
        expr: (node_filesystem_avail_bytes{mountpoint="/"} / node_filesystem_size_bytes{mountpoint="/"}) < 0.15
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Disk space low on {{ $labels.instance }}"
          description: "Less than 15% disk space remaining on root filesystem."

  - name: aiai-application
    interval: 30s
    rules:
      # API cost tracking
      - alert: HighAPICostRate
        expr: rate(aiai_api_cost_total[1h]) > 5
        for: 15m
        labels:
          severity: warning
        annotations:
          summary: "High API cost rate"
          description: "API costs exceeding $5/hour for 15 minutes. Current rate: {{ $value }}/hr"

      # Task failure rate
      - alert: HighTaskFailureRate
        expr: rate(aiai_task_failures_total[15m]) / rate(aiai_task_total[15m]) > 0.3
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "High task failure rate"
          description: "More than 30% of tasks failing over 15 minutes."

      # Evolution regression
      - alert: EvolutionRegression
        expr: aiai_evolution_score < aiai_evolution_score offset 1d
        for: 6h
        labels:
          severity: info
        annotations:
          summary: "Evolution score regression detected"
          description: "Evolution score is lower than 24 hours ago."
```

### Loki Configuration

```yaml
# /etc/loki/loki-config.yml
auth_enabled: false

server:
  http_listen_port: 3100
  grpc_listen_port: 9096

common:
  path_prefix: /var/lib/loki
  storage:
    filesystem:
      chunks_directory: /var/lib/loki/chunks
      rules_directory: /var/lib/loki/rules
  replication_factor: 1
  ring:
    instance_addr: 127.0.0.1
    kvstore:
      store: inmemory

query_range:
  results_cache:
    cache:
      embedded_cache:
        enabled: true
        max_size_mb: 100

schema_config:
  configs:
    - from: 2024-01-01
      store: tsdb
      object_store: filesystem
      schema: v13
      index:
        prefix: index_
        period: 24h

limits_config:
  retention_period: 30d
  max_query_series: 500

compactor:
  working_directory: /var/lib/loki/compactor
  compaction_interval: 10m
  retention_enabled: true
  retention_delete_delay: 2h
```

### Promtail Configuration (on each server)

```yaml
# /etc/promtail/promtail-config.yml
server:
  http_listen_port: 9080
  grpc_listen_port: 0

positions:
  filename: /var/lib/promtail/positions.yaml

clients:
  - url: http://10.0.1.20:3100/loki/api/v1/push

scrape_configs:
  # System journal
  - job_name: journal
    journal:
      max_age: 12h
      labels:
        job: systemd-journal
        host: ${HOSTNAME}
    relabel_configs:
      - source_labels: ['__journal__systemd_unit']
        target_label: 'unit'

  # aiai application logs
  - job_name: aiai
    static_configs:
      - targets:
          - localhost
        labels:
          job: aiai
          host: ${HOSTNAME}
          __path__: /var/log/aiai/*.log
```

### Grafana Dashboard for aiai

A minimal dashboard JSON snippet for aiai cost tracking:

```json
{
  "dashboard": {
    "title": "aiai - System Overview",
    "tags": ["aiai", "production"],
    "timezone": "utc",
    "panels": [
      {
        "title": "API Cost (Last 24h)",
        "type": "stat",
        "gridPos": { "h": 4, "w": 6, "x": 0, "y": 0 },
        "targets": [
          {
            "expr": "increase(aiai_api_cost_total[24h])",
            "legendFormat": "Total Cost"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "currencyUSD",
            "thresholds": {
              "steps": [
                { "color": "green", "value": null },
                { "color": "yellow", "value": 10 },
                { "color": "red", "value": 50 }
              ]
            }
          }
        }
      },
      {
        "title": "Tasks Completed (24h)",
        "type": "stat",
        "gridPos": { "h": 4, "w": 6, "x": 6, "y": 0 },
        "targets": [
          {
            "expr": "increase(aiai_task_total{status=\"success\"}[24h])",
            "legendFormat": "Completed"
          }
        ]
      },
      {
        "title": "Task Success Rate",
        "type": "gauge",
        "gridPos": { "h": 4, "w": 6, "x": 12, "y": 0 },
        "targets": [
          {
            "expr": "rate(aiai_task_total{status=\"success\"}[1h]) / rate(aiai_task_total[1h]) * 100",
            "legendFormat": "Success %"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "percent",
            "min": 0,
            "max": 100,
            "thresholds": {
              "steps": [
                { "color": "red", "value": null },
                { "color": "yellow", "value": 70 },
                { "color": "green", "value": 90 }
              ]
            }
          }
        }
      },
      {
        "title": "Evolution Score Over Time",
        "type": "timeseries",
        "gridPos": { "h": 8, "w": 12, "x": 0, "y": 4 },
        "targets": [
          {
            "expr": "aiai_evolution_score",
            "legendFormat": "{{ dimension }}"
          }
        ]
      },
      {
        "title": "API Cost by Model",
        "type": "timeseries",
        "gridPos": { "h": 8, "w": 12, "x": 12, "y": 4 },
        "targets": [
          {
            "expr": "rate(aiai_api_cost_total[1h]) * 3600",
            "legendFormat": "{{ model }}"
          }
        ],
        "fieldConfig": {
          "defaults": { "unit": "currencyUSD" }
        }
      },
      {
        "title": "Server Memory Usage",
        "type": "timeseries",
        "gridPos": { "h": 8, "w": 12, "x": 0, "y": 12 },
        "targets": [
          {
            "expr": "100 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes * 100)",
            "legendFormat": "{{ instance }}"
          }
        ],
        "fieldConfig": {
          "defaults": { "unit": "percent", "max": 100 }
        }
      },
      {
        "title": "CPU Usage",
        "type": "timeseries",
        "gridPos": { "h": 8, "w": 12, "x": 12, "y": 12 },
        "targets": [
          {
            "expr": "100 - (avg by(instance)(rate(node_cpu_seconds_total{mode=\"idle\"}[5m])) * 100)",
            "legendFormat": "{{ instance }}"
          }
        ],
        "fieldConfig": {
          "defaults": { "unit": "percent", "max": 100 }
        }
      }
    ],
    "time": { "from": "now-24h", "to": "now" },
    "refresh": "30s"
  }
}
```

### Custom Prometheus Metrics for aiai

Expose application-level metrics from the Python orchestrator:

```python
"""Custom Prometheus metrics for aiai services."""

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    Info,
    start_http_server,
)

# --- Task Metrics ---
TASK_TOTAL = Counter(
    'aiai_task_total',
    'Total number of tasks processed',
    ['status', 'complexity', 'agent_type']
)

TASK_DURATION = Histogram(
    'aiai_task_duration_seconds',
    'Time spent processing tasks',
    ['complexity', 'agent_type'],
    buckets=[1, 5, 15, 30, 60, 120, 300, 600, 1800]
)

# --- API Cost Metrics ---
API_COST = Counter(
    'aiai_api_cost_total',
    'Total API cost in USD',
    ['model', 'provider', 'complexity']
)

API_TOKENS_USED = Counter(
    'aiai_api_tokens_total',
    'Total API tokens consumed',
    ['model', 'direction']  # direction: input/output
)

API_REQUESTS = Counter(
    'aiai_api_requests_total',
    'Total API requests made',
    ['model', 'status']  # status: success/failure/timeout
)

# --- Evolution Metrics ---
EVOLUTION_SCORE = Gauge(
    'aiai_evolution_score',
    'Current evolution score',
    ['dimension']  # code_quality, test_coverage, cost_efficiency, etc.
)

COMMITS_TOTAL = Counter(
    'aiai_commits_total',
    'Total commits made by agents',
    ['type', 'agent']  # type: feat/fix/refactor/evolve/etc.
)

# --- System Metrics ---
ACTIVE_AGENTS = Gauge(
    'aiai_active_agents',
    'Number of currently active agents',
    ['type']  # builder, researcher, architect, evolver
)

QUEUE_DEPTH = Gauge(
    'aiai_task_queue_depth',
    'Number of tasks waiting in queue',
    ['priority']
)

BUILD_INFO = Info(
    'aiai_build',
    'Build information'
)


def init_metrics(port: int = 9091) -> None:
    """Initialize and start the Prometheus metrics HTTP server."""
    BUILD_INFO.info({
        'version': '0.1.0',
        'python_version': '3.11',
        'git_sha': 'unknown',  # populated at startup
    })
    start_http_server(port)
```

---

## 6. Hetzner Object Storage (S3-Compatible)

### Overview

Hetzner Object Storage provides S3-compatible storage at significantly lower cost than AWS S3. The pricing structure is simple:

- **Base price**: 5.99/month (includes 1 TB storage + 1 TB egress)
- **Additional storage**: ~0.0067/GB-hour beyond the first TB
- **Additional egress**: ~1.00/TB beyond the first TB
- **Available in**: Falkenstein (FSN1), Helsinki (HEL1), Nuremberg (NBG1)

For aiai, object storage serves as the repository for logs, metrics archives, build artifacts, and backups. The 1 TB included quota is generous for these use cases.

**References:**
- [Hetzner Object Storage Overview](https://docs.hetzner.com/storage/object-storage/overview/)
- [Hetzner Object Storage Pricing](https://www.hetzner.com/storage/object-storage/)

### boto3 Configuration

```python
"""Hetzner Object Storage client using boto3/aioboto3."""

import boto3
from botocore.config import Config


def get_s3_client():
    """Create a boto3 client configured for Hetzner Object Storage."""
    return boto3.client(
        's3',
        endpoint_url='https://fsn1.your-objectstorage.com',
        aws_access_key_id='YOUR_ACCESS_KEY',      # from env or secrets
        aws_secret_access_key='YOUR_SECRET_KEY',   # from env or secrets
        region_name='fsn1',
        config=Config(
            signature_version='s3v4',
            retries={'max_attempts': 3, 'mode': 'adaptive'},
            connect_timeout=5,
            read_timeout=30,
        ),
    )


def get_async_s3_client():
    """Create an aioboto3 session for async S3 operations."""
    import aioboto3

    session = aioboto3.Session()
    return session.client(
        's3',
        endpoint_url='https://fsn1.your-objectstorage.com',
        aws_access_key_id='YOUR_ACCESS_KEY',
        aws_secret_access_key='YOUR_SECRET_KEY',
        region_name='fsn1',
    )
```

### Python Examples for Common Operations

```python
"""Examples of reading and writing to Hetzner Object Storage."""

import json
import gzip
from datetime import datetime, timezone
from io import BytesIO

import boto3
from botocore.config import Config


class AiaiStorage:
    """Wrapper for aiai's object storage operations."""

    def __init__(self, endpoint_url: str, access_key: str, secret_key: str):
        self.client = boto3.client(
            's3',
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name='fsn1',
            config=Config(signature_version='s3v4'),
        )
        self.resource = boto3.resource(
            's3',
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name='fsn1',
            config=Config(signature_version='s3v4'),
        )

    def ensure_buckets(self) -> None:
        """Create required buckets if they do not exist."""
        buckets = ['aiai-logs', 'aiai-metrics', 'aiai-artifacts', 'aiai-backups']
        existing = {b['Name'] for b in self.client.list_buckets().get('Buckets', [])}
        for bucket in buckets:
            if bucket not in existing:
                self.client.create_bucket(Bucket=bucket)

    # --- Log Storage ---

    def upload_log(self, service: str, content: str) -> str:
        """Upload a compressed log file with date-based partitioning."""
        now = datetime.now(timezone.utc)
        key = f"{service}/{now:%Y/%m/%d}/{now:%H%M%S}-{service}.log.gz"

        compressed = gzip.compress(content.encode('utf-8'))

        self.client.put_object(
            Bucket='aiai-logs',
            Key=key,
            Body=compressed,
            ContentType='application/gzip',
            ContentEncoding='gzip',
            Metadata={
                'service': service,
                'timestamp': now.isoformat(),
            },
        )
        return key

    def list_logs(self, service: str, date: str) -> list[str]:
        """List log files for a service on a given date (YYYY/MM/DD)."""
        response = self.client.list_objects_v2(
            Bucket='aiai-logs',
            Prefix=f"{service}/{date}/",
        )
        return [obj['Key'] for obj in response.get('Contents', [])]

    def read_log(self, key: str) -> str:
        """Read and decompress a log file."""
        response = self.client.get_object(Bucket='aiai-logs', Key=key)
        compressed = response['Body'].read()
        return gzip.decompress(compressed).decode('utf-8')

    # --- Metrics Archival ---

    def archive_metrics_snapshot(self, metrics_data: dict) -> str:
        """Archive a Prometheus metrics snapshot as compressed JSON."""
        now = datetime.now(timezone.utc)
        key = f"snapshots/{now:%Y/%m/%d}/{now:%H%M%S}-metrics.json.gz"

        json_bytes = json.dumps(metrics_data, default=str).encode('utf-8')
        compressed = gzip.compress(json_bytes)

        self.client.put_object(
            Bucket='aiai-metrics',
            Key=key,
            Body=compressed,
            ContentType='application/json',
            ContentEncoding='gzip',
        )
        return key

    # --- Build Artifacts ---

    def upload_artifact(self, build_id: str, filename: str, data: bytes) -> str:
        """Upload a build artifact."""
        key = f"builds/{build_id}/{filename}"
        self.client.put_object(
            Bucket='aiai-artifacts',
            Key=key,
            Body=data,
        )
        return key

    def download_artifact(self, build_id: str, filename: str) -> bytes:
        """Download a build artifact."""
        key = f"builds/{build_id}/{filename}"
        response = self.client.get_object(Bucket='aiai-artifacts', Key=key)
        return response['Body'].read()

    # --- Lifecycle Management ---

    def configure_lifecycle(self) -> None:
        """Set retention policies on buckets."""
        # Logs: keep 90 days
        self.client.put_bucket_lifecycle_configuration(
            Bucket='aiai-logs',
            LifecycleConfiguration={
                'Rules': [
                    {
                        'ID': 'expire-old-logs',
                        'Status': 'Enabled',
                        'Filter': {'Prefix': ''},
                        'Expiration': {'Days': 90},
                    },
                ],
            },
        )

        # Metrics: keep 365 days
        self.client.put_bucket_lifecycle_configuration(
            Bucket='aiai-metrics',
            LifecycleConfiguration={
                'Rules': [
                    {
                        'ID': 'expire-old-metrics',
                        'Status': 'Enabled',
                        'Filter': {'Prefix': ''},
                        'Expiration': {'Days': 365},
                    },
                ],
            },
        )

        # Artifacts: keep 30 days
        self.client.put_bucket_lifecycle_configuration(
            Bucket='aiai-artifacts',
            LifecycleConfiguration={
                'Rules': [
                    {
                        'ID': 'expire-old-artifacts',
                        'Status': 'Enabled',
                        'Filter': {'Prefix': ''},
                        'Expiration': {'Days': 30},
                    },
                ],
            },
        )
```

### Async Operations with aioboto3

```python
"""Async S3 operations for high-throughput scenarios."""

import aioboto3
import asyncio
from botocore.config import Config


async def upload_many_logs(logs: list[tuple[str, str]], endpoint: str,
                           access_key: str, secret_key: str) -> list[str]:
    """Upload multiple log files concurrently."""
    session = aioboto3.Session()
    keys = []

    async with session.client(
        's3',
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name='fsn1',
        config=Config(signature_version='s3v4'),
    ) as client:
        tasks = []
        for service, content in logs:
            key = f"{service}/{datetime.now(timezone.utc):%Y/%m/%d/%H%M%S}.log"
            tasks.append(
                client.put_object(
                    Bucket='aiai-logs',
                    Key=key,
                    Body=content.encode(),
                )
            )
            keys.append(key)

        await asyncio.gather(*tasks)

    return keys
```

---

## 7. Security Hardening

### Threat Model for Autonomous AI Agents

aiai runs autonomously -- no human is watching every action in real time. This changes the security posture significantly:

1. **API keys are the crown jewels** -- OpenRouter, GitHub, Hetzner tokens grant autonomous action
2. **The system modifies its own code** -- a compromised evolution engine could introduce backdoors
3. **CI runners execute arbitrary code** -- from commits the AI makes itself
4. **Network exposure must be minimal** -- only necessary ports, only necessary interfaces

### Hetzner Cloud Firewall (External)

Hetzner Cloud Firewalls are stateful and operate at the network edge, before traffic reaches the server. They are free to use.

```bash
# Create the firewall
hcloud firewall create --name aiai-firewall

# SSH access (restrict to known IPs if possible)
hcloud firewall add-rule aiai-firewall \
    --direction in \
    --protocol tcp \
    --port 22 \
    --source-ips "YOUR.ADMIN.IP/32" \
    --description "SSH from admin"

# Allow private network traffic (Hetzner firewalls apply to public IPs only,
# but this documents the intent)
# Private network traffic (10.0.0.0/16) is NOT filtered by Hetzner Cloud Firewalls

# Block everything else (implicit deny for inbound)
# Hetzner firewalls have implicit deny -- no rule = no access

# Apply firewall to servers
hcloud firewall apply-to-resource aiai-firewall \
    --type server \
    --server aiai-orchestrator-1

hcloud firewall apply-to-resource aiai-firewall \
    --type server \
    --server aiai-orchestrator-2

hcloud firewall apply-to-resource aiai-firewall \
    --type server \
    --server aiai-monitoring

hcloud firewall apply-to-resource aiai-firewall \
    --type server \
    --server aiai-ci-runner
```

**References:**
- [Hetzner Cloud Firewalls (Docs)](https://docs.hetzner.com/cloud/firewalls/)
- [Hetzner Firewall Overview](https://docs.hetzner.com/cloud/firewalls/overview/)

### Host-Level Firewall (UFW)

Defense in depth -- run UFW on every server in addition to the Hetzner Cloud Firewall:

```bash
#!/usr/bin/env bash
# scripts/harden-server.sh -- Run on each server after provisioning
set -euo pipefail

echo "=== Hardening server: $(hostname) ==="

# --- UFW Firewall ---
ufw default deny incoming
ufw default allow outgoing

# SSH (rate limited)
ufw limit 22/tcp comment 'SSH rate-limited'

# Allow all private network traffic
ufw allow from 10.0.0.0/16 comment 'Private network'

# Enable
ufw --force enable

# --- SSH Hardening ---
cat > /etc/ssh/sshd_config.d/99-hardening.conf << 'EOF'
PermitRootLogin no
PasswordAuthentication no
PubkeyAuthentication yes
AuthenticationMethods publickey
X11Forwarding no
AllowTcpForwarding no
MaxAuthTries 3
LoginGraceTime 30
ClientAliveInterval 300
ClientAliveCountMax 2
AllowUsers aiai
EOF
systemctl restart sshd

# --- Fail2ban ---
apt-get install -y fail2ban
cat > /etc/fail2ban/jail.d/defaults.conf << 'EOF'
[DEFAULT]
bantime = 3600
findtime = 600
maxretry = 3

[sshd]
enabled = true
port = ssh
filter = sshd
logpath = /var/log/auth.log
EOF
systemctl enable --now fail2ban

# --- Disable unused services ---
systemctl disable --now snapd.service 2>/dev/null || true
systemctl disable --now snapd.socket 2>/dev/null || true

# --- Kernel hardening ---
cat > /etc/sysctl.d/99-aiai-hardening.conf << 'EOF'
# Disable IP forwarding
net.ipv4.ip_forward = 0

# Enable SYN flood protection
net.ipv4.tcp_syncookies = 1

# Ignore ICMP redirects
net.ipv4.conf.all.accept_redirects = 0
net.ipv4.conf.default.accept_redirects = 0

# Ignore source-routed packets
net.ipv4.conf.all.accept_source_route = 0
net.ipv4.conf.default.accept_source_route = 0

# Log suspicious packets
net.ipv4.conf.all.log_martians = 1
net.ipv4.conf.default.log_martians = 1

# Disable ICMP redirect sending
net.ipv4.conf.all.send_redirects = 0
net.ipv4.conf.default.send_redirects = 0

# Enable ASLR
kernel.randomize_va_space = 2

# Restrict core dumps
fs.suid_dumpable = 0
EOF
sysctl --system

echo "=== Hardening complete ==="
```

### Secret Management

Secrets must never appear in git history, cloud-init configs checked into the repo, or environment variables visible in `/proc`:

```bash
#!/usr/bin/env bash
# scripts/inject-secrets.sh -- Securely inject secrets to a server
# Run from a trusted admin workstation
set -euo pipefail

SERVER_IP="${1:?Usage: inject-secrets.sh <server-ip>}"

# Read secrets from local password manager / vault
OPENROUTER_KEY="$(op read 'op://aiai/openrouter/api-key')"  # 1Password example
GITHUB_TOKEN="$(op read 'op://aiai/github/token')"
HETZNER_TOKEN="$(op read 'op://aiai/hetzner/api-token')"

# Write secrets to server via SSH (never stored on disk locally)
ssh aiai@"${SERVER_IP}" "sudo tee /etc/aiai/env > /dev/null" << EOF
OPENROUTER_API_KEY=${OPENROUTER_KEY}
GITHUB_TOKEN=${GITHUB_TOKEN}
HETZNER_API_TOKEN=${HETZNER_TOKEN}
AIAI_ENVIRONMENT=production
AIAI_LOG_LEVEL=INFO
AIAI_METRICS_PORT=9091
EOF

# Set restrictive permissions
ssh aiai@"${SERVER_IP}" "sudo chmod 600 /etc/aiai/env && sudo chown root:aiai /etc/aiai/env"

# Restart services to pick up new secrets
ssh aiai@"${SERVER_IP}" "sudo systemctl restart aiai-orchestrator"

echo "Secrets injected and services restarted on ${SERVER_IP}"
```

### API Key Rotation

```python
"""API key rotation utility for aiai services."""

import subprocess
import json
from datetime import datetime, timezone


def rotate_hetzner_token(old_token: str) -> str:
    """Rotate the Hetzner API token.

    Creates a new token, updates servers, then deletes the old one.
    """
    # This must be done through the Hetzner Cloud Console
    # as the API does not support token self-rotation.
    # Document the process:
    #
    # 1. Log into Hetzner Cloud Console
    # 2. Go to Security -> API Tokens
    # 3. Generate new token with same permissions
    # 4. Run inject-secrets.sh with new token
    # 5. Verify all servers operational
    # 6. Delete old token from Console
    raise NotImplementedError("Hetzner tokens must be rotated via Console")


def rotate_github_token() -> str:
    """Rotate GitHub fine-grained personal access token.

    Fine-grained tokens can be created via API with limited scopes.
    """
    # GitHub fine-grained PATs must also be rotated through
    # the GitHub web interface or GitHub App installation tokens.
    # For automated rotation, use a GitHub App:
    #
    # 1. Create a GitHub App with required permissions
    # 2. Generate installation access tokens (1-hour expiry)
    # 3. Refresh automatically before expiry
    raise NotImplementedError("Use GitHub App installation tokens for auto-rotation")


def audit_secret_usage(server_ip: str) -> dict:
    """Audit when secrets were last rotated and accessed."""
    result = subprocess.run(
        ['ssh', f'aiai@{server_ip}',
         'sudo stat -c "%Y" /etc/aiai/env'],
        capture_output=True, text=True, check=True,
    )
    last_modified = datetime.fromtimestamp(
        int(result.stdout.strip()), tz=timezone.utc
    )
    return {
        'server': server_ip,
        'env_file_last_modified': last_modified.isoformat(),
        'age_days': (datetime.now(timezone.utc) - last_modified).days,
    }
```

### Network Isolation Principles

```
┌─────────────────────────────────────────────────┐
│               Public Internet                    │
│                                                  │
│  Only SSH (port 22) from admin IPs              │
│  Load Balancer public IP (ports 80/443)         │
└──────────┬──────────────────────┬───────────────┘
           │                      │
    ┌──────┴──────┐        ┌─────┴──────┐
    │ Hetzner     │        │ Hetzner    │
    │ Cloud       │        │ Load       │
    │ Firewall    │        │ Balancer   │
    └──────┬──────┘        └─────┬──────┘
           │                      │
    ┌──────┴──────────────────────┴───────────────┐
    │           Private Network 10.0.0.0/16        │
    │                                              │
    │  All inter-service communication here        │
    │  No public exposure for:                     │
    │  - Prometheus (9090)                         │
    │  - Grafana (3000)                            │
    │  - Loki (3100)                               │
    │  - Application metrics (9091)                │
    │  - Node exporter (9100)                      │
    │  - Orchestrator API (8000)                   │
    └──────────────────────────────────────────────┘
```

---

## 8. Automated Deployment

### Deployment Script

A complete deployment script that provisions and updates aiai on Hetzner:

```bash
#!/usr/bin/env bash
# scripts/deploy.sh -- Deploy aiai to Hetzner Cloud
# Usage: ./deploy.sh [create|update|rollback|status|destroy]
set -euo pipefail

# --- Configuration ---
PROJECT="aiai"
LOCATION="fsn1"
NETWORK="aiai-net"
SSH_KEY="aiai-deploy"
IMAGE="ubuntu-24.04"
GIT_REPO="https://github.com/your-org/aiai.git"

# Server definitions
declare -A SERVERS=(
    ["orchestrator-1"]="cx33"
    ["orchestrator-2"]="cx23"
    ["monitoring"]="cax21"
    ["ci-runner"]="cpx31"
)

declare -A SERVER_IPS=(
    ["orchestrator-1"]="10.0.1.10"
    ["orchestrator-2"]="10.0.1.11"
    ["monitoring"]="10.0.1.20"
    ["ci-runner"]="10.0.1.30"
)

# --- Functions ---

create_network() {
    echo "Creating private network..."
    hcloud network create --name "${NETWORK}" --ip-range 10.0.0.0/16 || true
    hcloud network add-subnet "${NETWORK}" \
        --type cloud \
        --network-zone eu-central \
        --ip-range 10.0.1.0/24 || true
}

create_firewall() {
    echo "Creating firewall..."
    hcloud firewall create --name "${PROJECT}-firewall" || true

    # SSH from anywhere (restrict in production)
    hcloud firewall add-rule "${PROJECT}-firewall" \
        --direction in \
        --protocol tcp \
        --port 22 \
        --source-ips "0.0.0.0/0" \
        --description "SSH" || true
}

create_server() {
    local name="${PROJECT}-${1}"
    local type="${2}"
    local ip="${3}"
    local cloud_init="${4:-}"

    echo "Creating server: ${name} (${type})..."

    local cmd=(
        hcloud server create
        --name "${name}"
        --type "${type}"
        --image "${IMAGE}"
        --location "${LOCATION}"
        --ssh-key "${SSH_KEY}"
        --network "${NETWORK}"
        --label "project=${PROJECT}"
        --label "role=${1}"
    )

    if [[ -n "${cloud_init}" && -f "${cloud_init}" ]]; then
        cmd+=(--user-data-from-file "${cloud_init}")
    fi

    "${cmd[@]}"

    # Attach to firewall
    hcloud firewall apply-to-resource "${PROJECT}-firewall" \
        --type server \
        --server "${name}" || true
}

create_all() {
    create_network
    create_firewall

    for role in "${!SERVERS[@]}"; do
        local cloud_init_file="cloud-init/${role}.yaml"
        create_server "${role}" "${SERVERS[$role]}" "${SERVER_IPS[$role]}" "${cloud_init_file}"
    done

    echo ""
    echo "=== Servers Created ==="
    hcloud server list --selector "project=${PROJECT}" -o columns=name,status,ipv4,server_type
}

update_service() {
    local server_name="${PROJECT}-${1}"
    local server_ip

    server_ip=$(hcloud server ip "${server_name}")

    echo "Updating ${server_name} (${server_ip})..."

    ssh "aiai@${server_ip}" << 'REMOTE_SCRIPT'
        set -euo pipefail
        cd /opt/aiai

        # Store current commit for rollback
        git rev-parse HEAD > /tmp/aiai-rollback-sha

        # Pull latest code
        git fetch origin main
        git reset --hard origin/main

        # Update dependencies
        .venv/bin/pip install -r requirements.txt --quiet

        # Restart service
        sudo systemctl restart aiai-orchestrator

        # Wait for health check
        sleep 5
        if systemctl is-active --quiet aiai-orchestrator; then
            echo "Service healthy after update"
        else
            echo "ERROR: Service failed to start, rolling back..."
            git reset --hard "$(cat /tmp/aiai-rollback-sha)"
            .venv/bin/pip install -r requirements.txt --quiet
            sudo systemctl restart aiai-orchestrator
            exit 1
        fi
REMOTE_SCRIPT
}

rolling_update() {
    echo "=== Rolling Update ==="

    # Update orchestrator-2 first (standby)
    update_service "orchestrator-2"

    # Then orchestrator-1 (primary)
    update_service "orchestrator-1"

    # Update monitoring
    update_service "monitoring"

    echo "=== Rolling Update Complete ==="
}

health_check() {
    echo "=== Health Check ==="
    for role in "${!SERVERS[@]}"; do
        local name="${PROJECT}-${role}"
        local ip
        ip=$(hcloud server ip "${name}" 2>/dev/null) || { echo "${name}: NOT FOUND"; continue; }

        local status
        status=$(ssh -o ConnectTimeout=5 "aiai@${ip}" \
            "systemctl is-active aiai-orchestrator 2>/dev/null || echo 'no-service'" 2>/dev/null) \
            || status="UNREACHABLE"

        echo "${name} (${ip}): ${status}"
    done
}

rollback() {
    local server_name="${PROJECT}-${1}"
    local server_ip
    server_ip=$(hcloud server ip "${server_name}")

    echo "Rolling back ${server_name}..."

    ssh "aiai@${server_ip}" << 'REMOTE_SCRIPT'
        set -euo pipefail
        cd /opt/aiai

        if [[ -f /tmp/aiai-rollback-sha ]]; then
            git reset --hard "$(cat /tmp/aiai-rollback-sha)"
            .venv/bin/pip install -r requirements.txt --quiet
            sudo systemctl restart aiai-orchestrator
            echo "Rolled back to $(cat /tmp/aiai-rollback-sha)"
        else
            echo "No rollback SHA found"
            exit 1
        fi
REMOTE_SCRIPT
}

destroy_all() {
    echo "WARNING: This will destroy all aiai servers!"
    read -rp "Type 'destroy' to confirm: " confirm
    if [[ "${confirm}" != "destroy" ]]; then
        echo "Aborted."
        exit 1
    fi

    for role in "${!SERVERS[@]}"; do
        hcloud server delete "${PROJECT}-${role}" || true
    done

    hcloud load-balancer delete "${PROJECT}-lb" || true
    hcloud network delete "${NETWORK}" || true
    hcloud firewall delete "${PROJECT}-firewall" || true

    echo "All resources destroyed."
}

status() {
    echo "=== aiai Infrastructure Status ==="
    echo ""
    echo "--- Servers ---"
    hcloud server list --selector "project=${PROJECT}" \
        -o columns=name,status,ipv4,server_type,datacenter
    echo ""
    echo "--- Networks ---"
    hcloud network list
    echo ""
    echo "--- Firewalls ---"
    hcloud firewall list
    echo ""
    echo "--- Load Balancers ---"
    hcloud load-balancer list
    echo ""
    health_check
}

# --- Main ---
case "${1:-status}" in
    create)   create_all ;;
    update)   rolling_update ;;
    rollback) rollback "${2:?Usage: deploy.sh rollback <role>}" ;;
    status)   status ;;
    destroy)  destroy_all ;;
    health)   health_check ;;
    *)
        echo "Usage: deploy.sh [create|update|rollback|status|destroy|health]"
        exit 1
        ;;
esac
```

### Blue-Green Deployment on a Budget

True blue-green deployment requires double the infrastructure. On a budget, use a lightweight version with the load balancer:

```bash
#!/usr/bin/env bash
# scripts/blue-green-deploy.sh -- Budget blue-green deployment
set -euo pipefail

# Current active server is "blue", new deployment goes to "green"
BLUE_SERVER="aiai-orchestrator-1"
GREEN_SERVER="aiai-orchestrator-2"
LB_NAME="aiai-lb"

echo "=== Blue-Green Deployment ==="
echo "Blue (current active): ${BLUE_SERVER}"
echo "Green (deploy target): ${GREEN_SERVER}"

# Step 1: Deploy to green
GREEN_IP=$(hcloud server ip "${GREEN_SERVER}")
echo "Deploying to green (${GREEN_IP})..."

ssh "aiai@${GREEN_IP}" << 'REMOTE_SCRIPT'
    set -euo pipefail
    cd /opt/aiai
    git fetch origin main
    git reset --hard origin/main
    .venv/bin/pip install -r requirements.txt --quiet
    sudo systemctl restart aiai-orchestrator
    sleep 5

    # Run smoke tests
    curl -sf http://localhost:8000/health || { echo "Health check failed"; exit 1; }
REMOTE_SCRIPT

echo "Green deployment successful, switching traffic..."

# Step 2: Remove blue from load balancer, add green
hcloud load-balancer remove-target "${LB_NAME}" --server "${BLUE_SERVER}" || true
hcloud load-balancer add-target "${LB_NAME}" --server "${GREEN_SERVER}" --use-private-ip

echo "Traffic switched to green. Monitoring for 60 seconds..."
sleep 60

# Step 3: Verify green is healthy under load
GREEN_STATUS=$(ssh "aiai@${GREEN_IP}" "systemctl is-active aiai-orchestrator")
if [[ "${GREEN_STATUS}" == "active" ]]; then
    echo "Green is healthy. Updating blue to match..."

    # Step 4: Update blue too (for next deployment cycle)
    BLUE_IP=$(hcloud server ip "${BLUE_SERVER}")
    ssh "aiai@${BLUE_IP}" << 'REMOTE_SCRIPT'
        set -euo pipefail
        cd /opt/aiai
        git fetch origin main
        git reset --hard origin/main
        .venv/bin/pip install -r requirements.txt --quiet
        sudo systemctl restart aiai-orchestrator
REMOTE_SCRIPT

    # Add blue back to load balancer
    hcloud load-balancer add-target "${LB_NAME}" --server "${BLUE_SERVER}" --use-private-ip

    echo "=== Blue-Green Deployment Complete ==="
    echo "Both servers running latest code, traffic balanced."
else
    echo "ERROR: Green is unhealthy. Rolling back to blue..."
    hcloud load-balancer remove-target "${LB_NAME}" --server "${GREEN_SERVER}" || true
    hcloud load-balancer add-target "${LB_NAME}" --server "${BLUE_SERVER}" --use-private-ip
    echo "Rolled back to blue."
    exit 1
fi
```

---

## 9. Cost Optimization on Hetzner

### Principles

1. **Right-size aggressively** -- start with the smallest server that works, scale up only when metrics justify it
2. **Use ARM (CAX) for steady workloads** -- monitoring, log aggregation, long-running services
3. **Use on-demand for bursty workloads** -- CI runners, one-off analysis tasks
4. **Keep everything in EU** -- 20 TB traffic included vs 1 TB in US
5. **Use private networking** -- free, unmetered, reduces public traffic usage
6. **Compress everything to object storage** -- gzip logs, compact metrics

### Monthly Cost Projections

#### Minimal Configuration (Development/Early Stage)

| Resource | Type | Monthly Cost |
|----------|------|-------------|
| Orchestrator | CX23 (2 vCPU, 4 GB) | ~3.49 |
| Monitoring | CX23 (2 vCPU, 4 GB) | ~3.49 |
| Object Storage | Base plan (1 TB) | ~5.99 |
| **Total** | | **~12.97** |

A full autonomous AI development system for under 15/month.

#### Standard Configuration (Production)

| Resource | Type | Monthly Cost |
|----------|------|-------------|
| Orchestrator primary | CX33 (4 vCPU, 8 GB) | ~5.49 |
| Orchestrator standby | CX23 (2 vCPU, 4 GB) | ~3.49 |
| Monitoring | CAX21 (4 ARM vCPU, 8 GB) | ~6.49 |
| CI Runner | CPX31 (4 vCPU, 8 GB) | ~16.49 |
| Load Balancer | LB11 | ~5.39 |
| Object Storage | Base plan (1 TB) | ~5.99 |
| Snapshots | ~50 GB | ~0.55 |
| **Total** | | **~43.89** |

#### Scaled Configuration (High Activity)

| Resource | Type | Monthly Cost |
|----------|------|-------------|
| Orchestrator primary | CCX23 (4 ded. vCPU, 16 GB) | ~24.49 |
| Orchestrator secondary | CX33 (4 vCPU, 8 GB) | ~5.49 |
| Monitoring | CAX31 (8 ARM vCPU, 16 GB) | ~12.49 |
| CI Runner x2 | CPX31 (4 vCPU, 8 GB) each | ~32.98 |
| Load Balancer | LB11 | ~5.39 |
| Object Storage | Base plan (1 TB) | ~5.99 |
| Volumes | 100 GB (Prometheus data) | ~4.40 |
| Snapshots | ~200 GB | ~2.20 |
| Backups | 20% of server costs | ~16.17 |
| **Total** | | **~109.60** |

### ARM (CAX) vs x86 Decision Matrix

| Workload | Use CAX (ARM)? | Rationale |
|----------|---------------|-----------|
| Monitoring (Prometheus/Grafana) | Yes | Steady workload, excellent ARM support |
| Loki log aggregation | Yes | I/O bound, ARM efficient |
| Python orchestrator | Maybe | Test first; most pure-Python works well on ARM |
| CI Runner | No | Too many x86-only dependencies in test toolchains |
| NumPy/SciPy workloads | No | Floating-point performance worse on ARM |
| Build tools (Docker) | Yes | Multi-arch images widely available |

**References:**
- [Hetzner CAX ARM64 Performance Review (WebP Cloud)](https://blog.webp.se/hetzner-arm64-en/)
- [Hetzner ARM64 Cloud Announcement](https://www.hetzner.com/news/arm64-cloud/)

### Scheduled Scaling

Spin down CI runners during off-hours to save costs:

```bash
#!/usr/bin/env bash
# scripts/scale-ci.sh -- Scale CI runners based on schedule
set -euo pipefail

ACTION="${1:?Usage: scale-ci.sh [up|down|status]}"
RUNNER_NAME="aiai-ci-runner"

case "${ACTION}" in
    up)
        echo "Starting CI runner..."
        hcloud server poweron "${RUNNER_NAME}"
        echo "Waiting for server to be ready..."
        sleep 30
        # Re-register the runner if ephemeral
        SERVER_IP=$(hcloud server ip "${RUNNER_NAME}")
        ssh "aiai@${SERVER_IP}" "sudo systemctl start github-runner"
        echo "CI runner is up."
        ;;
    down)
        echo "Stopping CI runner..."
        SERVER_IP=$(hcloud server ip "${RUNNER_NAME}")
        # Gracefully stop the runner (finish current job)
        ssh "aiai@${SERVER_IP}" "sudo systemctl stop github-runner" || true
        sleep 10
        hcloud server poweroff "${RUNNER_NAME}"
        echo "CI runner is powered off. (Note: you are still billed for the server)"
        ;;
    status)
        hcloud server describe "${RUNNER_NAME}" -o format='{{.Status}}'
        ;;
esac
```

Note: Hetzner bills for powered-off servers. To truly stop billing, you must delete the server and recreate it. Use snapshots to preserve state:

```bash
# Save state before deletion
hcloud server create-image --type snapshot --description "ci-runner-$(date +%Y%m%d)" aiai-ci-runner

# Delete server (stops billing)
hcloud server delete aiai-ci-runner

# Recreate from snapshot when needed
SNAPSHOT_ID=$(hcloud image list --type snapshot --selector "role=ci-runner" -o noheader -o columns=id | tail -1)
hcloud server create \
    --name aiai-ci-runner \
    --type cpx31 \
    --image "${SNAPSHOT_ID}" \
    --location fsn1 \
    --ssh-key aiai-deploy \
    --network aiai-net
```

### Volume vs Local Storage

| Feature | Local SSD | Volume (Block Storage) |
|---------|-----------|----------------------|
| Price | Included with server | 0.044/GB/month |
| Performance | Higher IOPS | Good, but network-attached |
| Persistence | Destroyed with server | Survives server deletion |
| Portability | None | Attach to any server in same DC |
| Best for | OS, application code | Prometheus data, databases, persistent state |

For aiai, use volumes for Prometheus data retention (survives server replacements):

```bash
# Create a volume for Prometheus data
hcloud volume create \
    --name aiai-prometheus-data \
    --size 50 \
    --server aiai-monitoring \
    --format ext4 \
    --automount

# Mount point will be /mnt/HC_Volume_XXXXX
# Configure Prometheus to use this path for storage
```

---

## 10. Disaster Recovery

### Recovery Strategy Overview

aiai is designed for rapid reconstruction. The key insight: **Git is the primary backup**. All code, configuration, agent definitions, and documentation live in the git repository. Server state is disposable. The recovery strategy optimizes for "rebuild from scratch in minutes" rather than "restore exact state."

### Recovery Priority Matrix

| Component | Recovery Method | Recovery Time | Data Loss Window |
|-----------|----------------|---------------|------------------|
| Application code | Git clone | < 1 minute | Zero (git is the source) |
| System config | Cloud-init replay | 5-10 minutes | Zero (config in repo) |
| Secrets | Re-inject from vault | 2 minutes | Zero (in password manager) |
| Prometheus metrics | Volume snapshot restore | 5 minutes | Up to 1 hour |
| Application logs | Object storage | 0 (already external) | Up to flush interval |
| Server OS state | Snapshot restore | 3-5 minutes | Up to snapshot interval |

### Hetzner Snapshots

```bash
#!/usr/bin/env bash
# scripts/backup.sh -- Create snapshots of all aiai servers
set -euo pipefail

TIMESTAMP=$(date +%Y%m%d-%H%M)
PROJECT="aiai"

echo "=== Creating snapshots at ${TIMESTAMP} ==="

# Get all aiai servers
SERVERS=$(hcloud server list --selector "project=${PROJECT}" -o noheader -o columns=name)

for server in ${SERVERS}; do
    echo "Snapshotting ${server}..."
    hcloud server create-image \
        --type snapshot \
        --description "${server}-${TIMESTAMP}" \
        --label "project=${PROJECT}" \
        --label "source=${server}" \
        --label "timestamp=${TIMESTAMP}" \
        "${server}"
done

# Clean up old snapshots (keep last 7)
echo "Cleaning old snapshots..."
for server in ${SERVERS}; do
    SNAPSHOTS=$(hcloud image list \
        --type snapshot \
        --selector "source=${server}" \
        --sort created:desc \
        -o noheader \
        -o columns=id \
        | tail -n +8)  # Skip the 7 most recent

    for snap_id in ${SNAPSHOTS}; do
        echo "Deleting old snapshot ${snap_id}"
        hcloud image delete "${snap_id}"
    done
done

echo "=== Snapshot backup complete ==="
```

Schedule this with a systemd timer or cron on the monitoring server:

```ini
# /etc/systemd/system/aiai-backup.service
[Unit]
Description=aiai Snapshot Backup

[Service]
Type=oneshot
ExecStart=/opt/aiai/scripts/backup.sh
User=aiai
EnvironmentFile=/etc/aiai/env
```

```ini
# /etc/systemd/system/aiai-backup.timer
[Unit]
Description=Daily aiai snapshot backup

[Timer]
OnCalendar=*-*-* 03:00:00
RandomizedDelaySec=1800
Persistent=true

[Install]
WantedBy=timers.target
```

### Off-Site Backups to Object Storage

For critical data that must survive even if the entire Hetzner account is compromised:

```python
"""Off-site backup of critical aiai data to object storage."""

import subprocess
import tarfile
import io
from datetime import datetime, timezone
from pathlib import Path


class OffSiteBackup:
    """Backup critical data to Hetzner Object Storage."""

    def __init__(self, storage_client):
        self.storage = storage_client
        self.bucket = 'aiai-backups'

    def backup_prometheus_data(self, data_dir: str = '/var/lib/prometheus') -> str:
        """Create a snapshot of Prometheus data and upload to S3."""
        # First, create a Prometheus snapshot via API
        subprocess.run(
            ['curl', '-s', '-XPOST',
             'http://localhost:9090/api/v1/admin/tsdb/snapshot'],
            check=True,
        )

        # Tar and compress the snapshot directory
        now = datetime.now(timezone.utc)
        key = f"prometheus/{now:%Y/%m/%d}/{now:%H%M%S}-prometheus-snapshot.tar.gz"

        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode='w:gz') as tar:
            snapshot_dir = Path(data_dir) / 'snapshots'
            if snapshot_dir.exists():
                for snapshot in snapshot_dir.iterdir():
                    tar.add(str(snapshot), arcname=snapshot.name)

        buf.seek(0)
        self.storage.client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=buf.getvalue(),
        )

        # Clean up local snapshot
        subprocess.run(
            ['rm', '-rf', str(Path(data_dir) / 'snapshots')],
            check=True,
        )

        return key

    def backup_git_bundle(self, repo_dir: str = '/opt/aiai') -> str:
        """Create a git bundle (complete repo backup) and upload."""
        now = datetime.now(timezone.utc)
        key = f"git-bundles/{now:%Y/%m/%d}/{now:%H%M%S}-aiai.bundle"

        bundle_path = '/tmp/aiai.bundle'
        subprocess.run(
            ['git', '-C', repo_dir, 'bundle', 'create',
             bundle_path, '--all'],
            check=True,
        )

        with open(bundle_path, 'rb') as f:
            self.storage.client.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=f.read(),
            )

        Path(bundle_path).unlink()
        return key

    def backup_configs(self) -> str:
        """Backup system configuration files (not secrets)."""
        now = datetime.now(timezone.utc)
        key = f"configs/{now:%Y/%m/%d}/{now:%H%M%S}-configs.tar.gz"

        buf = io.BytesIO()
        config_paths = [
            '/etc/prometheus',
            '/etc/grafana',
            '/etc/loki',
            '/etc/systemd/system/aiai-*.service',
            '/etc/systemd/system/aiai-*.timer',
        ]

        with tarfile.open(fileobj=buf, mode='w:gz') as tar:
            for path in config_paths:
                p = Path(path)
                if p.exists():
                    tar.add(str(p), arcname=str(p).lstrip('/'))

        buf.seek(0)
        self.storage.client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=buf.getvalue(),
        )
        return key
```

### Full Rebuild from Scratch

The ultimate disaster recovery test: can you rebuild the entire infrastructure from nothing?

```bash
#!/usr/bin/env bash
# scripts/rebuild-from-scratch.sh
# Rebuilds the entire aiai infrastructure from zero
# Requires: hcloud CLI configured, SSH key registered, secrets in password manager
set -euo pipefail

echo "=== Rebuilding aiai Infrastructure from Scratch ==="
echo "This will create all servers, networks, and services."
echo ""

TIMESTAMP_START=$(date +%s)

# Step 1: Create network infrastructure
echo "[1/6] Creating network infrastructure..."
./scripts/deploy.sh create

# Step 2: Wait for cloud-init to complete on all servers
echo "[2/6] Waiting for cloud-init provisioning (this takes 3-5 minutes)..."
for server in aiai-orchestrator-1 aiai-orchestrator-2 aiai-monitoring aiai-ci-runner; do
    IP=$(hcloud server ip "${server}")
    echo "  Waiting for ${server} (${IP})..."

    # Wait for SSH to become available
    for i in $(seq 1 60); do
        if ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no "aiai@${IP}" "echo ok" 2>/dev/null; then
            break
        fi
        sleep 5
    done

    # Wait for cloud-init to finish
    ssh "aiai@${IP}" "cloud-init status --wait" || true
done

# Step 3: Inject secrets
echo "[3/6] Injecting secrets..."
for server in aiai-orchestrator-1 aiai-orchestrator-2 aiai-ci-runner; do
    IP=$(hcloud server ip "${server}")
    ./scripts/inject-secrets.sh "${IP}"
done

# Step 4: Create load balancer
echo "[4/6] Creating load balancer..."
hcloud load-balancer create --name aiai-lb --type lb11 --location fsn1
hcloud load-balancer attach-to-network aiai-lb --network aiai-net --ip 10.0.1.5
hcloud load-balancer add-target aiai-lb --server aiai-orchestrator-1 --use-private-ip
hcloud load-balancer add-target aiai-lb --server aiai-orchestrator-2 --use-private-ip
hcloud load-balancer add-service aiai-lb \
    --protocol http --listen-port 80 --destination-port 8000 \
    --health-check-protocol http --health-check-port 8000 --health-check-path /health

# Step 5: Configure object storage
echo "[5/6] Setting up object storage buckets..."
# Object storage setup via Python
ssh "aiai@$(hcloud server ip aiai-orchestrator-1)" << 'REMOTE'
    cd /opt/aiai
    .venv/bin/python -c "
from src.storage import AiaiStorage
import os
s = AiaiStorage(
    endpoint_url=os.environ['S3_ENDPOINT'],
    access_key=os.environ['S3_ACCESS_KEY'],
    secret_key=os.environ['S3_SECRET_KEY'],
)
s.ensure_buckets()
s.configure_lifecycle()
print('Object storage configured.')
"
REMOTE

# Step 6: Verify everything
echo "[6/6] Running health checks..."
./scripts/deploy.sh health

TIMESTAMP_END=$(date +%s)
ELAPSED=$((TIMESTAMP_END - TIMESTAMP_START))

echo ""
echo "=== Rebuild Complete ==="
echo "Total time: ${ELAPSED} seconds ($((ELAPSED / 60)) minutes)"
echo ""
echo "Infrastructure:"
hcloud server list --selector "project=aiai" -o columns=name,status,ipv4,server_type
echo ""
echo "Load Balancer:"
hcloud load-balancer list
```

### What Git Preserves (No Backup Needed)

The following are always recoverable from a `git clone`:

- All Python source code (`src/`)
- All tests (`tests/`)
- All scripts (`scripts/`)
- All documentation (`docs/`)
- Agent definitions (`.claude/`)
- Model routing configuration (`config/`)
- CI/CD workflows (`.github/`)
- Cloud-init templates (`cloud-init/`)
- Systemd service templates
- Project conventions (`CLAUDE.md`)

This is why aiai's "everything in git" philosophy is also a disaster recovery strategy. The git repository is the single source of truth. Servers are disposable compute that get reconstructed from the repo.

### What Requires External Backup

| Data | Backup Method | Location |
|------|---------------|----------|
| API keys / secrets | Password manager (1Password, Vault) | External service |
| Prometheus metrics | Volume snapshots + S3 archive | Hetzner Volume + Object Storage |
| Application logs | Shipped to Loki, archived to S3 | Object Storage |
| Grafana dashboards | Export as JSON, store in git | Git repository |
| Hetzner SSH keys | Password manager | External service |
| DNS records | Documented in git, managed via API | Git + DNS provider |

### Recovery Runbook

When disaster strikes, follow this sequence:

1. **Assess damage** -- Which servers are affected? Is it a single server, a datacenter issue, or account-level?
2. **If single server**: Rebuild from snapshot or cloud-init. Takes 5-10 minutes.
3. **If datacenter issue**: Rebuild in a different location (Falkenstein, Nuremberg, or Helsinki). Change location in deploy scripts. Takes 10-15 minutes.
4. **If account-level**: Create new Hetzner account, register SSH keys, run rebuild script. Takes 15-20 minutes.
5. **Restore secrets** from password manager using `inject-secrets.sh`.
6. **Restore metrics data** from object storage backup if needed.
7. **Verify**: Run health checks, confirm CI pipeline works, confirm monitoring is scraping.
8. **Post-mortem**: Document what happened, update runbook if needed.

The target is: **full infrastructure rebuild in under 20 minutes, with zero code loss.**

---

## Appendix A: Quick Reference -- hcloud CLI Commands

```bash
# --- Server Management ---
hcloud server list                                    # List all servers
hcloud server create --name X --type cx33 --image ubuntu-24.04 --location fsn1
hcloud server delete my-server                        # Delete server
hcloud server poweroff my-server                      # Power off (still billed)
hcloud server poweron my-server                       # Power on
hcloud server rebuild my-server --image ubuntu-24.04  # Wipe and rebuild
hcloud server ip my-server                            # Get public IP
hcloud server ssh my-server                           # SSH into server
hcloud server describe my-server                      # Full server details

# --- Snapshots ---
hcloud server create-image --type snapshot --description "desc" my-server
hcloud image list --type snapshot                     # List snapshots
hcloud image delete IMAGE_ID                          # Delete snapshot

# --- Networks ---
hcloud network create --name net --ip-range 10.0.0.0/16
hcloud network add-subnet net --type cloud --network-zone eu-central --ip-range 10.0.1.0/24
hcloud server attach-to-network my-server --network net --ip 10.0.1.10

# --- Firewalls ---
hcloud firewall create --name fw
hcloud firewall add-rule fw --direction in --protocol tcp --port 22 --source-ips 0.0.0.0/0
hcloud firewall apply-to-resource fw --type server --server my-server

# --- Load Balancers ---
hcloud load-balancer create --name lb --type lb11 --location fsn1
hcloud load-balancer add-target lb --server my-server --use-private-ip
hcloud load-balancer add-service lb --protocol http --listen-port 80 --destination-port 8000

# --- Volumes ---
hcloud volume create --name vol --size 50 --server my-server --format ext4 --automount
hcloud volume list
hcloud volume delete vol

# --- SSH Keys ---
hcloud ssh-key create --name my-key --public-key-from-file ~/.ssh/id_ed25519.pub
hcloud ssh-key list
```

## Appendix B: Environment Variables Reference

```bash
# /etc/aiai/env -- Complete environment variable reference
# This file is deployed via inject-secrets.sh, never committed to git

# --- API Keys ---
OPENROUTER_API_KEY=sk-or-...           # OpenRouter API key for model access
GITHUB_TOKEN=ghp_...                    # GitHub PAT for repo operations
HETZNER_API_TOKEN=...                   # Hetzner Cloud API token

# --- Object Storage ---
S3_ENDPOINT=https://fsn1.your-objectstorage.com
S3_ACCESS_KEY=...
S3_SECRET_KEY=...

# --- Application Config ---
AIAI_ENVIRONMENT=production             # production | staging | development
AIAI_LOG_LEVEL=INFO                     # DEBUG | INFO | WARNING | ERROR
AIAI_METRICS_PORT=9091                  # Prometheus metrics endpoint port
AIAI_ORCHESTRATOR_PORT=8000             # Orchestrator API port

# --- Monitoring ---
GRAFANA_ADMIN_PASSWORD=...              # Grafana admin password
ALERTMANAGER_WEBHOOK_URL=...            # Webhook for alert notifications
```

## Appendix C: Estimated Total Monthly Costs by Scenario

```
Scenario 1: Bootstrapping (1 server + storage)
  CX23 orchestrator .............. 3.49
  Object Storage ................. 5.99
  ─────────────────────────────────────
  Total .......................... 9.48/month

Scenario 2: Standard Production
  CX33 orchestrator .............. 5.49
  CX23 standby ................... 3.49
  CAX21 monitoring ............... 6.49
  CPX31 CI runner ................ 16.49
  LB11 load balancer ............. 5.39
  Object Storage ................. 5.99
  50 GB snapshots ................ 0.55
  ─────────────────────────────────────
  Total .......................... 43.89/month

Scenario 3: Full Production with Redundancy
  CCX23 orchestrator ............. 24.49
  CX33 secondary orchestrator .... 5.49
  CAX31 monitoring ............... 12.49
  CPX31 CI runner x2 ............. 32.98
  LB11 load balancer ............. 5.39
  Object Storage ................. 5.99
  100 GB volume .................. 4.40
  200 GB snapshots ............... 2.20
  Server backups (20%) ........... 16.17
  ─────────────────────────────────────
  Total .......................... 109.60/month

For comparison, equivalent AWS infrastructure would cost
approximately 300-500/month for similar specs.
```

---

**Key takeaway**: Hetzner Cloud provides the infrastructure to run a fully autonomous AI development system for under 50/month in a standard production configuration. The combination of cheap compute, free private networking, included traffic, and S3-compatible object storage makes it the optimal choice for aiai's deployment. All infrastructure is code, all state is in git, and any server can be rebuilt from scratch in minutes.
