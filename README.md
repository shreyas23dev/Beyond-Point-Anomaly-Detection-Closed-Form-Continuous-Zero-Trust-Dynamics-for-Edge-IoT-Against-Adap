# Beyond Point Anomaly Detection: Closed-Form Continuous Zero-Trust Dynamics for Edge-IoT Against Adaptive Slow-Burn Attacks

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Docker: Containerized](https://img.shields.io/badge/Docker-Microservices-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com/)
[![Dataset: Edge--IIoTset](https://img.shields.io/badge/Dataset-Edge--IIoTset-success.svg)](https://doi.org/10.1109/ACCESS.2022.3165421)
[![Status: Journal Submission](https://img.shields.io/badge/Status-Under%20Review%20at%20Elsevier%20IoT-orange.svg)]()

> **Official Implementation** for the research paper:  
> *"Beyond Point Anomaly Detection: Closed-Form Continuous Zero-Trust Dynamics for Edge-IoT Against Adaptive Slow-Burn Attacks"*  
> **Authors:** Shreyas A, Arun Kumar B R  
> **Data DOI:** [10.21227/mbc1-1h68](https://doi.org/10.21227/mbc1-1h68)

---

## 📌 Overview

Conventional Zero-Trust Architecture (ZTA) implementations in IoT rely on **stateless, per-packet anomaly verdicts**. This makes them structurally blind to **slow-burn adversaries**—attackers who deliberately pace malicious requests below the static detection threshold ($A_t < A_{\text{thr}}$) to evade detection while accumulating damage.

This repository provides an end-to-end, containerized, and mathematically grounded **Dynamic Trust Engine** that wraps any underlying point anomaly classifier with:
1. **$O(1)$ Asymmetric Slow-Burn Accumulator:** Tracks smoothed anomaly energy ($E_t$) against an empirical baseline margin ($\gamma = 0.335 \approx \mu_{\text{normal}} + 3.05\sigma_{\text{EMA}}$) with an $11\times$ accumulation-to-decay asymmetry ($\lambda=0.55 \gg \delta=0.05$).
2. **Closed-Form Asymptotic Trust Recovery ($\alpha$):** Analytically derives the recovery coefficient from high-level security policy targets without heuristic gain-tuning (Theorem 1).
3. **Multiplicative Attack Decay:** Immediately degrades trust proportional to anomaly magnitude ($T_{t+1} = T_t(1 - A_t)$) for overt volumetric attacks.
4. **Mutually Exclusive State Machine:** Enforces progressive slow-burn mitigation without operational contradictions between recovery and penalties.
5. **Deterministic Auditability:** Logs a two-level mathematical trace ($\text{Formula ID}, E_t, SB_t, T_t, \text{State Transition}, \text{Reason String}$) for every single request at zero computational overhead.

---

## 🏛️ System Architecture

```mermaid
flowchart LR
    subgraph EdgeLayer["1. IoT Edge Device (ESP32)"]
        ESP[ESP32 Telemetry Generator] -->|HMAC-SHA256 Auth| POST[HTTPS POST :5000]
    end

    subgraph RelayLayer["2. Gateway Relay (WSL2)"]
        POST -->|IPTables DNAT/SNAT| NAT[NAT Forwarding Layer]
    end

    subgraph CloudLayer["3. Cloud Microservices (Docker on Azure)"]
        NAT --> APIGW["3.1 API Gateway<br/>(:8000 / :5000)"]
        APIGW -->|JSON Schema| ML["3.2 ML Anomaly Engine<br/>(Isolation Forest :9000)"]
        ML -->|Anomaly Score A_t| DTE["3.3 Dynamic Trust Engine<br/>(Algorithm 1 :8002)"]
        DTE -->|Trust Score T_t| PDP["3.4 ZTA Decision Engine<br/>(Policy Engine :7000)"]
        DTE -.->|Async Persist| DB[(PostgreSQL / Redis<br/>:5432)]
    end

    subgraph Enforcement["4. Policy Enforcement (PEP)"]
        PDP -->|ALLOW / VERIFY / BLOCK| Decision[Zero-Trust Response]
        Decision -.->|HTTP 200 / 202 / 403| ESP
    end

    style DTE fill:#e1f5fe,stroke:#0288d1,stroke-width:2px
    style ML fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    style PDP fill:#e8f5e9,stroke:#388e3c,stroke-width:2px
