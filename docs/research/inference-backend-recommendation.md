---
id: research-inference-backend
title: "Bundled Inference Backend for Feather Cloud (April 2026)"
status: research-complete
date: 2026-04-27
author: Hawky.ai (research agent)
---

# Bundled Inference Backend Recommendation

> Source: research workstream for Phase 9 / Cloud bundled-LLM tier. Decisions below feed the Cloud control plane's `inference` module.

## TL;DR

| Tier | Primary | Fallback | Notes |
|---|---|---|---|
| **Default (cost-optimized)** | **Fireworks AI** (Llama 3.3 70B) | **DeepInfra** | Smart router: latency budget + error rate triggers DeepInfra for ~60% cheaper bursts; auto-fail back to Fireworks |
| **Pro (premium quality)** | **Anthropic Sonnet 4** (passthrough) | OpenAI GPT-4.1 (passthrough alternate) | Customer-selectable; bill at cost-plus 30–40% margin |
| **VPC / regulated** | **Baseten managed vLLM** | Customer-hosted vLLM + Qwen 2.5 32B (AWQ-INT4) Helm chart | Default on-prem to Qwen 2.5 32B (single L40S/A100-80GB) so customer hardware bar stays low |
| **Speed (optional add-on)** | **Groq** | — | LPU sub-100ms TTFT for re-rankers, query rewriting; opt-in toggle |

**Eliminate from candidate list:** Anyscale Endpoints (deprecated late 2024) · OctoAI (acquired by NVIDIA, public endpoints sunset).

## Pricing snapshot (Q1 2026 verified — re-verify before procurement)

| Provider | Llama 3.3 70B ($/1M in/out) | Latency p95 | Trains on data? | VPC? |
|---|---|---|---|---|
| Together AI | $0.88 / $0.88 | 250–400ms | No | Yes (Together Dedicated) |
| **Fireworks AI** | **$0.90 / $0.90** | **200–350ms** | **No** | **Yes** |
| Groq | $0.59 / $0.79 | **50–100ms TTFT** | No | Limited |
| DeepInfra | **$0.23 / $0.40** | 300–500ms | No | No |
| Baseten | ~$1.10 / $1.10 (GPU-billed) | 200ms | No | **Yes — primary use case** |
| Self-host (1× H100, 1y CUD) | ~$6/hr; ~80M tok/day breakeven | 150–250ms | n/a | Native |

Pro-tier passthrough cost (no markup):
- Anthropic Claude Sonnet 4: $3 / $15 per 1M
- OpenAI GPT-4.1: $2.50 / $10 per 1M
- Gemini 2.5 Pro: $1.25 / $10 per 1M

## TCO estimate (5K queries/tenant/day, ~5.5M tokens/tenant/day)

| Tenants | Tokens/mo | DeepInfra | Fireworks | Self-host (CUD) |
|---|---|---|---|---|
| 10 | 1.65B | ~$520/mo | ~$1,485/mo | ~$4,400/mo (idle) |
| 100 | 16.5B | ~$5,200/mo | ~$14,850/mo | ~$8,800/mo (2× H100) |
| 1,000 | 165B | ~$52,000/mo | ~$148,500/mo | **~$25–35K/mo (cluster autoscaled)** |

**Inflection points:**
- **< 100 tenants:** stay 100% managed (Fireworks primary + DeepInfra fallback)
- **100–500:** hybrid (managed burst + reserved/self-host baseline)
- **> 500:** self-host dominates; managed becomes overflow/failover

## Per-provider risks

| Provider | Risk |
|---|---|
| Together AI | Above-market pricing; margin compression risk |
| **Fireworks AI** | Smaller than Together; vendor concentration |
| Anyscale | **Already deprecated for hosted endpoints — do not target** |
| OctoAI | **Defunct as standalone (NVIDIA acquisition)** |
| Groq | Capacity-constrained on demand spikes; narrow model catalog; LPU lock-in |
| DeepInfra | Lower SLA historically; tail-latency variance |
| Baseten | GPU-second billing spikes on poor utilization; cold-starts non-trivial |
| Self-host | Ops burden; H100 lead times still bursty |

**License clearance:** Llama 3.3 has 700M-MAU restriction (not a practical risk at our volumes); Qwen 2.5 is Apache-2.0 and license-clean.

## Implementation work for Cloud

- [ ] `feather_cloud/inference/router.py` — Fireworks → DeepInfra failover, latency-budget router
- [ ] `feather_cloud/inference/passthrough.py` — Anthropic / OpenAI Pro-tier billing
- [ ] `feather_cloud/inference/byol.py` — OpenAI-compatible endpoint adapter (any URL + key)
- [ ] `feather_cloud/inference/speed.py` — Groq adapter for re-ranker / rewriter sub-calls
- [ ] On-prem Helm chart: vLLM + Qwen 2.5 32B AWQ-INT4

## Sources (verify before procurement)

- [Together AI pricing](https://www.together.ai/pricing)
- [Fireworks AI pricing](https://fireworks.ai/pricing)
- [Groq pricing](https://groq.com/pricing/)
- [DeepInfra pricing](https://deepinfra.com/pricing)
- [Baseten pricing](https://www.baseten.co/pricing/)
- [Anthropic pricing](https://www.anthropic.com/pricing)
- [OpenAI pricing](https://openai.com/api/pricing/)
- [Llama 3.3 license](https://www.llama.com/llama3_3/license/)
