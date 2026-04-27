---
id: research-multimodal-embedder
title: "Multi-modal Embedder Recommendation for Marketing Vertical"
status: research-complete
date: 2026-04-27
author: Hawky.ai (research agent)
---

# Multi-Modal Embedder Recommendation for Feather DB Marketing Vertical

> Source: research workstream for Phase 9 / Cloud Marketing vertical. Decisions below feed `feather_cloud.verticals.marketing` and the OSS `feather_db.embedders.visual` module.

## TL;DR

| Tier | Choice | Why |
|---|---|---|
| **Default (hosted)** | **Cohere Embed-Vision-3** | $0.0001/image · 1024-dim · stable API · 100M-token free tier covers small tenants |
| **Premium (hosted)** | **Voyage Multimodal-3** *(or Gemini Multimodal Embedding for video-heavy)* | Top hosted retrieval quality · ~+1pp over Cohere on Flickr30k/COCO · Gemini wins for native video |
| **Self-host default (VPC)** | **SigLIP-2 so400m** | Apache-2.0 · fits L4 GPU · retrieval quality (Flickr30k 94.6 / COCO 81.8) actually beats Voyage on benchmarks |
| **Self-host premium** | **InternVL-2.5-8B Embedding** | MIT · #1 on MTEB-Vision Apr 2026 · needs A10 24GB / L40S |

**Avoid:** PaliGemma-2 (VLM, not pure embedder; license restrictions).

## Comparison snapshot

```
                          Image dim  Pricing                  Quality (Flickr30k R@1)
─────────────────────────────────────────────────────────────────────────────────────
Cohere Embed-Vision-3     1024       $0.0001/image            ~92  (default tier)
Voyage Multimodal-3       1024       $0.0001/image            ~93  (premium)
Gemini MM Embedding       1408       $0.0001/img · video-native ~93  (premium video)
SigLIP-2 so400m           1152       Self-host (~$0.50/hr L4)  94.6 (self-host default)
InternVL-2.5-8B           4096       Self-host (~$1/hr L40S)   95.1+ (self-host premium)
OpenCLIP ViT-G/14         1280       Self-host (~$1/hr L4)     92.9
```

## Video handling — tiered strategy

Inside Feather:

1. **Default** (works everywhere): client-side **uniform 8-frame sampling** → embed each → mean-pool → L2-normalize → store as single vector under `modality="visual"`. Same approach Voyage and Cohere recommend internally.
2. **Optional richer mode**: store all 8 frame vectors under a new `modality="video_frames"` pocket; aggregation (max/attention) applied at query time. Lets users find a *moment* inside an ad.
3. **Native-video premium path**: when customer is on Gemini/Vertex backend, route video bytes directly to `multimodalembedding@001` and store the 1408-dim vector. Higher quality for motion-heavy creatives (transitions, kinetic typography).
4. Helper: `db.add_video(id, video_path, embedder=...)` picks strategy by configured backend.

## Implementation work for OSS

- [ ] `feather_db/embedders/visual/cohere.py` — `CohereVisualEmbedder` (default hosted)
- [ ] `feather_db/embedders/visual/voyage.py` — `VoyageMultimodalEmbedder` (premium hosted)
- [ ] `feather_db/embedders/visual/gemini.py` — `GeminiMultimodalEmbedder` (with video path)
- [ ] `feather_db/embedders/visual/siglip.py` — `SigLIP2Embedder` (self-host)
- [ ] `feather_db/embedders/visual/__init__.py` — factory
- [ ] `db.add_video()` helper

## Sources

- [Cohere Embed v3 model card](https://cohere.com/pricing)
- [Voyage Multimodal-3 docs](https://docs.voyageai.com/docs/multimodal-embeddings)
- [Gemini multimodal embeddings](https://ai.google.dev/gemini-api/docs/embeddings)
- [SigLIP-2 paper, arXiv:2502.14786](https://arxiv.org/abs/2502.14786) · [HF: google/siglip2-so400m-patch14-384](https://huggingface.co/google/siglip2-so400m-patch14-384)
- [InternVL-2.5 technical report](https://huggingface.co/OpenGVLab/InternVL2_5-8B-MPO)
- [MTEB-Vision leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
