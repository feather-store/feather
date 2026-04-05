# Glean AI — Deep Research & Competitive Analysis

> Research date: April 2026  
> Context: Brainstorming session on enterprise AI search architecture and positioning of Feather DB.

---

## 1. The "Google Algorithm" Claim — What's Actually True

The common claim that "Glean uses Google's indexing algorithm" is **accurate in spirit but imprecise technically.**

**What it really means:**
- Glean's CEO **Arvind Jain** spent 10+ years at Google as a Distinguished Engineer working directly on Search ranking algorithms, Maps, and YouTube.
- Three of four co-founders came from Google/Facebook core ranking and infrastructure teams.
- They brought **institutional knowledge** of how Google Search works — not a license, not an investment relationship.
- Glean was built from scratch, applying Google Search principles to the enterprise problem.

**What it does NOT mean:**
- Glean does not license Google's search algorithm or codebase.
- Google has not invested in Glean.
- Glean is not a Google product.

---

## 2. Founding Team

| Founder | Role | Background |
|---------|------|-----------|
| **Arvind Jain** | CEO & Co-founder | 10+ years at Google (Search ranking, Maps, YouTube). Founded Rubrik (data security, now public). IIT Delhi + U. Washington. |
| **Tony Gentilcore** | Co-founder, Product Engineering | 10 years at Google. Modernized web search results from "ten blue links" era. Founded Chrome Speed Team. Contributed to HTML5 spec + W3C Web Performance Working Group. |
| **T.R. Vishwanath** | Co-founder, Technical Infrastructure | ~10 years at Facebook. Principal Engineer on News Feed ranking, Ads, Developer Platform. |
| **Piyush Prahladka** | Co-founder | ~10 years at Google. Technical lead on Maps and Search Quality Ranking (2005+). |

**Pattern**: The founding team has combinatorially deep experience in ranking + infrastructure + distributed systems — not just product intuition.

---

## 3. How Glean Works — Technical Architecture

Glean is a **5-layer enterprise search platform**, not just a vector database.

### Layer 1 — Data Collection (Connectors)
- 100+ pre-built connectors: Slack, Jira, Confluence, Salesforce, Google Drive, Workday, Databricks, Snowflake, GitHub, etc.
- Real-time crawlers that respect permission boundaries at object/record level.
- API-native (not URL scraping) — understands each app's data model natively.

### Layer 2 — Indexing & Embedding Pipeline
- **Apache Flink** for real-time stream processing.
- Text → **BM25 keyword index** (Okapi BM25).
- Text → **Dense vector embeddings** (Vertex AI / TPUs for training).
- Both indexes updated in near-real-time as new data flows in.

### Layer 3 — Enterprise Knowledge Graph
Glean's true differentiator. The graph models three pillars:

| Pillar | What it captures |
|--------|-----------------|
| **Content** | Documents, messages, tickets, entities |
| **People** | Identities, roles, teams, departments, org hierarchy |
| **Activity** | Creation/edit history, comments, searches, clicks, co-authorship |

Relationships are extracted and strengthened with "strong, repeatable evidence" — not just co-occurrence. The graph is rebuilt continuously as data flows in.

### Layer 4 — Hybrid Search & Ranking
When a query arrives:

1. **BM25 keyword search** — fast, explainable, exact-term matching.
2. **Dense semantic search** — embedding similarity for meaning-based retrieval.
3. **Reciprocal Rank Fusion (RRF)** — merges both result sets intelligently.
4. **60+ ranking signals applied:**
   - Graph signals: ownership, org proximity, link structure, team relationships
   - Activity signals: engagement patterns, recency, usage frequency
   - Personalization: user role, team membership, interaction history
   - Content: freshness, authority, term statistics
5. **Permission-aware filtering** — results filtered per user ACL at object level.

### Layer 5 — LLM Reasoning Layer
- Retrieved documents passed as context to LLM (supports 15+ models: GPT-4, Claude, Gemini, Llama).
- LLM generates fact-grounded answers with line-by-line citations.
- Increasingly powered by **in-house fine-tuned models** (Llama 3.2 base + Glean fine-tune via Cake platform).
- Fine-tuned model accuracy: improved from 70% → >90% vs. base models on enterprise tasks.

---

## 4. Tech Stack

| Component | Technology |
|-----------|-----------|
| Cloud platform | Google Cloud (BigQuery, Dataflow, Vertex AI) + AWS (RDS, S3) |
| Stream processing | Apache Flink |
| Embeddings training | Vertex AI (Google TPUs) |
| ML orchestration | Kubeflow + KServe |
| LLM serving | vLLM + Llama Factory (in-house) |
| LLM models | GPT-4, Claude, Gemini, Llama 3.2 (fine-tuned) |
| Data storage | AWS RDS + S3 (customer-owned for sovereignty) |

---

## 5. Competitive Moat

### Why Glean is Hard to Replicate

1. **Enterprise Knowledge Graph** — Years of ingested org data (people, teams, projects). Network effects: more data = better graph = better results.
2. **Permission enforcement** — Object/record-level, near-real-time. Table-stakes for enterprise governance.
3. **Deep connector engineering** — 100+ connectors with app-native data model understanding. Significant maintenance investment.
4. **60+ ranking signals** — Tuned across knowledge graph, activity, content. ML models trained on enterprise-specific distributions.
5. **In-house LLM fine-tuning** — Models trained on enterprise corpora outperform generic LLMs on enterprise retrieval.
6. **Model neutrality** — Supports 15+ LLMs. Positions Glean as the abstraction layer above any LLM.

---

## 6. Business & Market Position

| Metric | Value |
|--------|-------|
| ARR | $200M+ (doubled in 9 months, Feb 2026) |
| Valuation | $7.2B (Series F, June 2025) |
| Total funding | $765M across 6 rounds |
| Key investors | Sequoia Capital, Wellington, Altimeter, DST Global |

**Competitors**: Microsoft 365 Copilot, Amazon Q Business, ChatGPT Enterprise, Perplexity, Writer, GoSearch.

**Key differentiator vs. Microsoft Copilot**: Works across 100+ apps, not just the Microsoft ecosystem.  
**Key differentiator vs. ChatGPT Enterprise**: Deep permission graph + knowledge graph, not just LLM chat.

---

## 7. Glean vs. Feather DB — Honest Comparison

| Dimension | Feather DB | Glean |
|-----------|------------|-------|
| **What it is** | Embedded vector DB + living context engine | Full enterprise search platform |
| **Setup** | `pip install feather-db`, one file | 6-12 month enterprise deployment |
| **Permissions** | Developer-built | Native object-level, real-time |
| **Connectors** | Developer-built | 100+ out-of-the-box |
| **Ranking** | Cosine similarity + adaptive decay | 60+ signals + knowledge graph |
| **Organizational context** | Namespace/entity model | Full org hierarchy + people graph |
| **Price** | Open source / $0 | ~$15-40/user/month (enterprise) |
| **Target** | Developers building AI apps | IT buyers at Fortune 500 |
| **Deployment** | Embedded / single binary | Multi-tenant cloud service |
| **Customization** | Full (it's your code) | Limited (SaaS product) |

**Key insight**: Feather DB is NOT competing with Glean. Feather DB is what you would use *inside* the retrieval layer of a Glean-like system. Glean is the $7.2B full platform built around that kind of primitive.

---

## 8. What Glean Does NOT Solve Well

These are **genuine gaps** — and the opportunity space for Feather DB:

| Gap | Details |
|-----|---------|
| **Small teams / developers** | Glean is too expensive and heavy for startups or individual developers |
| **Domain-specific context** | Generic connectors don't capture custom business logic (e.g., ad performance signals, marketing graphs) |
| **Embedded use in apps** | You can't `pip install glean` into a Python agent or mobile app |
| **Living context / memory decay** | Glean doesn't model which memories matter more over time (adaptive decay) |
| **Lightweight deployment** | Glean requires enterprise infrastructure; Feather DB runs on a Raspberry Pi |
| **Custom graph topology** | Glean's knowledge graph is predefined; Feather DB lets you model any relationship structure |
| **Offline / air-gapped** | Glean is cloud-only; Feather DB works fully offline |

---

## 9. Strategic Positioning Recommendation for Feather DB

**"SQLite for enterprise AI context"** — the developer-native, domain-aware, embeddable alternative.

Glean = Oracle. Feather DB = SQLite/PostgreSQL.

**Target markets where Feather DB wins:**
1. **AI agent memory** — LLM agents need lightweight, persistent, searchable context stores. Glean is too heavy.
2. **Domain-specific intelligence** — Ad tech, fintech, healthcare apps with proprietary data models.
3. **Edge / embedded AI** — Mobile, IoT, air-gapped environments where Glean can't reach.
4. **Startups building AI products** — Developers who need vector DB + graph + decay without enterprise contracts.
5. **Plugin/middleware layer** — Used *inside* enterprise tools as the retrieval component (the role a vector DB plays in Glean's stack).

---

## 10. Key Takeaways

1. Glean's moat is the **knowledge graph + permissions layer** — not the vector search (which is commoditized).
2. The "Google algorithm" claim is about **people and culture**, not licensed code.
3. Hybrid search (BM25 + dense + RRF) is now the industry standard for enterprise retrieval — pure vector search is insufficient alone.
4. The enterprise AI middleware layer (retrieval + permissions + graph + LLM orchestration) is where the $7B+ value sits.
5. **Feather DB's unique angle**: living context (adaptive decay + recall_count), embedded deployment, and full developer control — none of which Glean offers.
