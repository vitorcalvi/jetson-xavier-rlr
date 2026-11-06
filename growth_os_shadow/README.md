# Jetson Growth OS Shadow Commander (standalone Python 3.8+)

**Status:** Shadow-only, no network clients, no social publication, no external costs. Isolated from active RLR/trading runtimes. This is not a production publishing gateway or verified Android attribution pipeline.

The commander provides SQLite WAL/full-sync durable job intent, a deterministic one-provider-per-channel map, app-release eligibility gating, non-production My Thoughs block, human review for Reddit/LinkedIn, zero budget, event validation and deduplication, global pause, a durable heartbeat and an immutable audit trail. It never marks a post PUBLISHED, never sends external messages, and does not include a public publishing API.

### Usage

Run from the growth_os_shadow directory:

    python3 growth_os_shadow.py tick
    python3 growth_os_shadow.py status
    python3 growth_os_shadow.py enqueue ava pilot variantA bluesky vitorcalvi 20261106T1800Z
    python3 growth_os_shadow.py simulate YOUR_JOB_KEY
    python3 growth_os_shadow.py event event1 first_open app ava --campaign pilot --variant variantA
    python3 -m unittest test_growth_os_shadow.py -v

Default SQLite path: ~/.local/share/jetson-growth-shadow/ledger.sqlite3. All attribution events are synthetic only. Real tracking requires consent-compliant Android integration and authenticated cloud ingestion. The --real switch intentionally fails closed.

### Rollout requirements

Keep the existing RLR process and signed operator policy untouched. Do not give this process Jetson root MCP credentials or broker secrets. For read-only scheduling, install the supplied systemd user timer after tests. Its service runs one tick and exits every five minutes; there is no long-running model inference or media rendering on the Jetson. The actual multimedia worker must run independently through the managed Linux/Dokploy lifecycle.

**Remaining gates:** app-side Play Install Referrer and meaningful activation, authenticated Cloudflare ingestion, durable external asset store, live Play metric sampling, publisher M2M credentials, provider/media budget contracts, GEPA live evaluator, explicit public publication approval. Never treat these test results as proof that those gates passed.
