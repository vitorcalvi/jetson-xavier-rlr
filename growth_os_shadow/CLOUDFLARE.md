# Cloudflare D1 shadow state — applied, NOT live attribution

The implementation used existing dyagnosys.com account D1 database ava-clicks. Only additive growth_shadow_* schema changes were made. No new D1 database, Worker, R2 bucket, Queue, billing plan, route or API credential was created. The existing referral_campaigns/clicks/campaign_click_totals rows remained unchanged as verified immediately after migration.

Synthetic validation:
- one growth_shadow_jobs job, inserted twice with unique key and reconciled into SHADOW_TESTED
- one growth_shadow_events CTA event inserted twice; count exactly one
- Ava 1.4.11 (116) production snapshot and My Thoughs 1.1.3 (98) test-only snapshot from Play Developer read-only release filters
- original referral_campaigns count 1, clicks count 0, campaign_click_totals count 0
- audit row appended with explicit synthetic scope

There is **no runtime bridge** between Jetson local SQLite and Cloudflare D1 yet. The Jetson ledger is local shadow intent; Cloudflare growth_shadow tables hold independent synthetic validation and Play snapshots only. Do NOT call either one a complete authoritative publisher or conversion ledger. A production gate must select one canonical write authority and implement authenticated, replay-protected M2M ingestion and reconciliation with separate credentials.

Real app event ingestion is intentionally forbidden by the synthetic-only constraint. The private Ava staging branch contains the unmerged native Play Referrer reader and a purely local parser; those are neither wired into actual first_open nor deployed. This separation prevents false conversion attribution.

Do not run a destructive rollback without checking newer dependencies; DROP TABLE removes audit history and is not an automatic rollback. Future migrations should be transactional and versioned with backup/recovery tests.
