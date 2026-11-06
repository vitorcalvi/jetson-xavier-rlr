-- Growth OS shadow namespace; already applied additively to existing ava-clicks D1 on 2026-11-06.
-- NON-PRODUCTION DATA ONLY. Do not use this schema to store real user telemetry.
-- Preserve pre-existing referral_campaigns, clicks, campaign_click_totals.
CREATE TABLE IF NOT EXISTS growth_shadow_events (
  source TEXT NOT NULL, external_event_id TEXT NOT NULL,
  product_id TEXT NOT NULL, campaign_id TEXT, variant_id TEXT, referral_id TEXT,
  event_type TEXT NOT NULL, occurred_at TEXT NOT NULL,
  synthetic INTEGER NOT NULL DEFAULT 1 CHECK(synthetic = 1),
  received_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY(source, external_event_id)
);
CREATE TABLE IF NOT EXISTS growth_shadow_jobs (
  job_key TEXT PRIMARY KEY, product_id TEXT NOT NULL, campaign_id TEXT NOT NULL,
  variant_id TEXT NOT NULL, channel_id TEXT NOT NULL, account_id TEXT NOT NULL,
  owner_provider TEXT NOT NULL, scheduled_slot TEXT NOT NULL,
  state TEXT NOT NULL CHECK(state IN (
    'PLANNED','WAITING_APPROVAL','BLOCKED_PRODUCT','HUMAN_REVIEW',
    'SHADOW_TESTED','UNKNOWN_OUTCOME','CANCELLED')),
  attempts INTEGER NOT NULL DEFAULT 0 CHECK(attempts = 0),
  synthetic INTEGER NOT NULL DEFAULT 1 CHECK(synthetic = 1),
  last_updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(product_id,campaign_id,variant_id,channel_id,account_id,scheduled_slot)
);
CREATE TABLE IF NOT EXISTS growth_shadow_product_snapshots (
  product_id TEXT PRIMARY KEY, package_name TEXT NOT NULL,
  observed_version TEXT, observed_code INTEGER,
  production_serving_observed INTEGER NOT NULL CHECK(production_serving_observed IN (0,1)),
  observed_at TEXT NOT NULL, source TEXT NOT NULL,
  synthetic INTEGER NOT NULL DEFAULT 0 CHECK(synthetic IN (0,1))
);
CREATE TABLE IF NOT EXISTS growth_shadow_audit (
  seq INTEGER PRIMARY KEY AUTOINCREMENT,
  occurred_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  actor TEXT NOT NULL, subject_id TEXT NOT NULL,
  decision TEXT NOT NULL, reason TEXT NOT NULL,
  synthetic INTEGER NOT NULL DEFAULT 1 CHECK(synthetic = 1)
);
CREATE INDEX IF NOT EXISTS growth_shadow_event_campaign_idx
  ON growth_shadow_events(product_id,campaign_id,event_type);
CREATE INDEX IF NOT EXISTS growth_shadow_job_state_idx
  ON growth_shadow_jobs(state,channel_id);
