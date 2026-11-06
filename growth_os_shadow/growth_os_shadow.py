#!/usr/bin/env python3
"""Jetson Growth OS shadow commander. Python 3.8+; stdlib only.

This process deliberately has NO network client and NO publish method.
It cannot post, spend, change Cloudflare, mutate Play releases, or access RLR.
Policy-only queue simulation is not evidence of an actual provider publication.
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import pathlib
import re
import sqlite3
import sys
import time

PROVIDERS = {
    "instagram": "postspread",
    "facebook": "postspread",
    "bluesky": "metricool",
    "pinterest": "composio",
    "tiktok": "composio",
    "threads": "composio",
    "youtube": "composio",
    "reddit": "composio",
    "linkedin": "composio",
}
HUMAN_CHANNELS = {"reddit", "linkedin"}
PRODUCTS = {
    "ava": ("com.dyagnosys.ava", "1.4.11", 116, 1),
    "mythoughs": ("com.mindsend.mythoughs", "1.1.3", 98, 0),
    "fireuptrader": ("com.mindsense.fireuptrader", "", 0, 0),
}
EVENT_SOURCES = {
    "content_published": {"publisher"},
    "landing_view": {"cloudflare"},
    "cta_click": {"cloudflare"},
    "install_referrer_received": {"app"},
    "first_open": {"app"},
    "onboarding_completed": {"app"},
    "qualified_activation": {"app"},
    "D7_retained": {"app"},
    "subscription_confirmed": {"billing"},
    "net_revenue": {"billing"},
}
IDENTIFIER = re.compile(r"^[a-zA-Z0-9_.:-]{1,96}$")
PUBLICATION_STATES = {"WAITING_APPROVAL", "BLOCKED_PRODUCT", "HUMAN_REVIEW",
                      "SHADOW_TESTED", "UNKNOWN_OUTCOME", "CANCELLED"}
DEFAULT_DB = os.path.expanduser("~/.local/share/jetson-growth-shadow/ledger.sqlite3")


def utcnow():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def ident(value, field):
    if not isinstance(value, str) or not IDENTIFIER.fullmatch(value):
        raise ValueError("invalid " + field)
    return value


def receipt_id(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class ShadowLedger:
    def __init__(self, path=DEFAULT_DB):
        self.path = os.path.abspath(os.path.expanduser(path))
        pathlib.Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, timeout=10, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript("""
          CREATE TABLE IF NOT EXISTS products(
            product_id TEXT PRIMARY KEY, package TEXT NOT NULL,
            version TEXT NOT NULL, version_code INTEGER NOT NULL,
            observed_production INTEGER NOT NULL CHECK(observed_production IN(0,1)),
            observed_at TEXT NOT NULL
          );
          CREATE TABLE IF NOT EXISTS publish_jobs(
            job_key TEXT PRIMARY KEY, product_id TEXT NOT NULL REFERENCES products(product_id),
            campaign_id TEXT NOT NULL, variant_id TEXT NOT NULL, channel TEXT NOT NULL,
            account_id TEXT NOT NULL, provider TEXT NOT NULL, scheduled_slot TEXT NOT NULL,
            state TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
            attempts INTEGER NOT NULL DEFAULT 0 CHECK(attempts=0),
            provider_post_id TEXT, reason TEXT NOT NULL DEFAULT '',
            UNIQUE(product_id,campaign_id,variant_id,channel,account_id,scheduled_slot)
          );
          CREATE TABLE IF NOT EXISTS events(
            event_key TEXT PRIMARY KEY, event_type TEXT NOT NULL, event_source TEXT NOT NULL,
            product_id TEXT NOT NULL, campaign_id TEXT, variant_id TEXT, referral_id TEXT,
            synthetic INTEGER NOT NULL CHECK(synthetic=1), occurred_at TEXT NOT NULL,
            received_at TEXT NOT NULL
          );
          CREATE TABLE IF NOT EXISTS audit(
            id INTEGER PRIMARY KEY AUTOINCREMENT, action TEXT NOT NULL,
            object_key TEXT, decision TEXT NOT NULL, detail TEXT NOT NULL,
            occurred_at TEXT NOT NULL
          );
          CREATE TABLE IF NOT EXISTS health(
            id INTEGER PRIMARY KEY CHECK(id=1), last_tick TEXT NOT NULL,
            global_pause INTEGER NOT NULL CHECK(global_pause=1)
          );
        """)
        now = utcnow()
        with self.db:
            for key, (package, version, code, available) in PRODUCTS.items():
                self.db.execute("""INSERT OR IGNORE INTO products
                    (product_id,package,version,version_code,observed_production,observed_at)
                    VALUES(?,?,?,?,?,?)""", (key, package, version, code, available, now))
            self.db.execute(
                "INSERT OR IGNORE INTO health(id,last_tick,global_pause) VALUES(1,?,1)", (now,)
            )

    def audit(self, action, key, decision, detail):
        self.db.execute("""INSERT INTO audit(action,object_key,decision,detail,occurred_at)
            VALUES(?,?,?,?,?)""", (action, key, decision, detail, utcnow()))

    def gate(self, product_id, channel, provider, cost_usd=0):
        ident(product_id, "product_id")
        ident(channel, "channel")
        ident(provider, "provider")
        if product_id not in PRODUCTS:
            return "BLOCKED_PRODUCT", "unknown_product"
        row = self.db.execute("SELECT observed_production FROM products WHERE product_id=?",
                              (product_id,)).fetchone()
        if row is None or row["observed_production"] != 1:
            return "BLOCKED_PRODUCT", "no_verified_production_release"
        if channel not in PROVIDERS or provider != PROVIDERS[channel]:
            return "BLOCKED_PRODUCT", "publisher_ownership_mismatch"
        if channel in HUMAN_CHANNELS:
            return "HUMAN_REVIEW", "community_review_required"
        if cost_usd is None or cost_usd != 0:
            return "BLOCKED_PRODUCT", "zero_spend_policy"
        return "WAITING_APPROVAL", "public_publishing_disabled"

    def enqueue(self, product, campaign, variant, channel, account, slot, provider=None):
        inputs = [product, campaign, variant, channel, account, slot]
        for k, v in zip(["product", "campaign", "variant", "channel", "account", "slot"], inputs):
            ident(v, k)
        if product not in PRODUCTS:
            raise ValueError("unknown_product")
        provider = provider or PROVIDERS.get(channel, "unsupported")
        state, reason = self.gate(product, channel, provider)
        payload = dict(product=product, campaign=campaign, variant=variant,
                       channel=channel, account=account, slot=slot)
        key = receipt_id(payload)
        now = utcnow()
        with self.db:
            self.db.execute("""INSERT OR IGNORE INTO publish_jobs(
                job_key,product_id,campaign_id,variant_id,channel,account_id,
                provider,scheduled_slot,state,created_at,updated_at,reason
              ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
              (key, product, campaign, variant, channel, account, provider,
               slot, state, now, now, reason))
            row = self.db.execute("SELECT state,reason,attempts FROM publish_jobs WHERE job_key=?",
                                  (key,)).fetchone()
            self.audit("enqueue", key, row["state"], row["reason"])
        return {"key": key, "state": row["state"], "reason": row["reason"], "attempts": row["attempts"]}

    def simulate(self, key):
        ident(key, "job_key")
        with self.db:
            row = self.db.execute(
                "SELECT state,product_id,channel,provider FROM publish_jobs WHERE job_key=?", (key,)
            ).fetchone()
            if row is None:
                raise ValueError("unknown_job")
            if row["state"] == "WAITING_APPROVAL":
                self.db.execute("UPDATE publish_jobs SET state='SHADOW_TESTED',updated_at=? WHERE job_key=?",
                                (utcnow(), key))
                self.audit("simulate", key, "SHADOW_TESTED", "no_external_action")
                return "SHADOW_TESTED"
            self.audit("simulate", key, "NOOP", "state=" + row["state"])
            return row["state"]

    def record_event(self, event_id, kind, source, product, campaign=None,
                     variant=None, referral=None, synthetic=True, occurred_at=None):
        for name, value in [("event_id", event_id), ("kind", kind),
                            ("source", source), ("product", product)]:
            ident(value, name)
        for name, value in [("campaign", campaign), ("variant", variant), ("referral", referral)]:
            if value is not None:
                ident(value, name)
        if kind not in EVENT_SOURCES or source not in EVENT_SOURCES[kind]:
            raise ValueError("event_source_mismatch")
        if product not in PRODUCTS:
            raise ValueError("unknown_product")
        if synthetic is not True:
            raise ValueError("shadow_only_synthetic_events")
        occurred_at = occurred_at or utcnow()
        dt.datetime.fromisoformat(occurred_at.replace("Z", "+00:00"))
        event_key = receipt_id({"source": source, "event_id": event_id})
        with self.db:
            before = self.db.total_changes
            self.db.execute("""INSERT OR IGNORE INTO events(
                event_key,event_type,event_source,product_id,campaign_id,variant_id,
                referral_id,synthetic,occurred_at,received_at) VALUES(?,?,?,?,?,?,?,?,?,?)""",
                (event_key, kind, source, product, campaign, variant, referral, 1,
                 occurred_at, utcnow()))
            inserted = self.db.total_changes > before
            self.audit("event", event_key, "INSERTED" if inserted else "DUPLICATE",
                       "synthetic_only")
        return {"event_key": event_key, "inserted": inserted, "synthetic": True}

    def tick(self):
        now = utcnow()
        with self.db:
            self.db.execute("UPDATE health SET last_tick=?, global_pause=1 WHERE id=1", (now,))
            self.audit("tick", "", "PASS", "shadow_mode_no_external_actions")
        return self.status()

    def status(self):
        products = [dict(r) for r in self.db.execute(
            "SELECT product_id,package,version,version_code,observed_production,observed_at FROM products")]
        jobs = [dict(r) for r in self.db.execute(
            "SELECT state,count(*) AS n FROM publish_jobs GROUP BY state")]
        events = [dict(r) for r in self.db.execute(
            "SELECT event_type,count(*) AS n FROM events WHERE synthetic=1 GROUP BY event_type")]
        row = self.db.execute("SELECT last_tick,global_pause FROM health WHERE id=1").fetchone()
        return {"mode": "SHADOW_ONLY", "public_publish_enabled": False,
                "external_paid_budget_usd": 0, "global_pause": bool(row["global_pause"]),
                "heartbeat_utc": row["last_tick"], "products": products, "jobs": jobs,
                "synthetic_events": events, "attribution_e2e_verified": False,
                "app_telemetry_live": False, "gepa_live_promotion_enabled": False,
                "channels": PROVIDERS}

    def close(self):
        self.db.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DEFAULT_DB)
    sub = ap.add_subparsers(dest="command")
    sub.add_parser("tick")
    sub.add_parser("status")
    p = sub.add_parser("enqueue")
    for arg in ("product", "campaign", "variant", "channel", "account", "slot"):
        p.add_argument(arg)
    p.add_argument("--provider")
    p = sub.add_parser("simulate")
    p.add_argument("key")
    p = sub.add_parser("event")
    for arg in ("event_id", "kind", "source", "product"):
        p.add_argument(arg)
    p.add_argument("--campaign")
    p.add_argument("--variant")
    p.add_argument("--referral")
    p.add_argument("--real", action="store_true", help="intentionally rejected: shadow-only")
    args = ap.parse_args()
    if not args.command:
        ap.print_help()
        return 2
    ledger = ShadowLedger(args.db)
    try:
        if args.command == "tick":
            result = ledger.tick()
        elif args.command == "status":
            result = ledger.status()
        elif args.command == "enqueue":
            result = ledger.enqueue(args.product, args.campaign, args.variant,
                                    args.channel, args.account, args.slot, args.provider)
        elif args.command == "simulate":
            result = {"state": ledger.simulate(args.key)}
        else:
            result = ledger.record_event(args.event_id, args.kind, args.source,
                                         args.product, args.campaign, args.variant,
                                         args.referral, synthetic=not args.real)
        print(json.dumps(result, sort_keys=True))
        return 0
    except (ValueError, sqlite3.Error) as exc:
        print(json.dumps({"error": str(exc), "status": "FAIL_CLOSED"}))
        return 1
    finally:
        ledger.close()


if __name__ == "__main__":
    sys.exit(main())
