package rlr

import future.keywords.if

base_input := {
	"proposal": {"action": "set_load_target", "percent": 60, "rationale": "test"},
	"telemetry": {"zones_c": [55], "ram_used_mib": 1000, "ram_total_mib": 8000, "acquisition_stalled": false},
	"settings": {"load_target": 50},
	"safety_state": "NORMAL",
	"recent_actions": [],
	"execution_failures": 0,
	"seconds_since_aggressive_action": null,
}

test_allow if {
	d := decision with input as base_input
	d.decision == "ALLOW"
}

test_cap_thermal if {
	x := object.union(base_input, {
		"proposal": {"action": "set_reply_tokens", "tokens": 96, "rationale": "test"},
		"telemetry": {"zones_c": [76], "ram_used_mib": 1000, "ram_total_mib": 8000, "acquisition_stalled": false},
	})
	d := decision with input as x
	d.decision == "CAP"
	d.effective_action.tokens == 48
}

test_missing_telemetry_abstains if {
	x := object.union(base_input, {"telemetry": {"zones_c": [], "ram_used_mib": null, "ram_total_mib": null, "acquisition_stalled": false}})
	d := decision with input as x
	d.decision == "ABSTAIN"
}

test_hard_thermal_denies if {
	x := object.union(base_input, {"telemetry": {"zones_c": [78], "ram_used_mib": 1000, "ram_total_mib": 8000, "acquisition_stalled": false}})
	d := decision with input as x
	d.decision == "DENY"
}

test_safe_cooldown_survives_degraded_telemetry if {
	x := object.union(base_input, {
		"proposal": {"action": "cooldown", "seconds": 30, "rationale": "safe"},
		"telemetry": {"zones_c": [], "ram_used_mib": null, "ram_total_mib": null, "acquisition_stalled": true},
	})
	d := decision with input as x
	d.decision == "ALLOW"
}

test_load_target_thermal_cap if {
	x := object.union(base_input, {
		"proposal": {"action": "set_load_target", "percent": 90, "rationale": "load"},
		"telemetry": {"zones_c": [71], "ram_used_mib": 1000, "ram_total_mib": 8000, "acquisition_stalled": false},
		"settings": {"load_target": 100},
	})
	d := decision with input as x
	d.decision == "CAP"
	d.effective_action.percent == 50
}
