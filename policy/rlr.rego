package rlr

import future.keywords.if

allowed_actions := {
	"observe", "set_vision_fps", "set_reply_tokens", "set_load_target",
	"remember", "recall", "reflect", "set_goal", "complete_goal",
	"cooldown", "abstain", "set_nvpmodel", "manage_service", "set_cgroup_value",
}

safe_actions := {"cooldown", "abstain"}
aggressive_actions := {"set_vision_fps", "set_reply_tokens", "set_load_target"}

critical_missing if count(input.telemetry.zones_c) == 0
critical_missing if input.telemetry.ram_total_mib == null
critical_missing if input.telemetry.acquisition_stalled == true

max_temp := max(input.telemetry.zones_c) if {
	count(input.telemetry.zones_c) > 0
}

decision := {
	"decision": "ALLOW",
	"effective_action": input.proposal,
	"reason_code": "safe_action",
	"reason": "Emergency safe action allowed",
} if {
	safe_actions[input.proposal.action]
} else := {
	"decision": "DENY",
	"effective_action": {},
	"reason_code": "unsupported_action",
	"reason": "Action is not supported by policy",
} if {
	not allowed_actions[input.proposal.action]
} else := {
	"decision": "ABSTAIN",
	"effective_action": {"action": "abstain", "reason": "telemetry_unavailable", "rationale": "Critical telemetry unavailable"},
	"reason_code": "telemetry_unavailable",
	"reason": "Critical telemetry is missing, invalid, or stalled",
} if {
	critical_missing
}

else := {
	"decision": "ABSTAIN",
	"effective_action": {"action": "abstain", "reason": "safe_mode", "rationale": "Safety state requires abstention"},
	"reason_code": "safe_mode",
	"reason": "Safety state requires abstention",
} if {
	input.safety_state == "SAFE_MODE"
} else := {
	"decision": "ABSTAIN",
	"effective_action": {"action": "abstain", "reason": "repeated_execution_failures", "rationale": "Repeated execution failures require operator review"},
	"reason_code": "repeated_execution_failures",
	"reason": "Repeated execution failures require abstention",
} if {
	input.execution_failures >= 3
} else := {
	"decision": "DENY",
	"effective_action": {},
	"reason_code": "thermal_cooldown",
	"reason": "Aggressive action denied during thermal cooldown",
} if {
	aggressive_actions[input.proposal.action]
	input.safety_state == "COOLDOWN"
}

else := {
	"decision": "DENY",
	"effective_action": {},
	"reason_code": "hard_thermal_limit",
	"reason": "Aggressive action denied at hard thermal threshold",
} if {
	aggressive_actions[input.proposal.action]
	max_temp >= 78
} else := {
	"decision": "CAP",
	"effective_action": object.union(input.proposal, {"fps": 1.0}),
	"reason_code": "vision_fps_cap",
	"reason": "Vision FPS capped by thermal policy",
} if {
	input.proposal.action == "set_vision_fps"
	max_temp >= 75
	input.proposal.fps > 1.0
} else := {
	"decision": "CAP",
	"effective_action": object.union(input.proposal, {"fps": 2.0}),
	"reason_code": "vision_fps_cap",
	"reason": "Vision FPS capped by thermal policy",
} if {
	input.proposal.action == "set_vision_fps"
	max_temp >= 70
	input.proposal.fps > 2.0
}

else := {
	"decision": "CAP",
	"effective_action": object.union(input.proposal, {"tokens": 48}),
	"reason_code": "reply_token_cap",
	"reason": "Reply tokens capped by thermal policy",
} if {
	input.proposal.action == "set_reply_tokens"
	max_temp >= 75
	input.proposal.tokens > 48
} else := {
	"decision": "CAP",
	"effective_action": object.union(input.proposal, {"tokens": 64}),
	"reason_code": "reply_token_cap",
	"reason": "Reply tokens capped by thermal policy",
} if {
	input.proposal.action == "set_reply_tokens"
	max_temp >= 70
	input.proposal.tokens > 64
} else := {
	"decision": "CAP",
	"effective_action": object.union(input.proposal, {"percent": 50}),
	"reason_code": "thermal_load_cap",
	"reason": "Load target capped while approaching thermal threshold",
} if {
	input.proposal.action == "set_load_target"
	max_temp >= 70
	input.proposal.percent > 50
}

else := {
	"decision": "CAP",
	"effective_action": object.union(input.proposal, {"percent": 50}),
	"reason_code": "memory_pressure",
	"reason": "Load target capped under high memory pressure",
} if {
	input.proposal.action == "set_load_target"
	input.telemetry.ram_total_mib > 0
	input.telemetry.ram_used_mib / input.telemetry.ram_total_mib >= 0.90
	input.proposal.percent > 50
} else := {
	"decision": "ABSTAIN",
	"effective_action": {"action": "abstain", "reason": "aggressive_rate_limit", "rationale": "Aggressive load increase requested too soon"},
	"reason_code": "aggressive_rate_limit",
	"reason": "Minimum interval between aggressive load increases has not elapsed",
} if {
	input.proposal.action == "set_load_target"
	input.proposal.percent > input.settings.load_target
	input.seconds_since_aggressive_action != null
	input.seconds_since_aggressive_action < 10
} else := {
	"decision": "CAP",
	"effective_action": object.union(input.proposal, {"percent": input.settings.load_target + 20}),
	"reason_code": "load_increase_rate_cap",
	"reason": "Load target increase capped to 20 percentage points per decision",
} if {
	input.proposal.action == "set_load_target"
	input.proposal.percent > input.settings.load_target + 20
}

else := {
	"decision": "ALLOW",
	"effective_action": input.proposal,
	"reason_code": "allowed",
	"reason": "Action satisfies deterministic policy",
} if {
	allowed_actions[input.proposal.action]
}
