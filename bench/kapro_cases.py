"""KAware-RLR case generation with state-derived ground truth.

The ground truth for a case is computed here from the fixture state alone,
before any model is called. Nothing in this file asks the agent what it thinks
it should do, so a case cannot be labelled by the system under test.

Each case supplies:
  * the bounded context the Acting branch will see, and
  * a task whose required action set T follows deterministically from that
    context (an internal case needs nothing external; an external case names a
    fact that is deliberately absent from the context; a hybrid case requires
    reading one field and then changing another).
"""
import random

# Actions that touch persistent state, and therefore require a real capability.
STATEFUL = {"remember", "recall", "set_goal", "complete_goal", "set_load_target",
            "set_vision_fps", "set_reply_tokens"}

# Deliberately unreachable knobs. A goal naming one of these can never be
# satisfied by the whitelist, which is the point of the impossible-action cases.
IMPOSSIBLE_ACTIONS = ["set_temperature", "run_shell", "change_nvpmodel",
                      "disable_thermal_guard", "write_arbitrary_file",
                      "external_http_request"]


def _internal(task, context, topology="single-hop", subtasks=None):
    return {"boundary": "internal", "topology": topology, "task": task,
            "context": context, "target_actions": [],
            "subtasks": subtasks or [{"id": "s1", "boundary": "internal",
                                      "required_actions": []}]}


def _external(task, context, actions, topology="single-hop", subtasks=None):
    return {"boundary": "external", "topology": topology, "task": task,
            "context": context, "target_actions": list(actions),
            "subtasks": subtasks or [{"id": "s1", "boundary": "external",
                                      "required_actions": list(actions)}]}


def _hybrid(task, context, actions, internal_sub, external_sub,
            topology="single-hop"):
    return {"boundary": "hybrid", "topology": topology, "task": task,
            "context": context, "target_actions": list(actions),
            "subtasks": [{"id": "s1", "boundary": "internal",
                          "required_actions": []},
                         {"id": "s2", "boundary": "external",
                          "required_actions": list(actions)}],
            "internal_subtask": internal_sub, "external_subtask": external_sub}


def base_context(rng):
    """A bounded self-model context: telemetry, settings, capabilities, goals.

    Values are drawn from a seeded RNG so a given seed always produces the same
    case, and every number the model can see is one the scorer also knows.
    """
    temp = round(rng.uniform(38.0, 55.0), 1)
    ram_used = round(rng.uniform(900.0, 2400.0), 1)
    ram_total = 8192.0
    return {
        "telemetry": {"zones_c": [temp, round(temp - 2.0, 1)],
                      "ram_used_mib": ram_used, "ram_total_mib": ram_total,
                      "power_mode": "MODE_10W_DESKTOP", "fan_rpm": 0},
        "max_temperature_c": temp,
        "thermal_level": "normal",
        "settings": {"vision_fps": 3.0, "reply_tokens": 64, "load_target": 50},
        "capabilities": [
            {"name": "qwen_1_5b_generation", "value": 15.98, "unit": "tokens/s"},
            {"name": "yolov8n_gpu_qps", "value": 67.8935, "unit": "queries/s"},
            {"name": "yolov8n_peak_ram", "value": 3282, "unit": "MiB"},
        ],
        "active_goals": [],
        "memories": [],
        "recent_actions": [],
    }


def internal_cases(ctx):
    """Tasks answerable entirely from the context already in hand."""
    goal = ctx["active_goals"][0] if ctx["active_goals"] else None
    cams = [c for c in ctx["capabilities"] if c["name"] == "yolov8n_gpu_qps"]
    return [
        _internal("Identify the currently active goal from the self-model context "
                  "you already have.", ctx, subtasks=[{"id": "goal", "boundary": "internal",
                                                       "required_actions": []}]),
        _internal("Report whether the vision capability is present in the supplied "
                  "capability list, without retrieving anything further.", ctx),
        _internal("Determine from the action schema whether set_temperature is one "
                  "of the actions you can perform.", ctx),
        _internal("State the current maximum temperature given in your context.", ctx),
        _internal("Decide whether the current thermal level permits full vision FPS, "
                  "using only the telemetry already supplied.", ctx),
        _internal("Read the configured load target out of your own settings and "
                  "report it.", ctx),
        _internal("Decide which of your whitelisted actions could change the SoC "
                  "temperature directly, from your action list alone.", ctx),
        _internal("Given the capability table, say whether a vision detection rate "
                  "value is present. Do not measure anything.", ctx),
    ]


def external_cases(ctx, rng):
    """Tasks absent from inference context but backed by seeded persistent memory."""
    fact = "calibration offset is 0.37"
    ctx = dict(ctx)
    ctx["memories"] = []
    ctx["memory_seed"] = [fact]
    recall_case = _external(
        "Retrieve the stored calibration record. It is not in your current "
        "context; it is in persistent memory.", ctx, ["recall"])
    recall_case["expected_recall_substring"] = fact
    return [
        recall_case,
        _external("Persist this durable fact for later runs: the enclosure is on the "
                  "upper shelf and must stay level.", ctx, ["remember"]),
        _external("Create an explicit runtime goal to audit the vision pipeline "
                  "tonight.", ctx, ["set_goal"]),
        _external("Record a new goal to re-measure prompt processing throughput.",
                  ctx, ["set_goal"]),
        _external("Complete the goal recorded in the goal store; its id is not in "
                  "your context, so look it up.", ctx, ["complete_goal"]),
        _external("Change the configured load target to 40 percent so the next "
                  "measurement is gentler.", ctx, ["set_load_target"]),
    ]


def hybrid_cases(ctx):
    """Tasks with an internal reading step and an external consequence."""
    temp = ctx["max_temperature_c"]
    return [
        _hybrid("From the temperature in your context, decide the device is warm, "
                "then lower the load target to 30 percent.",
                ctx, ["set_load_target"],
                internal_sub="Interpret the supplied temperature.",
                external_sub="Modify the allowed load target."),
        _hybrid("Your context does not contain the active goal's completion state. "
                "Work out that it is still active, then mark it complete.",
                ctx, ["recall"],
                internal_sub="Recognise the goal is not yet complete.",
                external_sub="Look up and complete the goal."),
        _hybrid("You can see the current reply token cap in your context. Decide it "
                "is above the safe value, then reduce it to 48.",
                ctx, ["set_reply_tokens"],
                internal_sub="Compare the visible token cap against the policy.",
                external_sub="Apply the reduced cap."),
    ]


def topology_variants(case, index):
    """Spread one case across the three reasoning topologies.

    parallel and multi-hop add a second, independent or dependent subtask. The
    target set only grows when the added subtask genuinely needs an action; a
    second internal subtask must not add one, or the topology manipulation would
    be quietly changing the difficulty as well as the structure.
    """
    topology = ("single-hop", "parallel", "multi-hop")[index % 3]
    if topology == "single-hop" or case["boundary"] == "internal":
        case["topology"] = topology
        return case
    if topology == "parallel":
        case["topology"] = "parallel"
        case["subtasks"] = case["subtasks"] + [
            {"id": "s3", "boundary": "internal", "required_actions": []}]
        case["parallel_note"] = "second subtask is answerable from context"
    else:
        case["topology"] = "multi-hop"
        case["subtasks"] = case["subtasks"] + [
            {"id": "s3", "boundary": "internal", "required_actions": []}]
        case["dependency"] = "the later decision depends on the earlier subtask"
    return case


def distractors_for(whitelist, count=10, seed=0):
    """Plausible non-executable action names for the noise condition.

    These exist only to be shown to the model during selection. They are never
    registered with the runtime, so selecting one can only produce an invalid
    action, which the harness records rather than executes.
    """
    synthetic = [
        ("get_temperature", "Read the current SoC temperature"),
        ("search_memory", "Search persistent memory"),
        ("set_temperature", "Set the SoC thermal limit"),
        ("inspect_goal", "Read a goal without changing it"),
        ("query_capability", "Query a capability value"),
        ("set_thermal_goal", "Set a target SoC temperature"),
        ("list_files", "List files on the data volume"),
        ("run_shell", "Execute a shell command"),
        ("restart_vision", "Restart the vision pipeline"),
        ("set_gpu_clock", "Set the GPU clock"),
        ("fetch_url", "Fetch a remote URL"),
        ("log_metric", "Write a metric to the log"),
    ]
    out = []
    for name, description in synthetic:
        if len(out) >= count:
            break
        out.append({"name": name, "description": description,
                    "executable": False})
    return out


def build_cases(seed, count, with_distractors=False):
    """Deterministically build `count` cases for one seed.

    The same seed and count must always produce byte-identical cases, so the
    ablation conditions can reuse them and any difference between conditions is
    attributable to the manipulation rather than to the sample.
    """
    rng = random.Random(seed)
    ctx = base_context(rng)
    # Interleave the three boundaries instead of concatenating them, so even a
    # small per-seed run still contains external and hybrid cases. Concatenating
    # made every case under the pool length internal, which would have measured
    # restraint alone and reported no external or hybrid behaviour at all.
    internal, external, hybrid = internal_cases(ctx), external_cases(ctx, rng), hybrid_cases(ctx)
    pool = []
    for i in range(max(len(internal), len(external), len(hybrid))):
        for group in (internal, external, hybrid):
            if i < len(group):
                pool.append(group[i])
    cases = []
    for i in range(count):
        base = dict(pool[i % len(pool)])
        base = topology_variants(base, i)
        memory_seed = list((base.get("context") or {}).get("memory_seed") or [])
        ctx_copy = base_context(random.Random(seed * 1000 + i))
        if memory_seed:
            ctx_copy["memory_seed"] = memory_seed
        base["context"] = ctx_copy
        base["case_id"] = "s%d-%03d" % (seed, i)
        base["seed"] = seed
        base["condition"] = "full_self_model"
        if with_distractors:
            base["condition"] = "distractors"
            base["candidate_actions"] = distractors_for(None, 10, seed)
        base["ground_truth"] = {"T": list(base["target_actions"]),
                                "subtasks": base.get("subtasks", [])}
        cases.append(base)
    return cases
