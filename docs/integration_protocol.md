# Cloud local navigation contract

The upstream cloud method remains authoritative for the seven-value order. `navigation.schema.STATE_NAMES` is tested against `inference.build_state_inference.STATE_NAMES`.

## Distinct inputs

| Consumer | Input | Meaning |
|---|---|---|
| Cloud VLM | RGB + `[x,y,d_goal,d_front,d_left,d_back,d_right]` | Seven values, metres, body-frame range directions |
| Cloud risk prompt | separate `yaw_rad` | Radians; not appended to the seven-value vector |
| Local GRU-IQN | four frames ×20 | relative goal2 + actual world velocity2 + four ranges4 + previous executed action one-hot12 |
| Local IQN head | encoded512 + external72 taus | 64 CVaR decision samples, 8 lower-half RTV samples |

The current navigation policy is fixed-yaw planar navigation. World +x/+y coincide with body front/left at yaw0. Arbitrary-yaw ray interpolation is NOT implemented. These are distinct interface semantics, not a change from a seven-input network to an eight-input network.

## Observation JSONL

One line per decision, no future action or future geometry:

```json
{"observation":{"session_id":"unique-mission-001","sequence":1,"timestamp_s":10.0,"position_m":[0.0,-3.0],"goal_m":[0.0,3.0],"velocity_mps":[0.0,0.0],"ranges_m":[2.0,3.0,4.0,1.0],"yaw_rad":0.0},"previous_executed_action_id":null,"image_path":"frames/frame001.png","frame_timestamp_s":10.0}
```

This is a schema example, not a captured flight record. Remove image fields if no RGB exists. Body ranges must be valid metric values clipped to the training range0..4. Missing/invalid ranges are not infinity. Sensor adapters must convert mm to m and classify invalid firmware readings before building an Observation.

Session IDs must be unique for every new mission/goal; sequence is positive and strictly increasing, sample timestamps increase. Changing the goal without a new session is rejected. Local real-time producers must use the same `time.monotonic()` clock as the consumer. File replay explicitly uses its own recorded virtual clock.

Only the first observation can bootstrap a session. Later mission changes require a trusted lifecycle event, not a different ID arriving in a sensor packet. In JSONL insert this event before the new mission's observations:

```json
{"type":"start_session","session_id":"unique-mission-002"}
```

Programmatic adapters must call both `Controller.start_session(id)` and `CloudRiskClient.start_session(id)`. Old/unknown observation sessions are rejected without changing the active mission, pending command or history. Retired IDs cannot be reactivated; generate a new ID for a new mission. Explicit lifecycle events must come from the trusted mission manager, not untrusted telemetry.

## POST /v1/semantic-risk

```json
{"schema_version":1,"request_id":"unique-request-id","session_id":"unique-mission-001","sequence":1,"timestamp_s":10.0,"frame_timestamp_s":10.0,"state_names":["x","y","d_goal","d_front","d_left","d_back","d_right"],"state":[0.0,-3.0,6.0,2.0,3.0,4.0,1.0],"yaw_rad":0.0,"task":"risk_analysis","image":{"media_type":"image/png","base64":"BASE64_OF_ACTUAL_SYNCHRONIZED_IMAGE"}}
```

PNG/JPEG only, up to2MiB decoded, max4096×4096 pixels; image/state timestamp mismatch above100ms is rejected. The server accepts image bytes, never an arbitrary client-supplied file path. Use authentication and HTTPS when crossing machines.

Successful response:

```json
{"schema_version":1,"request_id":"unique-request-id","session_id":"unique-mission-001","sequence":1,"timestamp_s":10.0,"status":"ok","risk":{"risk_score":0.8,"confidence":0.9,"risk_level":"high","reason":"example semantic risk explanation"}}
```

These risk values are illustrative. IDs, sequence, timestamp and version must exactly echo the request. Numerical values must be finite and in0..1; confidence must be present, not fabricated by the adapter. `risk_level` is low/medium/high/critical. JSON fences may be removed, but arbitrary prose is not salvaged with regex. Missing/invalid model fields are rejected and fall back to local-only decisions.

## Local fusion

`alpha_fused = min(alpha_ART, max(alpha_min, alpha_ART - lambda * confidence * risk_score))` only when session, sequence, freshness and confidence gates pass. Defaults are alpha_min0.1, lambda0.25, confidence threshold0.6, TTL2s; these are initial experiment parameters, not calibrated safety thresholds. Otherwise `alpha_fused = alpha_ART`. Even a custom alpha_min cannot raise the current ART alpha. Cloud cannot output motor commands. Existing RTV/EWAF state is updated only after the selected local action is actually accepted.

One request may be in flight. A stalled cloud never creates a queue of old frames, and the local controller does not wait for a response. Freshness is checked on original sample time in the local clock, not on cloud reply time or wall-clock synchronization. Network timeout is not a hardware watchdog.

`submit_path` schedules file reading, image encoding and HTTP in a disposable worker process, outside the local decision thread. Its total timeout includes interpreter startup, file I/O, DNS and slow/trickling responses; timeout/cancellation kills the worker and releases the request slot when it exits. Keep every submitted frame path immutable until that job finishes, or use `submit` with already captured bytes. Camera/read errors are reported asynchronously on a later poll, not necessarily on the submitting row. Custom transports must implement their own bounded I/O/cancellation contract.

## Action and executor contract

Action ID = direction×3 + speed; directions +x,+y,-x,-y; speeds `(exp((i+1)/3)-1)/(e-1)` m/s, approximately0.23024,0.55156,1.0. Do not rescale the historical model's speed table silently.

`Controller.step` proposes a command. An actual executor must either accept that exact command and acknowledge it, or reject it and execute a zero-horizontal-velocity hold. If it executes a different discrete action, its actual action must be provided through `previous_executed_action` on the next observation; never write a discarded prediction into history. Arbitrary continuous speed overrides cannot be faithfully encoded as a12-action one-hot and require a separate trained interface.

Model prediction and output validation use a candidate history; failure does not append a frame or advance previous-action state. Successful observation history is committed separately from acceptance of the proposed action. Capture age plus local computation must remain within the observation deadline (default200ms); otherwise the proposal is replaced by a zero-horizontal-velocity HOLD and is not left pending. The CLI logs `observation_age_s` and `deadline_exceeded`; camera jobs are queued only after a still-valid local proposal.

The shipped JSONL CLI is shadow mode only: it sends nothing to a drone, uses externally recorded actual actions, and does not commit proposals to ART. The software deadline is not a hard-real-time hardware guarantee or permission to execute an old command. A real executor needs an additional deadline check at dispatch, an independent watchdog, altitude/yaw/telemetry checks, override arbitration and takeoff/landing procedures.
