"""Shadow-mode JSONL bridge; proposes commands but NEVER connects to motors."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
from .art import FusionConfig
from .cloud import CloudRiskClient, RiskHTTPTransport
from .controller import Controller
from .schema import Observation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, help='JSONL observations, or - for live stdin')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--expected-sha256')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--cloud-endpoint')
    parser.add_argument('--risk-ttl', type=float, default=2.0)
    parser.add_argument('--cloud-interval', type=float, default=1.0)
    parser.add_argument('--cloud-timeout', type=float, default=3.0)
    parser.add_argument('--fusion-strength', type=float, default=0.25)
    parser.add_argument('--pace', action='store_true', help='Replay at original sample intervals')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output already exists; choose a new experiment output path')
    from .model import QuantilePredictor
    predictor = QuantilePredictor(args.checkpoint, args.device, expected_sha256=args.expected_sha256)
    fusion = FusionConfig(strength=args.fusion_strength, max_age_s=args.risk_ttl)
    controller = Controller(predictor, fusion)
    cloud = CloudRiskClient(RiskHTTPTransport(args.cloud_endpoint, os.getenv('SEMANTIC_API_TOKEN'),
                                             args.cloud_timeout), args.cloud_interval, args.risk_ttl) if args.cloud_endpoint else None
    input_path = None if args.input == '-' else Path(args.input).resolve()
    source = sys.stdin if input_path is None else input_path.open(encoding='utf-8')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    session_clock = None
    session_origin = None
    count = fused_count = images = 0
    try:
        with args.output.open('x', encoding='utf-8') as output:
            for line in source:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get('type') == 'start_session':
                    controller.start_session(row['session_id'])
                    if cloud:
                        cloud.start_session(row['session_id'])
                    session_clock = session_origin = None
                    continue
                obs = Observation.from_dict(row['observation'])
                if controller.session_id is None:
                    controller.start_session(obs.session_id)
                    if cloud:
                        cloud.start_session(obs.session_id)
                active = obs.session_id == controller.session_id
                if active and session_clock is None:
                    session_clock, session_origin = time.monotonic(), obs.timestamp_s
                if args.pace and input_path and active:
                    delay = session_clock + (obs.timestamp_s - session_origin) - time.monotonic()
                    if delay > 0:
                        time.sleep(delay)
                # Live producers must use the SAME local monotonic clock as this process.
                # Recorded replay explicitly uses virtual time; it is not a real-time latency test.
                now = time.monotonic() if input_path is None else obs.timestamp_s
                risk = cloud.poll(obs, now) if cloud and active else None
                started = time.monotonic()
                decision = controller.step(obs, now, risk,
                                           previous_executed_action=row.get('previous_executed_action_id'))
                finished = time.monotonic()
                duration = finished - started
                finished_sample_time = finished if input_path is None else now + duration
                observation_age = finished_sample_time - obs.timestamp_s
                deadline_exceeded = observation_age > controller.max_age
                if decision.status == 'proposed':
                    if deadline_exceeded:
                        decision = controller.hold(obs, 'deadline_exceeded')
                    else:
                        # A proposal was not actually flown in shadow mode.
                        controller.acknowledge(decision, False)
                if cloud and active and decision.status == 'proposed' and row.get('image_path'):
                    try:
                        path = Path(row['image_path'])
                        if not path.is_absolute():
                            path = (input_path.parent if input_path else Path.cwd()) / path
                        queued = cloud.submit_path(obs, path, row['frame_timestamp_s'], finished_sample_time,
                                                   'image/jpeg' if path.suffix.lower() in ('.jpg', '.jpeg') else 'image/png')
                        images += int(queued)
                    except (OSError, ValueError, TypeError, KeyError) as exc:
                        # Camera trouble must not stop the independent local navigator.
                        cloud.last_error = type(exc).__name__ + ': invalid_or_unavailable_camera_frame'
                data = decision.to_dict()
                data.update(mode='shadow', motor_command_sent=False, checkpoint_sha256=predictor.sha256,
                            local_inference_s=duration, observation_age_s=observation_age,
                            deadline_exceeded=deadline_exceeded or decision.fusion_status == 'deadline_exceeded',
                            virtual_replay_clock=input_path is not None,
                            cloud_error=cloud.last_error if cloud else None)
                output.write(json.dumps(data, ensure_ascii=False, allow_nan=False) + '\n')
                output.flush()
                count += 1
                fused_count += decision.fusion_status == 'semantic_fusion'
    finally:
        if source is not sys.stdin:
            source.close()
        if cloud:
            cloud.close()
    print(json.dumps({'decisions': count, 'semantic_fused': fused_count, 'camera_jobs_submitted': images,
                      'output': str(args.output), 'mode': 'shadow', 'motor_command_sent': False}))


if __name__ == '__main__':
    main()
