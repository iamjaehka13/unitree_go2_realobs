# Real Robot Integration (SDK2)

Place Unitree Go2 SDK2 here:

- `third_party/unitree_sdk2/`

Then run:

```bash
python3 unitree_go2_realobs/scripts/real/check_sdk_setup.py
```

If your SDK is in a different location:

```bash
python3 unitree_go2_realobs/scripts/real/check_sdk_setup.py --sdk-root /absolute/path/to/sdk
```

Architecture check examples:

```bash
python3 unitree_go2_realobs/scripts/real/check_sdk_setup.py --target-arch x86_64
python3 unitree_go2_realobs/scripts/real/check_sdk_setup.py --target-arch aarch64
```

Once this check passes, this repo supports:

- 500 Hz `LowState` ingest
- 50 Hz governor loop (thermal/voltage)
- teleop/replay command scaling
- hard-stop and experiment logging

---

## Live Wiring (Now Implemented)

This repo now includes a concrete bridge path:

1) SDK2 C++ UDP bridge (`LowState` -> UDP, UDP -> `SportClient.Move`)
2) Python live governor runner (same governor logic as replay)

Build bridge:

```bash
cmake -S unitree_go2_realobs/scripts/real/sdk2_bridge -B /tmp/go2_udp_bridge_build
cmake --build /tmp/go2_udp_bridge_build -j
```

Run bridge:

```bash
/tmp/go2_udp_bridge_build/go2_udp_bridge enp3s0
```

필요할 때만 자동 기립:

```bash
/tmp/go2_udp_bridge_build/go2_udp_bridge enp3s0 --auto-stand-up
```

`LowState.bit_flag` 기반 estop 연동이 필요하면:

```bash
/tmp/go2_udp_bridge_build/go2_udp_bridge enp3s0 --estop-bit-mask 1
```

Run live governor (fixed command profile, no YAML schedule file required):

```bash
python3 unitree_go2_realobs/scripts/real/run_governor_live_template.py \
  --state_host 127.0.0.1 --state_port 17001 \
  --cmd_host 127.0.0.1 --cmd_port 17002 \
  --lin_x 0.20 --lin_y 0.00 --yaw_rate 0.00 \
  --out_dir ./real_runs/go2_live
```

Default governor thresholds are `65/70/75 C` for `warn/crit/stop`.

주의:
- 러너는 **첫 상태 패킷 수신 전에는 항상 stop 명령**을 보냅니다.
- 상태가 `--state_timeout_s` 동안 안 들어오면 `state_timeout`으로 종료합니다.

Bridge reference docs:

- `unitree_go2_realobs/scripts/real/sdk2_bridge/README.md`

---

## Direct Python Path (`unitree_sdk2_python`)

If you prefer direct Python ingest on Jetson (e.g., Ubuntu 22.04), you can
log `rt/lowstate` without ROS2/bridge:

1) Install `unitree_sdk2_python` (or clone it and use `--sdk2py_root`).
   - Required Python deps from upstream:
     - `cyclonedds==0.10.2`
     - `numpy`
     - `opencv-python`
2) Run:

```bash
python3 unitree_go2_realobs/scripts/real/log_collector_sdk2py.py \
  --interface enp3s0 \
  --output_csv ./go2_full_log_sdk2py.csv \
  --log_hz 500
```

If `unitree_sdk2py` is only cloned (not pip-installed), add:

```bash
  --sdk2py_root /path/to/unitree_sdk2_python
```

Then convert to a 50 Hz command-schedule CSV:

```bash
python3 unitree_go2_realobs/scripts/real/log_to_replay_csv.py \
  --input_csv ./go2_full_log_sdk2py.csv \
  --output_csv ./go2_replay_50hz.csv
```

---

## Keep vs Remove (Go2-only)

Keep (required):

- `third_party/unitree_sdk2/CMakeLists.txt`
- `third_party/unitree_sdk2/cmake/`
- `third_party/unitree_sdk2/include/`
- `third_party/unitree_sdk2/lib/`
- `third_party/unitree_sdk2/thirdparty/`
- `third_party/unitree_sdk2/licenses/`
- `third_party/unitree_sdk2/example/go2/` (reference code)
- `third_party/unitree_sdk2/example/CMakeLists.txt`

Safe-to-remove for this project:

- `third_party/unitree_sdk2/.github/`
- `third_party/unitree_sdk2/.devcontainer/`
- non-Go2 example folders under `third_party/unitree_sdk2/example/`

Automated prune command:

```bash
python3 unitree_go2_realobs/scripts/real/prune_sdk2_for_go2.py --dry-run
python3 unitree_go2_realobs/scripts/real/prune_sdk2_for_go2.py --apply
```

---

## Utilities

1) Live governor runner (UDP bridge integration):

```bash
python3 unitree_go2_realobs/scripts/real/run_governor_live_template.py --out_dir ./real_runs
```

2) Convert raw 500Hz log to a command-schedule-friendly 50Hz CSV:

```bash
python3 unitree_go2_realobs/scripts/real/log_to_replay_csv.py \
  --input_csv <raw_500hz.csv> \
  --output_csv <replay_50hz.csv>
```

3) Convert rosbag2 `LowState` logs directly to raw/replay CSV:

```bash
python3 -m pip install --user rosbags

python3 unitree_go2_realobs/scripts/real/rosbag_lowstate_to_csv.py \
  --input_bag ./realdata/<bag_dir> \
  --output_replay_csv ./realdata/<bag_name>_50hz.csv \
  --cmd_vx 0.0 --cmd_vy 0.0 --cmd_wz 0.0
```

If the rosbag is nested one level deeper, the script resolves the inner directory
that contains `metadata.yaml`. It reads `/lowstate` by default. Use
`--output_raw_csv` as well if you need the full replay-friendly raw CSV.

When directory names already encode the command, e.g. `cmd_x1,y0,z-1data`, you
can use:

```bash
python3 unitree_go2_realobs/scripts/real/rosbag_lowstate_to_csv.py \
  --input_bag ./realdata/cmd_x1,y0,z-1data \
  --output_replay_csv ./realdata/cmd_x1y0z-1_50hz.csv \
  --infer_command_from_path
```

3) Offline governor evaluation from converted log:

```bash
python3 unitree_go2_realobs/scripts/real/offline_governor_eval_from_log.py \
  --input_csv <replay_50hz.csv> \
  --output_json <offline_summary.json> \
  --governor
```

The offline evaluator uses the same default governor thresholds: `65/70/75 C`.

For real-log capture requirements and bridge packet details, see:

- `unitree_go2_realobs/scripts/real/REAL_LOG_REQUIREMENTS.txt`
- `unitree_go2_realobs/scripts/real/sdk2_bridge/README.md`

## Python Dependencies

Minimal (no YAML command schedule, fixed command flags only):

```bash
# no extra package required
```

If you use YAML replay command files (`--command_file *.yaml`):

```bash
python3 -m pip install --user numpy pyyaml
```
