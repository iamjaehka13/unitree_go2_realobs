# SDK2 UDP Bridge (Go2)

This bridge connects Unitree SDK2 `LowState`/`SportClient` and the Python
governor runner.

## Data Flow

1. Bridge subscribes `rt/lowstate` (~500Hz).
2. Bridge sends aggregated state via UDP (default `127.0.0.1:17001`):
   - `ts,temp_max_c,vpack_v,vcell_min_v,wz_actual,estop,mode`
3. Python runner computes governor scale at 50Hz.
4. Python sends command UDP back (default `127.0.0.1:17002`):
   - `vx,vy,wz,stop`
5. Bridge applies `SportClient.Move(vx,vy,wz)` or `StopMove()`.

## Build

```bash
cmake -S unitree_go2_realobs/scripts/real/sdk2_bridge -B /tmp/go2_udp_bridge_build
cmake --build /tmp/go2_udp_bridge_build -j
```

Binary:

- `/tmp/go2_udp_bridge_build/go2_udp_bridge`

## Run Bridge

```bash
/tmp/go2_udp_bridge_build/go2_udp_bridge enp3s0
```

Optional ports/host:

```bash
/tmp/go2_udp_bridge_build/go2_udp_bridge enp3s0 \
  --tx-host 127.0.0.1 --tx-port 17001 --rx-port 17002
```

Optional auto stand-up once at bridge init:

```bash
/tmp/go2_udp_bridge_build/go2_udp_bridge enp3s0 --auto-stand-up
```

Optional estop mapping from `LowState.bit_flag`:

```bash
/tmp/go2_udp_bridge_build/go2_udp_bridge enp3s0 --estop-bit-mask 1
```

Use your SDK bit definition; default is disabled (`0`).

## Run Python Governor

Install dependencies first:

```bash
python3 -m pip install --user numpy pyyaml
```

```bash
python3 unitree_go2_realobs/scripts/real/run_governor_live_template.py \
  --lin_x 0.20 --lin_y 0.00 --yaw_rate 0.00 \
  --state_host 127.0.0.1 --state_port 17001 \
  --cmd_host 127.0.0.1 --cmd_port 17002 \
  --out_dir ./real_runs/go2_live
```

To auto-start bridge from Python:

```bash
python3 unitree_go2_realobs/scripts/real/run_governor_live_template.py \
  --bridge_cmd "/tmp/go2_udp_bridge_build/go2_udp_bridge enp3s0" \
  --lin_x 0.20 --lin_y 0.00 --yaw_rate 0.00
```
