# =============================================================================
# unitree_go2_realobs/motor_deg/constants.py
# Shared physical, thermal, and safety constants for the Go2 motor-degradation tasks.
# =============================================================================
"""Shared physical, thermal, and safety constants for the Go2 motor-degradation stack."""

# =============================================================================
# 1. Global Simulation Constants
# =============================================================================
NUM_MOTORS = 12
EPS = 1e-6          # 0 나누기 방지용 엡실론
T_AMB = 25.0        # [°C] 표준 대기 온도 (열역학 기준점)

# [Fix #13] 중력 상수 단일화: GRAVITY_VAL만 사용 (이전 alias GRAVITY, GRAVITY_MAG 제거)
GRAVITY_VAL = 9.81  # [m/s^2] 중력 가속도

# =============================================================================
# 2. Motor / Electrical Specs (Unitree Go2 Spec)
# =============================================================================
# --- Nominal motor parameters (25°C 기준) ---
R_NOMINAL = 0.15           # [Ohm] 권선 저항 (Source [9])
KT_NOMINAL = 0.08          # [Nm/A] 토크 상수 (Source [9])
GEAR_RATIO = 6.33          # [-] 감속기 기어비
MAX_CURRENT_INSTANT = 30.0  # [UNUSED] 순간 최대 전류 (A)

# --- Gearbox Friction Model ---
GEAR_EFFICIENCY = 0.92         # [UNUSED] 명목 효율 (compute_component_losses에서 미사용)
GEAR_FRICTION_COULOMB = 0.1    # [Nm] 쿨롱 마찰 (Source [9])
GEAR_FRICTION_VISCOUS = 0.01   # [Nms/rad] 점성 마찰

# --- Inverter (ESC) Parameters ---
R_MOSFET = 0.01            # [Ohm] MOSFET On-resistance
V_DROP = 0.7               # [V] Diode/Switching Voltage Drop

# --- Torque & Current Limits ---
NOMINAL_TORQUE = 12.0      # [UNUSED] [Nm] 연속 정격 토크
TORQUE_LIMIT = 23.7        # [Nm] 피크 토크 한계 (env_cfg.py effort_limit과 일치)
I_NO_LOAD = 0.5            # [UNUSED] [A] 무부하 전류

# --- Temperature Dependencies ---
ALPHA_CU = 0.00393         # [1/K] 구리 저항 온도 계수
ALPHA_MAG = -0.0012        # [1/K] 자석 토크 상수 온도 계수
LAMBDA_SAT = 0.1           # [UNUSED] [-] 자기 포화 감쇄 계수

# --- Loss Model Coefficients ---
K_HYST = 0.01              # 히스테리시스 손실 계수
K_EDDY = 0.0001            # 와전류 손실 계수
B_VISCOUS = 0.001          # 베어링 점성 마찰 계수 (Nominal)

# =============================================================================
# 3. Battery & Energy Management (Regen)
# =============================================================================
BATTERY_CAPACITY_WH = 150.0              # [Wh] 배터리 총 용량
BATTERY_CAPACITY_J = BATTERY_CAPACITY_WH * 3600.0  # [J] 배터리 총 용량 (줄 단위)

# utils.py 참조 [6]
REGEN_PEAK_EFFICIENCY = 0.60  # [-] 최대 회생 효율 (60%)
REGEN_OPTIMAL_SPEED = 12.0    # [rad/s] 최적 속도
REGEN_WIDTH = 10.0            # [rad/s] 효율 곡선 폭

# [Encoder Normalization]
MAX_POWER_W = 500.0           # [Watts] 관측 정규화용 최대 전력 (추정치)

# =============================================================================
# 4. Thermal Model (MotorDeg Core)
# =============================================================================
# --- Thresholds (interface.py 호환) ---
TEMP_CRITICAL_THRESHOLD = 90.0  # [°C] 임계 온도 (Source [8])
# [FIX] 중복 정의 제거 및 완화된 값 적용 (60.0 -> 75.0)
TEMP_WARN_THRESHOLD = 75.0      # [°C] 경고 온도 (Source [8])

# Alias for legacy compatibility
T_CRITICAL = TEMP_CRITICAL_THRESHOLD
T_WARN = TEMP_WARN_THRESHOLD

# --- Thermal Properties ---
# [Thermal Model v2] 2-node RC model (coil <-> case <-> ambient)
# - coil: 권선/자석 근처 hotspot 동역학
# - case: 하우징/케이스 온도 (RealObs에서 직접 대응되는 채널)
# 값들은 "정확한 실측 식별 전" 보수적 priors이며, 실로그 기반 보정 대상입니다.
C_THERMAL_COIL = 120.0          # [J/K] 권선 등가 열용량
C_THERMAL_CASE = 380.0          # [J/K] 케이스/하우징 등가 열용량
K_COIL_TO_CASE = 2.2            # [W/K] coil->case 열전달 계수
K_CASE_COOLING = 1.8            # [W/K] case 자연 냉각 계수
K_CASE_WIND = 0.03              # [W/(K*rad/s)] case 강제 대류 계수

# Legacy single-node alias (backward compatibility)
C_THERMAL = C_THERMAL_COIL
K_COOLING = K_CASE_COOLING
K_WIND = K_CASE_WIND

# --- Dynamics & Normalization ---
MAX_TEMP_RATE = 5.0             # [°C/s] 온도 변화율 정규화 기준 (Obs Norm)
THERMAL_STRESS_SCALE = 40.0     # [°C] 피로 가속 열 스트레스 분모 (degradation.py 전용, TEMP_WARN 기준; observation용 정규화는 thermal.py가 자체 계산)

# =============================================================================
# 5. Mechanical Health / Fatigue (Multi-physics MotorDeg)
# =============================================================================
# models/degradation.py [1] 필수 파라미터
FATIGUE_EXPONENT = 3.0          # 베어링 피로 지수 (L10 Life Theory)
FATIGUE_SCALE = 1e-4            # [Scale] 피로도 누적 스케일링
VIBRATION_FATIGUE_SCALE = 5.0   # [Scale] 진동에 의한 피로 가속 계수 (Source [10])

RATED_LOAD_FACTOR = 0.7         # [-] 정격 부하율 (최대 토크의 70%를 정격으로 가정)

# [FIX] 중복 정의 제거 및 노이즈 강건성 확보 (0.01 -> 0.1)
MIN_STALL_VELOCITY = 0.1        # [rad/s] Stall 상태에서의 최소 속도 보정값

# --- Advanced Friction & Wear (Moved from Bottom) ---
# 하단에 있던 정의들을 이곳으로 통합하여 관리
# [Fix #8] 50.0 → 5.0: fatigue=0.5일 때 마찰이 26배(!) 증가하던 비현실적 스케일 수정.
# 수정 후: fatigue=0.5 → 마찰 3.5배, fatigue=1.0 → 마찰 6배 (물리적으로 합리적).
# Kd 감소(감쇠력↓)와의 상호작용도 이제 현실적인 범위 내에서 동작합니다.
WEAR_FRICTION_GAIN = 5.0        # 피로도(0~1)에 따른 마찰 증가 민감도
FRICTION_HEAT_EFF = 1.0         # 마찰 손실 에너지가 열로 변환되는 효율 (1.0 = 100%, 현재 no-op; 향후 부분 열변환 모델링 시 조정)
STICTION_NOMINAL = 0.2          # [Nm] 새 베어링의 기본 정지 마찰 토크
STICTION_WEAR_FACTOR = 3.0      # [-] 피로도 Max 시 정지 마찰 증가 배수
BACKLASH_MAX_TORQUE = 0.5       # [UNUSED] [Nm] 피로도 Max 시 발생하는 유격(Deadzone) 크기
FRICTION_SMOOTHING = 20.0       # [UNUSED] [-] Tanh 함수의 기울기 (클수록 Step 함수에 가까움)

# [Updated] proprioception.py와의 정합성 (Normalization Range)
NOMINAL_VELOCITY = 30.0         # [rad/s] 정격 속도 (Operational Nominal)
MAX_VIBRATION_G = 5.0           # [G] 진동 관측 정규화 기준

# =============================================================================
# 6. Contact / Ground Interaction
# =============================================================================
NORMAL_AXIS = 2                 # Z-axis
CONTACT_THRESHOLD_N = 5.0       # [N] Ghost contact 제거용

NOMINAL_LOAD_N = 120.0          # [N] 기준 하중 (Single leg nominal load)

IMPACT_NOISE_THRESHOLD = 10.0   # [N/s] 충격량(Jerk) 노이즈 컷오프

# 관절 가속도 스케일링 (proprioception.py 참조)
ACC_SCALE = 0.01

# =============================================================================
# 7. Joint & Foot Naming
# =============================================================================
JOINT_NAMES = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
]

GO2_FOOT_NAMES = ["FR_foot", "FL_foot", "RR_foot", "RL_foot"]

# =============================================================================
# 8. Controller Constants
# =============================================================================
ACTION_SCALE = 0.25
JOINT_VEL_LIMIT = 30.0

# =============================================================================
# 9. Termination & Safety (MDP Standards)
# =============================================================================
# 로봇 베이스 충돌 감지를 위한 스캔 포인트 오프셋
CHASSIS_SCAN_OFFSETS = [
    [ 0.2,  0.0,  0.05],  # Front Center
    [-0.2,  0.0,  0.05],  # Back Center
    [ 0.0,  0.1,  0.05],  # Left Center
    [ 0.0, -0.1,  0.05],  # Right Center
    [ 0.0,  0.0,  0.05],  # Body Center
]

# Base height check
BASE_HEIGHT_MIN = 0.21           # [m] 낙상 판단 최소 높이
INVALID_TERRAIN_HEIGHT = -100.0    # [m] 맵 이탈 판단 높이
