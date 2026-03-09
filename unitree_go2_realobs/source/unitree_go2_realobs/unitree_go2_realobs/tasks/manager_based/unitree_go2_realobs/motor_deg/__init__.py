"""Motor-degradation runtime package for the real-observable Go2 tasks."""

# 1. 핵심 상태 클래스 노출
# MDP나 Env에서 'from .motor_deg import MotorDegState' 형태로 접근 가능하게 함
from .state import MotorDegState

# 2. Env와 연결되는 인터페이스 함수 노출
# unitree_go2_motor_deg_env.py의 step() 및 reset() 함수에서 호출되는 핵심 함수들입니다.
from .interface import (
    init_motor_deg_interface,
    update_motor_deg_dynamics,
    reset_motor_deg_interface,
    refresh_motor_deg_sensors,
    clear_motor_deg_step_metrics,
)

# 3. 유틸리티 및 상수 (선택적 노출)
# MDP 관측(Observation) 함수 등에서 자주 사용되는 계산 함수들입니다.
from .utils import (
    compute_battery_voltage,
    compute_component_losses,
    compute_kinematic_accel,
)

# 4. 하위 모듈 명시적 노출 (필요 시)
# 사용자가 `import motor_deg.constants`로 직접 접근할 수 있도록 함
from . import constants
from . import models
from . import buffers
