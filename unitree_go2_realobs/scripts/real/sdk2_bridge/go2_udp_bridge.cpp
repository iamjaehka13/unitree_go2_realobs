#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include <unitree/idl/go2/LowState_.hpp>
#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/robot/go2/sport/sport_client.hpp>

using unitree::robot::ChannelFactory;
using unitree::robot::ChannelSubscriber;
using unitree::robot::ChannelSubscriberPtr;

namespace {
constexpr const char *kTopicLowState = "rt/lowstate";

struct Options {
  std::string interface;
  std::string tx_host = "127.0.0.1";
  int tx_port = 17001;
  int rx_port = 17002;
  double state_hz = 50.0;
  double cmd_timeout_s = 0.5;
  bool auto_stand_up = false;
  int estop_bit_mask = 0;
};

struct Sample {
  double ts = 0.0;
  double temp_max_c = 25.0;
  double vpack_v = 33.6;
  double vcell_min_v = 4.2;
  double wz_actual = 0.0;
  int estop = 0;
  int mode = 0;
};

struct Command {
  double vx = 0.0;
  double vy = 0.0;
  double wz = 0.0;
  bool stop = true;
  double recv_ts = 0.0;
};

double MonotonicNow() {
  timespec ts{};
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return static_cast<double>(ts.tv_sec) + static_cast<double>(ts.tv_nsec) * 1e-9;
}

void SleepSeconds(double sec) {
  if (sec <= 0.0) {
    return;
  }
  const useconds_t us = static_cast<useconds_t>(sec * 1e6);
  usleep(us);
}

bool ParseInt(const char *s, int &out) {
  if (s == nullptr) {
    return false;
  }
  char *end = nullptr;
  const long v = strtol(s, &end, 10);
  if (end == s || *end != '\0') {
    return false;
  }
  out = static_cast<int>(v);
  return true;
}

bool ParseDouble(const char *s, double &out) {
  if (s == nullptr) {
    return false;
  }
  char *end = nullptr;
  const double v = strtod(s, &end);
  if (end == s || *end != '\0') {
    return false;
  }
  out = v;
  return true;
}

void PrintUsage(const char *prog) {
  std::cout << "Usage: " << prog << " <networkInterface> [options]\n"
            << "Options:\n"
            << "  --tx-host <host>         default: 127.0.0.1\n"
            << "  --tx-port <port>         default: 17001 (state -> python)\n"
            << "  --rx-port <port>         default: 17002 (python -> command)\n"
            << "  --state-hz <hz>          default: 50\n"
            << "  --cmd-timeout-s <sec>    default: 0.5\n"
            << "  --auto-stand-up          send StandUp once at init (default: off)\n"
            << "  --estop-bit-mask <int>   set estop when (bit_flag & mask)!=0 (default: 0 disabled)\n";
}

bool ParseArgs(int argc, char **argv, Options &opt) {
  if (argc < 2) {
    PrintUsage(argv[0]);
    return false;
  }
  opt.interface = argv[1];
  for (int i = 2; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--tx-host" && i + 1 < argc) {
      opt.tx_host = argv[++i];
      continue;
    }
    if (arg == "--tx-port" && i + 1 < argc) {
      if (!ParseInt(argv[++i], opt.tx_port)) {
        std::cerr << "[ERR] Invalid --tx-port\n";
        return false;
      }
      continue;
    }
    if (arg == "--rx-port" && i + 1 < argc) {
      if (!ParseInt(argv[++i], opt.rx_port)) {
        std::cerr << "[ERR] Invalid --rx-port\n";
        return false;
      }
      continue;
    }
    if (arg == "--state-hz" && i + 1 < argc) {
      if (!ParseDouble(argv[++i], opt.state_hz)) {
        std::cerr << "[ERR] Invalid --state-hz\n";
        return false;
      }
      continue;
    }
    if (arg == "--cmd-timeout-s" && i + 1 < argc) {
      if (!ParseDouble(argv[++i], opt.cmd_timeout_s)) {
        std::cerr << "[ERR] Invalid --cmd-timeout-s\n";
        return false;
      }
      continue;
    }
    if (arg == "--auto-stand-up") {
      opt.auto_stand_up = true;
      continue;
    }
    if (arg == "--estop-bit-mask" && i + 1 < argc) {
      if (!ParseInt(argv[++i], opt.estop_bit_mask)) {
        std::cerr << "[ERR] Invalid --estop-bit-mask\n";
        return false;
      }
      continue;
    }
    std::cerr << "[ERR] Unknown or incomplete argument: " << arg << "\n";
    PrintUsage(argv[0]);
    return false;
  }
  return true;
}

bool ParseCommandCsv(const std::string &text, Command &cmd_out) {
  // vx,vy,wz,stop
  std::stringstream ss(text);
  std::string item;
  std::vector<std::string> cols;
  while (std::getline(ss, item, ',')) {
    cols.push_back(item);
  }
  if (cols.size() < 4) {
    return false;
  }
  char *end = nullptr;
  cmd_out.vx = strtod(cols[0].c_str(), &end);
  if (end == cols[0].c_str()) {
    return false;
  }
  cmd_out.vy = strtod(cols[1].c_str(), &end);
  if (end == cols[1].c_str()) {
    return false;
  }
  cmd_out.wz = strtod(cols[2].c_str(), &end);
  if (end == cols[2].c_str()) {
    return false;
  }
  const long stop_i = strtol(cols[3].c_str(), &end, 10);
  if (end == cols[3].c_str()) {
    return false;
  }
  cmd_out.stop = (stop_i != 0);
  return true;
}

double CellRawToVolt(uint16_t raw) {
  if (raw == 0U) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  if (raw > 100U) {
    return static_cast<double>(raw) * 1e-3;
  }
  return static_cast<double>(raw);
}

Sample FromLowState(
    const unitree_go::msg::dds_::LowState_ &msg,
    double now_s,
    int estop_bit_mask) {
  Sample s;
  s.ts = now_s;
  s.mode = static_cast<int>(msg.level_flag());
  if (estop_bit_mask > 0) {
    s.estop = ((static_cast<int>(msg.bit_flag()) & estop_bit_mask) != 0) ? 1 : 0;
  } else {
    s.estop = 0;
  }

  double temp_max = 0.0;
  constexpr int kMotors = 12;
  for (int i = 0; i < kMotors; ++i) {
    const double t = static_cast<double>(msg.motor_state()[i].temperature());
    temp_max = std::max(temp_max, t);
  }
  s.temp_max_c = temp_max;

  s.vpack_v = static_cast<double>(msg.power_v());
  s.wz_actual = static_cast<double>(msg.imu_state().gyroscope()[2]);

  double cell_min = std::numeric_limits<double>::infinity();
  constexpr int kCells = 8;
  for (int i = 0; i < kCells; ++i) {
    const double v = CellRawToVolt(msg.bms_state().cell_vol()[i]);
    if (std::isfinite(v)) {
      cell_min = std::min(cell_min, v);
    }
  }
  if (!std::isfinite(cell_min)) {
    // Fallback if cell-voltage array is unavailable.
    cell_min = (s.vpack_v > 1e-3) ? (s.vpack_v / 8.0) : 4.2;
  }
  s.vcell_min_v = cell_min;

  return s;
}

Sample Aggregate(const std::vector<Sample> &samples, const Sample &fallback) {
  if (samples.empty()) {
    return fallback;
  }
  Sample out = samples.back();
  out.temp_max_c = 0.0;
  out.vcell_min_v = std::numeric_limits<double>::infinity();
  for (const auto &s : samples) {
    out.temp_max_c = std::max(out.temp_max_c, s.temp_max_c);
    out.vcell_min_v = std::min(out.vcell_min_v, s.vcell_min_v);
  }
  if (!std::isfinite(out.vcell_min_v)) {
    out.vcell_min_v = fallback.vcell_min_v;
  }
  return out;
}
}  // namespace

class Go2UdpBridge {
 public:
  explicit Go2UdpBridge(Options opt) : opt_(std::move(opt)) {}

  bool Init() {
    if (!InitSockets()) {
      return false;
    }

    ChannelFactory::Instance()->Init(0, opt_.interface);

    sport_client_.SetTimeout(10.0f);
    sport_client_.Init();
    sport_client_.StopMove();
    if (opt_.auto_stand_up) {
      sport_client_.StandUp();
    }

    lowstate_sub_.reset(new ChannelSubscriber<unitree_go::msg::dds_::LowState_>(kTopicLowState));
    lowstate_sub_->InitChannel(
        std::bind(&Go2UdpBridge::LowStateHandler, this, std::placeholders::_1), 1);

    std::cout << "[INFO] Bridge started. iface=" << opt_.interface
              << " tx=" << opt_.tx_host << ":" << opt_.tx_port
              << " rx=0.0.0.0:" << opt_.rx_port
              << " auto_stand_up=" << (opt_.auto_stand_up ? "on" : "off")
              << " estop_bit_mask=" << opt_.estop_bit_mask << "\n";
    return true;
  }

  void Run() {
    const double dt = 1.0 / std::max(opt_.state_hz, 1.0);
    double next_tick = MonotonicNow();
    last_cmd_.recv_ts = MonotonicNow();

    while (true) {
      PollCommands();

      std::vector<Sample> snapshot;
      Sample fallback{};
      bool has_lowstate = false;
      {
        std::lock_guard<std::mutex> lock(mu_);
        snapshot.swap(samples_);
        fallback = last_sample_;
        has_lowstate = has_lowstate_;
      }

      if (!has_lowstate) {
        // Never drive before the first real state packet is observed.
        sport_client_.StopMove();
        next_tick += dt;
        const double sleep_s = next_tick - MonotonicNow();
        if (sleep_s > 0.0) {
          SleepSeconds(sleep_s);
        }
        continue;
      }

      const Sample agg = Aggregate(snapshot, fallback);
      SendState(agg);
      ApplyCommand(agg.estop);

      next_tick += dt;
      const double sleep_s = next_tick - MonotonicNow();
      if (sleep_s > 0.0) {
        SleepSeconds(sleep_s);
      }
    }
  }

 private:
  bool InitSockets() {
    tx_fd_ = socket(AF_INET, SOCK_DGRAM, 0);
    if (tx_fd_ < 0) {
      std::cerr << "[ERR] socket(tx) failed: " << strerror(errno) << "\n";
      return false;
    }

    rx_fd_ = socket(AF_INET, SOCK_DGRAM, 0);
    if (rx_fd_ < 0) {
      std::cerr << "[ERR] socket(rx) failed: " << strerror(errno) << "\n";
      close(tx_fd_);
      tx_fd_ = -1;
      return false;
    }

    sockaddr_in rx_addr{};
    rx_addr.sin_family = AF_INET;
    rx_addr.sin_addr.s_addr = htonl(INADDR_ANY);
    rx_addr.sin_port = htons(static_cast<uint16_t>(opt_.rx_port));
    if (bind(rx_fd_, reinterpret_cast<sockaddr *>(&rx_addr), sizeof(rx_addr)) < 0) {
      std::cerr << "[ERR] bind(rx) failed: " << strerror(errno) << "\n";
      return false;
    }

    const int flags = fcntl(rx_fd_, F_GETFL, 0);
    if (flags >= 0) {
      fcntl(rx_fd_, F_SETFL, flags | O_NONBLOCK);
    }

    memset(&tx_addr_, 0, sizeof(tx_addr_));
    tx_addr_.sin_family = AF_INET;
    tx_addr_.sin_port = htons(static_cast<uint16_t>(opt_.tx_port));
    if (inet_pton(AF_INET, opt_.tx_host.c_str(), &tx_addr_.sin_addr) != 1) {
      std::cerr << "[ERR] Invalid --tx-host: " << opt_.tx_host << "\n";
      return false;
    }

    return true;
  }

  void LowStateHandler(const void *message) {
    const auto &msg = *reinterpret_cast<const unitree_go::msg::dds_::LowState_ *>(message);
    const Sample s = FromLowState(msg, MonotonicNow(), opt_.estop_bit_mask);
    std::lock_guard<std::mutex> lock(mu_);
    samples_.push_back(s);
    last_sample_ = s;
    has_lowstate_ = true;
  }

  void PollCommands() {
    char buf[256];
    while (true) {
      sockaddr_in src{};
      socklen_t src_len = sizeof(src);
      const ssize_t n =
          recvfrom(rx_fd_, buf, sizeof(buf) - 1, 0, reinterpret_cast<sockaddr *>(&src), &src_len);
      if (n < 0) {
        if (errno == EWOULDBLOCK || errno == EAGAIN) {
          break;
        }
        continue;
      }
      buf[n] = '\0';
      Command parsed{};
      if (!ParseCommandCsv(std::string(buf), parsed)) {
        continue;
      }
      parsed.recv_ts = MonotonicNow();
      last_cmd_ = parsed;
    }
  }

  void ApplyCommand(int estop) {
    const double now = MonotonicNow();
    const bool timeout = (now - last_cmd_.recv_ts) > opt_.cmd_timeout_s;
    const bool stop = timeout || (estop > 0) || last_cmd_.stop;

    if (stop) {
      sport_client_.StopMove();
      return;
    }
    sport_client_.Move(
        static_cast<float>(last_cmd_.vx),
        static_cast<float>(last_cmd_.vy),
        static_cast<float>(last_cmd_.wz));
  }

  void SendState(const Sample &s) {
    char line[256];
    const int n = snprintf(
        line,
        sizeof(line),
        "%.6f,%.3f,%.3f,%.3f,%.4f,%d,%d",
        s.ts,
        s.temp_max_c,
        s.vpack_v,
        s.vcell_min_v,
        s.wz_actual,
        s.estop,
        s.mode);
    if (n <= 0) {
      return;
    }
    sendto(
        tx_fd_,
        line,
        static_cast<size_t>(n),
        0,
        reinterpret_cast<sockaddr *>(&tx_addr_),
        sizeof(tx_addr_));
  }

 private:
  Options opt_;
  int tx_fd_ = -1;
  int rx_fd_ = -1;
  sockaddr_in tx_addr_{};

  std::mutex mu_;
  std::vector<Sample> samples_;
  Sample last_sample_{};
  bool has_lowstate_ = false;
  Command last_cmd_{};

  unitree::robot::go2::SportClient sport_client_;
  ChannelSubscriberPtr<unitree_go::msg::dds_::LowState_> lowstate_sub_;
};

int main(int argc, char **argv) {
  Options opt;
  if (!ParseArgs(argc, argv, opt)) {
    return 1;
  }

  Go2UdpBridge bridge(opt);
  if (!bridge.Init()) {
    return 1;
  }
  bridge.Run();
  return 0;
}
