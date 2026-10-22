// RobotlabLockstep: an ordering barrier for RobotLab's lockstep stepping.
//
// In lockstep the world is paused between periods and Python requests each
// burst of steps through the world control service, after processing the
// previous period (control, motor and wind commands). Commands travel as
// topic messages; the service request uses a separate connection, so under
// load Gazebo could start the burst before it had read the commands.
//
// Python therefore also publishes a grant (gz.msgs.Int64 on
// /robotlab/lockstep/grant: simulation time in ns up to which the world
// may run) after the period's commands and before the step request. The
// grant shares the commands' ordered transport connection, and gz-transport
// delivers one publisher process's messages in order on one reception
// thread, so when the grant callback runs every earlier command callback
// has run. PreUpdate (which sees the simulation time at the end of the
// step about to run) holds a running step until the grant covers it. The
// plugin must be the first system in the world so other systems'
// PreUpdate (motors, wind, wrenches) see the period's commands.
//
// Paused iterations are never held: Gazebo keeps iterating while paused,
// which lets rendered sensors finish and publish frames that Python waits
// for before it grants the next period. A grant of INT64_MAX releases the
// barrier for shutdown; while holding, a stop flag is rechecked every
// 50 ms.

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <mutex>
#include <string>

#include <gz/msgs/int64.pb.h>
#include <gz/plugin/Register.hh>
#include <gz/sim/System.hh>
#include <gz/transport/Node.hh>

namespace robotlab
{
class Lockstep : public gz::sim::System,
                 public gz::sim::ISystemConfigure,
                 public gz::sim::ISystemPreUpdate
{
public:
  ~Lockstep() override
  {
    this->stopping = true;
    this->granted.notify_all();
  }

  void Configure(const gz::sim::Entity &,
                 const std::shared_ptr<const sdf::Element> &_sdf,
                 gz::sim::EntityComponentManager &,
                 gz::sim::EventManager &) override
  {
    // ROBOTLAB_LOCKSTEP_DEBUG=1 reports grants and holds longer than 1 ms.
    const char *debugEnv = std::getenv("ROBOTLAB_LOCKSTEP_DEBUG");
    this->debug = debugEnv != nullptr && std::string(debugEnv) == "1";
    std::string topic = "/robotlab/lockstep/grant";
    if (_sdf->HasElement("topic"))
      topic = _sdf->Get<std::string>("topic");
    this->node.Subscribe(topic, &Lockstep::OnGrant, this);
  }

  void PreUpdate(const gz::sim::UpdateInfo &_info,
                 gz::sim::EntityComponentManager &) override
  {
    if (_info.paused)
      return;
    const int64_t stepEndNs =
        std::chrono::duration_cast<std::chrono::nanoseconds>(_info.simTime)
            .count();
    std::unique_lock<std::mutex> lock(this->mutex);
    const auto start = std::chrono::steady_clock::now();
    while (stepEndNs > this->grantNs && !this->stopping)
      this->granted.wait_for(lock, std::chrono::milliseconds(50));
    if (this->debug)
    {
      const auto held = std::chrono::steady_clock::now() - start;
      if (held > std::chrono::milliseconds(1))
        std::cerr << "[robotlab::Lockstep] held step to " << stepEndNs
                  << " ns for "
                  << std::chrono::duration_cast<std::chrono::microseconds>(held)
                         .count()
                  << " us (grant " << this->grantNs << ")" << std::endl;
    }
  }

private:
  void OnGrant(const gz::msgs::Int64 &_msg)
  {
    {
      std::lock_guard<std::mutex> lock(this->mutex);
      if (_msg.data() > this->grantNs)
        this->grantNs = _msg.data();
      if (this->debug)
        std::cerr << "[robotlab::Lockstep] grant " << _msg.data() << std::endl;
    }
    this->granted.notify_all();
  }

  gz::transport::Node node;
  std::mutex mutex;
  std::condition_variable granted;
  int64_t grantNs{0};
  std::atomic<bool> stopping{false};
  bool debug{false};
};
}  // namespace robotlab

GZ_ADD_PLUGIN(robotlab::Lockstep, gz::sim::System,
              robotlab::Lockstep::ISystemConfigure,
              robotlab::Lockstep::ISystemPreUpdate)
GZ_ADD_PLUGIN_ALIAS(robotlab::Lockstep, "robotlab::Lockstep")
