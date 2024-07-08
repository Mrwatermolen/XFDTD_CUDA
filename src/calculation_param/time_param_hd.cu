#include <xfdtd/calculation_param/time_param.h>

#include <xfdtd_cuda/calculation_param/time_param.cuh>
#include <xfdtd_cuda/calculation_param/time_param_hd.cuh>
#include <xfdtd_cuda/common.cuh>
#include <xfdtd_cuda/tensor_hd.cuh>

namespace xfdtd::cuda {

XFDTD_CUDA_GLOBAL auto __nextStep(xfdtd::cuda::TimeParam* time_param) -> void {
  time_param->nextStep();
}

TimeParamHD::~TimeParamHD() { releaseDevice(); }

auto TimeParamHD::copyHostToDevice() -> void {
  if (host() == nullptr) {
    throw std::runtime_error("TimeParamHD::copyHostToDevice()");
  }

  auto d = Device{};
  d._dt = host()->dt();
  d._start_time_step = host()->startTimeStep();
  d._size = host()->size();
  d._current_time_step = host()->currentTimeStep();

  copyToDevice(&d);
}

auto TimeParamHD::copyDeviceToHost() -> void {
  auto d = Device{};
  copyToHost(&d);

  if (d._current_time_step != host()->currentTimeStep()) {
    std::cout << "TimeParamHD::copyDeviceToHost(): current_time_step is not equal\n";
  }

  // while (d._current_time_step != host()->currentTimeStep()) {
  //   host()->nextStep();
  // }
}

auto TimeParamHD::releaseDevice() -> void { releaseBaseDevice(); }

auto TimeParamHD::nextStepInDevice() -> void { __nextStep<<<1, 1>>>(device()); }

}  // namespace xfdtd::cuda
