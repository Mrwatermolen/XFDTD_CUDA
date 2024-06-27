#ifndef __XFDTD_CUDA_TIME_PARAM_HD_CUH__
#define __XFDTD_CUDA_TIME_PARAM_HD_CUH__

#include <xfdtd/calculation_param/time_param.h>
#include <xfdtd/common/type_define.h>

#include <xfdtd_cuda/calculation_param/time_param.cuh>
#include <xfdtd_cuda/host_device_carrier.cuh>
#include <xfdtd_cuda/tensor_hd.cuh>

namespace xfdtd::cuda {
class TimeParamHD
    : public HostDeviceCarrier<xfdtd::TimeParam, xfdtd::cuda::TimeParam> {
 public:
  using Host = xfdtd::TimeParam;
  using Device = xfdtd::cuda::TimeParam;

 public:
  using HostDeviceCarrier<xfdtd::TimeParam,
                          xfdtd::cuda::TimeParam>::HostDeviceCarrier;

  TimeParamHD(TimeParamHD &&other) noexcept
      : HostDeviceCarrier<xfdtd::TimeParam, xfdtd::cuda::TimeParam>{
            std::move(other)} {}

  auto operator=(TimeParamHD &&other) noexcept -> TimeParamHD & {
    if (this != &other) {
      HostDeviceCarrier<xfdtd::TimeParam, xfdtd::cuda::TimeParam>::operator=(
          std::move(other));
    }
    return *this;
  }

  ~TimeParamHD() override { releaseDevice(); }

  auto copyHostToDevice() -> void override {
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

  auto copyDeviceToHost() -> void override {
    auto d = Device{};
    copyToHost(&d);

    while (d._current_time_step != host()->currentTimeStep()) {
      host()->nextStep();
    }
  }

  auto releaseDevice() -> void override {}
};

XFDTD_CUDA_GLOBAL auto __kernelCheckTimeParam(TimeParam *time_param) -> void {
  if (time_param == nullptr) {
    printf("TimeParam is nullptr\n");
    return;
  }

  printf(
      "TimeParam: dt=%.5e, start_time_step=%lu, size=%lu, "
      "current_time_step=%lu, "
      "end_time_step=%lu, remaining_time_step=%lu\n",
      time_param->dt(), time_param->startTimeStep(), time_param->size(),
      time_param->currentTimeStep(), time_param->endTimeStep(),
      time_param->remainingTimeStep());

  auto e_time = time_param->eTime();
  auto h_time = time_param->hTime();
  for (Index i = 0; i < time_param->size(); ++i) {
    printf("eTime[%lu]:%.5e, hTime[%lu]:%.2e\n", i, e_time(i), i, h_time(i));
  }

  time_param->nextStep();

  printf(
      "TimeParam: dt=%.5e, start_time_step=%lu, size=%lu, "
      "current_time_step=%lu, "
      "end_time_step=%lu, remaining_time_step=%lu\n",
      time_param->dt(), time_param->startTimeStep(), time_param->size(),
      time_param->currentTimeStep(), time_param->endTimeStep(),
      time_param->remainingTimeStep());
}
}  // namespace xfdtd::cuda

#endif  // __XFDTD_CUDA_TIME_PARAM_HD_CUH__
