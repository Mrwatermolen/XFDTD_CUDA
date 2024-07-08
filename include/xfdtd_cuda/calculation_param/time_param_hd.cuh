#ifndef __XFDTD_CUDA_TIME_PARAM_HD_CUH__
#define __XFDTD_CUDA_TIME_PARAM_HD_CUH__

#include <xfdtd/common/type_define.h>

#include <xfdtd_cuda/host_device_carrier.cuh>

namespace xfdtd {
class TimeParam;
}

namespace xfdtd::cuda {

class TimeParam;

class TimeParamHD
    : public HostDeviceCarrier<xfdtd::TimeParam, xfdtd::cuda::TimeParam> {
 public:
  using Host = xfdtd::TimeParam;
  using Device = xfdtd::cuda::TimeParam;

 public:
  using HostDeviceCarrier<xfdtd::TimeParam,
                          xfdtd::cuda::TimeParam>::HostDeviceCarrier;

  ~TimeParamHD() override;

  auto copyHostToDevice() -> void override;

  auto copyDeviceToHost() -> void override;

  auto releaseDevice() -> void override;

  auto nextStepInDevice() -> void;
};

}  // namespace xfdtd::cuda

#endif  // __XFDTD_CUDA_TIME_PARAM_HD_CUH__
