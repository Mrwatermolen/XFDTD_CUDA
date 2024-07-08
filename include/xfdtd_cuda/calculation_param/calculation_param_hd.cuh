#ifndef __XFDTD_CUDA_CALCULATION_PARAM_HD__
#define __XFDTD_CUDA_CALCULATION_PARAM_HD__

#include <memory>
#include <xfdtd_cuda/host_device_carrier.cuh>

namespace xfdtd {

class CalculationParam;

}

namespace xfdtd::cuda {

class TimeParamHD;
class FDTDCoefficientHD;
class MaterialParamHD;
class CalculationParam;

class CalculationParamHD
    : public HostDeviceCarrier<xfdtd::CalculationParam,
                               xfdtd::cuda::CalculationParam> {
  using Host = xfdtd::CalculationParam;
  using Device = xfdtd::cuda::CalculationParam;

 public:
  CalculationParamHD(Host* calculation_param);

  ~CalculationParamHD() override;

  auto copyHostToDevice() -> void override;

  auto copyDeviceToHost() -> void override;

  auto releaseDevice() -> void override;

  auto timeParamHD() { return _time_param_hd; }

  auto fdtdCoefficientHD() { return _fdtd_coefficient_hd; }

  auto timeParamHD() const { return _time_param_hd; }

  auto fdtdCoefficientHD() const { return _fdtd_coefficient_hd; }

 private:
  std::shared_ptr<TimeParamHD> _time_param_hd{};
  std::shared_ptr<FDTDCoefficientHD> _fdtd_coefficient_hd{};
};

}  // namespace xfdtd::cuda

#endif  //__XFDTD_CUDA_CALCULATION_PARAM_HD__
