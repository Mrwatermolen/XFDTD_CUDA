#ifndef __XFDTD_CUDA_CALCULATION_PARAM_HD__
#define __XFDTD_CUDA_CALCULATION_PARAM_HD__

#include <xfdtd/calculation_param/calculation_param.h>

#include <memory>
#include <xfdtd_cuda/calculation_param/calculation_param.cuh>
#include <xfdtd_cuda/calculation_param/fdtd_coefficient_hd.cuh>
#include <xfdtd_cuda/calculation_param/material_param_hd.cuh>
#include <xfdtd_cuda/calculation_param/time_param_hd.cuh>
#include <xfdtd_cuda/host_device_carrier.cuh>

namespace xfdtd::cuda {

class CalculationParamHD
    : public HostDeviceCarrier<xfdtd::CalculationParam,
                               xfdtd::cuda::CalculationParam> {
  using Host = xfdtd::CalculationParam;
  using Device = xfdtd::cuda::CalculationParam;

 public:
  CalculationParamHD(Host* calculation_param)
      : HostDeviceCarrier{calculation_param},
        _time_param_hd{std::make_shared<TimeParamHD>(
            calculation_param->timeParam().get())},
        _fdtd_coefficient_hd{std::make_shared<FDTDCoefficientHD>(
            calculation_param->fdtdCoefficient().get())} {}

  ~CalculationParamHD() override { releaseDevice(); }

  auto copyHostToDevice() -> void override {
    if (host() == nullptr) {
      throw std::runtime_error(
          "CalculationParamHD::copyHostToDevice(): "
          "Host data is not initialized");
    }

    _time_param_hd->copyHostToDevice();
    _fdtd_coefficient_hd->copyHostToDevice();

    auto d = Device{_time_param_hd->device(), nullptr,
                    _fdtd_coefficient_hd->device()};

    copyToDevice(&d);
  }

  auto copyDeviceToHost() -> void override {
    if (host() == nullptr) {
      throw std::runtime_error(
          "CalculationParamHD::copyDeviceToHost(): "
          "Host data is not initialized");
    }

    _time_param_hd->copyDeviceToHost();
    _fdtd_coefficient_hd->copyDeviceToHost();
  }

  auto releaseDevice() -> void override {
    _time_param_hd->releaseDevice();
    _fdtd_coefficient_hd->releaseDevice();

    releaseBaseDevice();
  }

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
