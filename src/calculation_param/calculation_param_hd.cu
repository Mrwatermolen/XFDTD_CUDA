#include <xfdtd/calculation_param/calculation_param.h>

#include <xfdtd_cuda/calculation_param/calculation_param.cuh>
#include <xfdtd_cuda/calculation_param/calculation_param_hd.cuh>
#include <xfdtd_cuda/calculation_param/fdtd_coefficient_hd.cuh>
#include <xfdtd_cuda/calculation_param/material_param_hd.cuh>
#include <xfdtd_cuda/calculation_param/time_param_hd.cuh>

namespace xfdtd::cuda {

CalculationParamHD::CalculationParamHD(Host* calculation_param)
    : HostDeviceCarrier{calculation_param},
      _time_param_hd{
          std::make_shared<TimeParamHD>(calculation_param->timeParam().get())},
      _fdtd_coefficient_hd{std::make_shared<FDTDCoefficientHD>(
          calculation_param->fdtdCoefficient().get())} {}

CalculationParamHD::~CalculationParamHD() { releaseDevice(); }

auto CalculationParamHD::copyHostToDevice() -> void {
  if (host() == nullptr) {
    throw std::runtime_error(
        "CalculationParamHD::copyHostToDevice(): "
        "Host data is not initialized");
  }

  _time_param_hd->copyHostToDevice();
  _fdtd_coefficient_hd->copyHostToDevice();

  auto d =
      Device{_time_param_hd->device(), nullptr, _fdtd_coefficient_hd->device()};

  copyToDevice(&d);
}

auto CalculationParamHD::copyDeviceToHost() -> void {
  if (host() == nullptr) {
    throw std::runtime_error(
        "CalculationParamHD::copyDeviceToHost(): "
        "Host data is not initialized");
  }

  _time_param_hd->copyDeviceToHost();
  _fdtd_coefficient_hd->copyDeviceToHost();
}

auto CalculationParamHD::releaseDevice() -> void {
  _time_param_hd->releaseDevice();
  _fdtd_coefficient_hd->releaseDevice();

  releaseBaseDevice();
}

}  // namespace xfdtd::cuda
