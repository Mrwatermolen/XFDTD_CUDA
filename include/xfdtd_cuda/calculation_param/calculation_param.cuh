#ifndef __XFDTD_CUDA_CALCULATION_PARAM_MATERIAL_PARAM_CUH__
#define __XFDTD_CUDA_CALCULATION_PARAM_MATERIAL_PARAM_CUH__

#include <xfdtd/calculation_param/calculation_param.h>

#include <xfdtd_cuda/calculation_param/fdtd_coefficient_hd.cuh>
#include <xfdtd_cuda/calculation_param/material_param_hd.cuh>
#include <xfdtd_cuda/calculation_param/time_param_hd.cuh>
#include <xfdtd_cuda/common.cuh>

namespace xfdtd {

namespace cuda {

class CalculationParam {
  friend class CalculationParamHD;

 public:
  CalculationParam() = default;

  CalculationParam(TimeParam* time_param, MaterialParam* material_param,
                   FDTDCoefficient* fdtd_coefficient)
      : _time_param{time_param},
        _material_param{material_param},
        _fdtd_coefficient{fdtd_coefficient} {}

  XFDTD_CUDA_DUAL auto timeParam() const -> const TimeParam* {
    return _time_param;
  }

  XFDTD_CUDA_DUAL auto materialParam() const -> const MaterialParam* {
    return _material_param;
  }

  XFDTD_CUDA_DUAL auto fdtdCoefficient() const -> const FDTDCoefficient* {
    return _fdtd_coefficient;
  }

  XFDTD_CUDA_DUAL auto timeParam() -> TimeParam* { return _time_param; }

  XFDTD_CUDA_DUAL auto materialParam() -> MaterialParam* {
    return _material_param;
  }

  XFDTD_CUDA_DUAL auto fdtdCoefficient() -> FDTDCoefficient* {
    return _fdtd_coefficient;
  }

  XFDTD_CUDA_DUAL auto setTimeParam(TimeParam* time_param) {
    _time_param = time_param;
  }

  XFDTD_CUDA_DUAL auto setMaterialParam(MaterialParam* material_param) {
    _material_param = material_param;
  }

  XFDTD_CUDA_DUAL auto setFDTDCoefficient(FDTDCoefficient* fdtd_coefficient) {
    _fdtd_coefficient = fdtd_coefficient;
  }

 private:
  TimeParam* _time_param{};
  MaterialParam* _material_param{};
  FDTDCoefficient* _fdtd_coefficient{};
};

class CalculationParamHD {
 public:
  using HostTimeParam = xfdtd::TimeParam;
  using HostMaterialParam = xfdtd::MaterialParam;
  using HostFDTDCoefficient = xfdtd::FDTDUpdateCoefficient;
  using Host = xfdtd::CalculationParam;
  using Device = xfdtd::cuda::CalculationParam;

 public:
  CalculationParamHD(Host* host);

  CalculationParamHD(const CalculationParamHD&) = delete;

  CalculationParamHD(CalculationParamHD&&) noexcept = delete;

  ~CalculationParamHD();

  CalculationParamHD& operator=(const CalculationParamHD&) = delete;

  auto operator=(CalculationParamHD&&) noexcept -> CalculationParamHD& = delete;

  auto copyHostToDevice() -> void;

  auto copyDeviceToHost() -> void;

  auto releaseDevice() -> void;

 private:
  Host* _host{};
  Device* _device{};
  TimeParamHD _time_param_hd;
  FDTDCoefficientHD _fdtd_coefficient_hd;
};

}  // namespace cuda

}  // namespace xfdtd

#endif  // __XFDTD_CUDA_CALCULATION_PARAM_MATERIAL_PARAM_CUH__
