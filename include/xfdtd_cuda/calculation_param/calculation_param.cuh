#ifndef __XFDTD_CUDA_CALCULATION_PARAM_MATERIAL_PARAM_CUH__
#define __XFDTD_CUDA_CALCULATION_PARAM_MATERIAL_PARAM_CUH__

#include <xfdtd_cuda/calculation_param/fdtd_coefficient.cuh>
#include <xfdtd_cuda/calculation_param/material_param.cuh>
#include <xfdtd_cuda/calculation_param/time_param.cuh>

namespace xfdtd::cuda {

class CalculationParam {
  friend class CalculationParamHD;

 public:
  CalculationParam(TimeParam* time_param, MaterialParam* material_param,
                   FDTDCoefficient* fdtd_coefficient)
      : _time_param{time_param},
        _material_param{material_param},
        _fdtd_coefficient{fdtd_coefficient} {}

  XFDTD_CUDA_DEVICE auto timeParam() const -> const TimeParam* {
    return _time_param;
  }

  XFDTD_CUDA_DEVICE auto materialParam() const -> const MaterialParam* {
    return _material_param;
  }

  XFDTD_CUDA_DEVICE auto fdtdCoefficient() const -> const FDTDCoefficient* {
    return _fdtd_coefficient;
  }

  XFDTD_CUDA_DEVICE auto timeParam() -> TimeParam* { return _time_param; }

  XFDTD_CUDA_DEVICE auto materialParam() -> MaterialParam* {
    return _material_param;
  }

  XFDTD_CUDA_DEVICE auto fdtdCoefficient() -> FDTDCoefficient* {
    return _fdtd_coefficient;
  }

 private:
  TimeParam* _time_param{};
  MaterialParam* _material_param{};
  FDTDCoefficient* _fdtd_coefficient{};
};

}  // namespace xfdtd::cuda

#endif  // __XFDTD_CUDA_CALCULATION_PARAM_MATERIAL_PARAM_CUH__
