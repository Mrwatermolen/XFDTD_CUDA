#ifndef __XFDTD_CUDA_TIME_PARAM_CUH__
#define __XFDTD_CUDA_TIME_PARAM_CUH__

#include <xfdtd/common/type_define.h>

#include <xfdtd_cuda/common.cuh>

namespace xfdtd {

namespace cuda {

class TimeParam {
  friend class TimeParamHD;

 public:
  TimeParam() = default;

  XFDTD_CUDA_DUAL TimeParam(Real dt, Index start_time_step, Index size,
                            Index current_time_step);

  XFDTD_CUDA_DUAL auto dt() const { return _dt; }

  XFDTD_CUDA_DUAL auto startTimeStep() const { return _start_time_step; }

  XFDTD_CUDA_DUAL auto size() const { return _size; }

  XFDTD_CUDA_DUAL auto currentTimeStep() const { return _current_time_step; }

  XFDTD_CUDA_DUAL auto endTimeStep() const { return _start_time_step + _size; }

  XFDTD_CUDA_DUAL auto remainingTimeStep() const {
    return _start_time_step + _size - _current_time_step;
  }

  XFDTD_CUDA_DUAL auto nextStep() { ++_current_time_step; }

  /**
   * @brief [1,2,3,...,size] * dt
   */
  XFDTD_CUDA_DUAL auto eTime() const -> Tensor<Real, 1>;

  /**
   * @brief [0.5,1.5,2.5,...,size-0.5] * dt
   */
  XFDTD_CUDA_DUAL auto hTime() const -> Tensor<Real, 1>;

 private:
  Real _dt{};
  Index _start_time_step{}, _size{}, _current_time_step{};
};

XFDTD_CUDA_GLOBAL auto __kernelCheckTimeParam(TimeParam *time_param) -> void;

}  // namespace cuda

}  // namespace xfdtd

#endif  // __XFDTD_CUDA_TIME_PARAM_CUH__
