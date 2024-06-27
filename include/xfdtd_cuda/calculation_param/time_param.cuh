#ifndef __XFDTD_CUDA_TIME_PARAM_CUH__
#define __XFDTD_CUDA_TIME_PARAM_CUH__

#include <xfdtd/common/type_define.h>

#include <xfdtd_cuda/common.cuh>
#include <xfdtd_cuda/tensor.cuh>

namespace xfdtd {

namespace cuda {

class TimeParam {
  friend class TimeParamHD;

public:
  TimeParam() = default;

  XFDTD_CUDA_DUAL TimeParam(Real dt, Index start_time_step, Index size,
                            Index current_time_step)
      : _dt{dt}, _start_time_step{start_time_step}, _size{size},
        _current_time_step{current_time_step} {}

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
  XFDTD_CUDA_DUAL auto eTime() const {
    auto interval = dt();
    auto e_time = Tensor<Real, 1>::from_shape({size()});
    for (Index i = 0; i < _size; ++i) {
      e_time(i) = interval * (i + 1);
    }

    return e_time;
  }

  /**
   * @brief [0.5,1.5,2.5,...,size-0.5] * dt
   */
  XFDTD_CUDA_DUAL auto hTime() const {
    auto interval = dt();
    auto h_time = Tensor<Real, 1>::from_shape({size()});
    for (Index i = 0; i < _size; ++i) {
      h_time(i) = (interval) * (i + 0.5);
    }

    return h_time;
  }

private:
  Real _dt{};
  Index _start_time_step{}, _size{}, _current_time_step{};
};

} // namespace cuda

} // namespace xfdtd

#endif // __XFDTD_CUDA_TIME_PARAM_CUH__
