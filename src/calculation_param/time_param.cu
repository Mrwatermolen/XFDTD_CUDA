#include <xfdtd_cuda/calculation_param/time_param.cuh>
#include <xfdtd_cuda/tensor.cuh>

namespace xfdtd::cuda {

XFDTD_CUDA_DUAL TimeParam::TimeParam(Real dt, Index start_time_step, Index size,
                                     Index current_time_step)
    : _dt{dt},
      _start_time_step{start_time_step},
      _size{size},
      _current_time_step{current_time_step} {}

XFDTD_CUDA_DUAL auto TimeParam::eTime() const -> Tensor<double, 1> {
  auto interval = dt();
  auto e_time = Tensor<Real, 1>::from_shape({size()});
  for (Index i = 0; i < _size; ++i) {
    e_time(i) = interval * (i + 1);
  }

  return e_time;
}

XFDTD_CUDA_DUAL auto TimeParam::hTime() const -> Tensor<double, 1> {
  auto interval = dt();
  auto h_time = Tensor<Real, 1>::from_shape({size()});
  for (Index i = 0; i < _size; ++i) {
    h_time(i) = (interval) * (i + 0.5);
  }

  return h_time;
}

XFDTD_CUDA_GLOBAL auto __kernelCheckTimeParam(
    xfdtd::cuda::TimeParam *time_param) -> void {
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
  for (xfdtd::Index i = 0; i < time_param->size(); ++i) {
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
