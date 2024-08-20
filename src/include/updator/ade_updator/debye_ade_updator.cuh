#ifndef __XFDTD_CUDA_DEBYE_ADE_UPDATOR_CUH__
#define __XFDTD_CUDA_DEBYE_ADE_UPDATOR_CUH__

#include <xfdtd_cuda/common.cuh>

#include "updator/ade_updator/ade_updator.cuh"

namespace xfdtd::cuda {

class DebyeADEUpdator : public ADEUpdator {
  friend class TemplateADEUpdateScheme;

 public:
  using ADEUpdator::ADEUpdator;

  XFDTD_CUDA_DEVICE auto updateE() -> void;

 private:
  template <xfdtd::Axis::XYZ xyz>
  XFDTD_CUDA_DEVICE auto updateJ(Index i, Index j, Index k, Real e_next,
                                 Real e_cur) -> void;

  template <xfdtd::Axis::XYZ xyz>
  XFDTD_CUDA_DEVICE auto recordEPrevious(Real e, Index i, Index j,
                                         Index k) -> void {}

  template <xfdtd::Axis::XYZ xyz>
  XFDTD_CUDA_DEVICE auto calculateJSum(Index i, Index j, Index k) -> Real;

  template <xfdtd::Axis::XYZ xyz>
  XFDTD_CUDA_DEVICE auto ePrevious(Index i, Index j, Index k) const -> Real {
    return 0.0;
  }

  XFDTD_CUDA_DEVICE auto coeffEPrev(Index i, Index j, Index k) const -> Real {
    return 0.0;
  }
};

}  // namespace xfdtd::cuda

#endif  // __XFDTD_CUDA_DEBYE_ADE_UPDATOR_CUH__
