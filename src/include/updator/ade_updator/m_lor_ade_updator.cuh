#ifndef __XFDTD_CUDA_M_LOR_ADE_UPDATOR_CUH__
#define __XFDTD_CUDA_M_LOR_ADE_UPDATOR_CUH__

#include "updator/ade_updator/ade_updator.cuh"

namespace xfdtd::cuda {

class MLorentzADEUpdator : public ADEUpdator {
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
                                         Index k) -> void;

  template <xfdtd::Axis::XYZ xyz>
  XFDTD_CUDA_DEVICE auto calculateJSum(Index i, Index j, Index k) -> Real;

  template <xfdtd::Axis::XYZ xyz>
  XFDTD_CUDA_DEVICE auto ePrevious(Index i, Index j, Index k) const -> Real;

  XFDTD_CUDA_DEVICE auto coeffEPrev(Index i, Index j, Index k) const -> Real;
};

}  // namespace xfdtd::cuda

#endif  // __XFDTD_CUDA_M_LOR_ADE_UPDATOR_CUH__
