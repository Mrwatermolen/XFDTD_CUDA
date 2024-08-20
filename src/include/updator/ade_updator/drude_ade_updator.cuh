#ifndef __XFDTD_CUDA_DRUDE_ADE_UPDATOR_CUH__
#define __XFDTD_CUDA_DRUDE_ADE_UPDATOR_CUH__

#include <xfdtd/coordinate_system/coordinate_system.h>

#include <xfdtd_cuda/index_task.cuh>

#include "updator/ade_updator/ade_updator.cuh"

namespace xfdtd::cuda {

class DrudeADEMethodStorage;

class DrudeADEUpdator : public ADEUpdator {
  friend class DrudeADEUpdatorHD;

  friend class TemplateADEUpdateScheme;

 public:
  DrudeADEUpdator(IndexTask task, GridSpace* grid_space,
                  CalculationParam* calculation_param, EMF* emf,
                  DrudeADEMethodStorage* storage);

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

#endif  // __XFDTD_CUDA_DRUDE_ADE_UPDATOR_CUH__
