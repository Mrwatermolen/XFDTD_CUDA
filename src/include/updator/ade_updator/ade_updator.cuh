#ifndef __XFDTD_CUDA_ADE_UPDATOR_CUH__
#define __XFDTD_CUDA_ADE_UPDATOR_CUH__

#include <xfdtd_cuda/common.cuh>

#include "updator/basic_updator/basic_updator.cuh"

namespace xfdtd::cuda {

class ADEMethodStorage;

class ADEUpdator : public BasicUpdator {
  friend class ADEUpdatorHD;

 public:
  ADEUpdator(IndexTask task, GridSpace* grid_space,
             CalculationParam* calculation_param, EMF* emf,
             ADEMethodStorage* storage);

  XFDTD_CUDA_DEVICE auto storage() { return _ade_method_storage; }

  XFDTD_CUDA_DEVICE auto storage() const { return _ade_method_storage; }

 private:
  ADEMethodStorage* _ade_method_storage{};
};

}  // namespace xfdtd::cuda

#endif  // __XFDTD_CUDA_ADE_UPDATOR_CUH__
