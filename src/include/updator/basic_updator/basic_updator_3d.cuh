#ifndef __XFDTD_CUDA_BASIC_UPDATOR_3D_CUH__
#define __XFDTD_CUDA_BASIC_UPDATOR_3D_CUH__

#include "updator/basic_updator/basic_updator.cuh"

namespace xfdtd::cuda {

class CalculationParam;
class EMF;

class BasicUpdator3D : public BasicUpdator {
  friend class BasicUpdator3DHD;

 public:
  using BasicUpdator::BasicUpdator;

  XFDTD_CUDA_DEVICE auto updateE() -> void;
};

}  // namespace xfdtd::cuda

#endif  // __XFDTD_CUDA_BASIC_UPDATOR_3D_CUH__
