#ifndef __XFDTD_CUDA_BASIC_UPDATOR_TE_CUH__
#define __XFDTD_CUDA_BASIC_UPDATOR_TE_CUH__

#include <xfdtd_cuda/common.cuh>

#include "updator/basic_updator/basic_updator.cuh"

namespace xfdtd::cuda {

class BasicUpdatorTE : public BasicUpdator {
  friend class BasicUpdatorTEHD;

 public:
  using BasicUpdator::BasicUpdator;

  XFDTD_CUDA_DEVICE auto updateE() -> void;
};

}  // namespace xfdtd::cuda

#endif  // __XFDTD_CUDA_BASIC_UPDATOR_TE_CUH__
