#ifndef __XFDTD_CUDA_BASIC_UPDATOR_TE_AGENCY__
#define __XFDTD_CUDA_BASIC_UPDATOR_TE_AGENCY__

#include "updator/updator_agency.cuh"

namespace xfdtd::cuda {

class BasicUpdatorTE;

class BasicUpdatorTEAgency : public UpdatorAgency {
 public:
  auto updateH(dim3 grid_size, dim3 block_size) -> void override;

  auto updateE(dim3 grid_size, dim3 block_size) -> void override;

  auto setDevice(BasicUpdatorTE* updator) -> void;

 private:
  BasicUpdatorTE* _updator{};
};

}  // namespace xfdtd::cuda

#endif  // __XFDTD_CUDA_BASIC_UPDATOR_TE_AGENCY__
