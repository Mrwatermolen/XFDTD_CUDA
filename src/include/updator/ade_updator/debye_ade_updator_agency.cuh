#ifndef __XFDTD_CUDA_DEBYE_ADE_UPDATOR_AGENCY_CUH__
#define __XFDTD_CUDA_DEBYE_ADE_UPDATOR_AGENCY_CUH__

#include "updator/ade_updator/debye_ade_updator.cuh"
#include "updator/updator_agency.cuh"

namespace xfdtd::cuda {

class DebeyeADEUpdatorAgency : public UpdatorAgency {
 public:
  auto updateE(dim3 grid_size, dim3 block_size) -> void override;

  auto updateH(dim3 grid_size, dim3 block_size) -> void override;

  auto setDevice(DebyeADEUpdator* updator) -> void;

 private:
  DebyeADEUpdator* _updator{};
};

}  // namespace xfdtd::cuda

#endif  // __XFDTD_CUDA_DEBYE_ADE_UPDATOR_AGENCY_CUH__
