#ifndef __XFDTD_CUDA_DRUDE_ADE_UPDATOR_AGENCY_CUH__
#define __XFDTD_CUDA_DRUDE_ADE_UPDATOR_AGENCY_CUH__

#include "updator/updator_agency.cuh"

namespace xfdtd::cuda {

class DrudeADEUpdator;

class DrudeADEUpdatorAgency : public UpdatorAgency {
 public:
  auto updateE(dim3 grid_size, dim3 block_size) -> void override;

  auto updateH(dim3 grid_size, dim3 block_size) -> void override;

  auto setDevice(DrudeADEUpdator* updator) -> void;

 private:
  DrudeADEUpdator* _updator{};
};

}  // namespace xfdtd::cuda

#endif  // __XFDTD_CUDA_DRUDE_ADE_UPDATOR_AGENCY_CUH__
