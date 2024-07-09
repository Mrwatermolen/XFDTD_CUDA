#ifndef __XFDTD_CUDA_BASIC_UPDATOR_3D_AGENCY_CUH__
#define __XFDTD_CUDA_BASIC_UPDATOR_3D_AGENCY_CUH__

#include "updator/updator_agency.cuh"

namespace xfdtd::cuda {

class BasicUpdator3D;

class BasicUpdator3DAgency : public UpdatorAgency {
 public:

  auto updateE(dim3 grid_size, dim3 block_size) -> void override;

  auto updateH(dim3 grid_size, dim3 block_size) -> void override;

  auto setDevice(BasicUpdator3D* updator) -> void;

 private:
  BasicUpdator3D* _updator{};
};

}  // namespace xfdtd::cuda

#endif  // __XFDTD_CUDA_BASIC_UPDATOR_3D_AGENCY_CUH__
