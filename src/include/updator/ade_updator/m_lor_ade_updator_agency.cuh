#ifndef __XFDTD_CUDA_M_LOR_ADE_UPDATOR_AGENCY_CUH__
#define __XFDTD_CUDA_M_LOR_ADE_UPDATOR_AGENCY_CUH__

#include "updator/ade_updator/m_lor_ade_updator.cuh"
#include "updator/updator_agency.cuh"

namespace xfdtd::cuda {

class MLorentzADEUpdatorAgency : public UpdatorAgency {
 public:
  explicit MLorentzADEUpdatorAgency(MLorentzADEUpdator* updator)
      : _updator{updator} {}

  auto updateE(dim3 grid_dim, dim3 block_dim) -> void override;

  auto updateH(dim3 grid_dim, dim3 block_dim) -> void override;

 private:
  MLorentzADEUpdator* _updator{};
};

}  // namespace xfdtd::cuda

#endif  // __XFDTD_CUDA_M_LOR_ADE_UPDATOR_AGENCY_CUH__
