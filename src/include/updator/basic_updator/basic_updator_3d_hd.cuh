#ifndef __XFDTD_CUDA_BASIC_UPDATOR_3D_HD_CUH__
#define __XFDTD_CUDA_BASIC_UPDATOR_3D_HD_CUH__

#include "updator/basic_updator/basic_updator_3d_agency.cuh"
#include "updator/basic_updator/basic_updator_hd.cuh"

namespace xfdtd::cuda {

class BasicUpdator3DHD : public BasicUpdatorHD {
 public:
  using BasicUpdatorHD::BasicUpdatorHD;

  ~BasicUpdator3DHD() override;

  auto getUpdatorAgency() -> UpdatorAgency* override;

 private:
  std::unique_ptr<BasicUpdator3DAgency> _updator_agency;
};

};  // namespace xfdtd::cuda

#endif  // __XFDTD_CUDA_BASIC_UPDATOR_3D_HD_CUH__
