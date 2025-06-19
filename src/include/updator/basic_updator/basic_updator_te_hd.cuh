#ifndef __XFDTD_CUDA_BASIC_UPDATOR_TE_HD_CUH__
#define __XFDTD_CUDA_BASIC_UPDATOR_TE_HD_CUH__

#include "updator/basic_updator/basic_updator_hd.cuh"
#include "updator/basic_updator/basic_updator_te_agency.cuh"

namespace xfdtd::cuda {

class BasicUpdatorTEHD : public BasicUpdatorHD {
 public:
  using BasicUpdatorHD::BasicUpdatorHD;

  ~BasicUpdatorTEHD() override;

  auto getUpdatorAgency() -> UpdatorAgency* override;

 private:
  std::unique_ptr<BasicUpdatorTEAgency> _updator_agency;
};

}  // namespace xfdtd::cuda

#endif  // __XFDTD_CUDA_BASIC_UPDATOR_TE_HD_CUH__
