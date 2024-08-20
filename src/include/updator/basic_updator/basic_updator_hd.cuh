#ifndef __XFDTD_CUDA_BASIC_UPDATOR_HD_CUH__
#define __XFDTD_CUDA_BASIC_UPDATOR_HD_CUH__

#include "updator/basic_updator/basic_updator.cuh"
#include "updator/updator_hd.cuh"

namespace xfdtd::cuda {

class BasicUpdatorHD : public UpdatorHD<BasicUpdator> {
 public:
  using UpdatorHD::UpdatorHD;

  auto copyHostToDevice() -> void override {
    auto device = Device{task(), gridSpaceHD()->device(),
                         calculationParamHD()->device(), emfHD()->device()};
    this->copyToDevice(&device);
  }
};

};  // namespace xfdtd::cuda

#endif  // __XFDTD_CUDA_BASIC_UPDATOR_HD_CUH__
