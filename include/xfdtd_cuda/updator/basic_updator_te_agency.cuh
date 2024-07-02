#ifndef __XFDTD_CUDA_BASIC_UPDATOR_TE_AGENCY__
#define __XFDTD_CUDA_BASIC_UPDATOR_TE_AGENCY__

#include <xfdtd_cuda/updator/basic_updator_te.cuh>
#include <xfdtd_cuda/updator/updator_agency.cuh>

namespace xfdtd::cuda {

XFDTD_CUDA_GLOBAL auto __basicUpdatorTEAgencyUpdateH(
    xfdtd::cuda::BasicUpdatorTE* updator) -> void {
  updator->updateH();
}

XFDTD_CUDA_GLOBAL auto __basicUpdatorTEAgencyUpdateE(
    xfdtd::cuda::BasicUpdatorTE* updator) -> void {
  updator->updateE();
}

class BasicUpdatorTEAgency : public UpdatorAgency {
 public:
  ~BasicUpdatorTEAgency() override = default;

  auto updateH(dim3 grid_size, dim3 block_size) -> void override {
    __basicUpdatorTEAgencyUpdateH<<<grid_size, block_size>>>(_updator);
  }

  auto updateE(dim3 grid_size, dim3 block_size) -> void override {
    __basicUpdatorTEAgencyUpdateE<<<grid_size, block_size>>>(_updator);
  }

  auto setDevice(BasicUpdatorTE* updator) -> void { _updator = updator; }

 private:
  BasicUpdatorTE* _updator{};
};

}  // namespace xfdtd::cuda

#endif  // __XFDTD_CUDA_BASIC_UPDATOR_TE_AGENCY__
