#ifndef __XFDTD_CUDA_UPDATOR_AGENCY_CUH__
#define __XFDTD_CUDA_UPDATOR_AGENCY_CUH__

namespace xfdtd::cuda {

class UpdatorAgency {
 public:
  virtual ~UpdatorAgency() = default;

  virtual auto updateH(dim3 grid_size, dim3 block_size) -> void = 0;

  virtual auto updateE(dim3 grid_size, dim3 block_size) -> void = 0;
};

}  // namespace xfdtd::cuda

#endif  // __XFDTD_CUDA_UPDATOR_AGENCY_CUH__
