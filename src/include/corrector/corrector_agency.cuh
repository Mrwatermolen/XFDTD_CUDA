#ifndef __XFDTD_CUDA_CORRECTOR_AGENCY_CUH__
#define __XFDTD_CUDA_CORRECTOR_AGENCY_CUH__

namespace xfdtd::cuda {

class CorrectorAgency {
 public:
  virtual ~CorrectorAgency() = default;

  virtual auto correctE(dim3 grid_size, dim3 block_size) -> void = 0;

  virtual auto correctH(dim3 grid_size, dim3 block_size) -> void = 0;
};

}  // namespace xfdtd::cuda

#endif  // __XFDTD_CUDA_CORRECTOR_AGENCY_CUH__
