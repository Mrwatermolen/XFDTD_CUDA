#ifndef __XFDTD_CUDA_MONITOR_AGENCY_CUH__
#define __XFDTD_CUDA_MONITOR_AGENCY_CUH__

#include <xfdtd_cuda/common.cuh>

namespace xfdtd::cuda {

class MonitorAgency {
 public:
  virtual ~MonitorAgency() = default;

  XFDTD_CUDA_HOST virtual auto update(dim3 grid_dim,
                                      dim3 block_dim) -> void = 0;
};

}  // namespace xfdtd::cuda

#endif  // __XFDTD_CUDA_MONITOR_AGENCY_CUH__
