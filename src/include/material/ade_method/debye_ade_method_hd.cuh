#ifndef __XFDTD_CUDA_DEBYE_ADE_METHOD_HD_CUH__
#define __XFDTD_CUDA_DEBYE_ADE_METHOD_HD_CUH__

#include <xfdtd/material/ade_method/debye_ade_method.h>

#include "material/ade_method/ade_method_hd.cuh"
#include "material/ade_method/debye_ade_method.cuh"

namespace xfdtd::cuda {

class DebyeADEMethodStorageHD : public ADEMethodStorageHD {
  using Host = xfdtd::DebyeADEMethodStorage;
  using Device = xfdtd::cuda::DebyeADEMethodStorage;

 public:
  explicit DebyeADEMethodStorageHD(Host* host);

  ~DebyeADEMethodStorageHD() override;

  auto copyHostToDevice() -> void override;

  auto copyDeviceToHost() -> void override;

  auto releaseDevice() -> void override;
};

}  // namespace xfdtd::cuda

#endif  // __XFDTD_CUDA_DEBYE_ADE_METHOD_HD_CUH__
