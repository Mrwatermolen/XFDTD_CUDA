#ifndef __XFDTD_CUDA_DRUDE_ADE_METHOD_HD_CUH__
#define __XFDTD_CUDA_DRUDE_ADE_METHOD_HD_CUH__

#include <xfdtd/material/ade_method/drude_ade_method.h>

#include <xfdtd_cuda/host_device_carrier.cuh>

#include "material/ade_method/ade_method_hd.cuh"
#include "material/ade_method/drude_ade_method.cuh"

namespace xfdtd::cuda {

class DrudeADEMethodStorageHD : public ADEMethodStorageHD {
  using Host = xfdtd::DrudeADEMethodStorage;
  using Device = xfdtd::cuda::DrudeADEMethodStorage;

 public:
  explicit DrudeADEMethodStorageHD(Host* host);

  ~DrudeADEMethodStorageHD() override;

  auto copyHostToDevice() -> void override;

  auto copyDeviceToHost() -> void override;

  auto releaseDevice() -> void override;
};

}  // namespace xfdtd::cuda

#endif  // __XFDTD_CUDA_DRUDE_ADE_METHOD_HD_CUH__
