#ifndef __XFDTD_CUDA_MATERIAL_PARAM_HD_CUH__
#define __XFDTD_CUDA_MATERIAL_PARAM_HD_CUH__

#include <xfdtd/calculation_param/calculation_param.h>
#include <xfdtd_cuda/host_device_carrier.cuh>

#include <xfdtd_cuda/calculation_param/material_param.cuh>
#include <xfdtd_cuda/common.cuh>

namespace xfdtd::cuda {

class MaterialParamHD : public HostDeviceCarrier<xfdtd::MaterialParam,
                                                 xfdtd::cuda::MaterialParam> {

public:
  using HostDeviceCarrier<xfdtd::MaterialParam,
                          xfdtd::cuda::MaterialParam>::HostDeviceCarrier;

  ~MaterialParamHD() override {}

  auto copyHostToDevice() -> void override {}

  auto copyDeviceToHost() -> void override {}

  auto releaseDevice() -> void override {}
};

} // namespace xfdtd::cuda

#endif // __XFDTD_CUDA_MATERIAL_PARAM_HD_CUH__
