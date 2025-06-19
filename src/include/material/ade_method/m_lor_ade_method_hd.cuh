#ifndef __XFDTD_CUDA_M_LOR_ADE_METHOD_HD_CUH__
#define __XFDTD_CUDA_M_LOR_ADE_METHOD_HD_CUH__

#include <xfdtd/material/ade_method/m_lor_ade_method.h>

#include "material/ade_method/ade_method_hd.cuh"
#include "material/ade_method/m_lor_ade_method.cuh"

namespace xfdtd::cuda {

class MLorentzADEMethodStorageHD : public ADEMethodStorageHD {
  using Host = xfdtd::MLorentzADEMethodStorage;
  using Device = xfdtd::cuda::MLorentzADEMethodStorage;

 public:
  explicit MLorentzADEMethodStorageHD(Host* host);

  ~MLorentzADEMethodStorageHD() override;

  auto copyHostToDevice() -> void override;

  auto copyDeviceToHost() -> void override;

  auto releaseDevice() -> void override;
};

}  // namespace xfdtd::cuda

#endif  // __XFDTD_CUDA_M_LOR_ADE_METHOD_HD_CUH__
