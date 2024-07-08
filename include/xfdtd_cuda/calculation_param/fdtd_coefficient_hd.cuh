#ifndef __XFDTD_CUDA_FDTD_COEFFICIENT_HD_CUH__
#define __XFDTD_CUDA_FDTD_COEFFICIENT_HD_CUH__

#include <xfdtd/common/type_define.h>

#include <xfdtd_cuda/host_device_carrier.cuh>
#include <xfdtd_cuda/tensor.cuh>
#include <xfdtd_cuda/tensor_hd.cuh>

namespace xfdtd {

class FDTDUpdateCoefficient;

}  // namespace xfdtd

namespace xfdtd::cuda {

class FDTDCoefficient;

class FDTDCoefficientHD
    : public HostDeviceCarrier<xfdtd::FDTDUpdateCoefficient,
                               xfdtd::cuda::FDTDCoefficient> {
 public:
  using Host = xfdtd::FDTDUpdateCoefficient;
  using Device = xfdtd::cuda::FDTDCoefficient;

  FDTDCoefficientHD(Host *host);

  ~FDTDCoefficientHD() override;

 public:
  auto copyHostToDevice() -> void override;

  auto copyDeviceToHost() -> void override;

  auto releaseDevice() -> void override;

 private:
  TensorHD<Real, 3> _cexe_hd, _cexhy_hd, _cexhz_hd, _ceye_hd, _ceyhz_hd,
      _ceyhx_hd, _ceze_hd, _cezhx_hd, _cezhy_hd;
  TensorHD<Real, 3> _chxh_hd, _chxey_hd, _chxez_hd, _chyh_hd, _chyez_hd,
      _chyex_hd, _chzh_hd, _chzex_hd, _chzey_hd;
};

}  // namespace xfdtd::cuda

#endif  // __XFDTD_CUDA_FDTD_COEFFICIENT_HD_CUH__
