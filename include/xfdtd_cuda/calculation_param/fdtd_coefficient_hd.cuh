#ifndef __XFDTD_CUDA_FDTD_COEFFICIENT_HD_CUH__
#define __XFDTD_CUDA_FDTD_COEFFICIENT_HD_CUH__

#include <xfdtd/common/type_define.h>

#include <xfdtd_cuda/host_device_carrier.cuh>
#include <xfdtd_cuda/tensor.cuh>
#include <xfdtd_cuda/tensor_hd.cuh>
#include <xfdtd_cuda/tensor_texture_ref.cuh>

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

  explicit FDTDCoefficientHD(Host *host);

  FDTDCoefficientHD(const FDTDCoefficientHD &) = delete;
  auto operator=(const FDTDCoefficientHD &) -> FDTDCoefficientHD & = delete;
  FDTDCoefficientHD(FDTDCoefficientHD &&) = delete;
  auto operator=(FDTDCoefficientHD &&) -> FDTDCoefficientHD & = delete;

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

  TensorTextureRef<Real, 3> _cexe_tex, _cexhy_tex, _cexhz_tex, _ceye_tex,
      _ceyhz_tex, _ceyhx_tex, _ceze_tex, _cezhx_tex, _cezhy_tex;
  TensorTextureRef<Real, 3> _chxh_tex, _chxey_tex, _chxez_tex, _chyh_tex,
      _chyez_tex, _chyex_tex, _chzh_tex, _chzex_tex, _chzey_tex;
};

}  // namespace xfdtd::cuda

#endif  // __XFDTD_CUDA_FDTD_COEFFICIENT_HD_CUH__
