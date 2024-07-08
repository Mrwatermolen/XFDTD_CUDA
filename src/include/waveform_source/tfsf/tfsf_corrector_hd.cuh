#ifndef __XFDTD_CUDA_TFSF_CORRECTOR_HD_CUH__
#define __XFDTD_CUDA_TFSF_CORRECTOR_HD_CUH__

#include <xfdtd_cuda/host_device_carrier.cuh>
#include <xfdtd_cuda/index_task.cuh>
#include <xfdtd_cuda/tensor_hd.cuh>

#include "waveform_source/tfsf/tfsf_corrector_agency.cuh"

namespace xfdtd {

class TFSF;

}

namespace xfdtd::cuda {

class CorrectorAgency;
class CalculationParam;
class EMF;
class TFSFCorrector;
class TFSFCorrectorAgency;

class TFSFCorrectorHD
    : public HostDeviceCarrier<xfdtd::TFSF, xfdtd::cuda::TFSFCorrector> {
  using Host = xfdtd::TFSF;
  using Device = xfdtd::cuda::TFSFCorrector;

 public:
  TFSFCorrectorHD(Host *host, xfdtd::cuda::CalculationParam *calculation_param,
                  xfdtd::cuda::EMF *emf);

  ~TFSFCorrectorHD() override;

  auto copyHostToDevice() -> void override;

  auto copyDeviceToHost() -> void override;

  auto releaseDevice() -> void override;

  auto getTFSFCorrector2DAgency() -> CorrectorAgency *;

 private:
  xfdtd::cuda::CalculationParam *_calculation_param;
  xfdtd::cuda::EMF *_emf;
  IndexTask _total_task{};
  TensorHD<Real, 1> _projection_x_int, _projection_y_int, _projection_z_int;
  TensorHD<Real, 1> _projection_x_half, _projection_y_half, _projection_z_half;
  TensorHD<Real, 2> _e_inc, _h_inc;
  std::unique_ptr<TFSFCorrectorAgency> _corrector_agency{};
};

}  // namespace xfdtd::cuda

#endif  // __XFDTD_CUDA_TFSF_CORRECTOR_HD_CUH__
