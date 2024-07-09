#ifndef __XFDTD_CUDA_TFSF_CORRECTOR_AGENCY_CUH__
#define __XFDTD_CUDA_TFSF_CORRECTOR_AGENCY_CUH__

#include <xfdtd/coordinate_system/coordinate_system.h>

#include "corrector/corrector_agency.cuh"

namespace xfdtd::cuda {

class TFSFCorrector;

class TFSFCorrectorAgency : public CorrectorAgency {
 public:
  TFSFCorrectorAgency(TFSFCorrector* device) : _device{device} {}

  auto device() -> TFSFCorrector*;

  auto device() const -> const TFSFCorrector*;

  auto setDevice(TFSFCorrector* device) -> void;

 private:
  TFSFCorrector* _device;
};

class TFSFCorrector2DAgency : public TFSFCorrectorAgency {
 public:
  using TFSFCorrectorAgency::TFSFCorrectorAgency;

  auto correctE(dim3 grid_size, dim3 block_size) -> void override;

  auto correctH(dim3 grid_size, dim3 block_size) -> void override;
};

class TFSFCorrector3DAgency : public TFSFCorrectorAgency {
 public:
  using TFSFCorrectorAgency::TFSFCorrectorAgency;

  auto correctE(dim3 grid_size, dim3 block_size) -> void override;

  auto correctH(dim3 grid_size, dim3 block_size) -> void override;
};

}  // namespace xfdtd::cuda

#endif  // __XFDTD_CUDA_TFSF_CORRECTOR_AGENCY_CUH__
