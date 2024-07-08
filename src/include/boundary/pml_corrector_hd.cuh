#ifndef __XFDTD_CUDA_PML_CORRECTOR_HD_CUH__
#define __XFDTD_CUDA_PML_CORRECTOR_HD_CUH__

#include <xfdtd/boundary/pml.h>
#include <xfdtd/common/type_define.h>
#include <xfdtd/coordinate_system/coordinate_system.h>

#include <memory>

#include "boundary/pml_corrector.cuh"
#include "xfdtd_cuda/electromagnetic_field/electromagnetic_field_hd.cuh"
#include "xfdtd_cuda/host_device_carrier.cuh"
#include "xfdtd_cuda/index_task.cuh"
#include "xfdtd_cuda/tensor_hd.cuh"

namespace xfdtd::cuda {

template <Axis::XYZ xyz>
class PMLCorrectorAgency;

class CorrectorAgency;

template <Axis::XYZ xyz>
class PMLCorrectorHD
    : public HostDeviceCarrier<xfdtd::PML, xfdtd::cuda::PMLCorrector<xyz>> {
  using Host = xfdtd::PML;
  using Device = xfdtd::cuda::PMLCorrector<xyz>;

 public:
  PMLCorrectorHD(Host* host, std::shared_ptr<EMFHD> emf_hd);

  ~PMLCorrectorHD() override;

  auto copyHostToDevice() -> void override;

  auto copyDeviceToHost() -> void override;

  auto releaseDevice() -> void override;

  auto getAgency() -> CorrectorAgency*;

 private:
  std::shared_ptr<EMFHD> _emf_hd;
  IndexTask _task;

  Index _pml_global_e_start{}, _pml_global_h_start{};
  Index _pml_node_e_start{}, _pml_node_h_start{};
  Index _offset_c{};

  TensorHD<Real, 1> _coeff_a_e_hd, _coeff_b_e_hd, _coeff_a_h_hd, _coeff_b_h_hd;
  TensorHD<Real, 3> _c_ea_psi_hb_hd, _c_eb_psi_ha_hd, _c_ha_psi_eb_hd,
      _c_hb_psi_ea_hd;
  TensorHD<Real, 3> _ea_psi_hb_hd, _eb_psi_ha_hd, _ha_psi_eb_hd, _hb_psi_ea_hd;

  std::unique_ptr<PMLCorrectorAgency<xyz>> _agency{};
};

}  // namespace xfdtd::cuda

#endif  // __XFDTD_CUDA_PML_CORRECTOR_HD_CUH__
