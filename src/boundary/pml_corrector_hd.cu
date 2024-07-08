#include "boundary/pml_corrector_agency.cuh"
#include "boundary/pml_corrector_hd.cuh"
#include "corrector/corrector_agency.cuh"

namespace xfdtd::cuda {

template <Axis::XYZ xyz>
PMLCorrectorHD<xyz>::PMLCorrectorHD(Host* host, std::shared_ptr<EMFHD> emf_hd)
    : HostDeviceCarrier<Host, Device>{host},
      _emf_hd{emf_hd},
      _task{xfdtd::cuda::IndexTask{
          xfdtd::cuda::IndexRange{host->task().xRange().start(),
                                  host->task().xRange().end()},
          xfdtd::cuda::IndexRange{host->task().yRange().start(),
                                  host->task().yRange().end()},
          xfdtd::cuda::IndexRange{host->task().zRange().start(),
                                  host->task().zRange().end()}}},
      _pml_global_e_start{host->globalENodeStartIndexMainAxis()},
      _pml_global_h_start{host->globalHNodeStartIndexMainAxis()},
      _pml_node_e_start{host->nodeENodeStartIndexMainAxis()},
      _pml_node_h_start{host->nodeHNodeStartIndexMainAxis()},
      _offset_c{host->offsetC()},
      _coeff_a_e_hd{host->coeffAE()},
      _coeff_b_e_hd{host->coeffBE()},
      _coeff_a_h_hd{host->coeffAH()},
      _coeff_b_h_hd{host->coeffBH()},
      _c_ea_psi_hb_hd{host->coeffEAPsiHB()},
      _c_eb_psi_ha_hd{host->coeffEBPsiHA()},
      _c_ha_psi_eb_hd{host->coeffHAPsiEB()},
      _c_hb_psi_ea_hd{host->coeffHBPsiEA()},
      _ea_psi_hb_hd{host->eaPsiHB()},
      _eb_psi_ha_hd{host->ebPsiHA()},
      _ha_psi_eb_hd{host->haPsiEB()},
      _hb_psi_ea_hd{host->hbPsiEA()} {}

template <Axis::XYZ xyz>
PMLCorrectorHD<xyz>::~PMLCorrectorHD() {
  releaseDevice();
}

template <Axis::XYZ xyz>
auto PMLCorrectorHD<xyz>::copyHostToDevice() -> void {
  if (this->host() == nullptr) {
    throw std::runtime_error("PMLCorrectorHD::copyHostToDevice: host is null!");
  }

  _coeff_a_e_hd.copyHostToDevice();
  _coeff_b_e_hd.copyHostToDevice();
  _coeff_a_h_hd.copyHostToDevice();
  _coeff_b_h_hd.copyHostToDevice();
  _c_ea_psi_hb_hd.copyHostToDevice();
  _c_eb_psi_ha_hd.copyHostToDevice();
  _c_ha_psi_eb_hd.copyHostToDevice();
  _c_hb_psi_ea_hd.copyHostToDevice();
  _ea_psi_hb_hd.copyHostToDevice();
  _eb_psi_ha_hd.copyHostToDevice();
  _ha_psi_eb_hd.copyHostToDevice();
  _hb_psi_ea_hd.copyHostToDevice();

  auto d = Device{};
  d._pml_global_e_start = _pml_global_e_start;
  d._pml_global_h_start = _pml_global_h_start;
  d._pml_node_e_start = _pml_node_e_start;
  d._pml_node_h_start = _pml_node_h_start;
  d._offset_c = _offset_c;
  d._emf = _emf_hd->device();
  d._task = _task;
  d._coeff_a_e = _coeff_a_e_hd.device();
  d._coeff_b_e = _coeff_b_e_hd.device();
  d._coeff_a_h = _coeff_a_h_hd.device();
  d._coeff_b_h = _coeff_b_h_hd.device();
  d._c_ea_psi_hb = _c_ea_psi_hb_hd.device();
  d._c_eb_psi_ha = _c_eb_psi_ha_hd.device();
  d._c_ha_psi_eb = _c_ha_psi_eb_hd.device();
  d._c_hb_psi_ea = _c_hb_psi_ea_hd.device();
  d._ea_psi_hb = _ea_psi_hb_hd.device();
  d._eb_psi_ha = _eb_psi_ha_hd.device();
  d._ha_psi_eb = _ha_psi_eb_hd.device();
  d._hb_psi_ea = _hb_psi_ea_hd.device();

  this->copyToDevice(&d);
}

template <Axis::XYZ xyz>
auto PMLCorrectorHD<xyz>::copyDeviceToHost() -> void {
  _ea_psi_hb_hd.copyDeviceToHost();
  _eb_psi_ha_hd.copyDeviceToHost();
  _ha_psi_eb_hd.copyDeviceToHost();
  _hb_psi_ea_hd.copyDeviceToHost();
}

template <Axis::XYZ xyz>
auto PMLCorrectorHD<xyz>::releaseDevice() -> void {
  _coeff_a_e_hd.releaseDevice();
  _coeff_b_e_hd.releaseDevice();
  _coeff_a_h_hd.releaseDevice();
  _coeff_b_h_hd.releaseDevice();
  _c_ea_psi_hb_hd.releaseDevice();
  _c_eb_psi_ha_hd.releaseDevice();
  _c_ha_psi_eb_hd.releaseDevice();
  _c_hb_psi_ea_hd.releaseDevice();
  _ea_psi_hb_hd.releaseDevice();
  _eb_psi_ha_hd.releaseDevice();
  _ha_psi_eb_hd.releaseDevice();
  _hb_psi_ea_hd.releaseDevice();
  this->releaseBaseDevice();
}

template <Axis::XYZ xyz>
auto PMLCorrectorHD<xyz>::getAgency() -> CorrectorAgency* {
  if (_agency == nullptr) {
    _agency = std::make_unique<PMLCorrectorAgency<xyz>>(this->device());
  }

  return _agency.get();
}

// explicit instantiation
template class PMLCorrectorHD<Axis::XYZ::X>;
template class PMLCorrectorHD<Axis::XYZ::Y>;
template class PMLCorrectorHD<Axis::XYZ::Z>;

}  // namespace xfdtd::cuda
