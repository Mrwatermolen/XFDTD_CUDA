#include <xfdtd/waveform_source/tfsf.h>

#include "waveform_source/tfsf/tfsf_corrector.cuh"
#include "waveform_source/tfsf/tfsf_corrector_agency.cuh"
#include "waveform_source/tfsf/tfsf_corrector_hd.cuh"

namespace xfdtd::cuda {

TFSFCorrectorHD::TFSFCorrectorHD(
    Host *host, xfdtd::cuda::CalculationParam *calculation_param,
    xfdtd::cuda::EMF *emf)
    : HostDeviceCarrier{host},
      _calculation_param{calculation_param},
      _emf{emf},
      _projection_x_int{host->projectionXInt()},
      _projection_y_int{host->projectionYInt()},
      _projection_z_int{host->projectionZInt()},
      _projection_x_half{host->projectionXHalf()},
      _projection_y_half{host->projectionYHalf()},
      _projection_z_half{host->projectionZHalf()},
      _e_inc{host->eInc()},
      _h_inc{host->hInc()}

{
  const auto &t = host->globalTask();
  _total_task = IndexTask{IndexRange{t.xRange().start(), t.xRange().end()},
                          IndexRange{t.yRange().start(), t.yRange().end()},
                          IndexRange{t.zRange().start(), t.zRange().end()}};
}

TFSFCorrectorHD::~TFSFCorrectorHD() { releaseDevice(); }

auto TFSFCorrectorHD::copyHostToDevice() -> void {
  if (host() == nullptr) {
    throw std::runtime_error("TFSFCorrectorHD::copyHostToDevice: host is null");
  }

  auto &&d = Device{};
  d._calculation_param = _calculation_param;
  d._emf = _emf;
  d._total_task = _total_task;
  _projection_x_int.copyHostToDevice();
  _projection_y_int.copyHostToDevice();
  _projection_z_int.copyHostToDevice();
  _projection_x_half.copyHostToDevice();
  _projection_y_half.copyHostToDevice();
  _projection_z_half.copyHostToDevice();
  _e_inc.copyHostToDevice();
  _h_inc.copyHostToDevice();

  d._projection_x_int = _projection_x_int.device();
  d._projection_y_int = _projection_y_int.device();
  d._projection_z_int = _projection_z_int.device();
  d._projection_x_half = _projection_x_half.device();
  d._projection_y_half = _projection_y_half.device();
  d._projection_z_half = _projection_z_half.device();
  d._e_inc = _e_inc.device();
  d._h_inc = _h_inc.device();
  d._cax = host()->cax();
  d._cbx = host()->cbx();
  d._cay = host()->cay();
  d._cby = host()->cby();
  d._caz = host()->caz();
  d._cbz = host()->cbz();
  d._transform_e_x = host()->transformE().x();
  d._transform_e_y = host()->transformE().y();
  d._transform_e_z = host()->transformE().z();
  d._transform_h_x = host()->transformH().x();
  d._transform_h_y = host()->transformH().y();
  d._transform_h_z = host()->transformH().z();

  copyToDevice(&d);
}

auto TFSFCorrectorHD::copyDeviceToHost() -> void {
  _e_inc.copyDeviceToHost();
  _h_inc.copyDeviceToHost();
}

auto TFSFCorrectorHD::releaseDevice() -> void {
  _projection_x_int.releaseDevice();
  _projection_y_int.releaseDevice();
  _projection_z_int.releaseDevice();
  _projection_x_half.releaseDevice();
  _projection_y_half.releaseDevice();
  _projection_z_half.releaseDevice();
  _e_inc.releaseDevice();
  _h_inc.releaseDevice();
  releaseBaseDevice();
}

auto TFSFCorrectorHD::getTFSFCorrector2DAgency() -> CorrectorAgency * {
  if (_corrector_agency != nullptr) {
    throw std::runtime_error(
        "TFSFCorrectorHD::getTFSFCorrector2DAgency: already created");
  }

  _corrector_agency = std::make_unique<TFSFCorrector2DAgency>(device());
  return _corrector_agency.get();
}

}  // namespace xfdtd::cuda
