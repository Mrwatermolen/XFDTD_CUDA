#include <xfdtd/calculation_param/calculation_param.h>

#include <xfdtd_cuda/calculation_param/fdtd_coefficient_hd.cuh>
#include <xfdtd_cuda/calculation_param/fdtd_coefficient.cuh>

namespace xfdtd::cuda {

FDTDCoefficientHD::FDTDCoefficientHD(Host *host)
    : HostDeviceCarrier{host},
      _cexe_hd{host->cexe()},
      _cexhy_hd{host->cexhy()},
      _cexhz_hd{host->cexhz()},
      _ceye_hd{host->ceye()},
      _ceyhz_hd{host->ceyhz()},
      _ceyhx_hd{host->ceyhx()},
      _ceze_hd{host->ceze()},
      _cezhx_hd{host->cezhx()},
      _cezhy_hd{host->cezhy()},
      _chxh_hd{host->chxh()},
      _chxey_hd{host->chxey()},
      _chxez_hd{host->chxez()},
      _chyh_hd{host->chyh()},
      _chyez_hd{host->chyez()},
      _chyex_hd{host->chyex()},
      _chzh_hd{host->chzh()},
      _chzex_hd{host->chzex()},
      _chzey_hd{host->chzey()} {}

FDTDCoefficientHD::~FDTDCoefficientHD() { releaseDevice(); }

auto FDTDCoefficientHD::copyHostToDevice() -> void {
  if (host() == nullptr) {
    throw std::runtime_error(
        "FDTDCoefficientHD::copyHostToDevice(): "
        "Host data is not initialized");
  }

  _cexe_hd.copyHostToDevice();
  _cexhy_hd.copyHostToDevice();
  _cexhz_hd.copyHostToDevice();
  _ceye_hd.copyHostToDevice();
  _ceyhz_hd.copyHostToDevice();
  _ceyhx_hd.copyHostToDevice();
  _ceze_hd.copyHostToDevice();
  _cezhx_hd.copyHostToDevice();
  _cezhy_hd.copyHostToDevice();

  _chxh_hd.copyHostToDevice();
  _chxey_hd.copyHostToDevice();
  _chxez_hd.copyHostToDevice();
  _chyh_hd.copyHostToDevice();
  _chyez_hd.copyHostToDevice();
  _chyex_hd.copyHostToDevice();
  _chzh_hd.copyHostToDevice();
  _chzex_hd.copyHostToDevice();
  _chzey_hd.copyHostToDevice();

  auto d = Device{_cexe_hd.device(), _cexhy_hd.device(), _cexhz_hd.device(),
                  _ceye_hd.device(), _ceyhz_hd.device(), _ceyhx_hd.device(),
                  _ceze_hd.device(), _cezhx_hd.device(), _cezhy_hd.device(),
                  _chxh_hd.device(), _chxey_hd.device(), _chxez_hd.device(),
                  _chyh_hd.device(), _chyez_hd.device(), _chyex_hd.device(),
                  _chzh_hd.device(), _chzex_hd.device(), _chzey_hd.device()};

  copyToDevice(&d);
}

auto FDTDCoefficientHD::copyDeviceToHost() -> void {
  if (host() == nullptr) {
    throw std::runtime_error(
        "FDTDCoefficientHD::copyDeviceToHost(): "
        "Host data is not initialized");
  }

  _cexe_hd.copyDeviceToHost();
  _cexhy_hd.copyDeviceToHost();
  _cexhz_hd.copyDeviceToHost();
  _ceye_hd.copyDeviceToHost();
  _ceyhz_hd.copyDeviceToHost();
  _ceyhx_hd.copyDeviceToHost();
  _ceze_hd.copyDeviceToHost();
  _cezhx_hd.copyDeviceToHost();
  _cezhy_hd.copyDeviceToHost();

  _chxh_hd.copyDeviceToHost();
  _chxey_hd.copyDeviceToHost();
  _chxez_hd.copyDeviceToHost();
  _chyh_hd.copyDeviceToHost();
  _chyez_hd.copyDeviceToHost();
  _chyex_hd.copyDeviceToHost();
  _chzh_hd.copyDeviceToHost();
  _chzex_hd.copyDeviceToHost();
  _chzey_hd.copyDeviceToHost();
}

auto FDTDCoefficientHD::releaseDevice() -> void {
  _cexe_hd.releaseDevice();
  _cexhy_hd.releaseDevice();
  _cexhz_hd.releaseDevice();
  _ceye_hd.releaseDevice();
  _ceyhz_hd.releaseDevice();
  _ceyhx_hd.releaseDevice();
  _ceze_hd.releaseDevice();
  _cezhx_hd.releaseDevice();
  _cezhy_hd.releaseDevice();

  _chxh_hd.releaseDevice();
  _chxey_hd.releaseDevice();
  _chxez_hd.releaseDevice();
  _chyh_hd.releaseDevice();
  _chyez_hd.releaseDevice();
  _chyex_hd.releaseDevice();
  _chzh_hd.releaseDevice();
  _chzex_hd.releaseDevice();
  _chzey_hd.releaseDevice();

  releaseBaseDevice();
}

}  // namespace xfdtd::cuda
