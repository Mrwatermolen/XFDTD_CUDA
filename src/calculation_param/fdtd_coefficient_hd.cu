#include <xfdtd/calculation_param/calculation_param.h>
#include <xfdtd/common/type_define.h>

#include <xfdtd_cuda/calculation_param/fdtd_coefficient.cuh>
#include <xfdtd_cuda/calculation_param/fdtd_coefficient_hd.cuh>

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
      _chzey_hd{host->chzey()} {
  _cexe_tex.setStrides(host->cexe().strides());
  _cexhz_tex.setStrides(host->cexhz().strides());
  _cexhy_tex.setStrides(host->cexhy().strides());

  _ceyhx_tex.setStrides(host->ceyhx().strides());
  _ceyhz_tex.setStrides(host->ceyhz().strides());
  _ceye_tex.setStrides(host->ceye().strides());

  _cezhy_tex.setStrides(host->cezhy().strides());
  _cezhx_tex.setStrides(host->cezhx().strides());
  _ceze_tex.setStrides(host->ceze().strides());

  _chxez_tex.setStrides(host->chxez().strides());
  _chxey_tex.setStrides(host->chxey().strides());
  _chxh_tex.setStrides(host->chxh().strides());

  _chyex_tex.setStrides(host->chyex().strides());
  _chyez_tex.setStrides(host->chyez().strides());
  _chyh_tex.setStrides(host->chyh().strides());

  _chzex_tex.setStrides(host->chzex().strides());
  _chzey_tex.setStrides(host->chzey().strides());
  _chzh_tex.setStrides(host->chzh().strides());
}

FDTDCoefficientHD::~FDTDCoefficientHD() {
  releaseDevice();
  _cexe_tex.release();
  _cexhy_tex.release();
  _cexhz_tex.release();
  _ceye_tex.release();
  _ceyhz_tex.release();
  _ceyhx_tex.release();
  _ceze_tex.release();
  _cezhx_tex.release();
  _cezhy_tex.release();
  _chxh_tex.release();
  _chxey_tex.release();
  _chxez_tex.release();
  _chyh_tex.release();
  _chyez_tex.release();
  _chyex_tex.release();
  _chzh_tex.release();
  _chzex_tex.release();
  _chzey_tex.release();
}

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

  auto bind_texture = [](auto &tex, const auto &hd) {
    Index size = 1;
    for (auto i = 0; i < hd.shape().size(); ++i) {
      size *= hd.shape()[i];
    }
    tex.bind(hd.deviceData(), size);
  };

  bind_texture(_cexe_tex, _cexe_hd);
  bind_texture(_cexhy_tex, _cexhy_hd);
  bind_texture(_cexhz_tex, _cexhz_hd);
  bind_texture(_ceye_tex, _ceye_hd);
  bind_texture(_ceyhz_tex, _ceyhz_hd);
  bind_texture(_ceyhx_tex, _ceyhx_hd);
  bind_texture(_ceze_tex, _ceze_hd);
  bind_texture(_cezhx_tex, _cezhx_hd);
  bind_texture(_cezhy_tex, _cezhy_hd);
  bind_texture(_chxh_tex, _chxh_hd);
  bind_texture(_chxey_tex, _chxey_hd);
  bind_texture(_chxez_tex, _chxez_hd);
  bind_texture(_chyh_tex, _chyh_hd);
  bind_texture(_chyez_tex, _chyez_hd);
  bind_texture(_chyex_tex, _chyex_hd);
  bind_texture(_chzh_tex, _chzh_hd);
  bind_texture(_chzex_tex, _chzex_hd);
  bind_texture(_chzey_tex, _chzey_hd);

  // auto d = Device{_cexe_hd.device(), _cexhy_hd.device(), _cexhz_hd.device(),
  //                 _ceye_hd.device(), _ceyhz_hd.device(), _ceyhx_hd.device(),
  //                 _ceze_hd.device(), _cezhx_hd.device(), _cezhy_hd.device(),
  //                 _chxh_hd.device(), _chxey_hd.device(), _chxez_hd.device(),
  //                 _chyh_hd.device(), _chyez_hd.device(), _chyex_hd.device(),
  //                 _chzh_hd.device(), _chzex_hd.device(), _chzey_hd.device()};
  auto d = Device{_cexe_tex,  _cexhy_tex, _cexhz_tex, _ceye_tex,  _ceyhz_tex,
                  _ceyhx_tex, _ceze_tex,  _cezhx_tex, _cezhy_tex, _chxh_tex,
                  _chxey_tex, _chxez_tex, _chyh_tex,  _chyez_tex, _chyex_tex,
                  _chzh_tex,  _chzex_tex, _chzey_tex};

  copyToDevice(&d);
}

auto FDTDCoefficientHD::copyDeviceToHost() -> void {
  // if (host() == nullptr) {
  //   throw std::runtime_error(
  //       "FDTDCoefficientHD::copyDeviceToHost(): "
  //       "Host data is not initialized");
  // }

  // _cexe_hd.copyDeviceToHost();
  // _cexhy_hd.copyDeviceToHost();
  // _cexhz_hd.copyDeviceToHost();
  // _ceye_hd.copyDeviceToHost();
  // _ceyhz_hd.copyDeviceToHost();
  // _ceyhx_hd.copyDeviceToHost();
  // _ceze_hd.copyDeviceToHost();
  // _cezhx_hd.copyDeviceToHost();
  // _cezhy_hd.copyDeviceToHost();

  // _chxh_hd.copyDeviceToHost();
  // _chxey_hd.copyDeviceToHost();
  // _chxez_hd.copyDeviceToHost();
  // _chyh_hd.copyDeviceToHost();
  // _chyez_hd.copyDeviceToHost();
  // _chyex_hd.copyDeviceToHost();
  // _chzh_hd.copyDeviceToHost();
  // _chzex_hd.copyDeviceToHost();
  // _chzey_hd.copyDeviceToHost();
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
