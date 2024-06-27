#include <xfdtd/calculation_param/calculation_param.h>

#include <xfdtd_cuda/calculation_param/fdtd_coefficient.cuh>

namespace xfdtd {

namespace cuda {

FDTDCoefficientHD::FDTDCoefficientHD(HostFDTDCoefficient* host_fdtd_coefficient)
    : _host_fdtd_coefficient{host_fdtd_coefficient},
      _cexe_hd{(host_fdtd_coefficient->cexe())},
      _cexhy_hd{(host_fdtd_coefficient->cexhy())},
      _cexhz_hd{(host_fdtd_coefficient->cexhz())},
      _ceye_hd{(host_fdtd_coefficient->ceye())},
      _ceyhz_hd{(host_fdtd_coefficient->ceyhz())},
      _ceyhx_hd{(host_fdtd_coefficient->ceyhx())},
      _ceze_hd{(host_fdtd_coefficient->ceze())},
      _cezhx_hd{(host_fdtd_coefficient->cezhx())},
      _cezhy_hd{(host_fdtd_coefficient->cezhy())},
      _chxh_hd{(host_fdtd_coefficient->chxh())},
      _chxey_hd{(host_fdtd_coefficient->chxey())},
      _chxez_hd{(host_fdtd_coefficient->chxez())},
      _chyh_hd{(host_fdtd_coefficient->chyh())},
      _chyez_hd{(host_fdtd_coefficient->chyez())},
      _chyex_hd{(host_fdtd_coefficient->chyex())},
      _chzh_hd{(host_fdtd_coefficient->chzh())},
      _chzex_hd{(host_fdtd_coefficient->chzex())},
      _chzey_hd{(host_fdtd_coefficient->chzey())} {
  std::cout << "FDTDCoefficientHD constructor\n";
}

auto FDTDCoefficientHD::copyHostToDevice() -> void {
  if (_device_fdtd_coefficient != nullptr) {
    XFDTD_CORE_CUDA_CHECK_CUDA_ERROR(cudaFree(_device_fdtd_coefficient));
  }

  XFDTD_CORE_CUDA_CHECK_CUDA_ERROR(
      cudaMalloc(&_device_fdtd_coefficient, sizeof(DeviceFDTDCoefficient)));

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

  auto d = DeviceFDTDCoefficient{};
  d._cexe = _cexe_hd.device();
  d._cexhy = _cexhy_hd.device();
  d._cexhz = _cexhz_hd.device();
  d._ceye = _ceye_hd.device();
  d._ceyhz = _ceyhz_hd.device();
  d._ceyhx = _ceyhx_hd.device();
  d._ceze = _ceze_hd.device();
  d._cezhx = _cezhx_hd.device();
  d._cezhy = _cezhy_hd.device();
  d._chxh = _chxh_hd.device();
  d._chxey = _chxey_hd.device();
  d._chxez = _chxez_hd.device();
  d._chyh = _chyh_hd.device();
  d._chyez = _chyez_hd.device();
  d._chyex = _chyex_hd.device();
  d._chzh = _chzh_hd.device();
  d._chzex = _chzex_hd.device();
  d._chzey = _chzey_hd.device();

  XFDTD_CORE_CUDA_CHECK_CUDA_ERROR(cudaMemcpy(_device_fdtd_coefficient, &d,
                                              sizeof(DeviceFDTDCoefficient),
                                              cudaMemcpyHostToDevice));

  d._cexe = nullptr;
  d._cexhy = nullptr;
  d._cexhz = nullptr;
  d._ceye = nullptr;
  d._ceyhz = nullptr;
  d._ceyhx = nullptr;
  d._ceze = nullptr;
  d._cezhx = nullptr;
  d._cezhy = nullptr;
  d._chxh = nullptr;
  d._chxey = nullptr;
  d._chxez = nullptr;
  d._chyh = nullptr;
  d._chyez = nullptr;
  d._chyex = nullptr;
  d._chzh = nullptr;
  d._chzex = nullptr;
  d._chzey = nullptr;
}

auto FDTDCoefficientHD::copyDeviceToHost() -> void {
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
  // Use RAII
}

}  // namespace cuda

}  // namespace xfdtd
