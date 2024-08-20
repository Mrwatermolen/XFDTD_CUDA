#include "material/ade_method/drude_ade_method_hd.cuh"

namespace xfdtd::cuda {

DrudeADEMethodStorageHD::DrudeADEMethodStorageHD(Host* host)
    : ADEMethodStorageHD{host} {};

DrudeADEMethodStorageHD::~DrudeADEMethodStorageHD() { releaseDevice(); }

auto DrudeADEMethodStorageHD::copyHostToDevice() -> void {
  auto host = this->host();

  _coeff_j_j_hd.copyHostToDevice();
  _coeff_j_e_hd.copyHostToDevice();
  _coeff_j_sum_j_hd.copyHostToDevice();
  _coeff_e_j_sum_hd.copyHostToDevice();
  _jx_arr_hd.copyHostToDevice();
  _jy_arr_hd.copyHostToDevice();
  _jz_arr_hd.copyHostToDevice();

  auto device = Device{};
  device._num_pole = host->numPole();
  device._coeff_j_j = _coeff_j_j_hd.device();
  device._coeff_j_e = _coeff_j_e_hd.device();
  device._coeff_j_sum_j = _coeff_j_sum_j_hd.device();
  device._coeff_e_j_sum = _coeff_e_j_sum_hd.device();
  device._jx_arr = _jx_arr_hd.device();
  device._jy_arr = _jy_arr_hd.device();
  device._jz_arr = _jz_arr_hd.device();

  this->copyToDevice(&device);
}

auto DrudeADEMethodStorageHD::copyDeviceToHost() -> void {
  _coeff_j_j_hd.copyDeviceToHost();
  _coeff_j_e_hd.copyDeviceToHost();
  _coeff_j_sum_j_hd.copyDeviceToHost();
  _coeff_e_j_sum_hd.copyDeviceToHost();
  _jx_arr_hd.copyDeviceToHost();
  _jy_arr_hd.copyDeviceToHost();
  _jz_arr_hd.copyDeviceToHost();
}

auto DrudeADEMethodStorageHD::releaseDevice() -> void {
  _coeff_j_j_hd.releaseDevice();
  _coeff_j_e_hd.releaseDevice();
  _coeff_j_sum_j_hd.releaseDevice();
  _coeff_e_j_sum_hd.releaseDevice();
  _jx_arr_hd.releaseDevice();
  _jy_arr_hd.releaseDevice();
  _jz_arr_hd.releaseDevice();
}

}  // namespace xfdtd::cuda
