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

  auto d = Device{host->numPole(),
                  _coeff_j_j_hd.device(),
                  _coeff_j_e_hd.device(),
                  _coeff_j_sum_j_hd.device(),
                  _coeff_e_j_sum_hd.device(),
                  _jx_arr_hd.device(),
                  _jy_arr_hd.device(),
                  _jz_arr_hd.device()};

  this->copyToDevice(&d);
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
