#include "material/ade_method/m_lor_ade_method_hd.cuh"

namespace xfdtd::cuda {

MLorentzADEMethodStorageHD::MLorentzADEMethodStorageHD(Host* host)
    : ADEMethodStorageHD{host} {}

MLorentzADEMethodStorageHD::~MLorentzADEMethodStorageHD() { releaseDevice(); }

auto MLorentzADEMethodStorageHD::copyHostToDevice() -> void {
  auto host = this->host();

  _coeff_j_j_hd.copyHostToDevice();
  _coeff_j_j_p_hd.copyHostToDevice();
  _coeff_j_e_n_hd.copyHostToDevice();
  _coeff_j_e_hd.copyHostToDevice();
  _coeff_j_e_p_hd.copyHostToDevice();
  _coeff_j_sum_j_hd.copyHostToDevice();
  _coeff_j_sum_j_p_hd.copyHostToDevice();
  _coeff_e_j_sum_hd.copyHostToDevice();
  _coeff_e_e_p_hd.copyHostToDevice();
  _ex_prev_hd.copyHostToDevice();
  _ey_prev_hd.copyHostToDevice();
  _ez_prev_hd.copyHostToDevice();
  _jx_arr_hd.copyHostToDevice();
  _jy_arr_hd.copyHostToDevice();
  _jz_arr_hd.copyHostToDevice();
  _jx_prev_arr_hd.copyHostToDevice();
  _jy_prev_arr_hd.copyHostToDevice();
  _jz_prev_arr_hd.copyHostToDevice();

  auto d = Device{host->numPole(),
                       _coeff_j_j_hd.device(),
                       _coeff_j_j_p_hd.device(),
                       _coeff_j_e_n_hd.device(),
                       _coeff_j_e_hd.device(),
                       _coeff_j_e_p_hd.device(),
                       _coeff_j_sum_j_hd.device(),
                       _coeff_j_sum_j_p_hd.device(),
                       _coeff_e_j_sum_hd.device(),
                       _coeff_e_e_p_hd.device(),
                       _ex_prev_hd.device(),
                       _ey_prev_hd.device(),
                       _ez_prev_hd.device(),
                       _jx_arr_hd.device(),
                       _jy_arr_hd.device(),
                       _jz_arr_hd.device(),
                       _jx_prev_arr_hd.device(),
                       _jy_prev_arr_hd.device(),
                       _jz_prev_arr_hd.device()};

  this->copyToDevice(&d);
}

auto MLorentzADEMethodStorageHD::copyDeviceToHost() -> void {
  _coeff_j_j_hd.copyDeviceToHost();
  _coeff_j_j_p_hd.copyDeviceToHost();
  _coeff_j_e_n_hd.copyDeviceToHost();
  _coeff_j_e_hd.copyDeviceToHost();
  _coeff_j_e_p_hd.copyDeviceToHost();
  _coeff_j_sum_j_hd.copyDeviceToHost();
  _coeff_j_sum_j_p_hd.copyDeviceToHost();
  _coeff_e_j_sum_hd.copyDeviceToHost();
  _coeff_e_e_p_hd.copyDeviceToHost();
  _ex_prev_hd.copyDeviceToHost();
  _ey_prev_hd.copyDeviceToHost();
  _ez_prev_hd.copyDeviceToHost();
  _jx_arr_hd.copyDeviceToHost();
  _jy_arr_hd.copyDeviceToHost();
  _jz_arr_hd.copyDeviceToHost();
  _jx_prev_arr_hd.copyDeviceToHost();
  _jy_prev_arr_hd.copyDeviceToHost();
  _jz_prev_arr_hd.copyDeviceToHost();
}

auto MLorentzADEMethodStorageHD::releaseDevice() -> void {
  _coeff_j_j_hd.releaseDevice();
  _coeff_j_j_p_hd.releaseDevice();
  _coeff_j_e_n_hd.releaseDevice();
  _coeff_j_e_hd.releaseDevice();
  _coeff_j_e_p_hd.releaseDevice();
  _coeff_j_sum_j_hd.releaseDevice();
  _coeff_j_sum_j_p_hd.releaseDevice();
  _coeff_e_j_sum_hd.releaseDevice();
  _coeff_e_e_p_hd.releaseDevice();
  _ex_prev_hd.releaseDevice();
  _ey_prev_hd.releaseDevice();
  _ez_prev_hd.releaseDevice();
  _jx_arr_hd.releaseDevice();
  _jy_arr_hd.releaseDevice();
  _jz_arr_hd.releaseDevice();
  _jx_prev_arr_hd.releaseDevice();
  _jy_prev_arr_hd.releaseDevice();
  _jz_prev_arr_hd.releaseDevice();
}

}  // namespace xfdtd::cuda
