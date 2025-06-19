#include "material/ade_method/debye_ade_method.cuh"

namespace xfdtd::cuda {

DebyeADEMethodStorage::DebyeADEMethodStorage(
    Index num_pole, Array4D<Real>* coeff_j_j, Array4D<Real>* coeff_j_e,
    Array4D<Real>* coeff_j_sum_j, Array3D<Real>* coeff_e_j_sum,
    Array4D<Real>* jx_arr, Array4D<Real>* jy_arr, Array4D<Real>* jz_arr)
    : ADEMethodStorage{num_pole,      coeff_j_j, nullptr,       nullptr,
                       coeff_j_e,     nullptr,   coeff_j_sum_j, nullptr,
                       coeff_e_j_sum, nullptr,   nullptr,       nullptr,
                       nullptr,       jx_arr,    jy_arr,        jz_arr,
                       nullptr,       nullptr,   nullptr}

{
  if (_coeff_j_j == nullptr) {
    throw std::runtime_error(
        "DebyeADMethodStorage::DebyeADMethodStorage: "
        "coeff_j_j is nullptr");
  }

  if (_coeff_j_e == nullptr) {
    throw std::runtime_error(
        "DebyeADEMethodStorage::DebyeADMethodStorage: "
        "coeff_j_e is nullptr");
  }

  if (_coeff_j_sum_j == nullptr) {
    throw std::runtime_error(
        "DebyeADMethodStorage::DebyeADMethodStorage: "
        "coeff_j_sum_j is nullptr");
  }

  if (_coeff_e_j_sum == nullptr) {
    throw std::runtime_error(
        "DebyeADMethodStorage::DebyeADMethodStorage: "
        "coeff_e_j_sum is nullptr");
  }

  if (_jx_arr == nullptr) {
    throw std::runtime_error(
        "DebyeADMethodStorage::DebyeADMethodStorage: "
        "jx_arr is nullptr");
  }

  if (_jy_arr == nullptr) {
    throw std::runtime_error(
        "DebyeADMethodStorage::DebyeADMethodStorage: "
        "jy_arr is nullptr");
  }

  if (_jz_arr == nullptr) {
    throw std::runtime_error(
        "DebyeADMethodStorage::DebyeADMethodStorage: "
        "jz_arr is nullptr");
  }
}

}  // namespace xfdtd::cuda
