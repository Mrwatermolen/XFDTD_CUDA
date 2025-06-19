#include "material/ade_method/ade_method_hd.cuh"

namespace xfdtd::cuda {

ADEMethodStorageHD::ADEMethodStorageHD(Host* host)
    : HostDeviceCarrier{host},
      _coeff_j_j_hd{host->coeffJJ()},
      _coeff_j_j_p_hd{host->coeffJJPrev()},
      _coeff_j_e_n_hd{host->coeffJENext()},
      _coeff_j_e_hd{host->coeffJE()},
      _coeff_j_e_p_hd{host->coeffJEPrev()},
      _coeff_j_sum_j_hd{host->coeffJSumJ()},
      _coeff_j_sum_j_p_hd{host->coeffJSumJPrev()},
      _coeff_e_j_sum_hd{host->coeffEJSum()},
      _coeff_e_e_p_hd{host->coeffEEPrev()},
      _ex_prev_hd{host->exPrev()},
      _ey_prev_hd{host->eyPrev()},
      _ez_prev_hd{host->ezPrev()},
      _jx_arr_hd{host->jxArr()},
      _jy_arr_hd{host->jyArr()},
      _jz_arr_hd{host->jzArr()},
      _jx_prev_arr_hd{host->jxPrevArr()},
      _jy_prev_arr_hd{host->jyPrevArr()},
      _jz_prev_arr_hd{host->jzPrevArr()} {}

}  // namespace xfdtd::cuda