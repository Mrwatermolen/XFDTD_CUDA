#ifndef __XFDTD_CUDA_DOMAIN_HD_CUH__
#define __XFDTD_CUDA_DOMAIN_HD_CUH__

#include <memory>
#include <xfdtd_cuda/calculation_param/calculation_param_hd.cuh>
#include <xfdtd_cuda/electromagnetic_field/electromagnetic_field_hd.cuh>
#include <xfdtd_cuda/grid_space/grid_space_hd.cuh>
#include <xfdtd_cuda/updator/basic_updator_te_hd.cuh>
#include <xfdtd_cuda/updator/updator_agency.cuh>

#include "xfdtd_cuda/calculation_param/time_param.cuh"
#include "xfdtd_cuda/common.cuh"

namespace xfdtd::cuda {

XFDTD_CUDA_GLOBAL auto __nextStep(xfdtd::cuda::TimeParam* time_param) -> void {
  time_param->nextStep();
}

class DomainHD {
 public:
  DomainHD(dim3 grid_dim, dim3 block_dim,
           std::shared_ptr<GridSpaceHD> grid_space_hd,
           std::shared_ptr<CalculationParamHD> calculation_param_hd,
           std::shared_ptr<EMFHD> emf_hd, UpdatorAgency* updator)
      : _grid_dim{grid_dim},
        _block_dim{block_dim},
        _grid_space_hd{grid_space_hd},
        _calculation_param_hd{calculation_param_hd},
        _emf_hd{emf_hd},
        _updator{updator} {}

  auto run() -> void {
    while (!isCalculationDone()) {
      updateH();
      correctH();
      updateE();
      correctE();
      record();
      nextStep();
    }
    cudaDeviceSynchronize();
  }

  auto updateH() -> void { _updator->updateH(_grid_dim, _block_dim); }

  auto correctH() -> void {}

  auto updateE() -> void { _updator->updateE(_grid_dim, _block_dim); }

  auto correctE() -> void {}

  auto record() -> void {}

  auto nextStep() -> void {
    __nextStep<<<1, 1>>>(_calculation_param_hd->timeParamHD()->device());
    _calculation_param_hd->timeParamHD()->host()->nextStep();
  }

  auto isCalculationDone() -> bool {
    return _calculation_param_hd->timeParamHD()->host()->endTimeStep() <=
           _calculation_param_hd->timeParamHD()->host()->currentTimeStep();
  }

 private:
  dim3 _grid_dim;
  dim3 _block_dim;
  std::shared_ptr<GridSpaceHD> _grid_space_hd;
  std::shared_ptr<CalculationParamHD> _calculation_param_hd;
  std::shared_ptr<EMFHD> _emf_hd;
  UpdatorAgency* _updator;
};

}  // namespace xfdtd::cuda

#endif  // __XFDTD_CUDA_DOMAIN_HD_CUH__
