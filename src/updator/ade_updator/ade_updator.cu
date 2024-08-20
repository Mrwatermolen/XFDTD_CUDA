#include <xfdtd_cuda/common.cuh>

#include "material/ade_method/ade_method.cuh"
#include "updator/ade_updator/ade_updator.cuh"

namespace xfdtd::cuda {

ADEUpdator::ADEUpdator(IndexTask task, GridSpace* grid_space,
                       CalculationParam* calculation_param, EMF* emf,
                       ADEMethodStorage* storage)
    : BasicUpdator{task, grid_space, calculation_param, emf},
      _ade_method_storage{storage} {
  if (storage == nullptr) {
    throw std::runtime_error("ADEUpdator::ADEUpdator: storage is null");
  }
}

}  // namespace xfdtd::cuda
