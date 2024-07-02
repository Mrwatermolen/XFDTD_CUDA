#ifndef __XFDTD_CUDA_SIMULATION_CUH__
#define __XFDTD_CUDA_SIMULATION_CUH__

#include <xfdtd_cuda/calculation_param/calculation_param.cuh>
#include <xfdtd_cuda/electromagnetic_field/electromagnetic_field.cuh>
#include <xfdtd_cuda/grid_space/grid_space.cuh>

#include "xfdtd_cuda/index_task.cuh"

namespace xfdtd {

namespace cuda {

class Simulation {
 public:
  friend class SimulationHD;

  auto gridSpace() const -> const xfdtd::cuda::GridSpaceData* {
    return _grid_space;
  }

  auto calculationParam() const -> const xfdtd::cuda::CalculationParam* {
    return _calculation_param;
  }

  auto emf() const -> const xfdtd::cuda::EMF* { return _emf; }

  auto gridSpace() -> xfdtd::cuda::GridSpaceData* { return _grid_space; }

  auto calculationParam() -> xfdtd::cuda::CalculationParam* {
    return _calculation_param;
  }

  auto emf() -> xfdtd::cuda::EMF* { return _emf; }

  auto task() const {
    auto nx = _grid_space->sizeX();
    auto ny = _grid_space->sizeY();
    auto nz = _grid_space->sizeZ();

    return makeTask(makeRange<Index>(0, nx), makeRange<Index>(0, ny),
                    makeRange<Index>(0, nz));
  }

 private:
  xfdtd::cuda::GridSpaceData* _grid_space{};
  xfdtd::cuda::CalculationParam* _calculation_param{};
  xfdtd::cuda::EMF* _emf{};
};

}  // namespace cuda

}  // namespace xfdtd

#endif  // __XFDTD_CUDA_SIMULATION_CUH__
