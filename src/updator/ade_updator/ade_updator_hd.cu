#include <xfdtd_cuda/calculation_param/calculation_param_hd.cuh>
#include <xfdtd_cuda/electromagnetic_field/electromagnetic_field_hd.cuh>
#include <xfdtd_cuda/grid_space/grid_space_hd.cuh>
#include <xfdtd_cuda/index_task.cuh>

#include "material/ade_method/ade_method_hd.cuh"
#include "updator/ade_updator/ade_updator.cuh"
#include "updator/ade_updator/ade_updator_hd.cuh"

namespace xfdtd::cuda {

ADEUpdatorHD::ADEUpdatorHD(
    IndexTask task, std::shared_ptr<GridSpaceHD> grid_space_hd,
    std::shared_ptr<CalculationParamHD> calculation_param_hd,
    std::shared_ptr<EMFHD> emf_hd,
    std::shared_ptr<ADEMethodStorageHD> storage_hd)
    : HostDeviceCarrier{nullptr},
      _task{task},
      _grid_space_hd{std::move(grid_space_hd)},
      _calculation_param_hd{std::move(calculation_param_hd)},
      _emf_hd{std::move(emf_hd)},
      _storage_hd{std::move(storage_hd)} {}

ADEUpdatorHD::~ADEUpdatorHD() = default;

}  // namespace xfdtd::cuda
