#include <utility>

#include "material/ade_method/ade_method_hd.cuh"
#include "updator/ade_updator/ade_updator.cuh"
#include "updator/ade_updator/ade_updator_hd.cuh"

namespace xfdtd::cuda {

ADEUpdatorHD::ADEUpdatorHD(
    IndexTask task, std::shared_ptr<GridSpaceHD> grid_space_hd,
    std::shared_ptr<CalculationParamHD> calculation_param_hd,
    std::shared_ptr<EMFHD> emf_hd,
    std::shared_ptr<ADEMethodStorageHD> storage_hd)
    : UpdatorHD{task, std::move(grid_space_hd), std::move(calculation_param_hd),
                std::move(emf_hd)},
      _storage_hd{std::move(storage_hd)} {}

ADEUpdatorHD::~ADEUpdatorHD() = default;

}  // namespace xfdtd::cuda
