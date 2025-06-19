#ifndef __XFDTD_CUDA_ADE_UPDATOR_HD_CUH__
#define __XFDTD_CUDA_ADE_UPDATOR_HD_CUH__

#include "material/ade_method/ade_method_hd.cuh"
#include "updator/ade_updator/ade_updator.cuh"
#include "updator/updator_hd.cuh"

namespace xfdtd::cuda {

class ADEUpdatorHD : public UpdatorHD<ADEUpdator> {
  using Host = void;
  using Device = ADEUpdator;

 public:
  ADEUpdatorHD(IndexTask task, std::shared_ptr<GridSpaceHD> grid_space_hd,
               std::shared_ptr<CalculationParamHD> calculation_param_hd,
               std::shared_ptr<EMFHD> emf_hd,
               std::shared_ptr<ADEMethodStorageHD> storage_hd);

  ~ADEUpdatorHD() override;

  auto storageHD() const { return _storage_hd; }

  auto storageHD() { return _storage_hd; }

  auto copyHostToDevice() -> void override {
    auto device =
        Device{task(), gridSpaceHD()->device(), calculationParamHD()->device(),
               emfHD()->device(), storageHD()->device()};
    this->copyToDevice(&device);
  }

 protected:
  std::shared_ptr<ADEMethodStorageHD> _storage_hd;
};

}  // namespace xfdtd::cuda

#endif  // __XFDTD_CUDA_ADE_UPDATOR_HD_CUH__
