#ifndef __XFDTD_CUDA_M_LOR_ADE_UPDATOR_HD_CUH__
#define __XFDTD_CUDA_M_LOR_ADE_UPDATOR_HD_CUH__

#include <memory>

#include "material/ade_method/m_lor_ade_method_hd.cuh"
#include "updator/ade_updator/ade_updator_hd.cuh"
#include "updator/ade_updator/m_lor_ade_updator.cuh"

namespace xfdtd::cuda {

class MLorentzADEUpdatorAgency;

class MLorentzADEUpdatorHD : public ADEUpdatorHD {
  using Host = void;
  using Device = MLorentzADEUpdator;

 public:
  MLorentzADEUpdatorHD(IndexTask task,
                       std::shared_ptr<GridSpaceHD> grid_space_hd,
                       std::shared_ptr<CalculationParamHD> calculation_param_hd,
                       std::shared_ptr<EMFHD> emf_hd,
                       std::shared_ptr<MLorentzADEMethodStorageHD> storage_hd);

  ~MLorentzADEUpdatorHD() override;

  auto copyHostToDevice() -> void override;

  auto getUpdatorAgency() -> UpdatorAgency* override;

 private:
  std::unique_ptr<MLorentzADEUpdatorAgency> _agency;
};

}  // namespace xfdtd::cuda

#endif  // __XFDTD_CUDA_M_LOR_ADE_UPDATOR_HD_CUH__
