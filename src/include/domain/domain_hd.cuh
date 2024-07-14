#ifndef __XFDTD_CUDA_DOMAIN_HD_CUH__
#define __XFDTD_CUDA_DOMAIN_HD_CUH__

#include <xfdtd/electromagnetic_field/electromagnetic_field.h>

#include <memory>
#include <vector>
#include <xfdtd_cuda/common.cuh>

namespace xfdtd::cuda {

class TimeParam;
class GridSpaceHD;
class CalculationParamHD;
class EMFHD;
class UpdatorAgency;
class CorrectorAgency;
class MonitorAgency;

template <xfdtd::EMF::Field F>
class MovieMonitorAgency;

class NF2FFFrequencyDomainAgency;
class NF2FFTimeDoaminAgency;

class DomainHD {
 public:
  DomainHD(dim3 grid_dim, dim3 block_dim,
           std::shared_ptr<GridSpaceHD> grid_space_hd,
           std::shared_ptr<CalculationParamHD> calculation_param_hd,
           std::shared_ptr<EMFHD> emf_hd, UpdatorAgency* updator);

  ~DomainHD();

  auto run() -> void;

  auto updateH() -> void;

  auto correctH() -> void;

  auto updateE() -> void;

  auto correctE() -> void;

  auto record() -> void;

  auto nextStep() -> void;

  auto isCalculationDone() -> bool;

  auto addCorrector(CorrectorAgency* corrector) -> void;

  auto addMonitor(MonitorAgency* monitor) -> void;

  auto addNF2FFFrequencyDomainAgency(NF2FFFrequencyDomainAgency* agency) -> void;

  auto addNF2FFTimeDoaminAgency(NF2FFTimeDoaminAgency* agency) -> void;

 private:
  dim3 _grid_dim;
  dim3 _block_dim;
  std::shared_ptr<GridSpaceHD> _grid_space_hd;
  std::shared_ptr<CalculationParamHD> _calculation_param_hd;
  std::shared_ptr<EMFHD> _emf_hd;
  UpdatorAgency* _updator;
  std::vector<CorrectorAgency*> _correctors;
  std::vector<MonitorAgency*> _monitors;
  std::vector<NF2FFFrequencyDomainAgency*> _nf2ff_agencies;
  std::vector<NF2FFTimeDoaminAgency*> _nf2ff_time_agencies;
};

}  // namespace xfdtd::cuda

#endif  // __XFDTD_CUDA_DOMAIN_HD_CUH__
