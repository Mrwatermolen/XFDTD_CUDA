#ifndef __XFDTD_CUDA_NF2FF_TIME_DOMAIN_HD_CUH__
#define __XFDTD_CUDA_NF2FF_TIME_DOMAIN_HD_CUH__

#include <xfdtd/common/type_define.h>
#include <xfdtd/coordinate_system/coordinate_system.h>
#include <xfdtd/nffft/nffft_time_domain.h>

#include <memory>
#include <xfdtd_cuda/calculation_param/calculation_param_hd.cuh>
#include <xfdtd_cuda/electromagnetic_field/electromagnetic_field_hd.cuh>
#include <xfdtd_cuda/grid_space/grid_space_hd.cuh>
#include <xfdtd_cuda/host_device_carrier.cuh>
#include <xfdtd_cuda/tensor_hd.cuh>

namespace xfdtd::cuda {

template <xfdtd::Axis::Direction D>
class NF2FFTImeDomainData;

class NF2FFTimeDoaminAgency;

struct SurfaceCurrentSetTD;

class NF2FFTimeDomainHD
    : public HostDeviceCarrier<xfdtd::NFFFTTimeDomain, void> {
  using Host = xfdtd::NFFFTTimeDomain;
  using Device = void;

 public:
  NF2FFTimeDomainHD(
      Host* host, std::shared_ptr<const GridSpaceHD> grid_space_hd,
      std::shared_ptr<const CalculationParamHD> calculation_param_hd,
      std::shared_ptr<const EMFHD> emf_hd);

  ~NF2FFTimeDomainHD() override;

  auto copyHostToDevice() -> void override;

  auto copyDeviceToHost() -> void override;

  auto releaseDevice() -> void override;

  auto agency() -> NF2FFTimeDoaminAgency*;

 private:
  std::shared_ptr<const GridSpaceHD> _grid_space_hd{};
  std::shared_ptr<const CalculationParamHD> _calculation_param_hd{};
  std::shared_ptr<const EMFHD> _emf_hd{};

  std::unique_ptr<SurfaceCurrentSetTD> _surface_current_set{};
  std::unique_ptr<NF2FFTimeDoaminAgency> _agency{};
};

}  // namespace xfdtd::cuda

#endif  // __XFDTD_CUDA_NF2FF_TIME_DOMAIN_HD_CUH__
