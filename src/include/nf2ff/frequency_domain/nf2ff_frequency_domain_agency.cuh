#ifndef __XFDTD_CUDA_NF2FF_FREQUENCY_DOMAIN_AGENCY_CUH__
#define __XFDTD_CUDA_NF2FF_FREQUENCY_DOMAIN_AGENCY_CUH__

#include <xfdtd/coordinate_system/coordinate_system.h>

namespace xfdtd::cuda {

template <xfdtd::Axis::Direction D>
class NF2FFFrequencyDomainData;

class NF2FFFrequencyDomainAgency {
 public:
  NF2FFFrequencyDomainAgency(
      NF2FFFrequencyDomainData<xfdtd::Axis::Direction::XN>* data_xn,
      NF2FFFrequencyDomainData<xfdtd::Axis::Direction::XP>* data_xp,
      NF2FFFrequencyDomainData<xfdtd::Axis::Direction::YN>* data_yn,
      NF2FFFrequencyDomainData<xfdtd::Axis::Direction::YP>* data_yp,
      NF2FFFrequencyDomainData<xfdtd::Axis::Direction::ZN>* data_zn,
      NF2FFFrequencyDomainData<xfdtd::Axis::Direction::ZP>* data_zp);

  auto update(dim3 grid_dim, dim3 block_dim) -> void;

 private:
  NF2FFFrequencyDomainData<xfdtd::Axis::Direction::XN>* _data_xn{};
  NF2FFFrequencyDomainData<xfdtd::Axis::Direction::XP>* _data_xp{};
  NF2FFFrequencyDomainData<xfdtd::Axis::Direction::YN>* _data_yn{};
  NF2FFFrequencyDomainData<xfdtd::Axis::Direction::YP>* _data_yp{};
  NF2FFFrequencyDomainData<xfdtd::Axis::Direction::ZN>* _data_zn{};
  NF2FFFrequencyDomainData<xfdtd::Axis::Direction::ZP>* _data_zp{};
};

}  // namespace xfdtd::cuda

#endif  // __XFDTD_CUDA_NF2FF_FREQUENCY_DOMAIN_AGENCY_CUH__
