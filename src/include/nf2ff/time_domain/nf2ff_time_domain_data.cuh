#ifndef __XFDTD_CUDA_NF2FF_TIME_DOMAIN_DATA__
#define __XFDTD_CUDA_NF2FF_TIME_DOMAIN_DATA__

#include <xfdtd/common/type_define.h>
#include <xfdtd/coordinate_system/coordinate_system.h>
#include <xfdtd/electromagnetic_field/electromagnetic_field.h>

#include <tuple>
#include <xfdtd_cuda/common.cuh>
#include <xfdtd_cuda/grid_space/grid_space.cuh>
#include <xfdtd_cuda/index_task.cuh>
#include <xfdtd_cuda/tensor.cuh>

namespace xfdtd::cuda {

class GridSpace;
class CalculationParam;
class EMF;

struct TempVector {
  Real _x;
  Real _y;
  Real _z;

  XFDTD_CUDA_DEVICE auto x() const { return _x; }
  XFDTD_CUDA_DEVICE auto y() const { return _y; }
  XFDTD_CUDA_DEVICE auto z() const { return _z; }
};

template <xfdtd::Axis::Direction D>
class NF2FFTimeDomainData {
  inline static constexpr auto xyz = xfdtd::Axis::fromDirectionToXYZ<D>();
  inline static constexpr auto xyz_a = xfdtd::Axis::tangentialAAxis<xyz>();
  inline static constexpr auto xyz_b = xfdtd::Axis::tangentialBAxis<xyz>();

 public:
  template <xfdtd::EMF::Attribute attribute>
  XFDTD_CUDA_DEVICE auto update() -> void;

  IndexTask _task{};
  const GridSpace* _grid_space{nullptr};
  const CalculationParam* _calculation_param{nullptr};
  const EMF* _emf{nullptr};
  Array2D<Real>*_ea_prev{nullptr}, *_eb_prev{nullptr}, *_ha_prev{nullptr},
      *_hb_prev{nullptr};
  Array1D<Real>*_wa{nullptr}, *_wb{nullptr}, *_ua{nullptr}, *_ub{nullptr};
  TempVector _r_unit;
  Range<Real> _distance_range_e, _distance_range_h;

 private:
  XFDTD_CUDA_DEVICE auto task() const -> IndexTask;

  template <xfdtd::EMF::Attribute attribute>
  XFDTD_CUDA_DEVICE auto potential()
      -> std::tuple<Array1D<Real>&, Array1D<Real>&> {
    if constexpr (attribute == xfdtd::EMF::Attribute::E) {
      return {*_ua, *_ub};
    }

    if constexpr (attribute == xfdtd::EMF::Attribute::H) {
      return {*_wa, *_wb};
    }
  }

  template <xfdtd::EMF::Attribute attribute>
  XFDTD_CUDA_DEVICE auto timeDelay(const TempVector& location) const -> Real {
    const auto& r_unit = _r_unit;
    if constexpr (attribute == xfdtd::EMF::Attribute::E) {
      return (_distance_range_e.end() -
              (location.x() * r_unit.x() + location.y() * r_unit.y() +
               location.z() * r_unit.z()));
    }

    if constexpr (attribute == xfdtd::EMF::Attribute::H) {
      return (_distance_range_h.end() -
              (location.x() * r_unit.x() + location.y() * r_unit.y() +
               location.z() * r_unit.z()));
    }
  }

  template <xfdtd::EMF::Attribute attribute>
  XFDTD_CUDA_DEVICE auto previousAValue(Index i, Index j,
                                        Index k) const -> Real {
    const auto& ea_prev = *_ea_prev;
    const auto& ha_prev = *_ha_prev;
    if constexpr (attribute == xfdtd::EMF::Attribute::E) {
      if constexpr (xyz == Axis::XYZ::X) {
        return ea_prev.at(j, k);
      } else if constexpr (xyz == Axis::XYZ::Y) {
        return ea_prev.at(k, i);
      } else if constexpr (xyz == Axis::XYZ::Z) {
        return ea_prev.at(i, j);
      }
    }

    if constexpr (attribute == xfdtd::EMF::Attribute::H) {
      if constexpr (xyz == Axis::XYZ::X) {
        return ha_prev.at(j, k);
      } else if constexpr (xyz == Axis::XYZ::Y) {
        return ha_prev.at(k, i);
      } else if constexpr (xyz == Axis::XYZ::Z) {
        return ha_prev.at(i, j);
      }
    }
  }

  template <xfdtd::EMF::Attribute attribute>
  XFDTD_CUDA_DEVICE auto previousBValue(Index i, Index j,
                                        Index k) const -> Real {
    const auto& eb_prev = *_eb_prev;
    const auto& hb_prev = *_hb_prev;

    if constexpr (attribute == xfdtd::EMF::Attribute::E) {
      if constexpr (xyz == Axis::XYZ::X) {
        return eb_prev.at(j, k);
      } else if constexpr (xyz == Axis::XYZ::Y) {
        return eb_prev.at(k, i);
      } else if constexpr (xyz == Axis::XYZ::Z) {
        return eb_prev.at(i, j);
      }
    }

    if constexpr (attribute == xfdtd::EMF::Attribute::H) {
      if constexpr (xyz == Axis::XYZ::X) {
        return hb_prev.at(j, k);
      } else if constexpr (xyz == Axis::XYZ::Y) {
        return hb_prev.at(k, i);
      } else if constexpr (xyz == Axis::XYZ::Z) {
        return hb_prev.at(i, j);
      }
    }
  }

  template <xfdtd::EMF::Attribute attribute>
  XFDTD_CUDA_DEVICE auto setPreviousAValue(Index i, Index j, Index k,
                                           Real value) -> void {
    auto&& ea_prev = *_ea_prev;
    auto&& ha_prev = *_ha_prev;

    if constexpr (attribute == xfdtd::EMF::Attribute::E) {
      if constexpr (xyz == Axis::XYZ::X) {
        ea_prev.at(j, k) = value;
      } else if constexpr (xyz == Axis::XYZ::Y) {
        ea_prev.at(k, i) = value;
      } else if constexpr (xyz == Axis::XYZ::Z) {
        ea_prev.at(i, j) = value;
      }
    }

    if constexpr (attribute == xfdtd::EMF::Attribute::H) {
      if constexpr (xyz == Axis::XYZ::X) {
        ha_prev.at(j, k) = value;
      } else if constexpr (xyz == Axis::XYZ::Y) {
        ha_prev.at(k, i) = value;
      } else if constexpr (xyz == Axis::XYZ::Z) {
        ha_prev.at(i, j) = value;
      }
    }
  }

  template <xfdtd::EMF::Attribute attribute>
  XFDTD_CUDA_DEVICE auto setPreviousBValue(Index i, Index j, Index k,
                                           Real value) -> void {
    auto&& eb_prev = *_eb_prev;
    auto&& hb_prev = *_hb_prev;

    if constexpr (attribute == xfdtd::EMF::Attribute::E) {
      if constexpr (xyz == Axis::XYZ::X) {
        eb_prev.at(j, k) = value;
      } else if constexpr (xyz == Axis::XYZ::Y) {
        eb_prev.at(k, i) = value;
      } else if constexpr (xyz == Axis::XYZ::Z) {
        eb_prev.at(i, j) = value;
      }
    }

    if constexpr (attribute == xfdtd::EMF::Attribute::H) {
      if constexpr (xyz == Axis::XYZ::X) {
        hb_prev.at(j, k) = value;
      } else if constexpr (xyz == Axis::XYZ::Y) {
        hb_prev.at(k, i) = value;
      } else if constexpr (xyz == Axis::XYZ::Z) {
        hb_prev.at(i, j) = value;
      }
    }
  }
};

}  // namespace xfdtd::cuda

#endif  // __XFDTD_CUDA_NF2FF_TIME_DOMAIN_DATA__
