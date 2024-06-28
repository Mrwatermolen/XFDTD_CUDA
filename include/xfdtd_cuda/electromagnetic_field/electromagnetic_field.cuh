#ifndef __XFDTD_CUDA_ELECTROMAGNETIC_FIELD_CUH__
#define __XFDTD_CUDA_ELECTROMAGNETIC_FIELD_CUH__

#include <xfdtd/common/type_define.h>
#include <xfdtd/coordinate_system/coordinate_system.h>
#include <xfdtd/electromagnetic_field/electromagnetic_field.h>

#include <xfdtd_cuda/common.cuh>

namespace xfdtd::cuda {

class EMF {
  friend class EMFHD;

 public:
  template <xfdtd::EMF::Attribute attribute, xfdtd::Axis::XYZ xyz>
  XFDTD_CUDA_DUAL auto field() const -> const Array3D<Real>& {
    constexpr auto component = xfdtd::EMF::xYZToComponent(xyz);
    constexpr auto f =
        xfdtd::EMF::attributeComponentToField(attribute, component);
    return field<f>();
  }

  template <xfdtd::EMF::Attribute attribute, xfdtd::Axis::XYZ xyz>
  XFDTD_CUDA_DUAL auto field() -> Array3D<Real>& {
    return const_cast<Array3D<Real>&>(
        const_cast<const EMF*>(this)->field<attribute, xyz>());
  }

  template <xfdtd::EMF::Field f>
  XFDTD_CUDA_DUAL auto field() const -> const Array3D<Real>& {
    static_assert(f != xfdtd::EMF::Field::EM && f != xfdtd::EMF::Field::HM,
                  "No Implementation");
    if constexpr (f == xfdtd::EMF::Field::EX) {
      return ex();
    } else if constexpr (f == xfdtd::EMF::Field::EY) {
      return ey();
    } else if constexpr (f == xfdtd::EMF::Field::EZ) {
      return ez();
    } else if constexpr (f == xfdtd::EMF::Field::HX) {
      return hx();
    } else if constexpr (f == xfdtd::EMF::Field::HY) {
      return hy();
    } else if constexpr (f == xfdtd::EMF::Field::HZ) {
      return hz();
    }
  }

  template <xfdtd::EMF::Field f>
  XFDTD_CUDA_DUAL auto field() -> Array3D<Real>& {
    return const_cast<Array3D<Real>&>(const_cast<const EMF*>(this)->field<f>());
  }

  XFDTD_CUDA_DUAL auto ex() const -> const Array3D<Real>& { return *_ex; }

  XFDTD_CUDA_DUAL auto ey() const -> const Array3D<Real>& { return *_ey; }

  XFDTD_CUDA_DUAL auto ez() const -> const Array3D<Real>& { return *_ez; }

  XFDTD_CUDA_DUAL auto hx() const -> const Array3D<Real>& { return *_hx; }

  XFDTD_CUDA_DUAL auto hy() const -> const Array3D<Real>& { return *_hy; }

  XFDTD_CUDA_DUAL auto hz() const -> const Array3D<Real>& { return *_hz; }

  XFDTD_CUDA_DUAL auto ex() -> Array3D<Real>& { return *_ex; }

  XFDTD_CUDA_DUAL auto ey() -> Array3D<Real>& { return *_ey; }

  XFDTD_CUDA_DUAL auto ez() -> Array3D<Real>& { return *_ez; }

  XFDTD_CUDA_DUAL auto hx() -> Array3D<Real>& { return *_hx; }

  XFDTD_CUDA_DUAL auto hy() -> Array3D<Real>& { return *_hy; }

  XFDTD_CUDA_DUAL auto hz() -> Array3D<Real>& { return *_hz; }

 private:
  Array3D<Real>*_ex{}, *_ey{}, *_ez{};
  Array3D<Real>*_hx{}, *_hy{}, *_hz{};
};

}  // namespace xfdtd::cuda

#endif  //__XFDTD_CUDA_ELECTROMAGNETIC_FIELD_CUH__
