#ifndef __XFDTD_CUDA_FDTD_COEFFICIENT_CUH__
#define __XFDTD_CUDA_FDTD_COEFFICIENT_CUH__

#include <xfdtd/common/type_define.h>
#include <xfdtd/coordinate_system/coordinate_system.h>
#include <xfdtd/electromagnetic_field/electromagnetic_field.h>
#include <xfdtd/util/transform.h>

#include <xfdtd_cuda/common.cuh>
#include <xfdtd_cuda/tensor.cuh>

namespace xfdtd::cuda {

class FDTDCoefficient {
 public:
  friend class FDTDCoefficientHD;

 public:
  FDTDCoefficient() = default;

  FDTDCoefficient(Array3D<Real> *cexe, Array3D<Real> *cexhy,
                  Array3D<Real> *cexhz, Array3D<Real> *ceye,
                  Array3D<Real> *ceyhz, Array3D<Real> *ceyhx,
                  Array3D<Real> *ceze, Array3D<Real> *cezhx,
                  Array3D<Real> *cezhy, Array3D<Real> *chxh,
                  Array3D<Real> *chxey, Array3D<Real> *chxez,
                  Array3D<Real> *chyh, Array3D<Real> *chyez,
                  Array3D<Real> *chyex, Array3D<Real> *chzh,
                  Array3D<Real> *chzex, Array3D<Real> *chzey);

 public:
  template <xfdtd::EMF::Attribute c, Axis::XYZ xyz>
  XFDTD_CUDA_DUAL auto coeff() const -> const Array3D<Real> & {
    constexpr auto f = transform::attributeXYZToField<c, xyz>();
    if constexpr (f == xfdtd::EMF::Field::EX) {
      return cexe();
    } else if constexpr (f == xfdtd::EMF::Field::EY) {
      return ceye();
    } else if constexpr (f == xfdtd::EMF::Field::EZ) {
      return ceze();
    } else if constexpr (f == xfdtd::EMF::Field::HX) {
      return chxh();
    } else if constexpr (f == xfdtd::EMF::Field::HY) {
      return chyh();
    } else if constexpr (f == xfdtd::EMF::Field::HZ) {
      return chzh();
    } else {
      static_assert(
          f == xfdtd::EMF::Field::EX || f == xfdtd::EMF::Field::EY ||
              f == xfdtd::EMF::Field::EZ || f == xfdtd::EMF::Field::HX ||
              f == xfdtd::EMF::Field::HY || f == xfdtd::EMF::Field::HZ,
          "FDTDUpdateCoefficient::coeff(): Invalid  xfdtd::EMF::Field");
    }
  }

  template <xfdtd::EMF::Attribute a, Axis::XYZ xyz_a, xfdtd::EMF::Attribute b,
            Axis::XYZ xyz_b>
  XFDTD_CUDA_DUAL auto coeff() const -> const Array3D<Real> & {
    constexpr auto f = transform::attributeXYZToField<a, xyz_a>();
    constexpr auto g = transform::attributeXYZToField<b, xyz_b>();
    if constexpr (f == xfdtd::EMF::Field::EX && g == xfdtd::EMF::Field::HY) {
      return cexhy();
    } else if constexpr (f == xfdtd::EMF::Field::EX &&
                         g == xfdtd::EMF::Field::HZ) {
      return cexhz();
    } else if constexpr (f == xfdtd::EMF::Field::EY &&
                         g == xfdtd::EMF::Field::HZ) {
      return ceyhz();
    } else if constexpr (f == xfdtd::EMF::Field::EY &&
                         g == xfdtd::EMF::Field::HX) {
      return ceyhx();
    } else if constexpr (f == xfdtd::EMF::Field::EZ &&
                         g == xfdtd::EMF::Field::HX) {
      return cezhx();
    } else if constexpr (f == xfdtd::EMF::Field::EZ &&
                         g == xfdtd::EMF::Field::HY) {
      return cezhy();
    } else if constexpr (f == xfdtd::EMF::Field::HX &&
                         g == xfdtd::EMF::Field::EY) {
      return chxey();
    } else if constexpr (f == xfdtd::EMF::Field::HX &&
                         g == xfdtd::EMF::Field::EZ) {
      return chxez();
    } else if constexpr (f == xfdtd::EMF::Field::HY &&
                         g == xfdtd::EMF::Field::EZ) {
      return chyez();
    } else if constexpr (f == xfdtd::EMF::Field::HY &&
                         g == xfdtd::EMF::Field::EX) {
      return chyex();
    } else if constexpr (f == xfdtd::EMF::Field::HZ &&
                         g == xfdtd::EMF::Field::EX) {
      return chzex();
    } else if constexpr (f == xfdtd::EMF::Field::HZ &&
                         g == xfdtd::EMF::Field::EY) {
      return chzey();
    } else {
      static_assert(
          f == xfdtd::EMF::Field::EX || f == xfdtd::EMF::Field::EY ||
              f == xfdtd::EMF::Field::EZ || f == xfdtd::EMF::Field::HX ||
              f == xfdtd::EMF::Field::HY || f == xfdtd::EMF::Field::HZ,
          "FDTDUpdateCoefficient::coeff(): Invalid  xfdtd::EMF::Field");
    }
  };

 public:
  XFDTD_CUDA_DUAL auto cexe() const -> const Array3D<Real> & { return *_cexe; }
  XFDTD_CUDA_DUAL auto cexhy() const -> const Array3D<Real> & {
    return *_cexhy;
  }
  XFDTD_CUDA_DUAL auto cexhz() const -> const Array3D<Real> & {
    return *_cexhz;
  }
  XFDTD_CUDA_DUAL auto ceye() const -> const Array3D<Real> & { return *_ceye; }
  XFDTD_CUDA_DUAL auto ceyhz() const -> const Array3D<Real> & {
    return *_ceyhz;
  }
  XFDTD_CUDA_DUAL auto ceyhx() const -> const Array3D<Real> & {
    return *_ceyhx;
  }
  XFDTD_CUDA_DUAL auto ceze() const -> const Array3D<Real> & { return *_ceze; }
  XFDTD_CUDA_DUAL auto cezhx() const -> const Array3D<Real> & {
    return *_cezhx;
  }
  XFDTD_CUDA_DUAL auto cezhy() const -> const Array3D<Real> & {
    return *_cezhy;
  }

  XFDTD_CUDA_DUAL auto chxh() const -> const Array3D<Real> & { return *_chxh; }
  XFDTD_CUDA_DUAL auto chxey() const -> const Array3D<Real> & {
    return *_chxey;
  }
  XFDTD_CUDA_DUAL auto chxez() const -> const Array3D<Real> & {
    return *_chxez;
  }
  XFDTD_CUDA_DUAL auto chyh() const -> const Array3D<Real> & { return *_chyh; }
  XFDTD_CUDA_DUAL auto chyez() const -> const Array3D<Real> & {
    return *_chyez;
  }
  XFDTD_CUDA_DUAL auto chyex() const -> const Array3D<Real> & {
    return *_chyex;
  }
  XFDTD_CUDA_DUAL auto chzh() const -> const Array3D<Real> & { return *_chzh; }
  XFDTD_CUDA_DUAL auto chzex() const -> const Array3D<Real> & {
    return *_chzex;
  }
  XFDTD_CUDA_DUAL auto chzey() const -> const Array3D<Real> & {
    return *_chzey;
  }

  XFDTD_CUDA_DUAL auto cexe() -> Array3D<Real> & { return *_cexe; }
  XFDTD_CUDA_DUAL auto cexhy() -> Array3D<Real> & { return *_cexhy; }
  XFDTD_CUDA_DUAL auto cexhz() -> Array3D<Real> & { return *_cexhz; }
  XFDTD_CUDA_DUAL auto ceye() -> Array3D<Real> & { return *_ceye; }
  XFDTD_CUDA_DUAL auto ceyhz() -> Array3D<Real> & { return *_ceyhz; }
  XFDTD_CUDA_DUAL auto ceyhx() -> Array3D<Real> & { return *_ceyhx; }
  XFDTD_CUDA_DUAL auto ceze() -> Array3D<Real> & { return *_ceze; }
  XFDTD_CUDA_DUAL auto cezhx() -> Array3D<Real> & { return *_cezhx; }
  XFDTD_CUDA_DUAL auto cezhy() -> Array3D<Real> & { return *_cezhy; }

  XFDTD_CUDA_DUAL auto chxh() -> Array3D<Real> & { return *_chxh; }
  XFDTD_CUDA_DUAL auto chxey() -> Array3D<Real> & { return *_chxey; }
  XFDTD_CUDA_DUAL auto chxez() -> Array3D<Real> & { return *_chxez; }
  XFDTD_CUDA_DUAL auto chyh() -> Array3D<Real> & { return *_chyh; }
  XFDTD_CUDA_DUAL auto chyez() -> Array3D<Real> & { return *_chyez; }
  XFDTD_CUDA_DUAL auto chyex() -> Array3D<Real> & { return *_chyex; }
  XFDTD_CUDA_DUAL auto chzh() -> Array3D<Real> & { return *_chzh; }
  XFDTD_CUDA_DUAL auto chzex() -> Array3D<Real> & { return *_chzex; }
  XFDTD_CUDA_DUAL auto chzey() -> Array3D<Real> & { return *_chzey; }

 private:
  Array3D<Real> *_cexe{};
  Array3D<Real> *_cexhy{};
  Array3D<Real> *_cexhz{};
  Array3D<Real> *_ceye{};
  Array3D<Real> *_ceyhz{};
  Array3D<Real> *_ceyhx{};
  Array3D<Real> *_ceze{};
  Array3D<Real> *_cezhx{};
  Array3D<Real> *_cezhy{};

  Array3D<Real> *_chxh{};
  Array3D<Real> *_chxey{};
  Array3D<Real> *_chxez{};
  Array3D<Real> *_chyh{};
  Array3D<Real> *_chyez{};
  Array3D<Real> *_chyex{};
  Array3D<Real> *_chzh{};
  Array3D<Real> *_chzex{};
  Array3D<Real> *_chzey{};
};

XFDTD_CUDA_GLOBAL auto __kernelCheckFDTDUpdateCoefficient(FDTDCoefficient *f)
    -> void;

}  // namespace xfdtd::cuda

#endif  // __XFDTD_CUDA_FDTD_COEFFICIENT_CUH__
