#ifndef __XFDTD_CUDA_FDTD_COEFFICIENT_CUH__
#define __XFDTD_CUDA_FDTD_COEFFICIENT_CUH__

#include <xfdtd/common/type_define.h>
#include <xfdtd/coordinate_system/coordinate_system.h>
#include <xfdtd/electromagnetic_field/electromagnetic_field.h>
#include <xfdtd/util/transform.h>

#include <xfdtd_cuda/common.cuh>
#include <xfdtd_cuda/tensor.cuh>
#include <xfdtd_cuda/tensor_texture_ref.cuh>

namespace xfdtd::cuda {

class FDTDCoefficient {
 public:
  friend class FDTDCoefficientHD;

 public:
  FDTDCoefficient() = default;

  FDTDCoefficient(
      TensorTextureRef<Real, 3> cexe, TensorTextureRef<Real, 3> cexhy,
      TensorTextureRef<Real, 3> cexhz, TensorTextureRef<Real, 3> ceye,
      TensorTextureRef<Real, 3> ceyhz, TensorTextureRef<Real, 3> ceyhx,
      TensorTextureRef<Real, 3> ceze, TensorTextureRef<Real, 3> cezhx,
      TensorTextureRef<Real, 3> cezhy, TensorTextureRef<Real, 3> chxh,
      TensorTextureRef<Real, 3> chxey, TensorTextureRef<Real, 3> chxez,
      TensorTextureRef<Real, 3> chyh, TensorTextureRef<Real, 3> chyez,
      TensorTextureRef<Real, 3> chyex, TensorTextureRef<Real, 3> chzh,
      TensorTextureRef<Real, 3> chzex, TensorTextureRef<Real, 3> chzey);

 public:
  XFDTD_CUDA_DUAL auto &cexe() const { return _cexe; }
  XFDTD_CUDA_DUAL auto &cexhy() const { return _cexhy; }
  XFDTD_CUDA_DUAL auto &cexhz() const { return _cexhz; }
  XFDTD_CUDA_DUAL auto &ceye() const { return _ceye; }
  XFDTD_CUDA_DUAL auto &ceyhz() const { return _ceyhz; }
  XFDTD_CUDA_DUAL auto &ceyhx() const { return _ceyhx; }
  XFDTD_CUDA_DUAL auto &ceze() const { return _ceze; }
  XFDTD_CUDA_DUAL auto &cezhx() const { return _cezhx; }
  XFDTD_CUDA_DUAL auto &cezhy() const { return _cezhy; }

  XFDTD_CUDA_DUAL auto &chxh() const { return _chxh; }
  XFDTD_CUDA_DUAL auto &chxey() const { return _chxey; }
  XFDTD_CUDA_DUAL auto &chxez() const { return _chxez; }
  XFDTD_CUDA_DUAL auto &chyh() const { return _chyh; }
  XFDTD_CUDA_DUAL auto &chyez() const { return _chyez; }
  XFDTD_CUDA_DUAL auto &chyex() const { return _chyex; }
  XFDTD_CUDA_DUAL auto &chzh() const { return _chzh; }
  XFDTD_CUDA_DUAL auto &chzex() const { return _chzex; }
  XFDTD_CUDA_DUAL auto &chzey() const { return _chzey; }

  XFDTD_CUDA_DUAL auto &cexe() { return _cexe; }
  XFDTD_CUDA_DUAL auto &cexhy() { return _cexhy; }
  XFDTD_CUDA_DUAL auto &cexhz() { return _cexhz; }
  XFDTD_CUDA_DUAL auto &ceye() { return _ceye; }
  XFDTD_CUDA_DUAL auto &ceyhz() { return _ceyhz; }
  XFDTD_CUDA_DUAL auto &ceyhx() { return _ceyhx; }
  XFDTD_CUDA_DUAL auto &ceze() { return _ceze; }
  XFDTD_CUDA_DUAL auto &cezhx() { return _cezhx; }
  XFDTD_CUDA_DUAL auto &cezhy() { return _cezhy; }

  XFDTD_CUDA_DUAL auto &chxh() { return _chxh; }
  XFDTD_CUDA_DUAL auto &chxey() { return _chxey; }
  XFDTD_CUDA_DUAL auto &chxez() { return _chxez; }
  XFDTD_CUDA_DUAL auto &chyh() { return _chyh; }
  XFDTD_CUDA_DUAL auto &chyez() { return _chyez; }
  XFDTD_CUDA_DUAL auto &chyex() { return _chyex; }
  XFDTD_CUDA_DUAL auto &chzh() { return _chzh; }
  XFDTD_CUDA_DUAL auto &chzex() { return _chzex; }
  XFDTD_CUDA_DUAL auto &chzey() { return _chzey; }

 public:
  template <xfdtd::EMF::Attribute c, Axis::XYZ xyz>
  XFDTD_CUDA_DUAL auto &coeff() const {
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
  XFDTD_CUDA_DUAL auto &coeff() const {
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

 private:
  TensorTextureRef<Real, 3> _cexe{};
  TensorTextureRef<Real, 3> _cexhy{};
  TensorTextureRef<Real, 3> _cexhz{};
  TensorTextureRef<Real, 3> _ceye{};
  TensorTextureRef<Real, 3> _ceyhz{};
  TensorTextureRef<Real, 3> _ceyhx{};
  TensorTextureRef<Real, 3> _ceze{};
  TensorTextureRef<Real, 3> _cezhx{};
  TensorTextureRef<Real, 3> _cezhy{};

  TensorTextureRef<Real, 3> _chxh{};
  TensorTextureRef<Real, 3> _chxey{};
  TensorTextureRef<Real, 3> _chxez{};
  TensorTextureRef<Real, 3> _chyh{};
  TensorTextureRef<Real, 3> _chyez{};
  TensorTextureRef<Real, 3> _chyex{};
  TensorTextureRef<Real, 3> _chzh{};
  TensorTextureRef<Real, 3> _chzex{};
  TensorTextureRef<Real, 3> _chzey{};
};

XFDTD_CUDA_GLOBAL auto __kernelCheckFDTDUpdateCoefficient(FDTDCoefficient *f)
    -> void;

}  // namespace xfdtd::cuda

#endif  // __XFDTD_CUDA_FDTD_COEFFICIENT_CUH__
