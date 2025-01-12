#include <xfdtd/calculation_param/calculation_param.h>

#include <utility>
#include <xfdtd_cuda/calculation_param/fdtd_coefficient.cuh>

namespace xfdtd::cuda {

FDTDCoefficient::FDTDCoefficient(
    TensorTextureRef<Real, 3> cexe, TensorTextureRef<Real, 3> cexhy,
    TensorTextureRef<Real, 3> cexhz, TensorTextureRef<Real, 3> ceye,
    TensorTextureRef<Real, 3> ceyhz, TensorTextureRef<Real, 3> ceyhx,
    TensorTextureRef<Real, 3> ceze, TensorTextureRef<Real, 3> cezhx,
    TensorTextureRef<Real, 3> cezhy, TensorTextureRef<Real, 3> chxh,
    TensorTextureRef<Real, 3> chxey, TensorTextureRef<Real, 3> chxez,
    TensorTextureRef<Real, 3> chyh, TensorTextureRef<Real, 3> chyez,
    TensorTextureRef<Real, 3> chyex, TensorTextureRef<Real, 3> chzh,
    TensorTextureRef<Real, 3> chzex, TensorTextureRef<Real, 3> chzey)
    : _cexe{std::move(cexe)},
      _cexhy{std::move(cexhy)},
      _cexhz{std::move(cexhz)},
      _ceye{std::move(ceye)},
      _ceyhz{std::move(ceyhz)},
      _ceyhx{std::move(ceyhx)},
      _ceze{std::move(ceze)},
      _cezhx{std::move(cezhx)},
      _cezhy{std::move(cezhy)},
      _chxh{std::move(chxh)},
      _chxey{std::move(chxey)},
      _chxez{std::move(chxez)},
      _chyh{std::move(chyh)},
      _chyez{std::move(chyez)},
      _chyex{std::move(chyex)},
      _chzh{std::move(chzh)},
      _chzex{std::move(chzex)},
      _chzey{std::move(chzey)} {}

XFDTD_CUDA_GLOBAL auto __kernelCheckFDTDUpdateCoefficient(FDTDCoefficient *f)
    -> void {
  // std::printf("FDTDCoefficient::cexe().size()=%lu\n", f->cexe().size());
  // std::printf("FDTDCoefficient::cexhy().size()=%lu\n",
  //             f->coeff<EMF::Attribute::E, Axis::XYZ::X>().size());
  // std::printf("FDTDCoefficient::cexhz().size()=%lu\n", f->cexhz().size());
  // std::printf("FDTDCoefficient::ceye().size()=%lu\n", f->ceye().size());
  // std::printf("FDTDCoefficient::ceyhz().size()=%lu\n", f->ceyhz().size());
  // std::printf("FDTDCoefficient::ceyhx().size()=%lu\n", f->ceyhx().size());

  // // print dim
  // std::printf("FDTDCoefficient::cexe().dim()=%lu\n", f->cexe().dimension());

  // // print shape
  // std::printf("FDTDCoefficient::cexe().shape()=[%lu,%lu,%lu]\n",
  //             f->cexe().shape()[0], f->cexe().shape()[1],
  //             f->cexe().shape()[2]);

  // auto max_ceze = std::numeric_limits<Real>::lowest();
  // auto min_ceze = std::numeric_limits<Real>::max();

  // for (Index i = 0; i < f->ceze().shape()[0]; ++i) {
  //   for (Index j = 0; j < f->ceze().shape()[1]; ++j) {
  //     for (Index k = 0; k < f->ceze().shape()[2]; ++k) {
  //       auto value = f->ceze().at(i, j, k);
  //       if (value > max_ceze) {
  //         max_ceze = value;
  //       }
  //       if (value < min_ceze) {
  //         min_ceze = value;
  //       }
  //     }
  //   }
  // }

  // std::printf("FDTDCoefficient::ceze().min()=%f\n", min_ceze);
  // std::printf("FDTDCoefficient::ceze().max()=%f\n", max_ceze);

  // auto max_cehx = std::numeric_limits<Real>::lowest();
  // auto min_cehx = std::numeric_limits<Real>::max();

  // for (Index i = 0; i < f->cezhx().shape()[0]; ++i) {
  //   for (Index j = 0; j < f->cezhx().shape()[1]; ++j) {
  //     for (Index k = 0; k < f->cezhx().shape()[2]; ++k) {
  //       auto value = f->cezhx().at(i, j, k);
  //       if (value > max_cehx) {
  //         max_cehx = value;
  //       }
  //       if (value < min_cehx) {
  //         min_cehx = value;
  //       }
  //     }
  //   }
  // }

  // std::printf("FDTDCoefficient::cezhx().min()=%f\n", min_cehx);
  // std::printf("FDTDCoefficient::cezhx().max()=%f\n", max_cehx);
}

}  // namespace xfdtd::cuda
