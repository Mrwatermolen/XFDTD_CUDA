#include <xfdtd/calculation_param/calculation_param.h>

#include <xfdtd_cuda/calculation_param/fdtd_coefficient.cuh>

namespace xfdtd::cuda {

FDTDCoefficient::FDTDCoefficient(Array3D<Real> *cexe, Array3D<Real> *cexhy,
                                 Array3D<Real> *cexhz, Array3D<Real> *ceye,
                                 Array3D<Real> *ceyhz, Array3D<Real> *ceyhx,
                                 Array3D<Real> *ceze, Array3D<Real> *cezhx,
                                 Array3D<Real> *cezhy, Array3D<Real> *chxh,
                                 Array3D<Real> *chxey, Array3D<Real> *chxez,
                                 Array3D<Real> *chyh, Array3D<Real> *chyez,
                                 Array3D<Real> *chyex, Array3D<Real> *chzh,
                                 Array3D<Real> *chzex, Array3D<Real> *chzey)
    : _cexe{cexe},
      _cexhy{cexhy},
      _cexhz{cexhz},
      _ceye{ceye},
      _ceyhz{ceyhz},
      _ceyhx{ceyhx},
      _ceze{ceze},
      _cezhx{cezhx},
      _cezhy{cezhy},
      _chxh{chxh},
      _chxey{chxey},
      _chxez{chxez},
      _chyh{chyh},
      _chyez{chyez},
      _chyex{chyex},
      _chzh{chzh},
      _chzex{chzex},
      _chzey{chzey} {}

XFDTD_CUDA_GLOBAL auto __kernelCheckFDTDUpdateCoefficient(FDTDCoefficient *f)
    -> void {
  std::printf("FDTDCoefficient::cexe().size()=%lu\n", f->cexe().size());
  std::printf("FDTDCoefficient::cexhy().size()=%lu\n",
              f->coeff<EMF::Attribute::E, Axis::XYZ::X>().size());
  std::printf("FDTDCoefficient::cexhz().size()=%lu\n", f->cexhz().size());
  std::printf("FDTDCoefficient::ceye().size()=%lu\n", f->ceye().size());
  std::printf("FDTDCoefficient::ceyhz().size()=%lu\n", f->ceyhz().size());
  std::printf("FDTDCoefficient::ceyhx().size()=%lu\n", f->ceyhx().size());

  // print dim
  std::printf("FDTDCoefficient::cexe().dim()=%lu\n", f->cexe().dimension());

  // print shape
  std::printf("FDTDCoefficient::cexe().shape()=[%lu,%lu,%lu]\n",
              f->cexe().shape()[0], f->cexe().shape()[1], f->cexe().shape()[2]);

  auto max_ceze = std::numeric_limits<Real>::lowest();
  auto min_ceze = std::numeric_limits<Real>::max();

  for (Index i = 0; i < f->ceze().shape()[0]; ++i) {
    for (Index j = 0; j < f->ceze().shape()[1]; ++j) {
      for (Index k = 0; k < f->ceze().shape()[2]; ++k) {
        auto value = f->ceze().at(i, j, k);
        if (value > max_ceze) {
          max_ceze = value;
        }
        if (value < min_ceze) {
          min_ceze = value;
        }
      }
    }
  }

  std::printf("FDTDCoefficient::ceze().min()=%f\n", min_ceze);
  std::printf("FDTDCoefficient::ceze().max()=%f\n", max_ceze);

  auto max_cehx = std::numeric_limits<Real>::lowest();
  auto min_cehx = std::numeric_limits<Real>::max();

  for (Index i = 0; i < f->cezhx().shape()[0]; ++i) {
    for (Index j = 0; j < f->cezhx().shape()[1]; ++j) {
      for (Index k = 0; k < f->cezhx().shape()[2]; ++k) {
        auto value = f->cezhx().at(i, j, k);
        if (value > max_cehx) {
          max_cehx = value;
        }
        if (value < min_cehx) {
          min_cehx = value;
        }
      }
    }
  }

  std::printf("FDTDCoefficient::cezhx().min()=%f\n", min_cehx);
  std::printf("FDTDCoefficient::cezhx().max()=%f\n", max_cehx);
}

}  // namespace xfdtd::cuda
