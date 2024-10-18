#include <complex.h>
#include <xfdtd/common/index_task.h>
#include <xfdtd/common/type_define.h>
#include <xfdtd/electromagnetic_field/electromagnetic_field.h>
#include <xfdtd/nffft/nffft_frequency_domain.h>

#include <complex>
#include <memory>

#include "nf2ff/frequency_domain/nf2ff_frequency_domain_agency.cuh"
#include "nf2ff/frequency_domain/nf2ff_frequency_domain_data.cuh"
#include "nf2ff/frequency_domain/nf2ff_frequency_domain_hd.cuh"
#include "xfdtd_cuda/common.cuh"
#include "xfdtd_cuda/grid_space/grid_space.cuh"
#include "xfdtd_cuda/host_device_carrier.cuh"
#include "xfdtd_cuda/index_task.cuh"
#include "xfdtd_cuda/tensor_hd.cuh"

namespace xfdtd::cuda {

static auto indexTaskToIndexTask(xfdtd::IndexTask task) -> IndexTask {
  return {
      {task.xRange().start(), task.xRange().end()},
      {task.yRange().start(), task.yRange().end()},
      {task.zRange().start(), task.zRange().end()},
  };
}

template <xfdtd::Axis::Direction D>
struct SurfaceCurrentFD : HostDeviceCarrier<void, NF2FFFrequencyDomainData<D>> {
  using Host = void;
  using Device = NF2FFFrequencyDomainData<D>;
  inline static constexpr auto xyz = xfdtd::Axis::fromDirectionToXYZ<D>();
  inline static constexpr auto xyz_a = xfdtd::Axis::tangentialAAxis<xyz>();
  inline static constexpr auto xyz_b = xfdtd::Axis::tangentialBAxis<xyz>();

  SurfaceCurrentFD(NFFFTFrequencyDomain* nf2ff, Index i,
                 const GridSpace* grid_space_device,
                 const CalculationParam* calculation_param_device,
                 const EMF* emf_device, IndexTask task)
      : HostDeviceCarrier<Host, Device>{nullptr},
        _grid_space_device{grid_space_device},
        _calculation_param_device{calculation_param_device},
        _emf_device{emf_device},
        _task{task},
        _ja_hd{
            nf2ff->equivalentSurfaceCurrent<D, xfdtd::EMF::Attribute::H, xyz_a>(
                i)},
        _jb_hd{
            nf2ff->equivalentSurfaceCurrent<D, xfdtd::EMF::Attribute::H, xyz_b>(
                i)},
        _ma_hd{
            nf2ff->equivalentSurfaceCurrent<D, xfdtd::EMF::Attribute::E, xyz_a>(
                i)},
        _mb_hd{
            nf2ff->equivalentSurfaceCurrent<D, xfdtd::EMF::Attribute::E, xyz_b>(
                i)} {}

  ~SurfaceCurrentFD() override { releaseDevice(); }

  auto copyHostToDevice() -> void override {
    _ja_hd.copyHostToDevice();
    _jb_hd.copyHostToDevice();
    _ma_hd.copyHostToDevice();
    _mb_hd.copyHostToDevice();

    auto d = Device{};
    d._grid_space = _grid_space_device;
    d._calculation_param = _calculation_param_device;
    d._emf = _emf_device;
    d._task = _task;
    static_assert(sizeof(thrust::complex<Real>) == sizeof(std::complex<Real>),
                  "CUDA complex type size mismatch");
    d._ja =
        reinterpret_cast<Tensor<thrust::complex<Real>, 3>*>(_ja_hd.device());
    d._jb =
        reinterpret_cast<Tensor<thrust::complex<Real>, 3>*>(_jb_hd.device());
    d._ma =
        reinterpret_cast<Tensor<thrust::complex<Real>, 3>*>(_ma_hd.device());
    d._mb =
        reinterpret_cast<Tensor<thrust::complex<Real>, 3>*>(_mb_hd.device());
    if (_transform_e_device == nullptr) {
      throw std::runtime_error(
          "SurfaceCurrentFD::_transform_e_device is nullptr");
    }
    if (_transform_h_device == nullptr) {
      throw std::runtime_error(
          "SurfaceCurrentFD::_transform_h_device is nullptr");
    }

    d._transform_e = reinterpret_cast<Tensor<thrust::complex<Real>, 1>*>(
        _transform_e_device);
    d._transform_h = reinterpret_cast<Tensor<thrust::complex<Real>, 1>*>(
        _transform_h_device);

    this->copyToDevice(&d);
  }

  auto copyDeviceToHost() -> void override {
    _ja_hd.copyDeviceToHost();
    _jb_hd.copyDeviceToHost();
    _ma_hd.copyDeviceToHost();
    _mb_hd.copyDeviceToHost();
  }

  auto releaseDevice() -> void override {
    _ja_hd.releaseDevice();
    _jb_hd.releaseDevice();
    _ma_hd.releaseDevice();
    _mb_hd.releaseDevice();
    _transform_e_device = nullptr;
    _transform_h_device = nullptr;
    this->releaseBaseDevice();
  }

  const GridSpace* _grid_space_device;
  const CalculationParam* _calculation_param_device;
  const EMF* _emf_device;
  IndexTask _task;
  TensorHD<std::complex<Real>, 3> _ja_hd, _jb_hd, _ma_hd, _mb_hd;

  TensorHD<std::complex<Real>, 1>::DeviceTensor *_transform_e_device{nullptr},
      *_transform_h_device{nullptr};
};

// XFDTD_CUDA_GLOBAL void __test(auto a) {
//   for (Index i = 0; i < 5; ++i) {
//     std::printf("a: h.real() = %.5e, h.imag() = %e\n", a->data()[i].real(),
//                 a->data()[i].imag());
//     auto b = (*a)(i) * static_cast<Real>(i);
//     std::printf("i:%f b: h.real() = %f, b.imag() = %f\n", static_cast<Real>(i),
//                 b.real(), b.imag());
//   }
// }

struct SurfaceCurrentSetFD : public HostDeviceCarrier<void, void> {
  SurfaceCurrentSetFD(NFFFTFrequencyDomain* nf2ff, Index i,
                    const GridSpace* grid_space_device,
                    const CalculationParam* calculation_param_device,
                    const EMF* emf_device)
      : HostDeviceCarrier<void, void>{nullptr},
        _xn{nf2ff,
            i,
            grid_space_device,
            calculation_param_device,
            emf_device,
            indexTaskToIndexTask(nf2ff->globalTaskSurfaceXN())},
        _xp{nf2ff,
            i,
            grid_space_device,
            calculation_param_device,
            emf_device,
            indexTaskToIndexTask(nf2ff->globalTaskSurfaceXP())},
        _yn{nf2ff,
            i,
            grid_space_device,
            calculation_param_device,
            emf_device,
            indexTaskToIndexTask(nf2ff->globalTaskSurfaceYN())},
        _yp{nf2ff,
            i,
            grid_space_device,
            calculation_param_device,
            emf_device,
            indexTaskToIndexTask(nf2ff->globalTaskSurfaceYP())},
        _zn{nf2ff,
            i,
            grid_space_device,
            calculation_param_device,
            emf_device,
            indexTaskToIndexTask(nf2ff->globalTaskSurfaceZN())},
        _zp{nf2ff,
            i,
            grid_space_device,
            calculation_param_device,
            emf_device,
            indexTaskToIndexTask(nf2ff->globalTaskSurfaceZP())},
        _transform_e_hd{nf2ff->transformE(i)},
        _transform_h_hd{nf2ff->transformH(i)} {}

  ~SurfaceCurrentSetFD() override { releaseDevice(); }

  auto copyHostToDevice() -> void override {
    _transform_e_hd.copyHostToDevice();
    _transform_h_hd.copyHostToDevice();

    _xn._transform_e_device = _transform_e_hd.device();
    _xp._transform_e_device = _transform_e_hd.device();
    _yn._transform_e_device = _transform_e_hd.device();
    _yp._transform_e_device = _transform_e_hd.device();
    _zn._transform_e_device = _transform_e_hd.device();
    _zp._transform_e_device = _transform_e_hd.device();

    _xn._transform_h_device = _transform_h_hd.device();
    _xp._transform_h_device = _transform_h_hd.device();
    _yn._transform_h_device = _transform_h_hd.device();
    _yp._transform_h_device = _transform_h_hd.device();
    _zn._transform_h_device = _transform_h_hd.device();
    _zp._transform_h_device = _transform_h_hd.device();

    _xn.copyHostToDevice();
    _xp.copyHostToDevice();
    _yn.copyHostToDevice();
    _yp.copyHostToDevice();
    _zn.copyHostToDevice();
    _zp.copyHostToDevice();
  }

  auto copyDeviceToHost() -> void override {
    _xn.copyDeviceToHost();
    _xp.copyDeviceToHost();
    _yn.copyDeviceToHost();
    _yp.copyDeviceToHost();
    _zn.copyDeviceToHost();
    _zp.copyDeviceToHost();
  }

  auto releaseDevice() -> void override {
    _xn.releaseDevice();
    _xp.releaseDevice();
    _yn.releaseDevice();
    _yp.releaseDevice();
    _zn.releaseDevice();
    _zp.releaseDevice();
    _transform_e_hd.releaseDevice();
    _transform_h_hd.releaseDevice();
  }

  auto agency() {
    if (_agency == nullptr) {
      _agency = std::make_unique<NF2FFFrequencyDomainAgency>(
          _xn.device(), _xp.device(), _yn.device(), _yp.device(), _zn.device(),
          _zp.device());
    }

    return _agency.get();
  }

  SurfaceCurrentFD<xfdtd::Axis::Direction::XN> _xn;
  SurfaceCurrentFD<xfdtd::Axis::Direction::XP> _xp;
  SurfaceCurrentFD<xfdtd::Axis::Direction::YN> _yn;
  SurfaceCurrentFD<xfdtd::Axis::Direction::YP> _yp;
  SurfaceCurrentFD<xfdtd::Axis::Direction::ZN> _zn;
  SurfaceCurrentFD<xfdtd::Axis::Direction::ZP> _zp;
  TensorHD<std::complex<Real>, 1> _transform_e_hd, _transform_h_hd;
  std::unique_ptr<NF2FFFrequencyDomainAgency> _agency;
};

NF2FFFrequencyDomainHD::NF2FFFrequencyDomainHD(
    Host* host, std::shared_ptr<const GridSpaceHD> grid_space_hd,
    std::shared_ptr<const CalculationParamHD> calculation_param_hd,
    std::shared_ptr<const EMFHD> emf_hd)
    : HostDeviceCarrier<Host, Device>{host},
      _grid_space_hd{grid_space_hd},
      _calculation_param_hd{calculation_param_hd},
      _emf_hd{emf_hd} {
  for (auto i{0}; i < host->freqCount(); ++i) {
    _surface_current_set.emplace_back(std::make_unique<SurfaceCurrentSetFD>(
        host, i, grid_space_hd->device(), calculation_param_hd->device(),
        emf_hd->device()));
  }
}

NF2FFFrequencyDomainHD::~NF2FFFrequencyDomainHD() { 
  // for(auto&& v : _surface_current_set) {
  //   delete v;
  //   v = nullptr;
  // }
  // _surface_current_set.resize(0);
  releaseDevice(); }

auto NF2FFFrequencyDomainHD::copyHostToDevice() -> void {
  for (auto&& s : _surface_current_set) {
    s->copyHostToDevice();
  }
}

auto NF2FFFrequencyDomainHD::copyDeviceToHost() -> void {
  for (auto&& s : _surface_current_set) {
    s->copyDeviceToHost();
  }
}

auto NF2FFFrequencyDomainHD::releaseDevice() -> void {
  for (auto&& s : _surface_current_set) {
    s->releaseDevice();
  }
}

auto NF2FFFrequencyDomainHD::agencies()
    -> std::vector<NF2FFFrequencyDomainAgency*>& {
  if (_agencies.size() == 0) {
    for (auto&& s : _surface_current_set) {
      _agencies.emplace_back(s->agency());
    }
  }

  return _agencies;
}

}  // namespace xfdtd::cuda