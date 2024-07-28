#include <xfdtd/common/type_define.h>
#include <xfdtd/electromagnetic_field/electromagnetic_field.h>
#include <xfdtd/nffft/nffft_time_domain.h>

#include "nf2ff/time_domain/nf2ff_time_domain_agency.cuh"
#include "nf2ff/time_domain/nf2ff_time_domain_data.cuh"
#include "nf2ff/time_domain/nf2ff_time_domain_hd.cuh"
#include "xfdtd_cuda/host_device_carrier.cuh"
#include "xfdtd_cuda/tensor_hd.cuh"

namespace xfdtd::cuda {

static auto rangeToRange(const xfdtd::Range<Real>& range)
    -> xfdtd::cuda::Range<Real> {
  return {range.start(), range.end()};
}

static auto indexTaskToIndexTask(xfdtd::IndexTask task) -> IndexTask {
  return {
      {task.xRange().start(), task.xRange().end()},
      {task.yRange().start(), task.yRange().end()},
      {task.zRange().start(), task.zRange().end()},
  };
}

template <xfdtd::Axis::Direction D>
struct SurfaceCurrentTD
    : public HostDeviceCarrier<void, NF2FFTimeDomainData<D>> {
  using Host = void;
  using Device = NF2FFTimeDomainData<D>;
  inline static constexpr auto xyz = xfdtd::Axis::fromDirectionToXYZ<D>();
  inline static constexpr auto xyz_a = xfdtd::Axis::tangentialAAxis<xyz>();
  inline static constexpr auto xyz_b = xfdtd::Axis::tangentialBAxis<xyz>();

  SurfaceCurrentTD(NFFFTTimeDomain* nf2ff, const GridSpace* grid_space_device,
                   const CalculationParam* calculation_param_device,
                   const EMF* emf_device, IndexTask task)
      : HostDeviceCarrier<Host, Device>{nullptr},
        _grid_space_device{grid_space_device},
        _calculation_param_device{calculation_param_device},
        _emf_device{emf_device},
        _task{task},
        _r_unit{nf2ff->observationDirection()},
        _distance_range_e{
            rangeToRange(nf2ff->distanceRange<xfdtd::EMF::Attribute::E>())},
        _distance_range_h{
            rangeToRange(nf2ff->distanceRange<xfdtd::EMF::Attribute::H>())},
        _wa_hd{nf2ff->equivalentSurfaceCurrent<D, xfdtd::EMF::Attribute::H,
                                               xyz_a>()},
        _wb_hd{nf2ff->equivalentSurfaceCurrent<D, xfdtd::EMF::Attribute::H,
                                               xyz_b>()},
        _ua_hd{nf2ff->equivalentSurfaceCurrent<D, xfdtd::EMF::Attribute::E,
                                               xyz_a>()},
        _ub_hd{nf2ff->equivalentSurfaceCurrent<D, xfdtd::EMF::Attribute::E,
                                               xyz_b>()},
        _ea_prev_hd{nf2ff->fieldPrev<D, xfdtd::EMF::Attribute::E, xyz_a>()},
        _eb_prev_hd{nf2ff->fieldPrev<D, xfdtd::EMF::Attribute::E, xyz_b>()},
        _ha_prev_hd{nf2ff->fieldPrev<D, xfdtd::EMF::Attribute::H, xyz_a>()},
        _hb_prev_hd{nf2ff->fieldPrev<D, xfdtd::EMF::Attribute::H, xyz_b>()} {}

  ~SurfaceCurrentTD() override { releaseDevice(); }

  auto copyHostToDevice() -> void override {
    _wa_hd.copyHostToDevice();
    _wb_hd.copyHostToDevice();
    _ua_hd.copyHostToDevice();
    _ub_hd.copyHostToDevice();

    _ea_prev_hd.copyHostToDevice();
    _eb_prev_hd.copyHostToDevice();
    _ha_prev_hd.copyHostToDevice();
    _hb_prev_hd.copyHostToDevice();

    auto d = Device{};
    d._grid_space = _grid_space_device;
    d._calculation_param = _calculation_param_device;
    d._emf = _emf_device;
    d._task = _task;
    TempVector r{_r_unit.x(), _r_unit.y(), _r_unit.z()};
    d._r_unit = r;
    d._distance_range_e = _distance_range_e;
    d._distance_range_h = _distance_range_h;

    d._wa = _wa_hd.device();
    d._wb = _wb_hd.device();
    d._ua = _ua_hd.device();
    d._ub = _ub_hd.device();

    d._ea_prev = _ea_prev_hd.device();
    d._eb_prev = _eb_prev_hd.device();
    d._ha_prev = _ha_prev_hd.device();
    d._hb_prev = _hb_prev_hd.device();

    this->copyToDevice(&d);
  };

  auto copyDeviceToHost() -> void override {
    _wa_hd.copyDeviceToHost();
    _wb_hd.copyDeviceToHost();
    _ua_hd.copyDeviceToHost();
    _ub_hd.copyDeviceToHost();
  };

  auto releaseDevice() -> void override {
    _wa_hd.releaseDevice();
    _wb_hd.releaseDevice();
    _ua_hd.releaseDevice();
    _ub_hd.releaseDevice();

    _ea_prev_hd.releaseDevice();
    _eb_prev_hd.releaseDevice();
    _ha_prev_hd.releaseDevice();
    _hb_prev_hd.releaseDevice();

    this->releaseBaseDevice();
  };

  const GridSpace* _grid_space_device;
  const CalculationParam* _calculation_param_device;
  const EMF* _emf_device;
  const Vector _r_unit;
  const Range<Real> _distance_range_e, _distance_range_h;
  IndexTask _task;

  TensorHD<Real, 1> _wa_hd, _wb_hd, _ua_hd, _ub_hd;
  TensorHD<Real, 2> _ea_prev_hd, _eb_prev_hd, _ha_prev_hd, _hb_prev_hd;
};

struct SurfaceCurrentSetTD : public HostDeviceCarrier<void, void> {
  SurfaceCurrentSetTD(NFFFTTimeDomain* nf2ff,
                      const GridSpace* grid_space_device,
                      const CalculationParam* calculation_param_device,
                      const EMF* emf_device)
      : HostDeviceCarrier<void, void>{nullptr},
        _xn{nf2ff, grid_space_device, calculation_param_device, emf_device,
            indexTaskToIndexTask(nf2ff->nodeTaskSurfaceXN())},
        _xp{nf2ff, grid_space_device, calculation_param_device, emf_device,
            indexTaskToIndexTask(nf2ff->nodeTaskSurfaceXP())},
        _yn{nf2ff, grid_space_device, calculation_param_device, emf_device,
            indexTaskToIndexTask(nf2ff->nodeTaskSurfaceYN())},
        _yp{nf2ff, grid_space_device, calculation_param_device, emf_device,
            indexTaskToIndexTask(nf2ff->nodeTaskSurfaceYP())},
        _zn{nf2ff, grid_space_device, calculation_param_device, emf_device,
            indexTaskToIndexTask(nf2ff->nodeTaskSurfaceZN())},
        _zp{nf2ff, grid_space_device, calculation_param_device, emf_device,
            indexTaskToIndexTask(nf2ff->nodeTaskSurfaceZP())} {}

  ~SurfaceCurrentSetTD() override { releaseDevice(); }

  auto agency() {
    if (_agency == nullptr) {
      _agency = std::make_unique<NF2FFTimeDoaminAgency>(
          _xn.device(), _xp.device(), _yn.device(), _yp.device(), _zn.device(),
          _zp.device());
    }

    return _agency.get();
  }

  auto copyHostToDevice() -> void override {
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
  }

  SurfaceCurrentTD<xfdtd::Axis::Direction::XN> _xn;
  SurfaceCurrentTD<xfdtd::Axis::Direction::XP> _xp;
  SurfaceCurrentTD<xfdtd::Axis::Direction::YN> _yn;
  SurfaceCurrentTD<xfdtd::Axis::Direction::YP> _yp;
  SurfaceCurrentTD<xfdtd::Axis::Direction::ZN> _zn;
  SurfaceCurrentTD<xfdtd::Axis::Direction::ZP> _zp;

  std::unique_ptr<NF2FFTimeDoaminAgency> _agency;
};

NF2FFTimeDomainHD::NF2FFTimeDomainHD(
    xfdtd::NFFFTTimeDomain* host,
    std::shared_ptr<const GridSpaceHD> grid_space_hd,
    std::shared_ptr<const CalculationParamHD> calculation_param_hd,
    std::shared_ptr<const EMFHD> emf_hd)
    : HostDeviceCarrier<xfdtd::NFFFTTimeDomain, void>{host},
      _grid_space_hd{grid_space_hd},
      _calculation_param_hd{calculation_param_hd},
      _emf_hd{emf_hd},
      _surface_current_set{new SurfaceCurrentSetTD(
          host, grid_space_hd->device(), calculation_param_hd->device(),
          emf_hd->device())} {}

NF2FFTimeDomainHD::~NF2FFTimeDomainHD() {
  releaseDevice();
  if (_surface_current_set != nullptr) {
    delete _surface_current_set;
    _surface_current_set = nullptr;
  }
}

auto NF2FFTimeDomainHD::copyHostToDevice() -> void {
  _surface_current_set->copyHostToDevice();
}

auto NF2FFTimeDomainHD::copyDeviceToHost() -> void {
  _surface_current_set->copyDeviceToHost();
}

auto NF2FFTimeDomainHD::releaseDevice() -> void {
  _surface_current_set->releaseDevice();
}

auto NF2FFTimeDomainHD::agency() -> NF2FFTimeDoaminAgency* {
  return _surface_current_set->agency();
}

}  // namespace xfdtd::cuda
