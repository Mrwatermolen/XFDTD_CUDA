#ifndef __XFDTD_CUDA_ELECTROMAGNETIC_FIELD_HD_CUH__
#define __XFDTD_CUDA_ELECTROMAGNETIC_FIELD_HD_CUH__

#include <xfdtd_cuda/electromagnetic_field/electromagnetic_field.cuh>
#include <xfdtd_cuda/host_device_carrier.cuh>
#include <xfdtd_cuda/tensor_hd.cuh>

namespace xfdtd::cuda {

class EMFHD : public HostDeviceCarrier<xfdtd::EMF, xfdtd::cuda::EMF> {
  using Host = xfdtd::EMF;
  using Device = xfdtd::cuda::EMF;

 public:
  EMFHD(Host* host)
      : HostDeviceCarrier<xfdtd::EMF, xfdtd::cuda::EMF>{host},
        _ex_hd{host->ex()},
        _ey_hd{host->ey()},
        _ez_hd{host->ez()},
        _hx_hd{host->hx()},
        _hy_hd{host->hy()},
        _hz_hd{host->hz()} {}

  auto copyHostToDevice() -> void override {
    if (host() == nullptr) {
      throw std::runtime_error(
          "EMFHD::copyHostToDevice(): Host data is not initialized");
    }

    _ex_hd.copyHostToDevice();
    _ey_hd.copyHostToDevice();
    _ez_hd.copyHostToDevice();
    _hx_hd.copyHostToDevice();
    _hy_hd.copyHostToDevice();
    _hz_hd.copyHostToDevice();

    auto d = Device{};
    d._ex = _ex_hd.device();
    d._ey = _ey_hd.device();
    d._ez = _ez_hd.device();
    d._hx = _hx_hd.device();
    d._hy = _hy_hd.device();
    d._hz = _hz_hd.device();

    copyToDevice(&d);
  }

  auto copyDeviceToHost() -> void override {
    if (host() == nullptr) {
      throw std::runtime_error(
          "EMFHD::copyDeviceToHost(): Host data is not initialized");
    }

    _ex_hd.copyDeviceToHost();
    _ey_hd.copyDeviceToHost();
    _ez_hd.copyDeviceToHost();
    _hx_hd.copyDeviceToHost();
    _hy_hd.copyDeviceToHost();
    _hz_hd.copyDeviceToHost();
  }

  auto releaseDevice() -> void override {
    _ex_hd.releaseDevice();
    _ey_hd.releaseDevice();
    _ez_hd.releaseDevice();
    _hx_hd.releaseDevice();
    _hy_hd.releaseDevice();
    _hz_hd.releaseDevice();

    releaseBaseDevice();
  }

 private:
  TensorHD<Real, 3> _ex_hd, _ey_hd, _ez_hd, _hx_hd, _hy_hd, _hz_hd;
};

}  // namespace xfdtd::cuda

#endif  //__XFDTD_CUDA_ELECTROMAGNETIC_FIELD_HD_CUH__
