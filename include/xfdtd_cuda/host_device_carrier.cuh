#ifndef __XFDTD_CUDA_HOST_DEVICE_CARRIER_CUH__
#define __XFDTD_CUDA_HOST_DEVICE_CARRIER_CUH__

#include <xfdtd_cuda/common.cuh>

namespace xfdtd::cuda {

template <typename Host, typename Device>
class HostDeviceCarrier {
 public:
  explicit HostDeviceCarrier(Host *host) : _host{host} {}

  HostDeviceCarrier(const HostDeviceCarrier &) = delete;

  auto operator=(const HostDeviceCarrier &) -> HostDeviceCarrier & = delete;

  HostDeviceCarrier(HostDeviceCarrier &&other) noexcept
      : _host{other._host}, _device{other._device} {
    other._host = nullptr;
    other._device = nullptr;
  }

  auto operator=(HostDeviceCarrier &&other) noexcept -> HostDeviceCarrier & {
    if (this != &other) {
      _host = other._host;
      _device = other._device;
      other._host = nullptr;
      other._device = nullptr;
    }
    return *this;
  }

  virtual ~HostDeviceCarrier() { releaseBaseDevice(); }

  auto host() { return _host; }

  auto device() { return _device; }

  auto host() const { return _host; }

  auto device() const { return _device; }

  virtual auto copyHostToDevice() -> void = 0;

  virtual auto copyDeviceToHost() -> void = 0;

  virtual auto releaseDevice() -> void = 0;

 protected:
  auto releaseBaseDevice() -> void {
    if (_device != nullptr) {
      XFDTD_CORE_CUDA_CHECK_CUDA_ERROR(cudaFree(_device));
      _device = nullptr;
    }
  }

  auto mallocDevice() -> void {
    releaseBaseDevice();
    XFDTD_CORE_CUDA_CHECK_CUDA_ERROR(cudaMalloc(&_device, sizeof(Device)));
  }

  auto copyToDevice(Device *data) -> void {
    if (_device == nullptr) {
      mallocDevice();
    }
    XFDTD_CORE_CUDA_CHECK_CUDA_ERROR(
        cudaMemcpy(_device, data, sizeof(Device), cudaMemcpyHostToDevice));
  }

  auto copyToHost(Device *data) -> void {
    if (_device == nullptr) {
      throw std::runtime_error("Device is not allocated.");
    }

    XFDTD_CORE_CUDA_CHECK_CUDA_ERROR(
        cudaMemcpy(data, _device, sizeof(Device), cudaMemcpyDeviceToHost));
  }

 private:
  Host *_host;
  Device *_device{};
};

}  // namespace xfdtd::cuda

#endif  // __XFDTD_CUDA_HOST_DEVICE_CARRIER_CUH__
