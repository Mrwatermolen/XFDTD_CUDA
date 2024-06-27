#include <xfdtd_cuda/calculation_param/calculation_param.cuh>

namespace xfdtd {

namespace cuda {

CalculationParamHD::CalculationParamHD(Host* host)
    : _host{host},
      _device{nullptr},
      _time_param_hd{host->timeParam().get()},
      _fdtd_coefficient_hd{host->fdtdCoefficient().get()} {}

CalculationParamHD::~CalculationParamHD() { releaseDevice(); }

auto CalculationParamHD::copyHostToDevice() -> void {
  if (_device != nullptr) {
    XFDTD_CORE_CUDA_CHECK_CUDA_ERROR(cudaFree(_device));
  }

  XFDTD_CORE_CUDA_CHECK_CUDA_ERROR(cudaMalloc(&_device, sizeof(Device)));

  _time_param_hd.copyHostToDevice();
  _fdtd_coefficient_hd.copyHostToDevice();

  auto d = Device{};
  d._time_param = _time_param_hd.device();
  d._fdtd_coefficient = _fdtd_coefficient_hd.device();

  XFDTD_CORE_CUDA_CHECK_CUDA_ERROR(
      cudaMemcpy(_device, &d, sizeof(Device), cudaMemcpyHostToDevice));
}

auto CalculationParamHD::copyDeviceToHost() -> void {
  if (_device == nullptr) {
    throw std::runtime_error("Device is nullptr");
  }

  // CalculationParam does't need any info, so don't copy metadata

  _time_param_hd.copyDeviceToHost();
  _fdtd_coefficient_hd.copyDeviceToHost();
}

auto CalculationParamHD::releaseDevice() -> void {
  _fdtd_coefficient_hd.releaseDevice();
  _time_param_hd.releaseDevice();

  if (_device != nullptr) {
    XFDTD_CORE_CUDA_CHECK_CUDA_ERROR(cudaFree(_device));
    _device = nullptr;
  }
}

}  // namespace cuda

}  // namespace xfdtd