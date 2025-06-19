#ifndef __XFDTD_CUDA_TENSOR_HD_CUH__
#define __XFDTD_CUDA_TENSOR_HD_CUH__

#include <xfdtd/exception/exception.h>

#include <cassert>
#include <sstream>
#include <string>
#include <xfdtd_cuda/common.cuh>
#include <xfdtd_cuda/fixed_array.cuh>
#include <xfdtd_cuda/memory.cuh>
#include <xfdtd_cuda/tensor.cuh>

namespace xfdtd {

namespace cuda {

class XFDTDCudaTensorHDException : public XFDTDException {
public:
  XFDTDCudaTensorHDException(const std::string &mes) : XFDTDException{mes} {}
};

template <typename T, SizeType N> class TensorHD {
public:
public:
  using DeviceTensor = Tensor<T, N>;

  template <typename WrappedTensor>
  TensorHD(const WrappedTensor &wrapped_tensor) {
    const auto &dim = wrapped_tensor.dimension();
    const auto &shape = wrapped_tensor.shape();
    const auto &stride = wrapped_tensor.strides();
    const auto size = wrapped_tensor.size();
    // remove const
    auto data = const_cast<T *>(wrapped_tensor.data());

    // check dim
    if (dim != DeviceTensor::dimension()) {
      std::stringstream ss;
      ss << "Wrong dim";
      throw XFDTDCudaTensorHDException{ss.str()};
    }

    for (SizeType i = 0; i < dim; ++i) {
      _shape[i] = shape[i];
    }

    _host_data = data;
  }

  // TensorHD(FixedArray<SizeType, N> shape) : _host{new HostTensor{shape}} {}

  ~TensorHD() { releaseDevice(); }

  XFDTD_CUDA_DUAL auto device() -> DeviceTensor * { return _device; }

  XFDTD_CUDA_DUAL auto device() const -> const DeviceTensor * {
    return _device;
  }

  auto hostData() { return _host_data; }

  auto hostData() const { return _host_data; }

  auto deviceData() { return _device_data; }

  auto deviceData() const { return _device_data; }

  auto shape() const { return _shape; }

  /**
   * @brief Reset host data. It doesn't free previous host data and copy data to
   * device. The only thing it does is to change the host data pointer.
   */
  auto resetHostData(T *data) { _host_data = data; }

  auto copyHostToDevice() -> void {
    if (_host_data == nullptr) {
      throw XFDTDCudaTensorHDException{"Host memory is not allocated"};
    }

    if (_device) {
      // Free previous device memory
      XFDTD_CORE_CUDA_CHECK_CUDA_ERROR(cudaFree(_device));
      _device = nullptr;
    }

    if (_device_data != nullptr) {
      XFDTD_CORE_CUDA_CHECK_CUDA_ERROR(cudaFree(_device_data));
      _device_data = nullptr;
    }

    // Malloc device
    XFDTD_CORE_CUDA_CHECK_CUDA_ERROR(
        cudaMalloc(&_device, sizeof(DeviceTensor)));

    // tensor metadata
    auto host_tensor_matedata = DeviceTensor{};
    host_tensor_matedata._shape = _shape;
    host_tensor_matedata._strides = host_tensor_matedata.makeStride(_shape);
    host_tensor_matedata._size = host_tensor_matedata.makeSize(_shape);
    host_tensor_matedata._data = nullptr;

    // Malloc device data
    XFDTD_CORE_CUDA_CHECK_CUDA_ERROR(
        cudaMalloc(&_device_data, host_tensor_matedata.size() * sizeof(T)));

    // Copy data
    XFDTD_CORE_CUDA_CHECK_CUDA_ERROR(cudaMemcpy(
        _device_data, _host_data, host_tensor_matedata.size() * sizeof(T),
        cudaMemcpyHostToDevice))

    try {
      host_tensor_matedata._data = _device_data;
      XFDTD_CORE_CUDA_CHECK_CUDA_ERROR(
          cudaMemcpy(_device, &host_tensor_matedata, sizeof(DeviceTensor),
                     cudaMemcpyHostToDevice));
    } catch (const std::exception &e) {
      host_tensor_matedata._data = nullptr;
      throw e;
    }

    host_tensor_matedata._data = nullptr;
  }

  auto copyDeviceToHost() -> void {
    if (!_device || !_device_data) {
      throw std::runtime_error("Device memory is not allocated");
    }

    // recieve meta data
    auto host_tensor_matedata = DeviceTensor{};
    XFDTD_CORE_CUDA_CHECK_CUDA_ERROR(cudaMemcpy(&host_tensor_matedata, _device,
                                                sizeof(DeviceTensor),
                                                cudaMemcpyDeviceToHost));
    host_tensor_matedata._data = nullptr; // can't receive data in device
    // assume that shape will be never changed. Just copy
    XFDTD_CORE_CUDA_CHECK_CUDA_ERROR(
        cudaMemcpy(_host_data, _device_data,
                   host_tensor_matedata.size() * sizeof(T),
                   cudaMemcpyDeviceToHost););
  }

  auto releaseDevice() -> void {
    if (_device) {
      XFDTD_CORE_CUDA_CHECK_CUDA_ERROR(cudaFree(_device));
      _device = nullptr;
    }

    if (_device_data) {
      XFDTD_CORE_CUDA_CHECK_CUDA_ERROR(cudaFree(_device_data));
      _device_data = nullptr;
    }
  }

protected:
private:
  DeviceTensor *_device{};

  T *_device_data{};
  T *_host_data{};

  FixedArray<SizeType, N> _shape;
};

template <typename WrappedTensor, SizeType N>
class TensorHDWrapped : public TensorHD<typename WrappedTensor::value_type, N> {
public:
public:
  using T = typename WrappedTensor::value_type;
  TensorHDWrapped(WrappedTensor tensor)
      : TensorHD<T, N>{tensor}, _host_tensor{std::move(tensor)} {}

  auto tensor() const -> const WrappedTensor & { return _host_tensor; }

  auto tensor() -> WrappedTensor & { return _host_tensor; }

private:
  WrappedTensor _host_tensor;
};

} // namespace cuda

} // namespace xfdtd

#endif // __XFDTD_CUDA_TENSOR_HD_CUH__
