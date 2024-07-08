#ifndef __XFDTD_CUDA_MOVIE_MONITOR_HD_CUH__
#define __XFDTD_CUDA_MOVIE_MONITOR_HD_CUH__

#include <xfdtd/calculation_param/calculation_param.h>
#include <xfdtd/calculation_param/time_param.h>
#include <xfdtd/common/type_define.h>
#include <xfdtd/electromagnetic_field/electromagnetic_field.h>
#include <xfdtd/monitor/field_monitor.h>
#include <xfdtd/monitor/movie_monitor.h>

#include <memory>
#include <xfdtd_cuda/electromagnetic_field/electromagnetic_field_hd.cuh>
#include <xfdtd_cuda/host_device_carrier.cuh>
#include <xfdtd_cuda/index_task.cuh>
#include <xfdtd_cuda/tensor_hd.cuh>

#include "monitor/monitor_agency.cuh"
#include "monitor/movie_monitor.cuh"
#include "monitor/movie_monitor_agency.cuh"

namespace xfdtd::cuda {

template <typename xfdtd::EMF::Field F>
class MovieMonitorHD : public HostDeviceCarrier<xfdtd::MovieMonitor,
                                                xfdtd::cuda::MovieMonitor<F>> {
  using Host = xfdtd::MovieMonitor;
  using Device = xfdtd::cuda::MovieMonitor<F>;

 public:
  MovieMonitorHD(Host *host, std::shared_ptr<xfdtd::cuda::EMFHD> emf_hd)
      : HostDeviceCarrier<Host, Device>{host},
        _field_monitor{
            dynamic_cast<xfdtd::FieldMonitor *>(host->frame().get())},
        _emf_hd{emf_hd},
        _task{
            IndexTask{IndexRange{_field_monitor->globalTask().xRange().start(),
                                 _field_monitor->globalTask().xRange().end()},
                      IndexRange{_field_monitor->globalTask().yRange().start(),
                                 _field_monitor->globalTask().yRange().end()},
                      IndexRange{_field_monitor->globalTask().zRange().start(),
                                 _field_monitor->globalTask().zRange().end()}}},
        _frame_interval{host->frameInterval()},
        _data_in_host{{host->calculationParam()->timeParam()->size() /
                           host->frameInterval(),
                       _field_monitor->globalTask().xRange().size(),
                       _field_monitor->globalTask().yRange().size(),
                       _field_monitor->globalTask().zRange().size()}},
        _data_hd{_data_in_host},
        _out_dir{host->outputDir()} {}

  ~MovieMonitorHD() override { releaseDevice(); }

  auto copyHostToDevice() -> void override {
    if (this->host() == nullptr) {
      throw std::runtime_error(
          "MovieMonitorHD::copyHostToDevice(): Host data is not initialized");
    }

    _data_hd.copyHostToDevice();

    auto d = Device{};
    d._task = _task;
    d._emf = _emf_hd->device();
    d._frame_interval = _frame_interval;
    d._frame_count = 0;
    d._data = _data_hd.device();

    this->copyToDevice(&d);
    d._data = nullptr;
  }

  auto copyDeviceToHost() -> void override {
    if (this->host() == nullptr) {
      throw std::runtime_error(
          "MovieMonitorHD::copyDeviceToHost(): Host data is not initialized");
    }

    auto d = Device{};
    this->copyToHost(&d);
    _data_hd.copyDeviceToHost();
  }

  auto releaseDevice() -> void override {
    _data_hd.releaseDevice();
    this->releaseBaseDevice();
  }

  auto output(std::string_view path_dir) const -> void {
    auto d = Device{};
    d._data = const_cast<Tensor<Real, 4> *>(&_data_in_host);
    d._frame_interval = _frame_interval;
    d.output(path_dir);
  }

  auto getAgency() -> MonitorAgency * {
    if (_movie_monitor_agency == nullptr) {
      _movie_monitor_agency =
          std::make_unique<MovieMonitorAgency<F>>(this->device());
    }

    return _movie_monitor_agency.get();
  }

  auto outDir() const -> std::string { return _out_dir; }

  auto output() -> void { output(_out_dir); }

 private:
  xfdtd::FieldMonitor *_field_monitor{nullptr};
  std::shared_ptr<xfdtd::cuda::EMFHD> _emf_hd;
  IndexTask _task;
  Index _frame_interval;
  std::string _out_dir;
  xfdtd::cuda::Array4D<Real> _data_in_host;
  TensorHD<Real, 4> _data_hd;
  std::unique_ptr<MovieMonitorAgency<F>> _movie_monitor_agency{};
};

}  // namespace xfdtd::cuda

#endif  // __XFDTD_CUDA_MOVIE_MONITOR_HD_CUH__
