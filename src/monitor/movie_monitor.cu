#include <cstdio>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <xfdtd_cuda/tensor.cuh>
#include <xtensor/xcsv.hpp>
#include <xtensor/xnpy.hpp>
#include <xtensor/xview.hpp>

#include "monitor/movie_monitor.cuh"
#include "monitor/movie_monitor_agency.cuh"

namespace xfdtd::cuda {

template <typename xfdtd::EMF::Field F>
XFDTD_CUDA_DEVICE auto MovieMonitor<F>::update() -> void {
  if (_frame_count % _frame_interval == 0) {
    auto index = _frame_count / _frame_interval;
    const auto& field = _emf->field<F>();
    auto task = this->task();
    if (!task.valid()) {
      return;
    }

    auto is = task.xRange().start();
    auto ie = task.xRange().end();
    auto js = task.yRange().start();
    auto je = task.yRange().end();
    auto ks = task.zRange().start();
    auto ke = task.zRange().end();
    auto offset_i = _task.xRange().start();
    auto offset_j = _task.yRange().start();
    auto offset_k = _task.zRange().start();

    for (Index x = is; x < ie; ++x) {
      for (Index y = js; y < je; ++y) {
        for (Index z = ks; z < ke; ++z) {
          (*data())(index, x - offset_i, y - offset_j, z - offset_k) =
              field(x, y, z);
        }
      }
    }
  }
}

template <typename xfdtd::EMF::Field F>
XFDTD_CUDA_DEVICE auto MovieMonitor<F>::task() const -> IndexTask {
  const auto& task = _task;
  // blcok
  auto size_x = static_cast<Index>(gridDim.x);
  auto size_y = static_cast<Index>(gridDim.y);
  auto size_z = static_cast<Index>(gridDim.z);
  auto id =
      blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
  auto block_task = decomposeTask(task, id, size_x, size_y, size_z);
  // thread
  size_x = static_cast<Index>(blockDim.x);
  size_y = static_cast<Index>(blockDim.y);
  size_z = static_cast<Index>(blockDim.z);
  id = threadIdx.x + threadIdx.y * blockDim.x +
       threadIdx.z * blockDim.x * blockDim.y;
  auto thread_task = decomposeTask(block_task, id, size_x, size_y, size_z);
  return thread_task;
}

template <typename xfdtd::EMF::Field F>
XFDTD_CUDA_HOST auto MovieMonitor<F>::output(std::string_view path_dir) const
    -> void {
  if (data() == nullptr) {
    throw std::runtime_error("MovieMonitor::output(): data is nullptr");
  }

  auto out_dir{std::filesystem::path(path_dir)};
  if (!std::filesystem::exists(out_dir)) {
    std::filesystem::create_directories(out_dir);
  }

  auto xtensor_data = xfdtd::Array4D<Real>::from_shape(
      {data()->shape()[0], data()->shape()[1], data()->shape()[2],
       data()->shape()[3]});

  std::memcpy(xtensor_data.data(), data()->data(),
              data()->size() * sizeof(Real));

  for (Index t = 0; t < data()->shape()[0]; ++t) {
    auto file = out_dir / formatFrameCount(t);
    xt::dump_npy(file.string() + ".npy",
                 xt::view(xtensor_data, t, xt::all(), xt::all(), xt::all()));
    // auto file_stream = std::ofstream(file.string() + ".csv");
    // xt::dump_csv(file_stream,
    //              xt::view(xtensor_data, t, xt::all(), xt::all(), 0));
  }
}

template <typename xfdtd::EMF::Field F>
XFDTD_CUDA_HOST auto MovieMonitor<F>::formatFrameCount(Index frame_count) const
    -> std::string {
  std::string frame_count_str{std::to_string(frame_count)};
  std::stringstream ss;
  ss << std::setw(5) << std::setfill('0') << frame_count_str;
  return ss.str();
}

// Agency

template <xfdtd::EMF::Field F>
XFDTD_CUDA_GLOBAL auto __movieMonitorUpdate(MovieMonitor<F>* movie_monitor)
    -> void {
  movie_monitor->update();
}

template <xfdtd::EMF::Field F>
XFDTD_CUDA_GLOBAL auto __movieMonitorNextCount(MovieMonitor<F>* movie_monitor)
    -> void {
  movie_monitor->nextCount();
}

template <xfdtd::EMF::Field F>
MovieMonitorAgency<F>::MovieMonitorAgency(MovieMonitor<F>* movie_monitor)
    : _movie_monitor{movie_monitor} {}

template <xfdtd::EMF::Field F>
XFDTD_CUDA_HOST auto MovieMonitorAgency<F>::update(dim3 grid_dim,
                                                   dim3 block_dim) -> void {
  __movieMonitorUpdate<<<grid_dim, block_dim>>>(_movie_monitor);
  __movieMonitorNextCount<<<1, 1>>>(_movie_monitor);
}

// explicit instantiation
template class MovieMonitor<xfdtd::EMF::Field::EX>;
template class MovieMonitor<xfdtd::EMF::Field::EY>;
template class MovieMonitor<xfdtd::EMF::Field::EZ>;
template class MovieMonitor<xfdtd::EMF::Field::HX>;
template class MovieMonitor<xfdtd::EMF::Field::HY>;
template class MovieMonitor<xfdtd::EMF::Field::HZ>;

template class MovieMonitorAgency<xfdtd::EMF::Field::EX>;
template class MovieMonitorAgency<xfdtd::EMF::Field::EY>;
template class MovieMonitorAgency<xfdtd::EMF::Field::EZ>;
template class MovieMonitorAgency<xfdtd::EMF::Field::HX>;
template class MovieMonitorAgency<xfdtd::EMF::Field::HY>;
template class MovieMonitorAgency<xfdtd::EMF::Field::HZ>;

}  // namespace xfdtd::cuda
