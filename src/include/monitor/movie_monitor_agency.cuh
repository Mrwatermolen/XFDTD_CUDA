#ifndef __XFDTD_CUDA_MOVIE_MONITOR_AGENCY_CUH__
#define __XFDTD_CUDA_MOVIE_MONITOR_AGENCY_CUH__

#include <xfdtd/electromagnetic_field/electromagnetic_field.h>

#include <xfdtd_cuda/common.cuh>

#include "monitor/monitor_agency.cuh"

namespace xfdtd::cuda {

template <xfdtd::EMF::Field F>
class MovieMonitor;

template <xfdtd::EMF::Field F>
class MovieMonitorAgency : public MonitorAgency {
 public:
  MovieMonitorAgency(MovieMonitor<F>* movie_monitor);

  XFDTD_CUDA_HOST auto update(dim3 grid_dim, dim3 block_dim) -> void override;

 private:
  MovieMonitor<F>* _movie_monitor;
};

};  // namespace xfdtd::cuda

#endif  // __XFDTD_CUDA_MOVIE_MONITOR_AGENCY_CUH__
