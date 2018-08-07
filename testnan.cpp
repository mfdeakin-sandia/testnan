
#include <limits>
#include <algorithm>

#include <signal.h>
#include <fenv.h>
#include <stdio.h>

#include <Kokkos_Core.hpp>
#include "vector/KokkosKernels_Vector.hpp"

void enable_fp_exceptions() {
  sigset_t fp_sig;
  sigemptyset(&fp_sig);
  sigaddset(&fp_sig, SIGFPE);
  sigprocmask(SIG_UNBLOCK, &fp_sig, nullptr);

  feenableexcept(-1);
  _MM_SET_EXCEPTION_MASK(_MM_GET_EXCEPTION_MASK() &
                         ~(_MM_MASK_INVALID | _MM_MASK_DIV_ZERO |
                           _MM_MASK_OVERFLOW | _MM_MASK_UNDERFLOW));
}

using ExecSpace = Kokkos::DefaultExecutionSpace::execution_space;

using VectorT = KokkosKernels::Batched::Experimental::Vector<
    KokkosKernels::Batched::Experimental::VectorTag<
        KokkosKernels::Batched::Experimental::AVX<double, ExecSpace>, 8> >;

VectorT pack_min(const VectorT &a, const VectorT &b) {
  return _mm512_min_pd(a, b);
}

int main(int argc, char **argv) {
  enable_fp_exceptions();

  constexpr double nan = std::numeric_limits<double>::quiet_NaN();

  constexpr int vlen = 50;
  VectorT a[vlen], b[vlen];
  for (int i = 0; i < vlen; i++) {
    a[i] = 1.0;
    b[i] = 4.0;
  }
  a[vlen - 1][4] = nan;

  for (int i = 0; i < vlen; i++) {
    printf("%d\n", i);
    pack_min(a[i], b[i]);
  }
  printf("No signal\n");
  double v = 0.0 / 0.0;
}
