#ifndef INCLUDE_BILINEARINTERPOLATOR
#define INCLUDE_BILINEARINTERPOLATOR

#include <glog/logging.h>

namespace mdso {

template <typename Grid> class BilinearInterpolator {
public:
  BilinearInterpolator(const Grid &grid)
      : grid(grid) {}

  void Evaluate(double r, double c, double *f) const {
    Evaluate(r, c, f, nullptr, nullptr);
  }

  template <typename JetT>
  void Evaluate(const JetT &r, const JetT &c, JetT *f) const {
    double dfdr, dfdc;
    Evaluate(r.a, c.a, &f->a, &dfdr, &dfdc);
    f->v = dfdr * r.v + dfdc * c.v;
  }

  void Evaluate(double r, double c, double *f, double *dfdr,
                double *dfdc) const {
    CHECK(dfdr && dfdc || !dfdr && !dfdc);
    int ri = r, ci = c;
    double dr = r - ri, dc = c - ci;
    double f00, f01, f10, f11;
    grid.GetValue(ri, ci, &f00);
    grid.GetValue(ri, ci + 1, &f01);
    grid.GetValue(ri + 1, ci, &f10);
    grid.GetValue(ri + 1, ci + 1, &f11);
    double f11_m_f10 = f11 - f10;
    double f01_m_f00 = f01 - f00;
    *f = (1 - dr) * (f00 + (f01_m_f00)*dc) + dr * (f10 + (f11_m_f10)*dc);
    if (dfdr && dfdc) {
      double f00_m_f01_m_f10_p_f11 = f11_m_f10 - f01_m_f00;
      *dfdr = f10 - f00 + f00_m_f01_m_f10_p_f11 * dc;
      *dfdc = f01_m_f00 + f00_m_f01_m_f10_p_f11 * dr;
    }
  }

private:
  const Grid &grid;
};

} // namespace mdso

#endif
