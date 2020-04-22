#ifndef INCLUDE_ACCUMULATOR
#define INCLUDE_ACCUMULATOR

#include "util/types.h"

namespace mdso::optimize {

template <typename BlockT> class Accumulator {
public:
  Accumulator()
      : mWasUsed(false)
      , mAccumulated(getZero()) {}

  Accumulator &operator+=(const BlockT &block) {
    mAccumulated += block;
    mWasUsed = true;
    return *this;
  }

  bool wasUsed() const { return mWasUsed; }
  const BlockT &accumulated() const { return mAccumulated; }

private:
  static BlockT getZero();

  BlockT mAccumulated;
  bool mWasUsed;
};

template <typename MatrixT> MatrixT Accumulator<MatrixT>::getZero() {
  return MatrixT::Zero();
}
template <> float Accumulator<float>::getZero();
template <> double Accumulator<double>::getZero();

} // namespace mdso::optimize

#endif