#ifndef INCLUDE_ACCUMULATOR
#define INCLUDE_ACCUMULATOR

#include "util/types.h"

namespace mdso::optimize {

template <typename BlockT> class Accumulator {
public:
  Accumulator();

  Accumulator &operator+=(const BlockT &block) {
    mAccumulated += block;
    mWasUsed = true;
    return *this;
  }

  bool wasUsed() const { return mWasUsed; }
  const BlockT &accumulated() const { return mAccumulated; }

private:
  BlockT mAccumulated;
  bool mWasUsed;
};

template <typename MatrixT>
Accumulator<MatrixT>::Accumulator()
    : mWasUsed(false)
    , mAccumulated(MatrixT::Zero()) {}

template <> Accumulator<T>::Accumulator();

} // namespace mdso::optimize

#endif