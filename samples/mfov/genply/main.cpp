#include "../reader/MultiFovReader.h"
#include "system/DsoSystem.h"
#include <iostream>

DEFINE_int32(start, 1, "Number of the starting frame.");
DEFINE_int32(count, 100, "Number of frames to process.");

int main(int argc, char **argv) {
  std::string usage =
      R"abacaba(Usage: genply data_dir
Where data_dir names a directory with MultiFoV fishseye dataset.
It should contain "info" and "data" subdirectories.)abacaba";

  gflags::ParseCommandLineFlags(&argc, &argv, true);
  gflags::SetUsageMessage(usage);
  google::InitGoogleLogging(argv[0]);

  if (argc != 2) {
    std::cerr << "Wrong number of arguments!\n" << usage << std::endl;
    return 1;
  }

  MultiFovReader reader(argv[1]);

  DsoSystem dso(reader.cam.get());
  for (int it = FLAGS_start; it < FLAGS_start + FLAGS_count; ++it) {
    std::cout << "add frame #" << it << std::endl;
    dso.addGroundTruthPose(it, reader.getWorldToFrameGT(it));
    dso.addFrame(reader.getFrame(it), it);
  }

  dso.fillRemainedHistory();
  std::ofstream ofs(FLAGS_output_directory + "/out.ply");
  dso.printPointsInPly(ofs);

  return 0;
}
