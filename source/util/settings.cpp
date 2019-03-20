#include "util/settings.h"
#include "util/defs.h"

#include <opencv2/core.hpp>

namespace fishdso {

const std::vector<double> Settings::PixelSelector::default_gradThresholds{
    20.0, 8.0, 5.0};
const std::vector<cv::Scalar> Settings::PixelSelector::default_pointColors{
    CV_GREEN, CV_BLUE, CV_RED};
const StdVector<Vec2> Settings::ResidualPattern::default_pattern{
    Vec2(0, 0), Vec2(0, -2), Vec2(-1, -1), Vec2(1, -1), Vec2(-2, 0),
    Vec2(2, 0), Vec2(-1, 1), Vec2(1, 1),   Vec2(0, 2)};

} // namespace fishdso

DEFINE_bool(
    output_reproj_CDF, false,
    "Output reprojection errors when doing keypoint-based stereomatching? If "
    "set to true, values will be in {output_directory}/reproj_err.txt");

DEFINE_bool(draw_inlier_matches, false, "Debug output stereo inlier matches.");

DEFINE_bool(debug_video, true, "Do we need to output debug video?");

DEFINE_double(rel_point_size, 0.004,
              "Relative to w+h point size on debug images.");
DEFINE_int32(debug_width, 1200, "Width of the debug image.");

DEFINE_double(debug_max_residual, 12.0,
              "Max tracking residual when displaying debug image.");
DEFINE_double(debug_max_stddev, 6.0,
              "Max predicted stddev when displaying debug image with stddevs.");

bool validateDepthsPart(const char *flagname, double value) {
  if (value >= 0 && value <= 1)
    return true;
  std::cerr << "Invalid value for --" << std::string(flagname) << ": " << value
            << "\nit should be in [0, 1]" << std::endl;
  return false;
}

DEFINE_double(red_depths_part, 0,
              "Part of contrast points that will be drawn red (i.e. they are "
              "too close to be distinguished)");
DEFINE_validator(red_depths_part, validateDepthsPart);

DEFINE_double(blue_depths_part, 0.7,
              "Part of contrast points that will NOT be drawn completely blue "
              "(i.e. they are not too far to be distinguished)");
DEFINE_validator(blue_depths_part, validateDepthsPart);

DEFINE_string(debug_img_dir, "output/default/debug",
              "Directory for debug images to be put into.");
DEFINE_string(
    track_img_dir, "output/default/track",
    "Directory for tracking residuals on all pyr levels to be put into.");

DEFINE_bool(show_interpolation, false,
            "Show interpolated depths after initialization?");
DEFINE_bool(show_track_base, false,
            "Show depths used for tracking after the new kf is inserted?");
DEFINE_bool(show_track_res, false,
            "Show tracking residuals on all levels of the pyramind?");
DEFINE_bool(show_debug_image, true,
            R"__(Show debug image? Structure of debug image:
+---+---+
| A | B |
+---+---+
| C | D |
+---+---+
Where
A displays color-coded depths projected onto the base frame;
B -- visible vs invizible OptimizedPoint-s projected onto the base frame;
C -- color-coded predicted disparities;
D -- color-coded tracking residuals on the finest pyramid level.)__");

DEFINE_bool(write_files, true,
            "Do we need to write output files into output_directory?");
DEFINE_string(output_directory, "output/default",
              "CO: \"it's dso's output directory!\"");
