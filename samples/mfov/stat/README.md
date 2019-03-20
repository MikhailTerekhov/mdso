## MultiFoV datasets statistics acquisition demo

Acquire errors of some of our methods from a range of MultiFoV dataset frames. 
Call it with
```
stat data_dir
```
where `data_dir` stands for MultiFoV dataset directory.

Tracking statistics is acquired with a range of base frames [`start_frame`, `end_frame`], and is output into the file _tracking_err.csv_. For each base frame ground truth depths in contrast points are used, multiplied by gaussian noize with mean=1 and stddev=`depths_noize`. For each base frame `track_count` consequential frames are tracked, and translational and rotational errors are measured. This way _tracking_err.csv_ has four columns: base frame number, reference frame number, the translational error in internal dataset length units, rotational error in degrees.

Stereo-matching errors are put into the file _stereo_err.csv_. They are measured for the same range [`start_frame`, `end_frame`] of base frames. For each base frame reference frames from the range [N+`min_stereo_displ`, N+`max_stereo_displ`] are used (N is the number of the baseframe). For each pair of base and reference frames stereo-matching is done. Scale multiplier between our system and dataset length units is measured from the median resulting depth. Considering this scale difference, the translational error is measured in dataset length units. Also, **relative** depths errors are measured. First, .25, .5, .75, .95 quantiles of errors in keypoints are measured. After that, we measure the same four quantiles for contrast points after triangulation. At last, translational and rotational errors and the same four quantiles are measured after BA. In total, each line in _stereo_err.csv_ contains 18 numbers: base frame number, relative frame number, the translational error before BA, rotational error before BA, the translational error after BA, rotational error after BA, .25, .5, .75, .95 quantiles of relative depth errors in keypoints, .25, .5, .75, .95 quantiles of relative depth errors in contrast points after triangulation, .25, .5, .75, .95 quantiles of relative depth errors in contrast points after BA.

`start_frame`, `end_frame`, `depths_noize`, `track_count`, `min_stereo_displ`, `max_stereo_displ` are flags that can be passed to the system.
