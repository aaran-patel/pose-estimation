std_msgs/Header header
int32 person_index
string state              # "STANDING" or "FALL_DETECTED"
float64 smoothed_torso_angle_deg
int32 sustained_frames
bool newly_confirmed       # true only on the frame it first flips to FALL_DETECTED
