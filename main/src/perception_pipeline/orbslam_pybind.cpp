 

 
#define GL_GLEXT_PROTOTYPES
#include <GL/glew.h>
#include <GL/gl.h>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "System.h"
#include <opencv2/opencv.hpp>

namespace py = pybind11;

class PyORBSLAM3 {
public:
    PyORBSLAM3(const std::string& vocab_path, const std::string& config_path) 
        : slam_system(vocab_path, config_path, ORB_SLAM3::System::RGBD, false) {}
        
    py::dict process_frame_rgb(py::array_t<uint8_t> rgb_frame, py::array_t<float> depth_frame, double timestamp) {
        cv::Mat cv_rgb = numpy_to_cvmat(rgb_frame);
        cv::Mat cv_depth = numpy_to_depth_cvmat(depth_frame);  // Depth image matrix (new helper)

        Sophus::SE3f current_pose = slam_system.TrackRGBD(cv_rgb, cv_depth, timestamp);

        py::dict result;
        result["current_pose"] = se3f_to_numpy(current_pose);
        result["tracking_status"] = get_tracking_status();
        result["visible_points"] = get_tracked_map_points();

        return result;
    }

     
    py::array_t<double> get_current_pose() {

        Sophus::SE3f identity_pose = Sophus::SE3f();
        return se3f_to_numpy(identity_pose);
    } 
    void save_trajectory_tum(const std::string& filename) {
        slam_system.SaveTrajectoryTUM(filename);
    }
 
    void save_trajectory_kitti(const std::string& filename) {
        slam_system.SaveTrajectoryKITTI(filename);
    }
 
    void save_keyframe_trajectory_tum(const std::string& filename) {
        slam_system.SaveKeyFrameTrajectoryTUM(filename);
    }
 
    py::dict get_tracking_status() {
        py::dict status;
        status["tracking_ok"] = (slam_system.GetTrackingState() == ORB_SLAM3::Tracking::OK);
        status["tracking_lost"] = slam_system.isLost();
        status["system_shutdown"] = slam_system.isShutDown();
        
        auto tracked_points = slam_system.GetTrackedMapPoints();
        status["visible_points_count"] = tracked_points.size();
        
        return status;
    }
     
    py::list get_tracked_map_points() {
        py::list points;
        auto tracked_points = slam_system.GetTrackedMapPoints();
        
        for (auto* point : tracked_points) {
            if (point && !point->isBad()) {
                Eigen::Vector3f world_pos = point->GetWorldPos();
                py::dict point_data;
                point_data["id"] = point->mnId;
                point_data["x"] = world_pos.x();
                point_data["y"] = world_pos.y();
                point_data["z"] = world_pos.z();
                points.append(point_data);
            }
        }
        return points;
    }
 
    void reset() { slam_system.Reset(); }
    void reset_active_map() { slam_system.ResetActiveMap(); }
    void shutdown() { slam_system.Shutdown(); }
 
    bool is_lost() { return slam_system.isLost(); }
    bool is_shutdown() { return slam_system.isShutDown(); }
    int get_tracking_state() { return slam_system.GetTrackingState(); }

private:
    ORB_SLAM3::System slam_system;
    
    cv::Mat numpy_to_depth_cvmat(py::array_t<float> input) {
        py::buffer_info buf = input.request();
        // Assuming depth is single channel float32
        return cv::Mat(buf.shape[0], buf.shape[1], CV_32FC1, buf.ptr);
    }

    
    cv::Mat numpy_to_cvmat(py::array_t<uint8_t>& input) {
    py::buffer_info buf = input.request();
    // Assuming RGB image is 3-channel uint8
    return cv::Mat(buf.shape[0], buf.shape[1], CV_8UC3, buf.ptr);
}

    py::array_t<double> se3f_to_numpy(const Sophus::SE3f& pose) {
        auto result = py::array_t<double>({4, 4});
        double* ptr = static_cast<double*>(result.request().ptr);
        Eigen::Matrix4f matrix = pose.matrix();
        for (int i = 0; i < 4; ++i) 
            for (int j = 0; j < 4; ++j) 
                ptr[i*4 + j] = matrix(j, i);
        return result;
    }
};

PYBIND11_MODULE(orbslam3, m) {
    py::class_<PyORBSLAM3>(m, "System")
        .def(py::init<const std::string&, const std::string&>())
        .def("process_frame", &PyORBSLAM3::process_frame_rgb)
        .def("get_current_pose", &PyORBSLAM3::get_current_pose)
        .def("get_tracking_status", &PyORBSLAM3::get_tracking_status)
        .def("get_tracked_map_points", &PyORBSLAM3::get_tracked_map_points)
        .def("save_trajectory_tum", &PyORBSLAM3::save_trajectory_tum)
        .def("save_trajectory_kitti", &PyORBSLAM3::save_trajectory_kitti)
        .def("save_keyframe_trajectory_tum", &PyORBSLAM3::save_keyframe_trajectory_tum)
        .def("reset", &PyORBSLAM3::reset)
        .def("reset_active_map", &PyORBSLAM3::reset_active_map)
        .def("shutdown", &PyORBSLAM3::shutdown)
        .def("is_lost", &PyORBSLAM3::is_lost)
        .def("is_shutdown", &PyORBSLAM3::is_shutdown)
        .def("get_tracking_state", &PyORBSLAM3::get_tracking_state);
}



 