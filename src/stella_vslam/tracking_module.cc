#include "stella_vslam/config.h"
#include "stella_vslam/system.h"
#include "stella_vslam/tracking_module.h"
#include "stella_vslam/mapping_module.h"
#include "stella_vslam/global_optimization_module.h"
#include "stella_vslam/camera/base.h"
#include "stella_vslam/data/landmark.h"
#include "stella_vslam/data/map_database.h"
#include "stella_vslam/data/bow_database.h"
#include "stella_vslam/match/projection.h"
#include "stella_vslam/module/local_map_updater.h"
#include "stella_vslam/optimize/pose_optimizer_factory.h"
#include "stella_vslam/util/yaml.h"

#include <chrono>
#include <unordered_map>

#include <spdlog/spdlog.h>

namespace stella_vslam {

tracking_module::tracking_module(const std::shared_ptr<config>& cfg, camera::base* camera, data::map_database* map_db,
                                 data::bow_vocabulary* bow_vocab, data::bow_database* bow_db)
    : camera_(camera),      // 相机模型
      tracking_yaml_(util::yaml_optional_ref(cfg->yaml_node_, "Tracking")),     // 配置文件
      reloc_distance_threshold_(tracking_yaml_["reloc_distance_threshold"].as<double>(0.2)),    // 重定位距离阈值
      reloc_angle_threshold_(tracking_yaml_["reloc_angle_threshold"].as<double>(0.45)),         // 重定位角度阈值
      init_retry_threshold_time_(tracking_yaml_["init_retry_threshold_time"].as<double>(5.0)),  // 初始化重试时间阈值
      enable_auto_relocalization_(tracking_yaml_["enable_auto_relocalization"].as<bool>(true)), // 是否启用自动重定位
      enable_temporal_keyframe_only_tracking_(tracking_yaml_["enable_temporal_keyframe_only_tracking"].as<bool>(false)),    // 是否仅对时间关键帧进行跟踪
      use_robust_matcher_for_relocalization_request_(tracking_yaml_["use_robust_matcher_for_relocalization_request"].as<bool>(false)),  // 重定位请求是否使用鲁棒匹配器
      max_num_local_keyfrms_(tracking_yaml_["max_num_local_keyfrms"].as<unsigned int>(60)),         // 局部关键帧的最大数量
      margin_local_map_projection_(tracking_yaml_["margin_local_map_projection"].as<float>(5.0)),   // 局部地图投影的边距
      margin_local_map_projection_unstable_(tracking_yaml_["margin_local_map_projection_unstable"].as<float>(20.0)),    // 不稳定时局部地图投影的边距
      map_db_(map_db), bow_vocab_(bow_vocab), bow_db_(bow_db),
      initializer_(map_db, bow_db, util::yaml_optional_ref(cfg->yaml_node_, "Initializer")),
      pose_optimizer_(optimize::pose_optimizer_factory::create(tracking_yaml_)),
      frame_tracker_(camera_, pose_optimizer_, 10, initializer_.get_use_fixed_seed(), tracking_yaml_["margin_last_frame_projection"].as<float>(20.0)),
      relocalizer_(pose_optimizer_, util::yaml_optional_ref(cfg->yaml_node_, "Relocalizer")),
      keyfrm_inserter_(util::yaml_optional_ref(cfg->yaml_node_, "KeyframeInserter")) {
    spdlog::debug("CONSTRUCT: tracking_module");
}

tracking_module::~tracking_module() {
    spdlog::debug("DESTRUCT: tracking_module");
}

// 设置跟踪模块和映射模块，更新关键帧插入器
void tracking_module::set_mapping_module(mapping_module* mapper) {
    mapper_ = mapper;
    keyfrm_inserter_.set_mapping_module(mapper);
}

// 设置全局优化模块
void tracking_module::set_global_optimization_module(global_optimization_module* global_optimizer) {
    global_optimizer_ = global_optimizer;
}

// 发起一个基于给定位姿的重定位请求，在跟踪失败后，根据外部信息提供的位姿来重定位
bool tracking_module::request_relocalize_by_pose(const Mat44_t& pose_cw) {
    std::lock_guard<std::mutex> lock(mtx_relocalize_by_pose_request_);
    // 如果有重定位工作正在进行，返回 false
    if (relocalize_by_pose_is_requested_) {
        spdlog::warn("Can not process new pose update request while previous was not finished");
        return false;
    }
    relocalize_by_pose_is_requested_ = true;    // 表示一个重定位请求已经被发起
    relocalize_by_pose_request_.mode_2d_ = false;
    relocalize_by_pose_request_.pose_cw_ = pose_cw;
    return true;
}

// 发起一个基于给定位姿和法向量的2D重定位请求
bool tracking_module::request_relocalize_by_pose_2d(const Mat44_t& pose_cw, const Vec3_t& normal_vector) {
    std::lock_guard<std::mutex> lock(mtx_relocalize_by_pose_request_);
    if (relocalize_by_pose_is_requested_) {
        spdlog::warn("Can not process new pose update request while previous was not finished");
        return false;
    }
    relocalize_by_pose_is_requested_ = true;
    relocalize_by_pose_request_.mode_2d_ = true;
    relocalize_by_pose_request_.pose_cw_ = pose_cw;
    relocalize_by_pose_request_.normal_vector_ = normal_vector;
    return true;
}

// 检查当前是否有基于位姿的重定位请求正在等待处理
bool tracking_module::relocalize_by_pose_is_requested() {
    std::lock_guard<std::mutex> lock(mtx_relocalize_by_pose_request_);
    return relocalize_by_pose_is_requested_;
}

// 提供对当前重定位对象的访问
pose_request& tracking_module::get_relocalize_by_pose_request() {
    std::lock_guard<std::mutex> lock(mtx_relocalize_by_pose_request_);
    return relocalize_by_pose_request_;
}

// 结束重定位请求
void tracking_module::finish_relocalize_by_pose_request() {
    std::lock_guard<std::mutex> lock(mtx_relocalize_by_pose_request_);
    relocalize_by_pose_is_requested_ = false;
}

// 重置跟踪模块
void tracking_module::reset() {
    spdlog::info("resetting system");

    // 系统初始化和关键帧插入器的重置
    initializer_.reset();
    keyfrm_inserter_.reset();

    // 调用映射模块的异步重置，然后使用 get 等待这些线程完成
    auto future_mapper_reset = mapper_->async_reset();
    auto future_global_optimizer_reset = global_optimizer_->async_reset();
    future_mapper_reset.get();
    future_global_optimizer_reset.get();

    // 清理词袋数据和地图数据
    bow_db_->clear();
    map_db_->clear();

    // 上次重定位的 id 和时间戳
    last_reloc_frm_id_ = 0;
    last_reloc_frm_timestamp_ = 0.0;

    tracking_state_ = tracker_state_t::Initializing;
}

// 处理传入的帧数据，更新系统的跟踪状态，返回相机在世界坐标系中的位姿
std::shared_ptr<Mat44_t> tracking_module::feed_frame(data::frame curr_frm) {
    // check if pause is requested
    // 检查是否需要暂停
    pause_if_requested();
    while (is_paused()) {
        std::this_thread::sleep_for(std::chrono::microseconds(5000));
    }

    // 更新当前帧
    curr_frm_ = curr_frm;

    // 根据当前状态判断初始化还是跟踪
    // 跟踪逻辑：
    // 锁定互斥量以防止在跟踪过程中插入关键帧。
    // 判断是否需要重定位。
    // 调用track函数进行帧跟踪。
    // 检查是否需要插入新的关键帧
    bool succeeded = false;
    if (tracking_state_ == tracker_state_t::Initializing) {
        succeeded = initialize();
    }
    else {
        std::lock_guard<std::mutex> lock(mtx_stop_keyframe_insertion_);
        bool relocalization_is_needed = tracking_state_ == tracker_state_t::Lost;
        SPDLOG_TRACE("tracking_module: start tracking");
        unsigned int num_tracked_lms = 0;
        unsigned int num_reliable_lms = 0;
        const unsigned int min_num_obs_thr = (3 <= map_db_->get_num_keyframes()) ? 3 : 2;
        succeeded = track(relocalization_is_needed, num_tracked_lms, num_reliable_lms, min_num_obs_thr);

        // check to insert the new keyframe derived from the current frame
        // 判断是否插入关键帧
        if (succeeded && !is_stopped_keyframe_insertion_ && new_keyframe_is_needed(num_tracked_lms, num_reliable_lms, min_num_obs_thr)) {
            keyfrm_inserter_.insert_new_keyframe(map_db_, curr_frm_);
        }
    }

    // state transition
    // 根据是否跟踪成功转换状态
    if (succeeded) {
        tracking_state_ = tracker_state_t::Tracking;
    }
    else if (tracking_state_ == tracker_state_t::Tracking) {
        tracking_state_ = tracker_state_t::Lost;

        spdlog::info("tracking lost: frame {}", curr_frm_.id_);
        // if tracking is failed within init_retry_threshold_time_ sec after initialization, reset the system
        if (!mapper_->is_paused() && curr_frm_.timestamp_ - initializer_.get_initial_frame_timestamp() < init_retry_threshold_time_) {
            spdlog::info("tracking lost within {} sec after initialization", init_retry_threshold_time_);
            reset();
            return nullptr;
        }
    }

    // 计算并返回相机在世界坐标系中的位姿
    std::shared_ptr<Mat44_t> cam_pose_wc = nullptr;
    // store the relative pose from the reference keyframe to the current frame
    // to update the camera pose at the beginning of the next tracking process
    if (curr_frm_.pose_is_valid()) {
        // 参考关键帧到当前帧的相对姿态
        last_cam_pose_from_ref_keyfrm_ = curr_frm_.get_pose_cw() * curr_frm_.ref_keyfrm_->get_pose_wc();
        cam_pose_wc = std::allocate_shared<Mat44_t>(Eigen::aligned_allocator<Mat44_t>(), curr_frm_.get_pose_wc());
    }

    // update last frame
    // 当前帧赋值给上一帧
    SPDLOG_TRACE("tracking_module: update last frame (curr_frm_={})", curr_frm_.id_);
    {
        std::lock_guard<std::mutex> lock(mtx_last_frm_);
        last_frm_ = curr_frm_;
    }
    SPDLOG_TRACE("tracking_module: finish tracking");

    return cam_pose_wc;
}

// 处理帧跟踪过程，包括是否需要重定位、执行跟踪、更新局部地图和位姿优化等
bool tracking_module::track(bool relocalization_is_needed,          // 是否需要重定位
                            unsigned int& num_tracked_lms,          // 跟踪到的地图点的数量
                            unsigned int& num_reliable_lms,         // 储存可靠的地图点的数量
                            const unsigned int min_num_obs_thr) {   // 最小观测数量阈值
    // LOCK the map database
    std::lock_guard<std::mutex> lock1(data::map_database::mtx_database_);
    std::lock_guard<std::mutex> lock2(mtx_last_frm_);

    // update the camera pose of the last frame
    // because the mapping module might optimize the camera pose of the last frame's reference keyframe
    SPDLOG_TRACE("tracking_module: update the camera pose of the last frame (curr_frm_={})", curr_frm_.id_);
    // 更新上一帧的位姿
    update_last_frame();

    // set the reference keyframe of the current frame
    // 把上一帧初参考关键帧设置为当前帧参考关键帧
    curr_frm_.ref_keyfrm_ = last_frm_.ref_keyfrm_;

    bool succeeded = false;
    if (relocalize_by_pose_is_requested()) {
        // Force relocalization by pose
        // 如果有重定位请求，根据位姿重定位
        succeeded = relocalize_by_pose(get_relocalize_by_pose_request());
    }
    else if (!relocalization_is_needed) {
        SPDLOG_TRACE("tracking_module: track_current_frame (curr_frm_={})", curr_frm_.id_);
        // 如果不需要重定位，直接跟踪当前帧
        succeeded = track_current_frame();
    }
    else if (enable_auto_relocalization_) {
        // 如果启用了自动重定位，则计算词袋模型，然后尝试重定位
        // Compute the BoW representations to perform relocalization
        SPDLOG_TRACE("tracking_module: Compute the BoW representations to perform relocalization (curr_frm_={})", curr_frm_.id_);
        if (!curr_frm_.bow_is_available()) {
            curr_frm_.compute_bow(bow_vocab_);
        }
        // try to relocalize
        SPDLOG_TRACE("tracking_module: try to relocalize (curr_frm_={})", curr_frm_.id_);
        succeeded = relocalizer_.relocalize(bow_db_, curr_frm_);
        if (succeeded) {
            last_reloc_frm_id_ = curr_frm_.id_;
            last_reloc_frm_timestamp_ = curr_frm_.timestamp_;
        }
    }

    // update the local map and optimize current camera pose
    // 更新局部地图，优化当前帧的位姿
    unsigned int fixed_keyframe_id_threshold = map_db_->get_fixed_keyframe_id_threshold();
    unsigned int num_temporal_keyfrms = 0;
    if (succeeded) {
        succeeded = track_local_map(num_tracked_lms, num_reliable_lms, num_temporal_keyfrms, min_num_obs_thr, fixed_keyframe_id_threshold);
    }

    // update the local map and optimize current camera pose without temporal keyframes
    if (fixed_keyframe_id_threshold > 0 && succeeded && num_temporal_keyfrms > 0) {
        succeeded = track_local_map_without_temporal_keyframes(num_tracked_lms, num_reliable_lms, min_num_obs_thr, fixed_keyframe_id_threshold);
    }

    // update the motion model
    // 更新运动模型
    if (succeeded) {
        SPDLOG_TRACE("tracking_module: update_motion_model (curr_frm_={})", curr_frm_.id_);
        update_motion_model();
    }

    // update the frame statistics
    // 更新当前帧的统计信息
    SPDLOG_TRACE("tracking_module: update_frame_statistics (curr_frm_={})", curr_frm_.id_);
    map_db_->update_frame_statistics(curr_frm_, !succeeded);

    return succeeded;
}

// 处理局部地图跟踪过程，包括更新局部地图、搜索局部地标和优化当前帧的位姿
bool tracking_module::track_local_map(unsigned int& num_tracked_lms,        // 存储跟踪到的地图点的数量
                                      unsigned int& num_reliable_lms,       // 存储可靠的地图点的数量
                                      unsigned int& num_temporal_keyfrms,   // 存储临时关键帧的数量
                                      const unsigned int min_num_obs_thr,   // 最小观测数量阈值
                                      const unsigned int fixed_keyframe_id_threshold) {     // 固定关键帧的ID阈值
    // 首先更新局部地图，可能包括插入新的关键帧、优化关键帧的位姿和添加新的地图点，然后返回跟踪是否成功，以及时间关键帧的数量
    bool succeeded = false;
    SPDLOG_TRACE("tracking_module: update_local_map (curr_frm_={})", curr_frm_.id_);
    succeeded = update_local_map(fixed_keyframe_id_threshold, num_temporal_keyfrms);

    if (succeeded) {
        // 如果更新局部地图成功，搜索匹配的地图点
        succeeded = search_local_landmarks(fixed_keyframe_id_threshold);
    }

    // 优化当前位姿
    if (succeeded) {
        SPDLOG_TRACE("tracking_module: optimize_current_frame_with_local_map (curr_frm_={})", curr_frm_.id_);
        succeeded = optimize_current_frame_with_local_map(num_tracked_lms, num_reliable_lms, min_num_obs_thr);
    }

    // 返回跟踪状态
    if (!succeeded) {
        spdlog::info("local map tracking failed (curr_frm_={})", curr_frm_.id_);
    }
    return succeeded;
}

// 不使用临时关键帧的局部地图跟踪
bool tracking_module::track_local_map_without_temporal_keyframes(unsigned int& num_tracked_lms,
                                                                 unsigned int& num_reliable_lms,
                                                                 const unsigned int min_num_obs_thr,
                                                                 const unsigned int fixed_keyframe_id_threshold) {
    bool succeeded = false;
    SPDLOG_TRACE("tracking_module: update_local_map without temporal keyframes (curr_frm_={})", curr_frm_.id_);
    succeeded = search_local_landmarks(fixed_keyframe_id_threshold);

    if (enable_temporal_keyframe_only_tracking_ && !succeeded) {
        SPDLOG_TRACE("temporal keyframe only tracking (curr_frm_={})", curr_frm_.id_);
        return true;
    }

    if (succeeded) {
        SPDLOG_TRACE("tracking_module: optimize_current_frame_with_local_map without temporal keyframes (curr_frm_={})", curr_frm_.id_);
        succeeded = optimize_current_frame_with_local_map(num_tracked_lms, num_reliable_lms, min_num_obs_thr);
    }

    if (!succeeded) {
        spdlog::info("local map tracking (without temporal keyframes) failed (curr_frm_={})", curr_frm_.id_);
    }
    return succeeded;
}

// 使用当前帧尝试初始化、处理初始化失败的情况以及将关键帧传递给映射模块
bool tracking_module::initialize() {
    {
        // LOCK the map database
        std::lock_guard<std::mutex> lock1(data::map_database::mtx_database_);
        std::lock_guard<std::mutex> lock2(mtx_stop_keyframe_insertion_);

        // try to initialize with the current frame
        initializer_.initialize(camera_->setup_type_, bow_vocab_, curr_frm_);
    }

    // if map building was failed -> reset the map database
    if (initializer_.get_state() == module::initializer_state_t::Wrong) {
        reset();
        return false;
    }

    // if initializing was failed -> try to initialize with the next frame
    if (initializer_.get_state() != module::initializer_state_t::Succeeded) {
        return false;
    }

    // pass all of the keyframes to the mapping module
    // 传递关键帧给映射模块
    assert(!is_stopped_keyframe_insertion_);
    for (const auto& keyfrm : curr_frm_.ref_keyfrm_->graph_node_->get_keyframes_from_root()) {
        auto future = mapper_->async_add_keyframe(keyfrm);
        future.get();
    }

    // succeeded
    return true;
}

// 跟踪当前帧
bool tracking_module::track_current_frame() {
    bool succeeded = false;

    // Tracking mode
    // 如果运动模型有效，则根据当前帧与上一帧的运动模型来跟踪
    if (twist_is_valid_) {
        // if the motion model is valid
        succeeded = frame_tracker_.motion_based_track(curr_frm_, last_frm_, twist_);
    }

    // 如果运动模型跟踪失败，则计算词袋模型，根据词袋来匹配
    if (!succeeded) {
        // Compute the BoW representations to perform the BoW match
        if (!curr_frm_.bow_is_available()) {
            curr_frm_.compute_bow(bow_vocab_);
        }
        succeeded = frame_tracker_.bow_match_based_track(curr_frm_, last_frm_, curr_frm_.ref_keyfrm_);
    }

    // 如果两种方式都失败，则用基于鲁棒的匹配来跟踪
    if (!succeeded) {
        succeeded = frame_tracker_.robust_match_based_track(curr_frm_, last_frm_, curr_frm_.ref_keyfrm_);
    }

    return succeeded;
}

// 基于给定位姿重定位
bool tracking_module::relocalize_by_pose(const pose_request& request) {
    bool succeeded = false;
    // 设置当前帧的位姿
    curr_frm_.set_pose_cw(request.pose_cw_);

    // 如果词袋不可用，计算词袋模型
    if (!curr_frm_.bow_is_available()) {
        curr_frm_.compute_bow(bow_vocab_);
    }

    // 获取候选关键帧
    const auto candidates = get_close_keyframes(request);
    for (const auto& candidate : candidates) {
        spdlog::debug("relocalize_by_pose: candidate = {}", candidate->id_);
    }

    // 如果候选关键帧不为空，尝试使用这些关键帧俩进行重定位
    if (!candidates.empty()) {
        succeeded = relocalizer_.reloc_by_candidates(curr_frm_, candidates, use_robust_matcher_for_relocalization_request_);
        // 如果重定位成功，记录重定位帧的 id 和时间戳
        if (succeeded) {
            last_reloc_frm_id_ = curr_frm_.id_;
            last_reloc_frm_timestamp_ = curr_frm_.timestamp_;
            // If the initial pose was given manually, use motion_based_track, expecting that the camera is not moving.
            last_frm_ = curr_frm_;
        }
    }
    else {
        // 设置当前帧为无效
        curr_frm_.invalidate_pose();
    }

    // 结束重定位请求
    finish_relocalize_by_pose_request();
    return succeeded;
}

// 从地图数据库中获取最近的关键帧
std::vector<std::shared_ptr<data::keyframe>> tracking_module::get_close_keyframes(const pose_request& request) {
    if (request.mode_2d_) {
        return map_db_->get_close_keyframes_2d(
            request.pose_cw_,
            request.normal_vector_,
            reloc_distance_threshold_,
            reloc_angle_threshold_);
    }
    else {
        return map_db_->get_close_keyframes(
            request.pose_cw_,
            reloc_distance_threshold_,
            reloc_angle_threshold_);
    }
}

// 更新运动模型，即当前帧与上一帧的运动关系，储存在 twist_ 中
void tracking_module::update_motion_model() {
    if (last_frm_.pose_is_valid()) {
        Mat44_t last_frm_cam_pose_wc = Mat44_t::Identity();
        last_frm_cam_pose_wc.block<3, 3>(0, 0) = last_frm_.get_rot_wc();
        last_frm_cam_pose_wc.block<3, 1>(0, 3) = last_frm_.get_trans_wc();
        twist_is_valid_ = true;
        twist_ = curr_frm_.get_pose_cw() * last_frm_cam_pose_wc;
    }
    else {
        twist_is_valid_ = false;
        twist_ = Mat44_t::Identity();
    }
}

// 在当前帧的观察中替换地标点，更新 上一帧的地标
void tracking_module::replace_landmarks_in_last_frm(nondeterministic::unordered_map<std::shared_ptr<data::landmark>, std::shared_ptr<data::landmark>>& replaced_lms) {
    std::lock_guard<std::mutex> lock(mtx_last_frm_);
    // 遍历观察到的地标，检查是否有映射关系在 replaced_lms 中，如果有，将旧地标替换为新地标
    for (unsigned int idx = 0; idx < last_frm_.frm_obs_.undist_keypts_.size(); ++idx) {
        const auto& lm = last_frm_.get_landmark(idx);
        if (!lm) {
            continue;
        }

        if (replaced_lms.count(lm)) {
            auto replaced_lm = replaced_lms[lm];
            if (last_frm_.has_landmark(replaced_lm)) {
                last_frm_.erase_landmark(replaced_lm);
            }
            last_frm_.add_landmark(replaced_lm, idx);
        }
    }
}

// 更新上一帧的相机位姿
void tracking_module::update_last_frame() {
    auto last_ref_keyfrm = last_frm_.ref_keyfrm_;
    if (!last_ref_keyfrm) {
        return;
    }
    last_frm_.set_pose_cw(last_cam_pose_from_ref_keyfrm_ * last_ref_keyfrm->get_pose_cw());
}

bool tracking_module::optimize_current_frame_with_local_map(unsigned int& num_tracked_lms,
                                                            unsigned int& num_reliable_lms,
                                                            const unsigned int min_num_obs_thr) {
    // optimize the pose
    Mat44_t optimized_pose;
    std::vector<bool> outlier_flags;
    pose_optimizer_->optimize(curr_frm_, optimized_pose, outlier_flags);
    curr_frm_.set_pose_cw(optimized_pose);

    // Reject outliers
    for (unsigned int idx = 0; idx < curr_frm_.frm_obs_.undist_keypts_.size(); ++idx) {
        if (!outlier_flags.at(idx)) {
            continue;
        }
        curr_frm_.erase_landmark_with_index(idx);
    }

    // count up the number of tracked landmarks
    num_tracked_lms = 0;
    num_reliable_lms = 0;
    for (unsigned int idx = 0; idx < curr_frm_.frm_obs_.undist_keypts_.size(); ++idx) {
        const auto& lm = curr_frm_.get_landmark(idx);
        if (!lm) {
            continue;
        }
        if (lm->will_be_erased()) {
            continue;
        }

        // the observation has been considered as inlier in the pose optimization
        assert(lm->has_observation());
        // count up
        if (0 < min_num_obs_thr) {
            if (min_num_obs_thr <= lm->num_observations()) {
                ++num_reliable_lms;
            }
        }
        ++num_tracked_lms;
        // increment the number of tracked frame
        lm->increase_num_observed();
    }

    constexpr unsigned int num_tracked_lms_thr = 20;

    // if recently relocalized, use the more strict threshold
    if (curr_frm_.timestamp_ < last_reloc_frm_timestamp_ + 1.0 && num_tracked_lms < 2 * num_tracked_lms_thr) {
        spdlog::debug("local map tracking failed: {} matches < {}", num_tracked_lms, 2 * num_tracked_lms_thr);
        return false;
    }

    // check the threshold of the number of tracked landmarks
    if (num_tracked_lms < num_tracked_lms_thr) {
        spdlog::debug("local map tracking failed: {} matches < {}", num_tracked_lms, num_tracked_lms_thr);
        return false;
    }

    return true;
}

bool tracking_module::update_local_map(unsigned int fixed_keyframe_id_threshold,
                                       unsigned int& num_temporal_keyfrms) {
    // clean landmark associations
    for (unsigned int idx = 0; idx < curr_frm_.frm_obs_.undist_keypts_.size(); ++idx) {
        const auto& lm = curr_frm_.get_landmark(idx);
        if (!lm) {
            continue;
        }
        if (lm->will_be_erased()) {
            curr_frm_.erase_landmark_with_index(idx);
            continue;
        }
    }

    // acquire the current local map
    local_landmarks_.clear();
    auto local_map_updater = module::local_map_updater(max_num_local_keyfrms_);
    if (!local_map_updater.acquire_local_map(curr_frm_.get_landmarks(), fixed_keyframe_id_threshold, num_temporal_keyfrms)) {
        return false;
    }
    // update the variables
    local_landmarks_ = local_map_updater.get_local_landmarks();
    auto nearest_covisibility = local_map_updater.get_nearest_covisibility();

    // update the reference keyframe for the current frame
    if (nearest_covisibility) {
        curr_frm_.ref_keyfrm_ = nearest_covisibility;
    }

    map_db_->set_local_landmarks(local_landmarks_);
    return true;
}

bool tracking_module::search_local_landmarks(unsigned int fixed_keyframe_id_threshold) {
    // select the landmarks which can be reprojected from the ones observed in the current frame
    std::unordered_set<unsigned int> curr_landmark_ids;
    for (const auto& lm : curr_frm_.get_landmarks()) {
        if (!lm) {
            continue;
        }
        if (lm->will_be_erased()) {
            continue;
        }

        // this landmark cannot be reprojected
        // because already observed in the current frame
        curr_landmark_ids.insert(lm->id_);

        // this landmark is observable from the current frame
        lm->increase_num_observable();
    }

    bool found_proj_candidate = false;
    // temporary variables
    Vec2_t reproj;
    float x_right;
    unsigned int pred_scale_level;
    eigen_alloc_unord_map<unsigned int, Vec2_t> lm_to_reproj;
    std::unordered_map<unsigned int, float> lm_to_x_right;
    std::unordered_map<unsigned int, unsigned int> lm_to_scale;
    for (const auto& lm : local_landmarks_) {
        if (curr_landmark_ids.count(lm->id_)) {
            continue;
        }
        if (lm->will_be_erased()) {
            continue;
        }
        if (fixed_keyframe_id_threshold > 0) {
            const auto observations = lm->get_observations();
            unsigned int temporal_observations = 0;
            for (auto obs : observations) {
                auto keyfrm = obs.first.lock();
                if (keyfrm->id_ >= fixed_keyframe_id_threshold) {
                    ++temporal_observations;
                }
            }
            const double temporal_ratio_thr = 0.5;
            double temporal_ratio = static_cast<double>(temporal_observations) / observations.size();
            if (temporal_ratio > temporal_ratio_thr) {
                continue;
            }
        }

        // check the observability
        if (curr_frm_.can_observe(lm, 0.5, reproj, x_right, pred_scale_level)) {
            lm_to_reproj[lm->id_] = reproj;
            lm_to_x_right[lm->id_] = x_right;
            lm_to_scale[lm->id_] = pred_scale_level;

            // this landmark is observable from the current frame
            lm->increase_num_observable();

            found_proj_candidate = true;
        }
    }

    if (!found_proj_candidate) {
        spdlog::warn("projection candidate not found");
        return false;
    }

    // acquire more 2D-3D matches by projecting the local landmarks to the current frame
    match::projection projection_matcher(0.8);
    const float margin = (curr_frm_.id_ < last_reloc_frm_id_ + 2)
                             ? margin_local_map_projection_unstable_
                             : margin_local_map_projection_;
    projection_matcher.match_frame_and_landmarks(curr_frm_, local_landmarks_, lm_to_reproj, lm_to_x_right, lm_to_scale, margin);
    return true;
}

bool tracking_module::new_keyframe_is_needed(unsigned int num_tracked_lms,
                                             unsigned int num_reliable_lms,
                                             const unsigned int min_num_obs_thr) const {
    // cannnot insert the new keyframe in a second after relocalization
    if (curr_frm_.timestamp_ < last_reloc_frm_timestamp_ + 1.0) {
        return false;
    }

    // check the new keyframe is needed
    return keyfrm_inserter_.new_keyframe_is_needed(map_db_, curr_frm_, num_tracked_lms, num_reliable_lms, *curr_frm_.ref_keyfrm_, min_num_obs_thr);
}

std::future<void> tracking_module::async_stop_keyframe_insertion() {
    auto future_stop_keyframe_insertion = std::async(
        std::launch::async,
        [this]() {
            std::lock_guard<std::mutex> lock(mtx_stop_keyframe_insertion_);
            SPDLOG_TRACE("tracking_module: stop keyframe insertion");
            is_stopped_keyframe_insertion_ = true;
        });
    return future_stop_keyframe_insertion;
}

std::future<void> tracking_module::async_start_keyframe_insertion() {
    auto future_stop_keyframe_insertion = std::async(
        std::launch::async,
        [this]() {
            std::lock_guard<std::mutex> lock(mtx_stop_keyframe_insertion_);
            SPDLOG_TRACE("tracking_module: start keyframe insertion");
            is_stopped_keyframe_insertion_ = false;
        });
    return future_stop_keyframe_insertion;
}

std::shared_future<void> tracking_module::async_pause() {
    std::lock_guard<std::mutex> lock(mtx_pause_);
    pause_is_requested_ = true;
    if (!future_pause_.valid()) {
        future_pause_ = promise_pause_.get_future().share();
    }

    std::shared_future<void> future_pause = future_pause_;
    if (is_paused_) {
        promise_pause_.set_value();
        // Clear request
        promise_pause_ = std::promise<void>();
        future_pause_ = std::shared_future<void>();
    }
    return future_pause;
}

bool tracking_module::pause_is_requested() const {
    std::lock_guard<std::mutex> lock(mtx_pause_);
    return pause_is_requested_;
}

bool tracking_module::is_paused() const {
    std::lock_guard<std::mutex> lock(mtx_pause_);
    return is_paused_;
}

void tracking_module::resume() {
    std::lock_guard<std::mutex> lock(mtx_pause_);

    is_paused_ = false;
    pause_is_requested_ = false;

    spdlog::info("resume tracking module");
}

bool tracking_module::pause_if_requested() {
    std::lock_guard<std::mutex> lock(mtx_pause_);
    if (pause_is_requested_) {
        is_paused_ = true;
        spdlog::info("pause tracking module");
        promise_pause_.set_value();
        promise_pause_ = std::promise<void>();
        future_pause_ = std::shared_future<void>();
        return true;
    }
    else {
        return false;
    }
}

} // namespace stella_vslam
