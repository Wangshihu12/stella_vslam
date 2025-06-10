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
    : camera_(camera),
      tracking_yaml_(util::yaml_optional_ref(cfg->yaml_node_, "Tracking")),
      reloc_distance_threshold_(tracking_yaml_["reloc_distance_threshold"].as<double>(0.2)),
      reloc_angle_threshold_(tracking_yaml_["reloc_angle_threshold"].as<double>(0.45)),
      init_retry_threshold_time_(tracking_yaml_["init_retry_threshold_time"].as<double>(5.0)),
      enable_auto_relocalization_(tracking_yaml_["enable_auto_relocalization"].as<bool>(true)),
      enable_temporal_keyframe_only_tracking_(tracking_yaml_["enable_temporal_keyframe_only_tracking"].as<bool>(false)),
      use_robust_matcher_for_relocalization_request_(tracking_yaml_["use_robust_matcher_for_relocalization_request"].as<bool>(false)),
      max_num_local_keyfrms_(tracking_yaml_["max_num_local_keyfrms"].as<unsigned int>(60)),
      margin_local_map_projection_(tracking_yaml_["margin_local_map_projection"].as<float>(5.0)),
      margin_local_map_projection_unstable_(tracking_yaml_["margin_local_map_projection_unstable"].as<float>(20.0)),
      map_db_(map_db), bow_vocab_(bow_vocab), bow_db_(bow_db),
      initializer_(map_db, util::yaml_optional_ref(cfg->yaml_node_, "Initializer")),
      pose_optimizer_(optimize::pose_optimizer_factory::create(tracking_yaml_)),
      frame_tracker_(camera_, pose_optimizer_, 10, initializer_.get_use_fixed_seed(), tracking_yaml_["margin_last_frame_projection"].as<float>(20.0)),
      relocalizer_(pose_optimizer_, util::yaml_optional_ref(cfg->yaml_node_, "Relocalizer")),
      keyfrm_inserter_(util::yaml_optional_ref(cfg->yaml_node_, "KeyframeInserter")) {
    spdlog::debug("CONSTRUCT: tracking_module");
}

tracking_module::~tracking_module() {
    spdlog::debug("DESTRUCT: tracking_module");
}

void tracking_module::set_mapping_module(mapping_module* mapper) {
    mapper_ = mapper;
    keyfrm_inserter_.set_mapping_module(mapper);
}

void tracking_module::set_global_optimization_module(global_optimization_module* global_optimizer) {
    global_optimizer_ = global_optimizer;
}

bool tracking_module::request_relocalize_by_pose(const Mat44_t& pose_cw) {
    std::lock_guard<std::mutex> lock(mtx_relocalize_by_pose_request_);
    if (relocalize_by_pose_is_requested_) {
        spdlog::warn("Can not process new pose update request while previous was not finished");
        return false;
    }
    relocalize_by_pose_is_requested_ = true;
    relocalize_by_pose_request_.mode_2d_ = false;
    relocalize_by_pose_request_.pose_cw_ = pose_cw;
    return true;
}

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

bool tracking_module::relocalize_by_pose_is_requested() {
    std::lock_guard<std::mutex> lock(mtx_relocalize_by_pose_request_);
    return relocalize_by_pose_is_requested_;
}

pose_request& tracking_module::get_relocalize_by_pose_request() {
    std::lock_guard<std::mutex> lock(mtx_relocalize_by_pose_request_);
    return relocalize_by_pose_request_;
}

void tracking_module::finish_relocalize_by_pose_request() {
    std::lock_guard<std::mutex> lock(mtx_relocalize_by_pose_request_);
    relocalize_by_pose_is_requested_ = false;
}

void tracking_module::reset() {
    spdlog::info("resetting system");

    initializer_.reset();
    keyfrm_inserter_.reset();

    if (global_optimizer_) {
        auto future_mapper_reset = mapper_->async_reset();
        auto future_global_optimizer_reset = global_optimizer_->async_reset();
        future_mapper_reset.get();
        future_global_optimizer_reset.get();
    }
    else {
        auto future_mapper_reset = mapper_->async_reset();
        future_mapper_reset.get();
    }

    if (bow_db_) {
        bow_db_->clear();
    }
    map_db_->clear();

    last_reloc_frm_id_ = 0;
    last_reloc_frm_timestamp_ = 0.0;

    tracking_state_ = tracker_state_t::Initializing;
}

std::shared_ptr<Mat44_t> tracking_module::feed_frame(data::frame curr_frm) {
    // 检查是否有暂停请求，如果有则暂停执行
    pause_if_requested();
    // 如果当前处于暂停状态，则等待直到恢复
    while (is_paused()) {
        std::this_thread::sleep_for(std::chrono::microseconds(5000));
    }

    // 将当前输入帧设置为成员变量，供后续处理使用
    curr_frm_ = curr_frm;

    bool succeeded = false;
    // 根据跟踪状态选择处理方式
    if (tracking_state_ == tracker_state_t::Initializing) {
        // 如果系统处于初始化状态，尝试进行地图初始化
        succeeded = initialize();
    }
    else {
        // 如果系统已经初始化完成，进行正常的跟踪流程
        // 使用互斥锁防止关键帧插入操作的并发冲突
        std::lock_guard<std::mutex> lock(mtx_stop_keyframe_insertion_);
        
        // 判断是否需要重定位（当跟踪状态为Lost时）
        bool relocalization_is_needed = tracking_state_ == tracker_state_t::Lost;
        SPDLOG_TRACE("tracking_module: start tracking");
        
        // 初始化跟踪统计变量
        unsigned int num_tracked_lms = 0;      // 跟踪到的路标点数量
        unsigned int num_reliable_lms = 0;     // 可靠的路标点数量
        // 设置最小观测次数阈值：如果关键帧数量>=3则阈值为3，否则为2
        const unsigned int min_num_obs_thr = (3 <= map_db_->get_num_keyframes()) ? 3 : 2;
        
        // 执行主要的跟踪逻辑
        succeeded = track(relocalization_is_needed, num_tracked_lms, num_reliable_lms, min_num_obs_thr);

        // 检查是否需要插入新的关键帧
        if (succeeded && !is_stopped_keyframe_insertion_ && new_keyframe_is_needed(num_tracked_lms, num_reliable_lms, min_num_obs_thr)) {
            // 如果跟踪成功且未停止关键帧插入且满足新关键帧条件，则插入新关键帧
            keyfrm_inserter_.insert_new_keyframe(map_db_, curr_frm_);
        }
    }

    // 根据跟踪结果进行状态转换
    if (succeeded) {
        // 跟踪成功，设置状态为正在跟踪
        tracking_state_ = tracker_state_t::Tracking;
    }
    else if (tracking_state_ == tracker_state_t::Tracking) {
        // 如果之前是跟踪状态但现在失败了，设置为丢失状态
        tracking_state_ = tracker_state_t::Lost;

        spdlog::info("tracking lost: frame {}", curr_frm_.id_);
        // 如果在初始化后的短时间内（init_retry_threshold_time_秒内）跟踪失败，重置整个系统
        if (!mapper_->is_paused() && curr_frm_.timestamp_ - initializer_.get_initial_frame_timestamp() < init_retry_threshold_time_) {
            spdlog::info("tracking lost within {} sec after initialization", init_retry_threshold_time_);
            reset();
            return nullptr;
        }
    }

    std::shared_ptr<Mat44_t> cam_pose_wc = nullptr;
    // 如果当前帧的位姿有效，保存相对位姿信息并创建返回的相机位姿
    if (curr_frm_.pose_is_valid()) {
        // 计算并保存当前帧相对于参考关键帧的位姿变换
        // 这个信息将用于下一次跟踪时更新相机位姿
        last_cam_pose_from_ref_keyfrm_ = curr_frm_.get_pose_cw() * curr_frm_.ref_keyfrm_->get_pose_wc();
        // 创建世界坐标系到相机坐标系的变换矩阵的智能指针
        cam_pose_wc = std::allocate_shared<Mat44_t>(Eigen::aligned_allocator<Mat44_t>(), curr_frm_.get_pose_wc());
    }

    // 更新上一帧信息，使用互斥锁保证线程安全
    SPDLOG_TRACE("tracking_module: update last frame (curr_frm_={})", curr_frm_.id_);
    {
        std::lock_guard<std::mutex> lock(mtx_last_frm_);
        last_frm_ = curr_frm_;  // 将当前帧保存为上一帧，供下次跟踪使用
    }
    SPDLOG_TRACE("tracking_module: finish tracking");

    // 返回相机位姿（如果跟踪成功）或nullptr（如果跟踪失败）
    return cam_pose_wc;
}

// 跟踪函数：这是整个跟踪模块的核心函数，负责处理当前帧的跟踪逻辑
// 参数说明：
// - relocalization_is_needed: 是否需要重定位（当跟踪丢失时为true）
// - num_tracked_lms: 输出参数，记录成功跟踪到的路标点数量
// - num_reliable_lms: 输出参数，记录可靠的路标点数量（观测次数足够多）
// - min_num_obs_thr: 最小观测次数阈值，用于判断路标点是否可靠
bool tracking_module::track(bool relocalization_is_needed,
                            unsigned int& num_tracked_lms,
                            unsigned int& num_reliable_lms,
                            const unsigned int min_num_obs_thr) {
    
    // 第一步：加锁保护共享资源
    // lock1: 锁定地图数据库，防止在跟踪过程中其他线程修改地图数据
    std::lock_guard<std::mutex> lock1(data::map_database::mtx_database_);
    // lock2: 锁定上一帧数据，防止并发访问冲突 
    std::lock_guard<std::mutex> lock2(mtx_last_frm_);

    // 第二步：更新上一帧的相机位姿
    // 由于建图模块可能会优化上一帧参考关键帧的位姿，所以需要更新上一帧的相机位姿
    // 这确保了跟踪使用的是最新优化后的位姿信息
    SPDLOG_TRACE("tracking_module: update the camera pose of the last frame (curr_frm_={})", curr_frm_.id_);
    update_last_frame();

    // 第三步：设置当前帧的参考关键帧
    // 将上一帧的参考关键帧作为当前帧的参考关键帧
    // 参考关键帧用于计算相对位姿变换
    curr_frm_.ref_keyfrm_ = last_frm_.ref_keyfrm_;

    // 第四步：根据不同情况选择跟踪策略
    bool succeeded = false;  // 跟踪成功标志
    
    // 策略1：强制基于位姿的重定位
    // 如果存在BoW数据库且有外部位姿重定位请求，则执行强制重定位
    if (bow_db_ && relocalize_by_pose_is_requested()) {
        // 使用外部提供的位姿进行强制重定位
        // 这通常用于手动指定相机位置或从外部传感器获得位姿信息的情况
        succeeded = relocalize_by_pose(get_relocalize_by_pose_request());
    }
    // 策略2：正常跟踪模式
    // 如果不需要重定位，则使用正常的帧间跟踪
    else if (!relocalization_is_needed) {
        SPDLOG_TRACE("tracking_module: track_current_frame (curr_frm_={})", curr_frm_.id_);
        // 基于运动模型、BoW匹配或特征匹配进行帧间跟踪
        succeeded = track_current_frame();
    }
    // 策略3：自动重定位模式
    // 如果跟踪丢失且启用了自动重定位功能，则尝试重定位
    else if (bow_db_ && enable_auto_relocalization_) {
        // 计算当前帧的BoW表示，用于与关键帧数据库进行匹配
        SPDLOG_TRACE("tracking_module: Compute the BoW representations to perform relocalization (curr_frm_={})", curr_frm_.id_);
        if (!curr_frm_.bow_is_available()) {
            // 如果当前帧还没有计算BoW特征，则先计算
            curr_frm_.compute_bow(bow_vocab_);
        }
        
        // 尝试通过BoW数据库进行重定位
        SPDLOG_TRACE("tracking_module: try to relocalize (curr_frm_={})", curr_frm_.id_);
        succeeded = relocalizer_.relocalize(bow_db_, curr_frm_);
        
        // 如果重定位成功，记录重定位信息
        if (succeeded) {
            last_reloc_frm_id_ = curr_frm_.id_;           // 记录重定位的帧ID
            last_reloc_frm_timestamp_ = curr_frm_.timestamp_;  // 记录重定位的时间戳
        }
    }

    // 第五步：局部地图跟踪和位姿优化
    // 获取固定关键帧ID阈值，用于区分固定关键帧和临时关键帧
    unsigned int fixed_keyframe_id_threshold = map_db_->get_fixed_keyframe_id_threshold();
    unsigned int num_temporal_keyfrms = 0;  // 临时关键帧数量
    
    // 如果前面的跟踪步骤成功，则进行局部地图跟踪
    if (succeeded) {
        // 更新局部地图，搜索局部路标点，并优化当前帧位姿
        // 这一步会找到更多的特征匹配，提高跟踪的鲁棒性
        succeeded = track_local_map(num_tracked_lms, num_reliable_lms, num_temporal_keyfrms, 
                                   min_num_obs_thr, fixed_keyframe_id_threshold);
    }

    // 第六步：不包含临时关键帧的局部地图跟踪
    // 如果存在固定关键帧阈值、跟踪成功且有临时关键帧，则进行第二次优化
    // 这次优化排除临时关键帧，只使用更稳定的固定关键帧
    if (fixed_keyframe_id_threshold > 0 && succeeded && num_temporal_keyfrms > 0) {
        succeeded = track_local_map_without_temporal_keyframes(num_tracked_lms, num_reliable_lms, 
                                                              min_num_obs_thr, fixed_keyframe_id_threshold);
    }

    // 第七步：更新运动模型
    // 如果跟踪成功，则计算当前帧和上一帧之间的位姿变换
    // 这个运动模型将用于下一帧的初始位姿预测
    if (succeeded) {
        SPDLOG_TRACE("tracking_module: update_motion_model (curr_frm_={})", curr_frm_.id_);
        update_motion_model();
    }

    // 第八步：更新帧统计信息
    // 记录当前帧的跟踪状态和统计数据到地图数据库
    // 这些信息用于后续的关键帧选择和地图优化决策
    SPDLOG_TRACE("tracking_module: update_frame_statistics (curr_frm_={})", curr_frm_.id_);
    map_db_->update_frame_statistics(curr_frm_, !succeeded);  // 第二个参数表示是否跟踪失败

    // 返回跟踪是否成功
    return succeeded;
}

bool tracking_module::track_local_map(unsigned int& num_tracked_lms,
                                      unsigned int& num_reliable_lms,
                                      unsigned int& num_temporal_keyfrms,
                                      const unsigned int min_num_obs_thr,
                                      const unsigned int fixed_keyframe_id_threshold) {
    bool succeeded = false;
    SPDLOG_TRACE("tracking_module: update_local_map (curr_frm_={})", curr_frm_.id_);
    succeeded = update_local_map(fixed_keyframe_id_threshold, num_temporal_keyfrms);

    if (succeeded) {
        succeeded = search_local_landmarks(fixed_keyframe_id_threshold);
    }

    if (succeeded) {
        SPDLOG_TRACE("tracking_module: optimize_current_frame_with_local_map (curr_frm_={})", curr_frm_.id_);
        succeeded = optimize_current_frame_with_local_map(num_tracked_lms, num_reliable_lms, min_num_obs_thr);
    }

    if (!succeeded) {
        spdlog::info("local map tracking failed (curr_frm_={})", curr_frm_.id_);
    }
    return succeeded;
}

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
    assert(!is_stopped_keyframe_insertion_);
    for (const auto& keyfrm : curr_frm_.ref_keyfrm_->graph_node_->get_keyframes_from_root()) {
        auto future = mapper_->async_add_keyframe(keyfrm);
        future.get();
    }

    // succeeded
    return true;
}

// 当前帧跟踪函数：这是正常跟踪模式下的核心函数
// 该函数实现了三种不同的跟踪策略，按照从快到慢、从简单到复杂的顺序依次尝试
// 只有当前一种方法失败时，才会尝试下一种方法，这样可以平衡跟踪速度和鲁棒性
bool tracking_module::track_current_frame() {
    bool succeeded = false;  // 跟踪成功标志

    // 策略1：基于运动模型的跟踪（最快速的方法）
    // 前提条件：运动模型有效（即能够预测相机的运动趋势）
    if (twist_is_valid_) {
        // 使用运动模型进行跟踪
        // twist_ 是从上一帧到当前帧的位姿变换矩阵（运动模型）
        // 这种方法假设相机运动具有连续性，通过预测下一帧位姿来进行特征匹配
        // 适用于相机运动平滑、帧率较高的场景
        succeeded = frame_tracker_.motion_based_track(curr_frm_, last_frm_, twist_);
    }
    
    // 策略2：基于BoW（词袋模型）的跟踪（中等速度的方法）
    // 如果运动模型跟踪失败，尝试使用BoW特征进行匹配
    if (!succeeded) {
        // 首先确保当前帧的BoW特征已经计算
        // BoW特征是将图像特征描述子量化到视觉词典中的表示方法
        if (bow_vocab_ && !curr_frm_.bow_is_available()) {
            // 如果存在BoW词典且当前帧还没有计算BoW特征，则先计算
            curr_frm_.compute_bow(bow_vocab_);
        }
        
        // 进行BoW匹配跟踪的前提条件：
        // 1. 当前帧有BoW特征表示
        // 2. 参考关键帧也有BoW特征表示
        if (curr_frm_.bow_is_available() && curr_frm_.ref_keyfrm_->bow_is_available()) {
            // 使用BoW特征进行帧间跟踪
            // 这种方法通过比较当前帧和参考关键帧的BoW特征来建立特征对应关系
            // 相比运动模型，这种方法对相机运动的连续性要求较低
            // 但计算量比运动模型跟踪大，速度较慢
            succeeded = frame_tracker_.bow_match_based_track(curr_frm_, last_frm_, curr_frm_.ref_keyfrm_);
        }
    }
    
    // 策略3：基于鲁棒匹配的跟踪（最慢但最鲁棒的方法）
    // 如果前两种方法都失败，使用最鲁棒但最耗时的匹配方法
    if (!succeeded) {
        // 使用鲁棒匹配进行跟踪
        // 这种方法不依赖于运动模型或BoW特征，而是直接进行特征描述子匹配
        // 通常使用更严格的匹配策略和异常值检测机制
        // 虽然计算量最大、速度最慢，但在困难场景下具有最好的鲁棒性
        // 适用于相机运动剧烈、纹理较少或光照变化较大的场景
        succeeded = frame_tracker_.robust_match_based_track(curr_frm_, last_frm_, curr_frm_.ref_keyfrm_);
    }

    // 返回跟踪结果
    // true: 至少有一种跟踪策略成功，当前帧位姿估计有效
    // false: 所有跟踪策略都失败，需要进入重定位模式
    return succeeded;
}

bool tracking_module::relocalize_by_pose(const pose_request& request) {
    bool succeeded = false;
    curr_frm_.set_pose_cw(request.pose_cw_);

    if (!curr_frm_.bow_is_available()) {
        curr_frm_.compute_bow(bow_vocab_);
    }
    const auto candidates = get_close_keyframes(request);
    for (const auto& candidate : candidates) {
        spdlog::debug("relocalize_by_pose: candidate = {}", candidate->id_);
    }

    if (!candidates.empty()) {
        succeeded = relocalizer_.reloc_by_candidates(curr_frm_, candidates, use_robust_matcher_for_relocalization_request_);
        if (succeeded) {
            last_reloc_frm_id_ = curr_frm_.id_;
            last_reloc_frm_timestamp_ = curr_frm_.timestamp_;
            // If the initial pose was given manually, use motion_based_track, expecting that the camera is not moving.
            last_frm_ = curr_frm_;
        }
    }
    else {
        curr_frm_.invalidate_pose();
    }
    finish_relocalize_by_pose_request();
    return succeeded;
}

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

void tracking_module::replace_landmarks_in_last_frm(nondeterministic::unordered_map<std::shared_ptr<data::landmark>, std::shared_ptr<data::landmark>>& replaced_lms) {
    std::lock_guard<std::mutex> lock(mtx_last_frm_);
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

// 搜索局部路标点函数：在局部地图中寻找可以投影到当前帧的路标点
// 这是局部地图跟踪的核心步骤之一，用于增加当前帧与地图点的匹配数量
// 参数 fixed_keyframe_id_threshold: 固定关键帧ID阈值，用于区分固定关键帧和临时关键帧
bool tracking_module::search_local_landmarks(unsigned int fixed_keyframe_id_threshold) {
    
    // 第一步：收集当前帧中已经观测到的路标点ID
    // 这样做是为了避免重复投影已经在当前帧中观测到的路标点
    std::unordered_set<unsigned int> curr_landmark_ids;
    
    // 遍历当前帧中所有已经观测到的路标点
    for (const auto& lm : curr_frm_.get_landmarks()) {
        // 跳过空指针路标点
        if (!lm) {
            continue;
        }
        // 跳过即将被删除的路标点
        // 这些路标点在优化过程中被标记为异常值或质量不佳
        if (lm->will_be_erased()) {
            continue;
        }

        // 记录这个路标点的ID，表示它不能被重新投影
        // 因为它已经在当前帧中被观测到了
        curr_landmark_ids.insert(lm->id_);

        // 增加该路标点的可观测次数统计
        // 这个统计信息用于评估路标点的质量和可靠性
        lm->increase_num_observable();
    }

    // 第二步：准备投影匹配的相关变量
    bool found_proj_candidate = false;  // 是否找到可投影的候选路标点
    
    // 临时变量，用于存储投影计算结果
    Vec2_t reproj;                      // 重投影坐标
    float x_right;                      // 右相机的x坐标（双目相机使用）
    unsigned int pred_scale_level;      // 预测的尺度层级
    
    // 存储路标点投影信息的映射表
    eigen_alloc_unord_map<unsigned int, Vec2_t> lm_to_reproj;      // 路标点ID -> 投影坐标
    std::unordered_map<unsigned int, float> lm_to_x_right;         // 路标点ID -> 右相机x坐标
    std::unordered_map<unsigned int, unsigned int> lm_to_scale;    // 路标点ID -> 尺度层级
    
    // 第三步：遍历所有局部路标点，寻找可投影的候选点
    for (const auto& lm : local_landmarks_) {
        // 跳过已经在当前帧中观测到的路标点
        if (curr_landmark_ids.count(lm->id_)) {
            continue;
        }
        // 跳过即将被删除的路标点
        if (lm->will_be_erased()) {
            continue;
        }
        
        // 第四步：处理固定关键帧阈值的约束
        // 如果设置了固定关键帧阈值，需要过滤掉主要由临时关键帧观测到的路标点
        if (fixed_keyframe_id_threshold > 0) {
            // 获取该路标点的所有观测信息
            const auto observations = lm->get_observations();
            unsigned int temporal_observations = 0;  // 临时关键帧观测次数
            
            // 统计有多少观测来自临时关键帧（ID >= 阈值的关键帧）
            for (auto obs : observations) {
                auto keyfrm = obs.first.lock();  // 获取观测到该路标点的关键帧
                if (keyfrm->id_ >= fixed_keyframe_id_threshold) {
                    ++temporal_observations;
                }
            }
            
            // 计算临时关键帧观测的比例
            const double temporal_ratio_thr = 0.5;  // 临时观测比例阈值（50%）
            double temporal_ratio = static_cast<double>(temporal_observations) / observations.size();
            
            // 如果临时观测比例过高，跳过这个路标点
            // 这样做是为了优先使用由固定关键帧观测到的更稳定的路标点
            if (temporal_ratio > temporal_ratio_thr) {
                continue;
            }
        }

        // 第五步：检查路标点的可观测性
        // 判断该路标点是否可以从当前帧的视角观测到
        // 参数0.5是视角阈值，用于判断观测角度是否合适
        if (curr_frm_.can_observe(lm, 0.5, reproj, x_right, pred_scale_level)) {
            // 如果可以观测到，保存投影信息
            lm_to_reproj[lm->id_] = reproj;           // 保存投影坐标
            lm_to_x_right[lm->id_] = x_right;         // 保存右相机坐标（双目用）
            lm_to_scale[lm->id_] = pred_scale_level;  // 保存预测的尺度层级

            // 增加该路标点的可观测次数统计
            lm->increase_num_observable();

            // 标记找到了可投影的候选点
            found_proj_candidate = true;
        }
    }

    // 第六步：检查是否找到了候选投影点
    if (!found_proj_candidate) {
        // 如果没有找到任何可投影的路标点，输出警告并返回失败
        spdlog::warn("projection candidate not found");
        return false;
    }

    // 第七步：执行投影匹配
    // 创建投影匹配器，参数0.8是匹配阈值（越小越严格）
    match::projection projection_matcher(0.8);
    
    // 根据当前帧是否在重定位后的不稳定期来设置投影边界
    // 如果是重定位后的前2帧，使用更大的搜索边界以提高匹配成功率
    const float margin = (curr_frm_.id_ < last_reloc_frm_id_ + 2)
                             ? margin_local_map_projection_unstable_  // 不稳定期的大边界
                             : margin_local_map_projection_;          // 正常的小边界
    
    // 执行帧与路标点的投影匹配
    // 这一步会在当前帧中寻找与局部路标点对应的特征点
    // 成功的匹配会建立新的2D-3D对应关系，增强跟踪的鲁棒性
    projection_matcher.match_frame_and_landmarks(curr_frm_, local_landmarks_, 
                                                lm_to_reproj, lm_to_x_right, lm_to_scale, margin);
    
    // 返回成功
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
