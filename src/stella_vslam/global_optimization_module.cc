#include "stella_vslam/global_optimization_module.h"
#include "stella_vslam/mapping_module.h"
#include "stella_vslam/tracking_module.h"
#include "stella_vslam/data/keyframe.h"
#include "stella_vslam/data/landmark.h"
#include "stella_vslam/data/map_database.h"
#include "stella_vslam/match/fuse.h"
#include "stella_vslam/util/converter.h"
#include "stella_vslam/util/yaml.h"

#include <spdlog/spdlog.h>

namespace stella_vslam {

global_optimization_module::global_optimization_module(data::map_database* map_db, data::bow_database* bow_db,
                                                       data::bow_vocabulary* bow_vocab, const YAML::Node& yaml_node,
                                                       const bool fix_scale)
    : loop_detector_(new module::loop_detector(bow_db, bow_vocab, util::yaml_optional_ref(yaml_node, "LoopDetector"), fix_scale)),
      loop_bundle_adjuster_(new module::loop_bundle_adjuster(
          map_db,
          util::yaml_optional_ref(yaml_node, "GlobalOptimizer")["num_iter"].as<unsigned int>(10),
          util::yaml_optional_ref(yaml_node, "GlobalOptimizer")["use_huber_kernel"].as<bool>(false),
          util::yaml_optional_ref(yaml_node, "GlobalOptimizer")["verbose"].as<bool>(false))),
      map_db_(map_db),
      graph_optimizer_(new optimize::graph_optimizer(util::yaml_optional_ref(yaml_node, "GraphOptimizer"), fix_scale)),
      thr_neighbor_keyframes_(util::yaml_optional_ref(yaml_node, "GlobalOptimizer")["thr_neighbor_keyframes"].as<unsigned int>(15)) {
    spdlog::debug("CONSTRUCT: global_optimization_module");
}

global_optimization_module::~global_optimization_module() {
    abort_loop_BA();
    if (thread_for_loop_BA_) {
        thread_for_loop_BA_->join();
    }
    spdlog::debug("DESTRUCT: global_optimization_module");
}

void global_optimization_module::set_tracking_module(tracking_module* tracker) {
    tracker_ = tracker;
}

void global_optimization_module::set_mapping_module(mapping_module* mapper) {
    mapper_ = mapper;
    loop_bundle_adjuster_->set_mapping_module(mapper);
}

void global_optimization_module::enable_loop_detector() {
    spdlog::info("enable loop detector");
    loop_detector_->enable_loop_detector();
}

void global_optimization_module::disable_loop_detector() {
    spdlog::info("disable loop detector");
    loop_detector_->disable_loop_detector();
}

bool global_optimization_module::loop_detector_is_enabled() const {
    return loop_detector_->is_enabled();
}

bool global_optimization_module::request_loop_closure(unsigned int keyfrm1_id, unsigned int keyfrm2_id) {
    std::lock_guard<std::mutex> lock(mtx_loop_closure_request_);
    if (loop_closure_is_requested_) {
        spdlog::warn("Can not process new loop closure request while previous was not finished");
        return false;
    }
    loop_closure_is_requested_ = true;
    loop_closure_request_.keyfrm1_id_ = keyfrm1_id;
    loop_closure_request_.keyfrm2_id_ = keyfrm2_id;
    return true;
}

bool global_optimization_module::loop_closure_is_requested() {
    std::lock_guard<std::mutex> lock(mtx_loop_closure_request_);
    return loop_closure_is_requested_;
}

loop_closure_request& global_optimization_module::get_loop_closure_request() {
    std::lock_guard<std::mutex> lock(mtx_loop_closure_request_);
    return loop_closure_request_;
}

void global_optimization_module::finish_loop_closure_request() {
    std::lock_guard<std::mutex> lock(mtx_loop_closure_request_);
    loop_closure_is_requested_ = false;
}

/**
 * [功能描述]：执行回环闭合操作，处理回环检测请求并进行回环校正
 * @param request：回环闭合请求，包含两个可能形成回环的关键帧ID（keyfrm1_id_ 和 keyfrm2_id_）
 * @return bool：回环闭合是否成功，true 表示成功执行回环校正，false 表示回环检测失败或关键帧无效
 */
bool global_optimization_module::loop_closure(const loop_closure_request& request) {
    {
        // 加锁保护地图数据库，防止在回环检测和校正期间数据被修改
        std::lock_guard<std::mutex> lock(data::map_database::mtx_database_);

        // 从请求中获取当前关键帧ID（较大的ID，即较新的帧）和候选关键帧ID（较小的ID，即较旧的帧）
        unsigned int curr_keyfrm_id = std::max(request.keyfrm1_id_, request.keyfrm2_id_);
        unsigned int candidate_keyfrm_id = std::min(request.keyfrm1_id_, request.keyfrm2_id_);

        // 从地图数据库中获取当前关键帧，确保在回环检测和校正期间不被删除
        cur_keyfrm_ = map_db_->get_keyframe(curr_keyfrm_id);
        if (cur_keyfrm_ == nullptr) {
            // 当前关键帧不存在，记录日志并返回失败
            spdlog::info("keyframe {} not found", curr_keyfrm_id);
            return false;
        }
        // 设置当前关键帧不可删除，防止在回环处理过程中被意外移除
        cur_keyfrm_->set_not_to_be_erased();
        // 将当前关键帧设置到回环检测器中
        loop_detector_->set_current_keyframe(cur_keyfrm_);

        // 从地图数据库中获取候选关键帧（潜在的回环帧）
        auto candidate_keyfrm = map_db_->get_keyframe(candidate_keyfrm_id);
        if (candidate_keyfrm == nullptr) {
            // 候选关键帧不存在，记录日志并返回失败
            spdlog::info("candidate keyframe {} not found", candidate_keyfrm_id);
            return false;
        }
        // 将候选关键帧添加到回环检测器的候选列表中
        loop_detector_->add_loop_candidate(candidate_keyfrm);

        // 验证候选帧并从中选择一个有效的回环候选
        if (!loop_detector_->validate_candidates()) {
            // 验证失败，未找到有效的回环候选
            // 允许删除当前关键帧（恢复其可删除状态）
            cur_keyfrm_->set_to_be_erased();
            return false;
        }
    }
    // 锁释放后执行回环校正（包括位姿图优化和地图点融合）

    // 执行回环校正，修正累积漂移误差
    correct_loop();
    // 完成回环闭合请求的后续处理
    finish_loop_closure_request();
    return true;
}

/**
 * [功能描述]：全局优化模块的主运行循环
 *            负责处理关键帧队列、执行回环检测和回环校正
 *            该函数在独立线程中运行，持续监控并处理回环闭合任务
 */
void global_optimization_module::run() {
    spdlog::info("start global optimization module");

    // 初始化终止标志为false，表示模块正在运行
    is_terminated_ = false;

    // 主循环：持续运行直到收到终止请求
    while (true) {
        // 每次循环休眠5ms，降低CPU占用率
        std::this_thread::sleep_for(std::chrono::milliseconds(5));

        // ==================== 检查终止请求 ====================
        if (terminate_is_requested()) {
            // 收到终止请求，执行终止操作并退出循环
            terminate();
            break;
        }

        // ==================== 检查外部回环闭合请求 ====================
        // 处理来自外部的显式回环闭合请求（如用户指定的回环）
        if (loop_closure_is_requested()) {
            loop_closure(get_loop_closure_request());
        }

        // ==================== 检查暂停请求 ====================
        if (pause_is_requested()) {
            // 执行暂停操作
            pause();
            // 暂停期间持续等待，直到收到终止或重置请求
            while (is_paused() && !terminate_is_requested() && !reset_is_requested()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(3));
            }
        }

        // ==================== 检查重置请求 ====================
        if (reset_is_requested()) {
            // 执行重置操作（清空队列、重置状态等）
            reset();
            continue;
        }

        // ==================== 检查关键帧队列 ====================
        // 如果队列为空，跳过后续处理，继续下一次循环
        if (!keyframe_is_queued()) {
            continue;
        }

        // ==================== 从队列中取出关键帧 ====================
        {
            // 加锁保护关键帧队列
            std::lock_guard<std::mutex> lock(mtx_keyfrm_queue_);
            // 取出队首关键帧作为当前处理帧
            cur_keyfrm_ = keyfrms_queue_.front();
            // 将该关键帧从队列中移除
            keyfrms_queue_.pop_front();
        }

        // ==================== 执行回环检测 ====================
        {
            // 加锁保护地图数据库
            std::lock_guard<std::mutex> lock(data::map_database::mtx_database_);

            // 设置当前关键帧不可删除，确保在回环检测和校正期间不被移除
            cur_keyfrm_->set_not_to_be_erased();

            // 将当前关键帧传递给回环检测器
            loop_detector_->set_current_keyframe(cur_keyfrm_);

            // 使用词袋模型（BoW）检测回环候选帧
            if (!loop_detector_->detect_loop_candidates()) {
                // 未找到回环候选，允许删除当前关键帧
                cur_keyfrm_->set_to_be_erased();
                continue;
            }

            // 验证候选帧并从中选择一个有效的回环帧
            if (!loop_detector_->validate_candidates()) {
                // 验证失败，未找到有效回环，允许删除当前关键帧
                cur_keyfrm_->set_to_be_erased();
                continue;
            }
        }

        // ==================== 执行回环校正 ====================
        // 回环检测成功，执行回环校正（位姿优化、路标点融合等）
        correct_loop();
    }

    spdlog::info("terminate global optimization module");
}

void global_optimization_module::queue_keyframe(const std::shared_ptr<data::keyframe>& keyfrm) {
    std::lock_guard<std::mutex> lock(mtx_keyfrm_queue_);
    keyfrms_queue_.push_back(keyfrm);
}

bool global_optimization_module::keyframe_is_queued() const {
    std::lock_guard<std::mutex> lock(mtx_keyfrm_queue_);
    return !keyfrms_queue_.empty();
}

/**
 * [功能描述]：执行回环校正，修正由于累积漂移导致的位姿误差
 *            该函数是回环闭合的核心，包含位姿校正、路标点融合、位姿图优化和全局BA等步骤
 */
void global_optimization_module::correct_loop() {
    // 获取经过验证的最终回环候选关键帧
    auto final_candidate_keyfrm = loop_detector_->get_selected_candidate_keyframe();

    // 输出检测到的回环信息：候选帧ID和当前帧ID
    spdlog::info("detect loop: keyframe {} - keyframe {}", final_candidate_keyfrm->id_, cur_keyfrm_->id_);

    // 检查当前帧和候选帧是否属于同一个生成树
    // 如果不是同一棵树，则无法进行回环校正（合并两棵生成树的功能尚未实现）
    if (cur_keyfrm_->graph_node_->get_spanning_root() != final_candidate_keyfrm->graph_node_->get_spanning_root()) {
        spdlog::warn("The feature to merge two spanning trees has not yet been implemented.");
        return;
    }

    // ==================== 步骤0：预处理 ====================
    // 0-1. 暂停建图模块和之前的回环BA优化器

    // 异步请求暂停建图模块，避免在回环校正期间产生新的关键帧和路标点
    SPDLOG_TRACE("global_optimization_module: pause the mapping module");
    auto future_pause = mapper_->async_pause();

    // 如果之前的回环BA线程存在或正在运行，则中止它
    if (thread_for_loop_BA_ || loop_bundle_adjuster_->is_running()) {
        SPDLOG_TRACE("global_optimization_module: abort loop bundle adjustment");
        abort_loop_BA();
    }
    // 等待建图模块完全暂停
    future_pause.get();

    // ==================== 步骤1：计算并校正共视关键帧的Sim3变换 ====================
    // 计算当前关键帧共视帧的Sim3变换（回环检测器已估计出当前帧的Sim3）
    // 然后将共视帧移动到校正后的位置
    // 最后使用校正前后的相机位姿将观测到的路标点也移动到正确位置

    SPDLOG_TRACE("global_optimization_module: compute the Sim3 of the covisibilities of the current keyframe whose Sim3 is already estimated by the loop detector");

    // 获取当前关键帧的共视关键帧（共享路标点数量超过阈值的帧）
    std::vector<std::shared_ptr<data::keyframe>> curr_neighbors = cur_keyfrm_->graph_node_->get_covisibilities_over_min_num_shared_lms(thr_neighbor_keyframes_);
    // 将当前关键帧也加入共视帧列表
    curr_neighbors.push_back(cur_keyfrm_);

    // 存储回环校正前的Sim3相机位姿（从世界坐标系到相机坐标系的变换）
    module::keyframe_Sim3_pairs_t Sim3s_nw_before_correction;
    // 存储回环校正后的Sim3相机位姿
    module::keyframe_Sim3_pairs_t Sim3s_nw_after_correction;

    // 存储找到的路标点ID到参考关键帧ID的映射
    std::unordered_map<unsigned int, unsigned int> found_lm_to_ref_keyfrm_id;
    // 获取回环检测器计算的校正后的Sim3变换（世界坐标系到当前帧）
    const auto g2o_Sim3_cw_after_correction = loop_detector_->get_Sim3_world_to_current();
    {
        // 加锁保护地图数据库
        std::lock_guard<std::mutex> lock(data::map_database::mtx_database_);

        // 获取回环校正前当前关键帧的相机位姿（从相机坐标系到世界坐标系）
        const Mat44_t cam_pose_wc_before_correction = cur_keyfrm_->get_pose_wc();

        // 计算所有共视帧在回环校正前的Sim3位姿
        Sim3s_nw_before_correction = get_Sim3s_before_loop_correction(curr_neighbors);
        // 计算所有共视帧在回环校正后的Sim3位姿（基于当前帧的校正量传播到共视帧）
        Sim3s_nw_after_correction = get_Sim3s_after_loop_correction(cam_pose_wc_before_correction, g2o_Sim3_cw_after_correction, curr_neighbors);

        // 校正共视帧观测到的路标点位置
        correct_covisibility_landmarks(Sim3s_nw_before_correction, Sim3s_nw_after_correction, found_lm_to_ref_keyfrm_id);
        // 校正共视关键帧的相机位姿
        correct_covisibility_keyframes(Sim3s_nw_after_correction);
    }

    // ==================== 步骤2：解决回环融合导致的路标点重复问题 ====================

    SPDLOG_TRACE("global_optimization_module: resolve duplications of landmarks caused by loop fusion");
    // 获取当前帧与候选帧匹配的路标点
    const auto curr_match_lms_observed_in_cand = loop_detector_->current_matched_landmarks_observed_in_candidate();
    // 替换重复的路标点（将当前帧观测到的点与候选帧的对应点融合）
    replace_duplicated_landmarks(curr_match_lms_observed_in_cand, Sim3s_nw_after_correction);

    // ==================== 步骤3：提取回环融合后创建的新连接 ====================

    SPDLOG_TRACE("global_optimization_module: extract the new connections created after loop fusion");
    // 提取因路标点融合而产生的新的共视关系
    const auto new_connections = extract_new_connections(curr_neighbors);

    // ==================== 步骤4：位姿图优化 ====================

    SPDLOG_TRACE("global_optimization_module: pose graph optimization");
    // 执行位姿图优化，优化所有关键帧的位姿以满足新的回环约束
    graph_optimizer_->optimize(final_candidate_keyfrm, cur_keyfrm_, Sim3s_nw_before_correction, Sim3s_nw_after_correction, new_connections, found_lm_to_ref_keyfrm_id);

    // 在候选帧和当前帧之间添加双向回环边，建立回环约束关系
    final_candidate_keyfrm->graph_node_->add_loop_edge(cur_keyfrm_);
    cur_keyfrm_->graph_node_->add_loop_edge(final_candidate_keyfrm);

    // ==================== 步骤5：启动回环BA（全局光束法平差） ====================

    SPDLOG_TRACE("global_optimization_module: wait for loop BA");
    // 等待之前的回环BA完成
    while (loop_bundle_adjuster_->is_running()) {
        std::this_thread::sleep_for(std::chrono::microseconds(1000));
    }
    // 如果之前的BA线程存在，等待其结束并释放资源
    if (thread_for_loop_BA_) {
        SPDLOG_TRACE("global_optimization_module: wait for last loop BA");
        thread_for_loop_BA_->join();
        thread_for_loop_BA_.reset(nullptr);
    }
    // 在新线程中启动回环BA优化，进一步优化所有关键帧位姿和路标点位置
    SPDLOG_TRACE("global_optimization_module: launch loop BA");
    thread_for_loop_BA_ = std::unique_ptr<std::thread>(new std::thread(&module::loop_bundle_adjuster::optimize, loop_bundle_adjuster_.get(), cur_keyfrm_));

    // ==================== 步骤6：后处理 ====================

    SPDLOG_TRACE("global_optimization_module: resume the mapping module");
    // 恢复建图模块的运行
    mapper_->resume();

    // 将当前帧ID记录为最近一次回环校正的关键帧ID
    loop_detector_->set_loop_correct_keyframe_id(cur_keyfrm_->id_);
}

module::keyframe_Sim3_pairs_t global_optimization_module::get_Sim3s_before_loop_correction(const std::vector<std::shared_ptr<data::keyframe>>& neighbors) const {
    module::keyframe_Sim3_pairs_t Sim3s_nw_before_loop_correction;

    for (const auto& neighbor : neighbors) {
        // camera pose of `neighbor` BEFORE loop correction
        const Mat44_t cam_pose_nw = neighbor->get_pose_cw();
        // create Sim3 from SE3
        const Mat33_t& rot_nw = cam_pose_nw.block<3, 3>(0, 0);
        const Vec3_t& trans_nw = cam_pose_nw.block<3, 1>(0, 3);
        const g2o::Sim3 Sim3_nw_before_correction(rot_nw, trans_nw, 1.0);
        Sim3s_nw_before_loop_correction[neighbor] = Sim3_nw_before_correction;
    }

    return Sim3s_nw_before_loop_correction;
}

module::keyframe_Sim3_pairs_t global_optimization_module::get_Sim3s_after_loop_correction(const Mat44_t& cam_pose_wc_before_correction,
                                                                                          const g2o::Sim3& g2o_Sim3_cw_after_correction,
                                                                                          const std::vector<std::shared_ptr<data::keyframe>>& neighbors) const {
    module::keyframe_Sim3_pairs_t Sim3s_nw_after_loop_correction;

    for (auto neighbor : neighbors) {
        // camera pose of `neighbor` BEFORE loop correction
        const Mat44_t cam_pose_nw_before_correction = neighbor->get_pose_cw();
        // create the relative Sim3 from the current to `neighbor`
        const Mat44_t cam_pose_nc = cam_pose_nw_before_correction * cam_pose_wc_before_correction;
        const Mat33_t& rot_nc = cam_pose_nc.block<3, 3>(0, 0);
        const Vec3_t& trans_nc = cam_pose_nc.block<3, 1>(0, 3);
        const g2o::Sim3 Sim3_nc(rot_nc, trans_nc, 1.0);
        // compute the camera poses AFTER loop correction of the neighbors
        const g2o::Sim3 Sim3_nw_after_correction = Sim3_nc * g2o_Sim3_cw_after_correction;
        Sim3s_nw_after_loop_correction[neighbor] = Sim3_nw_after_correction;
    }

    return Sim3s_nw_after_loop_correction;
}

void global_optimization_module::correct_covisibility_landmarks(const module::keyframe_Sim3_pairs_t& Sim3s_nw_before_correction,
                                                                const module::keyframe_Sim3_pairs_t& Sim3s_nw_after_correction,
                                                                std::unordered_map<unsigned int, unsigned int>& found_lm_to_ref_keyfrm_id) const {
    for (const auto& t : Sim3s_nw_after_correction) {
        auto neighbor = t.first;
        // neighbor->world AFTER loop correction
        const auto Sim3_wn_after_correction = t.second.inverse();
        // world->neighbor BEFORE loop correction
        const auto& Sim3_nw_before_correction = Sim3s_nw_before_correction.at(neighbor);

        const auto ngh_landmarks = neighbor->get_landmarks();
        for (const auto& lm : ngh_landmarks) {
            if (!lm) {
                continue;
            }
            if (lm->will_be_erased()) {
                continue;
            }

            // avoid duplication
            if (found_lm_to_ref_keyfrm_id.count(lm->id_)) {
                continue;
            }
            // record the reference keyframe used in loop fusion of landmarks
            found_lm_to_ref_keyfrm_id[lm->id_] = neighbor->id_;

            // correct position of `lm`
            const Vec3_t pos_w_before_correction = lm->get_pos_in_world();
            const Vec3_t pos_w_after_correction = Sim3_wn_after_correction.map(Sim3_nw_before_correction.map(pos_w_before_correction));
            lm->set_pos_in_world(pos_w_after_correction);
            // update geometry
            lm->update_mean_normal_and_obs_scale_variance();
        }
    }
}

void global_optimization_module::correct_covisibility_keyframes(const module::keyframe_Sim3_pairs_t& Sim3s_nw_after_correction) const {
    for (const auto& t : Sim3s_nw_after_correction) {
        auto neighbor = t.first;
        const auto Sim3_nw_after_correction = t.second;

        const auto s_nw = Sim3_nw_after_correction.scale();
        const Mat33_t rot_nw = Sim3_nw_after_correction.rotation().toRotationMatrix();
        const Vec3_t trans_nw = Sim3_nw_after_correction.translation() / s_nw;
        const Mat44_t cam_pose_nw = util::converter::to_eigen_pose(rot_nw, trans_nw);
        neighbor->set_pose_cw(cam_pose_nw);
    }
}

void global_optimization_module::replace_duplicated_landmarks(const std::vector<std::shared_ptr<data::landmark>>& curr_match_lms_observed_in_cand,
                                                              const module::keyframe_Sim3_pairs_t& Sim3s_nw_after_correction) const {
    nondeterministic::unordered_map<std::shared_ptr<data::landmark>, std::shared_ptr<data::landmark>> replaced_lms;
    // resolve duplications of landmarks between the current keyframe and the loop candidate
    {
        std::lock_guard<std::mutex> lock(data::map_database::mtx_database_);

        for (unsigned int idx = 0; idx < cur_keyfrm_->frm_obs_.undist_keypts_.size(); ++idx) {
            auto curr_match_lm_in_cand = curr_match_lms_observed_in_cand.at(idx);
            if (!curr_match_lm_in_cand) {
                continue;
            }
            if (curr_match_lm_in_cand->will_be_erased()) {
                continue;
            }

            if (curr_match_lm_in_cand->is_observed_in_keyframe(cur_keyfrm_)) {
                cur_keyfrm_->erase_landmark(curr_match_lm_in_cand);
                curr_match_lm_in_cand->erase_observation(map_db_, cur_keyfrm_);
            }

            const auto& lm_in_curr = cur_keyfrm_->get_landmark(idx);
            if (lm_in_curr) {
                // if the landmark corresponding `idx` exists,
                // replace it with `curr_match_lm_in_cand` (observed in the candidate)
                if (lm_in_curr->id_ != curr_match_lm_in_cand->id_) {
                    replaced_lms[lm_in_curr] = curr_match_lm_in_cand;
                    lm_in_curr->replace(curr_match_lm_in_cand, map_db_);
                    if (!curr_match_lm_in_cand->has_representative_descriptor()) {
                        curr_match_lm_in_cand->compute_descriptor();
                    }
                    if (!curr_match_lm_in_cand->has_valid_prediction_parameters()) {
                        curr_match_lm_in_cand->update_mean_normal_and_obs_scale_variance();
                    }
                }
            }
            else {
                // if landmark corresponding `idx` does not exists,
                // add association between the current keyframe and `curr_match_lm_in_cand`
                curr_match_lm_in_cand->connect_to_keyframe(cur_keyfrm_, idx);
                curr_match_lm_in_cand->update_mean_normal_and_obs_scale_variance();
                curr_match_lm_in_cand->compute_descriptor();
            }
        }
    }

    // resolve duplications of landmarks between the current keyframe and the candidates of the loop candidate
    auto curr_match_lms_observed_in_cand_covis = loop_detector_->current_matched_landmarks_observed_in_candidate_covisibilities();
    match::fuse fuse_matcher(0.8);
    for (const auto& t : Sim3s_nw_after_correction) {
        auto neighbor = t.first;
        const Mat44_t Sim3_nw_after_correction = util::converter::to_eigen_mat(t.second);

        // reproject the landmarks observed in the current keyframe to the neighbor,
        // then search duplication of the landmarks
        std::unordered_map<std::shared_ptr<data::landmark>, std::shared_ptr<data::landmark>> duplicated_lms_in_keyfrm;
        std::unordered_map<unsigned int, std::shared_ptr<data::landmark>> new_connections;
        // Convert Sim3 into SE3
        const Mat33_t s_rot_cw = Sim3_nw_after_correction.block<3, 3>(0, 0);
        const auto s_cw = std::sqrt(s_rot_cw.block<1, 3>(0, 0).dot(s_rot_cw.block<1, 3>(0, 0)));
        const Mat33_t rot_cw = s_rot_cw / s_cw;
        const Vec3_t trans_cw = Sim3_nw_after_correction.block<3, 1>(0, 3) / s_cw;
        fuse_matcher.detect_duplication(neighbor, rot_cw, trans_cw, curr_match_lms_observed_in_cand_covis, 4.0, duplicated_lms_in_keyfrm, new_connections);

        std::lock_guard<std::mutex> lock(data::map_database::mtx_database_);

        for (const auto& best_idx_lm : new_connections) {
            const auto& best_idx = best_idx_lm.first;
            const auto& lm = best_idx_lm.second;
            lm->connect_to_keyframe(neighbor, best_idx);
            lm->update_mean_normal_and_obs_scale_variance();
            lm->compute_descriptor();
        }

        // if any landmark duplication is found, replace it
        for (const auto& lms_pair : duplicated_lms_in_keyfrm) {
            const auto& lm_to_replace = lms_pair.first;
            const auto& lm_in_neighbor = lms_pair.second;
            if (lm_to_replace->id_ != lm_in_neighbor->id_) {
                replaced_lms[lm_to_replace] = lm_in_neighbor;
                lm_to_replace->replace(lm_in_neighbor, map_db_);
                if (!lm_in_neighbor->has_representative_descriptor()) {
                    lm_in_neighbor->compute_descriptor();
                }
                if (!lm_in_neighbor->has_valid_prediction_parameters()) {
                    lm_in_neighbor->update_mean_normal_and_obs_scale_variance();
                }
            }
        }
    }
    tracker_->replace_landmarks_in_last_frm(replaced_lms);
}

auto global_optimization_module::extract_new_connections(const std::vector<std::shared_ptr<data::keyframe>>& covisibilities) const
    -> std::map<std::shared_ptr<data::keyframe>, std::set<std::shared_ptr<data::keyframe>>> {
    std::map<std::shared_ptr<data::keyframe>, std::set<std::shared_ptr<data::keyframe>>> new_connections;

    for (auto covisibility : covisibilities) {
        // acquire neighbors BEFORE loop fusion (because update_connections() is not called yet)
        const auto neighbors_before_update = covisibility->graph_node_->get_covisibilities();

        // call update_connections()
        covisibility->graph_node_->update_connections(map_db_->get_min_num_shared_lms());
        // acquire neighbors AFTER loop fusion
        new_connections[covisibility] = covisibility->graph_node_->get_connected_keyframes();

        // remove covisibilities
        for (const auto& keyfrm_to_erase : covisibilities) {
            new_connections.at(covisibility).erase(keyfrm_to_erase);
        }
        // remove nighbors before loop fusion
        for (const auto& keyfrm_to_erase : neighbors_before_update) {
            new_connections.at(covisibility).erase(keyfrm_to_erase);
        }
    }

    return new_connections;
}

std::shared_future<void> global_optimization_module::async_reset() {
    std::lock_guard<std::mutex> lock(mtx_reset_);
    reset_is_requested_ = true;
    if (!future_reset_.valid()) {
        future_reset_ = promise_reset_.get_future().share();
    }
    return future_reset_;
}

bool global_optimization_module::reset_is_requested() const {
    std::lock_guard<std::mutex> lock(mtx_reset_);
    return reset_is_requested_;
}

void global_optimization_module::reset() {
    std::lock_guard<std::mutex> lock(mtx_reset_);
    spdlog::info("reset global optimization module");
    keyfrms_queue_.clear();
    loop_detector_->set_loop_correct_keyframe_id(0);
    reset_is_requested_ = false;
    promise_reset_.set_value();
    promise_reset_ = std::promise<void>();
    future_reset_ = std::shared_future<void>();
}

std::shared_future<void> global_optimization_module::async_pause() {
    std::lock_guard<std::mutex> lock1(mtx_pause_);
    pause_is_requested_ = true;
    if (!future_pause_.valid()) {
        future_pause_ = promise_pause_.get_future().share();
    }
    return future_pause_;
}

bool global_optimization_module::pause_is_requested() const {
    std::lock_guard<std::mutex> lock(mtx_pause_);
    return pause_is_requested_;
}

bool global_optimization_module::is_paused() const {
    std::lock_guard<std::mutex> lock(mtx_pause_);
    return is_paused_;
}

void global_optimization_module::pause() {
    std::lock_guard<std::mutex> lock(mtx_pause_);
    spdlog::info("pause global optimization module");
    is_paused_ = true;
    promise_pause_.set_value();
    promise_pause_ = std::promise<void>();
    future_pause_ = std::shared_future<void>();
}

void global_optimization_module::resume() {
    std::lock_guard<std::mutex> lock1(mtx_pause_);
    std::lock_guard<std::mutex> lock2(mtx_terminate_);

    // if it has been already terminated, cannot resume
    if (is_terminated_) {
        return;
    }

    is_paused_ = false;
    pause_is_requested_ = false;

    spdlog::info("resume global optimization module");
}

std::shared_future<void> global_optimization_module::async_terminate() {
    std::lock_guard<std::mutex> lock(mtx_terminate_);
    terminate_is_requested_ = true;
    if (!future_terminate_.valid()) {
        future_terminate_ = promise_terminate_.get_future().share();
    }
    return future_terminate_;
}

bool global_optimization_module::is_terminated() const {
    std::lock_guard<std::mutex> lock(mtx_terminate_);
    return is_terminated_;
}

bool global_optimization_module::terminate_is_requested() const {
    std::lock_guard<std::mutex> lock(mtx_terminate_);
    return terminate_is_requested_;
}

void global_optimization_module::terminate() {
    std::lock_guard<std::mutex> lock(mtx_terminate_);
    is_terminated_ = true;
    promise_terminate_.set_value();
    promise_terminate_ = std::promise<void>();
    future_terminate_ = std::shared_future<void>();
}

bool global_optimization_module::loop_BA_is_running() const {
    return loop_bundle_adjuster_->is_running();
}

void global_optimization_module::abort_loop_BA() {
    loop_bundle_adjuster_->abort();
}

} // namespace stella_vslam
