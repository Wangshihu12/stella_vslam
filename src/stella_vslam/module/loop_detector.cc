#include "stella_vslam/data/bow_database.h"
#include "stella_vslam/data/bow_vocabulary.h"
#include "stella_vslam/data/keyframe.h"
#include "stella_vslam/data/landmark.h"
#include "stella_vslam/match/bow_tree.h"
#include "stella_vslam/match/projection.h"
#include "stella_vslam/match/robust.h"
#include "stella_vslam/module/loop_detector.h"
#include "stella_vslam/optimize/pose_optimizer_factory.h"
#include "stella_vslam/solve/pnp_solver.h"
#include "stella_vslam/util/converter.h"
#include "stella_vslam/util/fancy_index.h"

#include <spdlog/spdlog.h>

namespace stella_vslam {
namespace module {

loop_detector::loop_detector(data::bow_database* bow_db, data::bow_vocabulary* bow_vocab, const YAML::Node& yaml_node, const bool fix_scale_in_Sim3_estimation)
    : bow_db_(bow_db), bow_vocab_(bow_vocab), transform_optimizer_(fix_scale_in_Sim3_estimation), pose_optimizer_(optimize::pose_optimizer_factory::create(yaml_node)),
      loop_detector_is_enabled_(yaml_node["enabled"].as<bool>(true)),
      fix_scale_in_Sim3_estimation_(fix_scale_in_Sim3_estimation),
      num_final_matches_thr_(yaml_node["num_final_matches_threshold"].as<unsigned int>(40)),
      min_continuity_(yaml_node["min_continuity"].as<unsigned int>(3)),
      reject_by_graph_distance_(yaml_node["reject_by_graph_distance"].as<bool>(false)),
      min_distance_on_graph_(yaml_node["min_distance_on_graph"].as<unsigned int>(50)),
      num_matches_thr_(yaml_node["num_matches_thr"].as<unsigned int>(20)),
      num_matches_thr_brute_force_(yaml_node["num_matches_thr_robust_matcher"].as<unsigned int>(0)),
      num_optimized_inliers_thr_(yaml_node["num_optimized_inliers_thr"].as<unsigned int>(20)),
      top_n_covisibilities_to_search_(yaml_node["top_n_covisibilities_to_search"].as<unsigned int>(0)),
      use_fixed_seed_(yaml_node["use_fixed_seed"].as<bool>(false)),
      num_common_words_thr_ratio_(yaml_node["num_common_words_thr_ratio"].as<float>(0.8f)) {
    spdlog::debug("CONSTRUCT: loop_detector");
}

void loop_detector::enable_loop_detector() {
    loop_detector_is_enabled_ = true;
}

void loop_detector::disable_loop_detector() {
    loop_detector_is_enabled_ = false;
}

bool loop_detector::is_enabled() const {
    return loop_detector_is_enabled_;
}

void loop_detector::set_current_keyframe(const std::shared_ptr<data::keyframe>& keyfrm) {
    cur_keyfrm_ = keyfrm;
}

bool loop_detector::detect_loop_candidates() {
    auto succeeded = detect_loop_candidates_impl();
    // register to the BoW database
    bow_db_->add_keyframe(cur_keyfrm_);
    return succeeded;
}

void loop_detector::add_loop_candidate(const std::shared_ptr<data::keyframe>& keyfrm) {
    if (top_n_covisibilities_to_search_ > 0) {
        loop_candidates_to_validate_.insert(keyfrm);
        auto covisibilities = keyfrm->graph_node_->get_top_n_covisibilities(top_n_covisibilities_to_search_);
        for (const auto& covisibility : covisibilities) {
            loop_candidates_to_validate_.insert(covisibility);
        }
    }
    else {
        loop_candidates_to_validate_.insert(keyfrm);
    }
}

/**
 * [功能描述]：检测回环候选关键帧的内部实现
 *            通过词袋模型（BoW）查询相似关键帧，并利用连续检测机制提高检测的鲁棒性
 * @return bool：是否找到有效的回环候选帧，true表示找到，false表示未找到
 */
bool loop_detector::detect_loop_candidates_impl() {
    // 检查回环检测是否可用：
    // 1. 回环检测器是否启用
    // 2. 当前帧ID是否距离上次回环校正超过10帧（避免频繁检测）
    if (!loop_detector_is_enabled_ || cur_keyfrm_->id_ < prev_loop_correct_keyfrm_id_ + 10) {
        return false;
    }

    // ==================== 步骤1：通过BoW数据库搜索回环候选 ====================

    // ---------- 1-1. 计算当前帧与其共视帧之间的最小BoW相似度分数 ----------
    // 该分数作为查询阈值，确保候选帧的相似度至少与共视帧相当
    const float min_score = compute_min_score_in_covisibilities(cur_keyfrm_);

    // ---------- 1-2. 向BoW数据库查询相似关键帧 ----------

    // 构建需要排除的关键帧集合（避免搜索时间上相近的帧）
    std::set<std::shared_ptr<data::keyframe>> keyfrms_to_reject;
    if (!reject_by_graph_distance_) {
        // 简单模式：排除当前帧及其所有直接连接的关键帧
        keyfrms_to_reject = cur_keyfrm_->graph_node_->get_connected_keyframes();
        keyfrms_to_reject.insert(cur_keyfrm_);
    }
    else {
        // 基于图距离的排除模式：使用BFS遍历，排除图距离小于阈值的所有帧
        std::vector<std::pair<std::shared_ptr<data::keyframe>, int>> targets;  // 待处理的<关键帧, 距离>对
        targets.emplace_back(cur_keyfrm_, 0);
        keyfrms_to_reject.insert(cur_keyfrm_);

        // BFS遍历共视图
        while (!targets.empty()) {
            auto keyfrm_distance_pair = targets.back();
            targets.pop_back();
            auto& keyfrm = keyfrm_distance_pair.first;
            auto& distance = keyfrm_distance_pair.second;

            // 如果下一步距离仍在阈值范围内，继续扩展
            if (distance + 1 < min_distance_on_graph_) {
                // 搜索生成树父节点
                const auto parent = keyfrm->graph_node_->get_spanning_parent();
                if (parent && !static_cast<bool>(keyfrms_to_reject.count(parent))) {
                    keyfrms_to_reject.insert(parent);
                    targets.emplace_back(parent, distance + 1);
                }
                // 搜索回环边连接的节点
                for (const auto& node : keyfrm->graph_node_->get_loop_edges()) {
                    if (static_cast<bool>(keyfrms_to_reject.count(node))) {
                        continue;
                    }
                    keyfrms_to_reject.insert(node);
                    targets.emplace_back(node, distance + 1);
                }
                // 搜索生成树子节点
                for (const auto& child : keyfrm->graph_node_->get_spanning_children()) {
                    if (static_cast<bool>(keyfrms_to_reject.count(child))) {
                        continue;
                    }
                    keyfrms_to_reject.insert(child);
                    targets.emplace_back(child, distance + 1);
                }
            }
        }
    }

    // 向BoW数据库查询：返回相似度高于min_score且不在排除集合中的关键帧
    const auto init_loop_candidates = bow_db_->acquire_keyframes(cur_keyfrm_->bow_vec_, min_score, num_common_words_thr_ratio_, keyfrms_to_reject);

    // ---------- 1-3. 如果没有找到候选帧，清空缓存并返回 ----------

    if (init_loop_candidates.empty()) {
        // 清空连续检测缓存，因为没有找到任何候选
        cont_detected_keyfrm_sets_.clear();
        return false;
    }

    // ==================== 步骤2：利用连续检测机制提高鲁棒性 ====================
    // 将每个候选帧视为"关键帧集合"，每次调用时统计每个候选集合被检测到的次数
    // 如果某个集合在上一次调用时也被检测到，它会保存在 cont_detected_keyfrm_sets_ 中
    // 注意："两个关键帧集合匹配"意味着它们的交集非空

    // 查找连续被检测到的关键帧集合，并更新它们的连续检测计数
    const auto curr_cont_detected_keyfrm_sets = find_continuously_detected_keyframe_sets(cont_detected_keyfrm_sets_, init_loop_candidates);

    // ==================== 步骤3：筛选满足连续性阈值的候选 ====================
    // 如果某个候选集合的连续检测次数达到或超过阈值 min_continuity_，则采纳为回环候选

    loop_candidates_to_validate_.clear();
    for (auto& curr : curr_cont_detected_keyfrm_sets) {
        const auto candidate_keyfrm = curr.lead_keyfrm_;   // 候选集合的代表关键帧
        const auto continuity = curr.continuity_;          // 连续检测次数
        // 检查连续检测次数是否达到阈值
        if (min_continuity_ <= continuity) {
            // 采纳为回环候选
            loop_candidates_to_validate_.insert(candidate_keyfrm);
        }
    }

    // ==================== 步骤4：更新成员变量，供下次调用使用 ====================

    // 保存当前的连续检测关键帧集合，供下次检测时比对
    cont_detected_keyfrm_sets_ = curr_cont_detected_keyfrm_sets;

    // ==================== 步骤5：扩展候选集合（添加候选帧的共视帧） ====================

    if (top_n_covisibilities_to_search_ > 0) {
        // 复制当前候选集合（避免在遍历时修改）
        auto candidates = loop_candidates_to_validate_;
        for (auto& keyfrm : candidates) {
            // 获取每个候选帧的前N个共视帧
            auto covisibilities = keyfrm->graph_node_->get_top_n_covisibilities(top_n_covisibilities_to_search_);
            for (const auto& covisibility : covisibilities) {
                // 如果共视帧不在排除集合中，则添加到候选集合
                if (!static_cast<bool>(keyfrms_to_reject.count(covisibility))) {
                    loop_candidates_to_validate_.insert(covisibility);
                }
            }
        }
    }

    // 返回是否找到有效的回环候选
    return !loop_candidates_to_validate_.empty();
}

/**
 * [功能描述]：验证回环候选关键帧，从候选集中选择一个有效的回环帧
 *            在验证前保护候选帧不被删除，验证后根据结果恢复删除权限
 * @return bool：验证是否成功，true表示找到有效回环帧，false表示验证失败
 */
bool loop_detector::validate_candidates() {
    // 在验证开始前，设置所有候选帧为不可删除状态
    // 防止在验证过程中被其他线程（如建图模块）意外删除
    for (const auto& candidate : loop_candidates_to_validate_) {
        candidate->set_not_to_be_erased();
    }

    // 调用验证的具体实现（计算Sim3变换、验证几何一致性等）
    auto succeeded = validate_candidates_impl();

    if (succeeded) {
        // 验证成功：恢复除被选中帧外的所有候选帧的可删除状态
        for (const auto& loop_candidate : loop_candidates_to_validate_) {
            // 跳过被选中的回环帧，保持其不可删除状态
            if (*loop_candidate == *selected_candidate_) {
                continue;
            }
            // 其他未被选中的候选帧恢复为可删除状态
            loop_candidate->set_to_be_erased();
        }
    }
    else {
        // 验证失败：恢复所有候选帧的可删除状态
        for (const auto& loop_candidate : loop_candidates_to_validate_) {
            loop_candidate->set_to_be_erased();
        }
    }
    return succeeded;
}

/**
 * [功能描述]：回环候选帧验证的具体实现
 *            通过Sim3估计验证候选帧的几何一致性，并通过重投影匹配获取更多2D-3D对应
 * @return bool：验证是否成功，true表示找到足够匹配的有效回环帧
 */
bool loop_detector::validate_candidates_impl() {
    // ==================== 步骤1：估计并验证Sim3变换，选择一个候选帧 ====================
    // 对每个候选帧，利用观测到的路标点估计其与当前帧之间的Sim3变换
    // 验证变换的有效性后，从中选择一个最佳候选

    // 通过Sim3估计选择回环候选帧
    // 输出：selected_candidate_（被选中的候选帧）、g2o_Sim3_world_to_curr_（Sim3变换）、
    //       curr_match_lms_observed_in_cand_（当前帧特征点与候选帧路标点的匹配）
    const bool candidate_is_found = select_loop_candidate_via_Sim3(loop_candidates_to_validate_, selected_candidate_,
                                                                   g2o_Sim3_world_to_curr_, curr_match_lms_observed_in_cand_);
    // 将g2o格式的Sim3转换为Eigen矩阵格式
    Sim3_world_to_curr_ = util::converter::to_eigen_mat(g2o_Sim3_world_to_curr_);

    // 如果没有找到有效候选，返回失败
    if (!candidate_is_found) {
        return false;
    }

    spdlog::debug("detect loop candidate via Sim3 estimation: keyframe {} - keyframe {}", selected_candidate_->id_, cur_keyfrm_->id_);

    // ==================== 步骤2：重投影匹配，获取更多2D-3D对应 ====================
    // 将候选帧共视帧中观测到的路标点重投影到当前帧，获取额外的2D-3D匹配

    // 清空存储候选帧共视区域路标点的容器
    curr_match_lms_observed_in_cand_covis_.clear();

    // 获取候选帧的所有共视帧，并将候选帧本身也加入列表
    auto cand_covisibilities = selected_candidate_->graph_node_->get_covisibilities();
    cand_covisibilities.push_back(selected_candidate_);

    // 收集候选帧及其共视帧观测到的所有路标点（去重）
    std::unordered_set<std::shared_ptr<data::landmark>> already_inserted;  // 用于去重的集合
    for (const auto& covisibility : cand_covisibilities) {
        // 获取该共视帧观测到的所有路标点
        const auto lms_in_covisibility = covisibility->get_landmarks();
        for (const auto& lm : lms_in_covisibility) {
            // 跳过空指针
            if (!lm) {
                continue;
            }
            // 跳过即将被删除的路标点
            if (lm->will_be_erased()) {
                continue;
            }

            // 跳过已添加的路标点（去重）
            if (already_inserted.count(lm)) {
                continue;
            }
            // 添加到候选共视区域路标点列表
            curr_match_lms_observed_in_cand_covis_.push_back(lm);
            already_inserted.insert(lm);
        }
    }

    // 使用Sim3变换将候选帧共视区域的路标点重投影到当前帧，获取额外的2D-3D匹配
    // 注意：已在 curr_match_lms_observed_in_cand_ 中匹配的路标点将被排除，避免重复匹配
    match::projection projection_matcher(0.75);  // 创建投影匹配器，描述子距离比值阈值为0.75
    projection_matcher.match_by_Sim3_transform(cur_keyfrm_, Sim3_world_to_curr_, curr_match_lms_observed_in_cand_covis_,
                                               curr_match_lms_observed_in_cand_, 10);  // 搜索半径为10像素

    // 统计最终匹配数量
    unsigned int num_final_matches = 0;
    for (const auto& curr_assoc_lm_in_cand : curr_match_lms_observed_in_cand_) {
        if (curr_assoc_lm_in_cand) {
            ++num_final_matches;
        }
    }

    spdlog::debug("acquired {} matches after projection-match", num_final_matches);

    // 判断匹配数量是否达到阈值
    if (num_final_matches_thr_ <= num_final_matches) {
        // 匹配数量足够，验证成功
        return true;
    }
    else {
        // 匹配数量不足，验证失败
        spdlog::debug("destruct loop candidate because enough matches not acquired (< {})", num_final_matches_thr_);
        return false;
    }
}

float loop_detector::compute_min_score_in_covisibilities(const std::shared_ptr<data::keyframe>& keyfrm) const {
    // the maximum of score is 1.0
    float min_score = 1.0;

    // search the mininum score among covisibilities
    const auto covisibilities = keyfrm->graph_node_->get_covisibilities();
    const auto& bow_vec_1 = keyfrm->bow_vec_;
    for (const auto& covisibility : covisibilities) {
        if (covisibility->will_be_erased()) {
            continue;
        }
        const auto& bow_vec_2 = covisibility->bow_vec_;

        const auto score = data::bow_vocabulary_util::score(bow_vocab_, bow_vec_1, bow_vec_2);
        if (score < min_score) {
            min_score = score;
        }
    }

    return min_score;
}

/**
 * [功能描述]：查找连续被检测到的关键帧集合，用于提高回环检测的鲁棒性
 *            通过统计每个候选关键帧集合被连续检测到的次数，过滤掉偶然匹配的候选
 * @param prev_cont_detected_keyfrm_sets：上一次检测时记录的连续检测关键帧集合
 * @param keyfrms_to_search：本次BoW查询返回的候选关键帧列表
 * @return keyframe_sets：更新后的连续检测关键帧集合，包含新的连续计数
 */
keyframe_sets loop_detector::find_continuously_detected_keyframe_sets(const keyframe_sets& prev_cont_detected_keyfrm_sets,
                                                                      const std::vector<std::shared_ptr<data::keyframe>>& keyfrms_to_search) const {
    // 统计每个关键帧集合被检测到的次数

    // 存储当前轮次的连续检测结果（包含关键帧集合、代表帧和连续计数）
    keyframe_sets curr_cont_detected_keyfrm_sets;

    // 标记每个历史集合是否已被匹配，防止同一集合被重复计数
    std::map<std::set<std::shared_ptr<data::keyframe>>, bool> already_checked;
    for (const auto& prev : prev_cont_detected_keyfrm_sets) {
        already_checked[prev.keyfrm_set_] = false;
    }

    // 遍历本次查询到的每个候选关键帧
    for (const auto& keyfrm_to_search : keyfrms_to_search) {
        // 将单个候选帧扩展为"关键帧集合"（包含该帧及其所有共视帧）
        const auto keyfrm_set = keyfrm_to_search->graph_node_->get_connected_keyframes();

        // 标记是否需要初始化（即该集合是首次被检测到）
        bool initialization_is_needed = true;

        // 遍历上一轮检测到的所有关键帧集合，检查连续性
        for (const auto& prev : prev_cont_detected_keyfrm_sets) {
            // prev.keyfrm_set_：历史关键帧集合
            // prev.lead_keyfrm_：该集合的代表关键帧
            // prev.continuity_：该集合的连续检测次数

            // 检查该历史集合是否已被匹配过（避免重复计数）
            if (already_checked.at(prev.keyfrm_set_)) {
                continue;
            }

            // 计算历史集合与当前集合的交集，检查是否为空
            // 如果交集为空，说明不是同一个回环候选区域
            if (prev.intersection_is_empty(keyfrm_set)) {
                continue;
            }

            // 交集非空，说明找到了匹配的历史集合，无需初始化
            initialization_is_needed = false;

            // 创建新的统计记录：连续计数加1
            const auto curr_continuity = prev.continuity_ + 1;
            curr_cont_detected_keyfrm_sets.emplace_back(
                keyframe_set{keyfrm_set, keyfrm_to_search, curr_continuity});

            // 标记该历史集合已被匹配，防止重复计数
            already_checked.at(prev.keyfrm_set_) = true;
        }

        // 如果没有找到匹配的历史集合，说明是首次检测到该区域，初始化连续计数为0
        if (initialization_is_needed) {
            curr_cont_detected_keyfrm_sets.emplace_back(
                keyframe_set{keyfrm_set, keyfrm_to_search, 0});
        }
    }

    return curr_cont_detected_keyfrm_sets;
}

/**
 * [功能描述]：通过Sim3估计从候选帧集合中选择一个有效的回环候选帧
 *            采用多轮匹配和位姿优化来验证候选帧的几何一致性，最终估计Sim3变换
 * @param loop_candidates：回环候选关键帧集合
 * @param selected_candidate：[输出] 被选中的回环候选帧
 * @param g2o_Sim3_world_to_curr：[输出] 从世界坐标系到当前帧的Sim3变换
 * @param curr_match_lms_observed_in_cand：[输出] 当前帧特征点与候选帧路标点的匹配关系
 * @return bool：是否成功找到有效的回环候选帧
 */
bool loop_detector::select_loop_candidate_via_Sim3(const std::unordered_set<std::shared_ptr<data::keyframe>>& loop_candidates,
                                                   std::shared_ptr<data::keyframe>& selected_candidate,
                                                   g2o::Sim3& g2o_Sim3_world_to_curr,
                                                   std::vector<std::shared_ptr<data::landmark>>& curr_match_lms_observed_in_cand) const {
    // 使用观测到的路标点估计当前帧与每个候选帧之间的Sim3变换
    // Sim3通过线性和非线性两种方式估计
    // 如果估计后的内点数低于阈值，则丢弃该候选

    // 创建三种匹配器：鲁棒匹配器、词袋树匹配器、投影匹配器
    match::robust robust_matcher(0.75, false);      // 描述子距离比值阈值0.75，不检查方向
    match::bow_tree bow_matcher(0.75, false);       // 基于词袋树的匹配
    match::projection projection_matcher(0.75, false);  // 基于投影的匹配

    // 遍历所有候选帧
    for (const auto& candidate : loop_candidates) {
        // 跳过即将被删除的候选帧
        if (candidate->will_be_erased()) {
            continue;
        }

        // ==================== 阶段1：初始特征匹配 ====================
        // 使用词袋树匹配当前帧特征点与候选帧观测到的路标点
        curr_match_lms_observed_in_cand.clear();
        const auto num_matches = bow_matcher.match_keyframes(cur_keyfrm_, candidate, curr_match_lms_observed_in_cand);

        // 检查匹配数量是否达到阈值
        if (num_matches < num_matches_thr_) {
            continue;
        }

        spdlog::debug("Checking if the loop candidate is appropriate: keyframe {} - keyframe {} (num_matches: {})", candidate->id_, cur_keyfrm_->id_, num_matches);

        // 可选：使用暴力匹配获取更多对应点
        if (num_matches_thr_brute_force_ > 0) {
            const auto num_matches_brute_force = robust_matcher.match_keyframes(cur_keyfrm_, candidate, curr_match_lms_observed_in_cand, false);

            spdlog::debug("num_matches_brute_force: {}", num_matches_brute_force);

            if (num_matches_brute_force < num_matches_thr_brute_force_) {
                continue;
            }
        }

        // ==================== 阶段2：PnP求解初始位姿 ====================
        // 收集有效匹配的索引（排除空指针和即将删除的路标点）
        std::vector<unsigned int> valid_indices;
        valid_indices.reserve(curr_match_lms_observed_in_cand.size());
        for (unsigned int idx = 0; idx < curr_match_lms_observed_in_cand.size(); ++idx) {
            auto lm = curr_match_lms_observed_in_cand.at(idx);
            if (!lm) {
                continue;
            }
            if (lm->will_be_erased()) {
                continue;
            }
            valid_indices.push_back(idx);
        }

        // 根据有效索引重采样相关数据
        const auto valid_bearings = util::resample_by_indices(cur_keyfrm_->frm_obs_.bearings_, valid_indices);  // 方位向量
        const auto valid_keypts = util::resample_by_indices(cur_keyfrm_->frm_obs_.undist_keypts_, valid_indices);  // 去畸变特征点
        // 提取金字塔层级（用于尺度信息）
        std::vector<int> octaves(valid_indices.size());
        for (unsigned int i = 0; i < valid_indices.size(); ++i) {
            octaves.at(i) = valid_keypts.at(i).octave;
        }
        const auto valid_assoc_lms = util::resample_by_indices(curr_match_lms_observed_in_cand, valid_indices);  // 关联的路标点
        // 提取路标点的3D世界坐标
        eigen_alloc_vector<Vec3_t> valid_points(valid_indices.size());
        for (unsigned int i = 0; i < valid_indices.size(); ++i) {
            valid_points.at(i) = valid_assoc_lms.at(i)->get_pos_in_world();
        }

        // 创建PnP求解器，使用RANSAC进行鲁棒求解
        auto pnp_solver = std::unique_ptr<solve::pnp_solver>(new solve::pnp_solver(valid_bearings, octaves, valid_points,
                                                                                   cur_keyfrm_->orb_params_->scale_factors_,
                                                                                   10, use_fixed_seed_));

        // 执行RANSAC PnP求解，最多30次迭代
        pnp_solver->find_via_ransac(30, false);
        if (!pnp_solver->solution_is_valid()) {
            spdlog::debug("solution is not valid.");
            continue;
        }

        // 获取RANSAC内点的索引
        const auto inlier_indices = util::resample_by_indices(valid_indices, pnp_solver->get_inlier_flags());

        // 为位姿优化设置2D-3D匹配（只保留内点）
        auto lms_in_cand = std::vector<std::shared_ptr<data::landmark>>(cur_keyfrm_->frm_obs_.undist_keypts_.size(), nullptr);
        for (const auto idx : inlier_indices) {
            lms_in_cand.at(idx) = curr_match_lms_observed_in_cand.at(idx);
        }
        curr_match_lms_observed_in_cand = lms_in_cand;

        // ==================== 阶段3：第一轮位姿优化 ====================
        std::vector<bool> outlier_flags;
        Mat44_t optimized_pose;
        auto num_valid_obs = pose_optimizer_->optimize(pnp_solver->get_best_cam_pose(), cur_keyfrm_->frm_obs_, cur_keyfrm_->orb_params_, cur_keyfrm_->camera_,
                                                       curr_match_lms_observed_in_cand, optimized_pose, outlier_flags);

        // 检查优化后的内点数量
        const int min_num_matches_after_pose_optimize = 10;
        if (num_valid_obs < min_num_matches_after_pose_optimize) {
            spdlog::debug("1. Number of inliers ({}) < threshold ({})", num_valid_obs, min_num_matches_after_pose_optimize);
            continue;
        }

        // 剔除外点
        for (unsigned int idx = 0; idx < cur_keyfrm_->frm_obs_.undist_keypts_.size(); idx++) {
            if (!outlier_flags.at(idx)) {
                continue;
            }
            lms_in_cand.at(idx) = nullptr;
        }

        // 记录已关联的路标点（用于后续投影匹配时排除）
        std::set<std::shared_ptr<data::landmark>> already_found_landmarks;
        for (const auto idx : inlier_indices) {
            if (outlier_flags.at(idx)) {
                continue;
            }
            already_found_landmarks.insert(curr_match_lms_observed_in_cand.at(idx));
        }

        // ==================== 阶段4：第一轮投影匹配扩展 ====================
        // 基于优化后的位姿进行投影匹配，获取更多2D-3D对应
        auto num_found = projection_matcher.match_frame_and_keyframe(optimized_pose, cur_keyfrm_->camera_, cur_keyfrm_->frm_obs_,
                                                                     cur_keyfrm_->orb_params_, curr_match_lms_observed_in_cand,
                                                                     candidate, already_found_landmarks, 10, 100);
        // 检查匹配数量阈值
        const unsigned int min_num_valid_obs1 = 25;
        if (already_found_landmarks.size() + num_found < min_num_valid_obs1) {
            spdlog::debug("2. Number of matches ({}) < threshold ({})",
                          already_found_landmarks.size() + num_found, min_num_valid_obs1);
            continue;
        }

        // ==================== 阶段5：第二轮位姿优化 ====================
        Mat44_t optimized_pose1;
        std::vector<bool> outlier_flags1;
        auto num_valid_obs1 = pose_optimizer_->optimize(optimized_pose,
                                                        cur_keyfrm_->frm_obs_, cur_keyfrm_->orb_params_, cur_keyfrm_->camera_,
                                                        curr_match_lms_observed_in_cand, optimized_pose1, outlier_flags1);

        if (num_valid_obs1 < min_num_valid_obs1) {
            spdlog::debug("2. Number of inliers ({}) < threshold ({})", num_valid_obs1, min_num_valid_obs1);
            continue;
        }

        // ==================== 阶段6：第二轮投影匹配扩展 ====================
        // 更新已关联的路标点集合
        std::set<std::shared_ptr<data::landmark>> already_found_landmarks1;
        for (unsigned int idx = 0; idx < cur_keyfrm_->frm_obs_.undist_keypts_.size(); ++idx) {
            if (!curr_match_lms_observed_in_cand.at(idx)) {
                continue;
            }
            already_found_landmarks1.insert(curr_match_lms_observed_in_cand.at(idx));
        }
        // 使用更严格的参数再次进行投影匹配
        auto num_additional = projection_matcher.match_frame_and_keyframe(optimized_pose1, cur_keyfrm_->camera_, cur_keyfrm_->frm_obs_,
                                                                          cur_keyfrm_->orb_params_, curr_match_lms_observed_in_cand,
                                                                          candidate, already_found_landmarks, 3, 64);

        const unsigned int min_num_valid_obs2 = 40;
        // 检查总匹配数量是否达到更高阈值
        if (num_valid_obs1 + num_additional < min_num_valid_obs2) {
            spdlog::debug("3. Number of matches ({}) < threshold ({})", num_valid_obs1 + num_additional, min_num_valid_obs2);
            return false;
        }

        // ==================== 阶段7：第三轮位姿优化 ====================
        Mat44_t optimized_pose2;
        std::vector<bool> outlier_flags2;
        auto num_valid_obs2 = pose_optimizer_->optimize(optimized_pose1,
                                                        cur_keyfrm_->frm_obs_, cur_keyfrm_->orb_params_, cur_keyfrm_->camera_,
                                                        curr_match_lms_observed_in_cand, optimized_pose2, outlier_flags2);

        // 检查最终内点数量
        if (num_valid_obs2 < min_num_valid_obs2) {
            spdlog::debug("3. Number of inliers ({}) < threshold ({})", num_valid_obs2, min_num_valid_obs2);
            return false;
        }

        // 剔除最终的外点
        for (unsigned int idx = 0; idx < cur_keyfrm_->frm_obs_.undist_keypts_.size(); ++idx) {
            if (!outlier_flags2.at(idx)) {
                continue;
            }
            curr_match_lms_observed_in_cand.at(idx) = nullptr;
        }

        // ==================== 阶段8：计算尺度因子 ====================
        // 从优化后的位姿中提取旋转和平移
        const Mat44_t pose_1w_in_cand = optimized_pose2;
        const Mat33_t rot_1w_in_cand = pose_1w_in_cand.block<3, 3>(0, 0);
        const Vec3_t trans_1w_in_cand = pose_1w_in_cand.block<3, 1>(0, 3);

        // 获取当前帧观测到的路标点
        auto lms_curr = cur_keyfrm_->get_landmarks();
        std::vector<float> scales;  // 存储计算得到的尺度因子

        // 通过匹配的路标点对计算尺度因子
        for (unsigned int idx = 0; idx < lms_curr.size(); ++idx) {
            auto& lm_curr = lms_curr.at(idx);      // 当前帧观测到的路标点
            auto& lm_cand = curr_match_lms_observed_in_cand.at(idx);  // 候选帧对应的路标点
            if (!lm_cand || !lm_curr) {
                continue;
            }
            if (lm_cand->will_be_erased() || lm_curr->will_be_erased()) {
                continue;
            }

            // 获取两个路标点的世界坐标
            const Vec3_t pos_w_lm_cand = lm_cand->get_pos_in_world();
            const Vec3_t pos_w_lm_curr = lm_curr->get_pos_in_world();

            // 将路标点变换到相机坐标系
            const Vec3_t pos_1_in_cand = rot_1w_in_cand * pos_w_lm_cand + trans_1w_in_cand;  // 使用候选帧位姿变换
            const Vec3_t pos_1_in_curr = cur_keyfrm_->get_rot_cw() * pos_w_lm_curr + cur_keyfrm_->get_trans_cw();  // 使用当前帧位姿变换

            // 计算视差角的余弦值
            const float norm_pos_1_in_cand = pos_1_in_cand.norm();
            const float norm_pos_1_in_curr = pos_1_in_curr.norm();
            const float cos_parallax = pos_1_in_cand.dot(pos_1_in_curr) / (norm_pos_1_in_cand * norm_pos_1_in_curr);

            // 视差角阈值：cos(0.5度) ≈ 0.99996
            // 只使用视差角足够小的点对来计算尺度（视差过大会导致尺度估计不准确）
            constexpr float cos_parallax_thr = 0.99996192306;
            const bool parallax_is_small = cos_parallax_thr < cos_parallax;
            if (!parallax_is_small) {
                continue;
            }
            // 计算尺度因子：当前帧坐标系下的深度 / 候选帧位姿下的深度
            scales.push_back(norm_pos_1_in_curr / norm_pos_1_in_cand);
        }

        // 如果没有足够的尺度参考点，跳过该候选
        if (scales.size() < 1) {
            spdlog::debug("not enough scale references {}", scales.size());
            continue;
        }

        // ==================== 阶段9：计算Sim3相对变换 ====================
        // 计算从候选帧到当前帧的相对旋转和平移
        const Mat33_t rot_12 = rot_1w_in_cand * candidate->get_rot_cw().transpose();
        const Vec3_t trans_12 = -rot_12 * candidate->get_trans_cw() + trans_1w_in_cand;

        // 使用中值作为最终尺度因子（鲁棒估计）
        std::sort(scales.begin(), scales.end());
        const float scale_12 = scales[(scales.size() - 1) / 2];

        // ==================== 阶段10：Sim3非线性优化 ====================

        // 使用Sim3变换进行双向投影匹配
        projection_matcher.match_keyframes_mutually(cur_keyfrm_, candidate, curr_match_lms_observed_in_cand,
                                                    scale_12, rot_12, trans_12, 7.5);

        // 构建初始Sim3并进行非线性优化
        g2o::Sim3 g2o_sim3_12(rot_12, trans_12, scale_12);
        const auto num_optimized_inliers = transform_optimizer_.optimize(cur_keyfrm_, candidate, curr_match_lms_observed_in_cand,
                                                                         g2o_sim3_12, 10);

        // 检查优化后的内点数量是否达到阈值
        if (num_optimized_inliers < num_optimized_inliers_thr_) {
            continue;
        }

        spdlog::debug("found loop candidate via nonlinear Sim3 optimization: keyframe {} - keyframe {} (num_optimized_inliers: {})", candidate->id_, cur_keyfrm_->id_, num_optimized_inliers);

        // ==================== 阶段11：输出结果 ====================
        selected_candidate = candidate;

        // 将Sim3从"候选帧→当前帧"转换为"世界坐标系→当前帧"
        // 该Sim3表示回环校正后当前帧的正确相机位姿
        g2o_Sim3_world_to_curr = g2o_sim3_12 * g2o::Sim3(candidate->get_rot_cw(), candidate->get_trans_cw(), 1.0);

        return true;
    }

    // 遍历所有候选后仍未找到有效回环
    return false;
}

std::shared_ptr<data::keyframe> loop_detector::get_selected_candidate_keyframe() const {
    return selected_candidate_;
}

g2o::Sim3 loop_detector::get_Sim3_world_to_current() const {
    return g2o_Sim3_world_to_curr_;
}

std::vector<std::shared_ptr<data::landmark>> loop_detector::current_matched_landmarks_observed_in_candidate() const {
    return curr_match_lms_observed_in_cand_;
}

std::vector<std::shared_ptr<data::landmark>> loop_detector::current_matched_landmarks_observed_in_candidate_covisibilities() const {
    return curr_match_lms_observed_in_cand_covis_;
}

void loop_detector::set_loop_correct_keyframe_id(const unsigned int loop_correct_keyfrm_id) {
    prev_loop_correct_keyfrm_id_ = loop_correct_keyfrm_id;
}

} // namespace module
} // namespace stella_vslam
