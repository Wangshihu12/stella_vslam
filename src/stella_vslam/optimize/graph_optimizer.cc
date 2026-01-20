#include "stella_vslam/data/keyframe.h"
#include "stella_vslam/data/landmark.h"
#include "stella_vslam/data/map_database.h"
#include "stella_vslam/optimize/graph_optimizer.h"
#include "stella_vslam/optimize/terminate_action.h"
#include "stella_vslam/optimize/internal/sim3/shot_vertex.h"
#include "stella_vslam/optimize/internal/sim3/graph_opt_edge.h"
#include "stella_vslam/util/converter.h"

#include <Eigen/StdVector>
#include <g2o/core/solver.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/sparse_optimizer_terminate_action.h>

namespace stella_vslam {
namespace optimize {

graph_optimizer::graph_optimizer(const YAML::Node& yaml_node, const bool fix_scale)
    : fix_scale_(fix_scale),
      min_num_shared_lms_(yaml_node["min_num_shared_lms"].as<unsigned int>(100)) {}

/**
 * [功能描述]：执行位姿图优化，通过优化所有关键帧的Sim3位姿来消除回环闭合引入的误差
 *            使用g2o图优化框架，构建位姿图并进行非线性最小二乘优化
 * @param loop_keyfrm：回环关键帧（较旧的帧），作为固定顶点
 * @param curr_keyfrm：当前关键帧（较新的帧），作为固定顶点
 * @param non_corrected_Sim3s：回环校正前的Sim3位姿，用于计算相对位姿约束
 * @param pre_corrected_Sim3s：回环校正后的Sim3位姿（已应用回环约束的帧）
 * @param loop_connections：回环融合后新建立的共视关系映射
 * @param found_lm_to_ref_keyfrm_id：路标点ID到参考关键帧ID的映射，用于更新点云位置
 */
void graph_optimizer::optimize(const std::shared_ptr<data::keyframe>& loop_keyfrm, const std::shared_ptr<data::keyframe>& curr_keyfrm,
                               const module::keyframe_Sim3_pairs_t& non_corrected_Sim3s,
                               const module::keyframe_Sim3_pairs_t& pre_corrected_Sim3s,
                               const std::map<std::shared_ptr<data::keyframe>, std::set<std::shared_ptr<data::keyframe>>>& loop_connections,
                               std::unordered_map<unsigned int, unsigned int>& found_lm_to_ref_keyfrm_id) const {
    // ==================== 步骤1：构建优化器 ====================

    // 创建线性求解器，使用CSparse稀疏矩阵求解器
    // BlockSolver_7_3：顶点维度为7（Sim3），路标点维度为3
    auto linear_solver = stella_vslam::make_unique<g2o::LinearSolverCSparse<g2o::BlockSolver_7_3::PoseMatrixType>>();
    // 创建块求解器
    auto block_solver = stella_vslam::make_unique<g2o::BlockSolver_7_3>(std::move(linear_solver));
    // 创建Levenberg-Marquardt优化算法
    auto algorithm = new g2o::OptimizationAlgorithmLevenberg(std::move(block_solver));

    // 创建稀疏优化器
    g2o::SparseOptimizer optimizer;
    // 创建终止动作，设置增益阈值为1e-3（当优化增益小于此值时提前终止）
    auto terminateAction = new terminate_action;
    terminateAction->setGainThreshold(1e-3);
    // 添加迭代后动作和设置优化算法
    optimizer.addPostIterationAction(terminateAction);
    optimizer.setAlgorithm(algorithm);

    // ==================== 步骤2：添加顶点 ====================

    // 从当前关键帧的生成树根节点获取所有关键帧
    const auto all_keyfrms = curr_keyfrm->graph_node_->get_keyframes_from_root();

    // 收集所有关键帧观测到的路标点（用于后续更新点云位置）
    std::unordered_set<unsigned int> already_found_landmark_ids;  // 已找到的路标点ID集合，用于去重
    std::vector<std::shared_ptr<data::landmark>> all_lms;         // 存储所有有效路标点
    for (const auto& keyfrm : all_keyfrms) {
        for (const auto& lm : keyfrm->get_landmarks()) {
            // 跳过空指针
            if (!lm) {
                continue;
            }
            // 跳过即将被删除的路标点
            if (lm->will_be_erased()) {
                continue;
            }
            // 跳过已经添加过的路标点（去重）
            if (already_found_landmark_ids.count(lm->id_)) {
                continue;
            }

            already_found_landmark_ids.insert(lm->id_);
            all_lms.push_back(lm);
        }
    }

    // 存储所有关键帧优化前的Sim3位姿（从世界坐标系到相机坐标系）
    eigen_alloc_unord_map<unsigned int, g2o::Sim3> Sim3s_cw;
    // 存储已添加的顶点指针，用于后续添加边时查找
    std::unordered_map<unsigned int, internal::sim3::shot_vertex*> vertices;

    // 遍历所有关键帧，为每个关键帧创建顶点
    for (auto keyfrm : all_keyfrms) {
        // 跳过即将被删除的关键帧
        if (keyfrm->will_be_erased()) {
            continue;
        }
        // 创建Sim3位姿顶点
        auto keyfrm_vtx = new internal::sim3::shot_vertex();

        const auto id = keyfrm->id_;

        // 检查该关键帧的位姿是否已经被回环校正修改过
        const auto iter = pre_corrected_Sim3s.find(keyfrm);
        if (iter != pre_corrected_Sim3s.end()) {
            // 如果已修改，使用修改后的Sim3位姿作为顶点初始估计
            Sim3s_cw[id] = iter->second;
            keyfrm_vtx->setEstimate(iter->second);
        }
        else {
            // 如果未修改，将SE3位姿转换为Sim3（尺度因子为1.0）
            const Mat33_t rot_cw = keyfrm->get_rot_cw();
            const Vec3_t trans_cw = keyfrm->get_trans_cw();
            const g2o::Sim3 Sim3_cw(rot_cw, trans_cw, 1.0);

            Sim3s_cw[id] = Sim3_cw;
            keyfrm_vtx->setEstimate(Sim3_cw);
        }

        // 固定回环关键帧、当前关键帧或生成树根节点（作为参考基准，不参与优化）
        if (*keyfrm == *loop_keyfrm || *keyfrm == *curr_keyfrm || keyfrm->graph_node_->is_spanning_root()) {
            keyfrm_vtx->setFixed(true);
        }

        // 设置顶点ID和尺度固定标志，并添加到优化器
        keyfrm_vtx->setId(id);
        keyfrm_vtx->fix_scale_ = fix_scale_;

        optimizer.addVertex(keyfrm_vtx);
        vertices[id] = keyfrm_vtx;
    }

    // ==================== 步骤3：添加边（约束） ====================

    // 存储已插入的边对，用于避免重复添加边
    std::set<std::pair<unsigned int, unsigned int>> inserted_edge_pairs;

    // Lambda函数：添加Sim3约束边
    // 参数：id1, id2 - 两个关键帧的ID；Sim3_21 - 从帧1到帧2的相对Sim3变换
    const auto insert_edge =
        [&optimizer, &vertices, &inserted_edge_pairs](unsigned int id1, unsigned int id2, const g2o::Sim3& Sim3_21) {
            // 创建位姿图优化边
            auto edge = new internal::sim3::graph_opt_edge();
            // 设置边连接的两个顶点
            edge->setVertex(0, vertices.at(id1));
            edge->setVertex(1, vertices.at(id2));
            // 设置测量值（相对位姿）
            edge->setMeasurement(Sim3_21);

            // 设置信息矩阵为单位矩阵（7x7，对应Sim3的7个自由度）
            edge->information() = MatRC_t<7, 7>::Identity();

            optimizer.addEdge(edge);
            // 记录已插入的边对
            inserted_edge_pairs.insert(std::make_pair(std::min(id1, id2), std::max(id1, id2)));
        };

    // ---------- 3.1 添加回环融合产生的新连接边 ----------
    for (const auto& loop_connection : loop_connections) {
        auto keyfrm = loop_connection.first;
        const auto& connected_keyfrms = loop_connection.second;

        const auto id1 = keyfrm->id_;
        // 获取帧1的Sim3位姿及其逆变换
        const g2o::Sim3& Sim3_1w = Sim3s_cw.at(id1);
        const g2o::Sim3 Sim3_w1 = Sim3_1w.inverse();

        for (auto connected_keyfrm : connected_keyfrms) {
            const auto id2 = connected_keyfrm->id_;

            // 对于非当前帧-回环帧的边，需要检查共享路标点数量是否超过阈值
            // 当前帧与回环帧之间的边无条件添加
            if (!(id1 == curr_keyfrm->id_ && id2 == loop_keyfrm->id_)
                && keyfrm->graph_node_->get_num_shared_landmarks(connected_keyfrm) < min_num_shared_lms_) {
                continue;
            }

            // 计算相对位姿：Sim3_21 = Sim3_2w * Sim3_w1
            const g2o::Sim3& Sim3_2w = Sim3s_cw.at(id2);
            const g2o::Sim3 Sim3_21 = Sim3_2w * Sim3_w1;

            // 添加约束边
            insert_edge(id1, id2, Sim3_21);
        }
    }

    // ---------- 3.2 添加非回环连接边（生成树边、历史回环边、共视边） ----------
    for (auto keyfrm : all_keyfrms) {
        const auto id1 = keyfrm->id_;

        // 获取未修改的Sim3位姿（用于计算正确的相对位姿约束）
        // 如果该帧在non_corrected_Sim3s中，使用未校正的位姿；否则使用当前位姿
        const auto iter1 = non_corrected_Sim3s.find(keyfrm);
        const g2o::Sim3 Sim3_w1 = ((iter1 != non_corrected_Sim3s.end()) ? iter1->second : Sim3s_cw.at(id1)).inverse();

        // ---------- 3.2.1 添加生成树边（父子关系） ----------
        auto parent_node = keyfrm->graph_node_->get_spanning_parent();
        if (parent_node) {
            const auto id2 = parent_node->id_;

            // 使用未修改的位姿计算相对位姿（确保约束的正确性）
            const auto iter2 = non_corrected_Sim3s.find(parent_node);
            const g2o::Sim3& Sim3_2w = (iter2 != non_corrected_Sim3s.end()) ? iter2->second : Sim3s_cw.at(id2);

            // 计算并添加相对位姿约束
            const g2o::Sim3 Sim3_21 = Sim3_2w * Sim3_w1;
            insert_edge(id1, id2, Sim3_21);
        }

        // ---------- 3.2.2 添加历史回环边 ----------
        const auto loop_edges = keyfrm->graph_node_->get_loop_edges();
        for (auto connected_keyfrm : loop_edges) {
            const auto id2 = connected_keyfrm->id_;

            // 避免重复添加（只在id1 > id2时添加）
            if (id1 <= id2) {
                continue;
            }

            // 使用未修改的位姿计算相对位姿
            const auto iter2 = non_corrected_Sim3s.find(connected_keyfrm);
            const g2o::Sim3& Sim3_2w = (iter2 != non_corrected_Sim3s.end()) ? iter2->second : Sim3s_cw.at(id2);

            const g2o::Sim3 Sim3_21 = Sim3_2w * Sim3_w1;
            insert_edge(id1, id2, Sim3_21);
        }

        // ---------- 3.2.3 添加共视边（共享路标点数量超过阈值） ----------
        const auto connected_keyfrms = keyfrm->graph_node_->get_covisibilities_over_min_num_shared_lms(min_num_shared_lms_);
        for (auto connected_keyfrm : connected_keyfrms) {
            // 空指针检查
            if (!connected_keyfrm || !parent_node) {
                continue;
            }
            // 排除已添加的父子边
            if (*connected_keyfrm == *parent_node
                || keyfrm->graph_node_->has_spanning_child(connected_keyfrm)) {
                continue;
            }
            // 排除已添加的回环边
            if (static_cast<bool>(loop_edges.count(connected_keyfrm))) {
                continue;
            }

            // 跳过即将被删除的关键帧
            if (connected_keyfrm->will_be_erased()) {
                continue;
            }

            const auto id2 = connected_keyfrm->id_;

            // 避免重复添加（只在id1 > id2时添加）
            if (id1 <= id2) {
                continue;
            }
            // 检查是否已插入过该边
            if (static_cast<bool>(inserted_edge_pairs.count(std::make_pair(std::min(id1, id2), std::max(id1, id2))))) {
                continue;
            }

            // 使用未修改的位姿计算相对位姿
            const auto iter2 = non_corrected_Sim3s.find(connected_keyfrm);
            const g2o::Sim3& Sim3_2w = (iter2 != non_corrected_Sim3s.end()) ? iter2->second : Sim3s_cw.at(id2);

            const g2o::Sim3 Sim3_21 = Sim3_2w * Sim3_w1;
            insert_edge(id1, id2, Sim3_21);
        }
    }

    // ==================== 步骤4：执行位姿图优化 ====================

    // 初始化优化器
    optimizer.initializeOptimization();
    // 执行最多50次迭代优化
    optimizer.optimize(50);

    // 释放终止动作对象
    delete terminateAction;

    // ==================== 步骤5：更新相机位姿和点云 ====================

    {
        // 加锁保护地图数据库
        std::lock_guard<std::mutex> lock(data::map_database::mtx_database_);

        // 存储优化后的Sim3位姿（从相机坐标系到世界坐标系），用于更新点云
        std::unordered_map<unsigned int, g2o::Sim3> corrected_Sim3s_wc;

        // ---------- 5.1 更新所有关键帧的相机位姿 ----------
        for (auto keyfrm : all_keyfrms) {
            const auto id = keyfrm->id_;

            // 获取优化后的顶点估计值
            auto keyfrm_vtx = static_cast<internal::sim3::shot_vertex*>(optimizer.vertex(id));

            // 从Sim3中提取旋转、平移和尺度
            const g2o::Sim3& corrected_Sim3_cw = keyfrm_vtx->estimate();
            const float s = corrected_Sim3_cw.scale();
            const Mat33_t rot_cw = corrected_Sim3_cw.rotation().toRotationMatrix();
            // 平移向量需要除以尺度因子以还原真实平移
            const Vec3_t trans_cw = corrected_Sim3_cw.translation() / s;

            // 转换为4x4位姿矩阵并更新关键帧位姿
            const Mat44_t cam_pose_cw = util::converter::to_eigen_pose(rot_cw, trans_cw);
            keyfrm->set_pose_cw(cam_pose_cw);

            // 保存位姿的逆变换，用于更新点云
            corrected_Sim3s_wc[id] = corrected_Sim3_cw.inverse();
        }

        // ---------- 5.2 更新所有路标点的世界坐标 ----------
        for (const auto& lm : all_lms) {
            // 跳过即将被删除的路标点
            if (lm->will_be_erased()) {
                continue;
            }

            // 获取该路标点的参考关键帧ID
            // 如果在found_lm_to_ref_keyfrm_id中有记录则使用，否则使用路标点自身的参考关键帧
            const auto id = (found_lm_to_ref_keyfrm_id.count(lm->id_))
                                ? found_lm_to_ref_keyfrm_id.at(lm->id_)
                                : lm->get_ref_keyframe()->id_;

            // 获取优化前后的Sim3变换
            const g2o::Sim3& Sim3_cw = Sim3s_cw.at(id);                    // 优化前：世界->相机
            const g2o::Sim3& corrected_Sim3_wc = corrected_Sim3s_wc.at(id); // 优化后：相机->世界

            // 计算校正后的世界坐标：
            // 1. 先用优化前的位姿将点从世界坐标系变换到相机坐标系
            // 2. 再用优化后的位姿将点从相机坐标系变换回世界坐标系
            const Vec3_t pos_w = lm->get_pos_in_world();
            const Vec3_t corrected_pos_w = corrected_Sim3_wc.map(Sim3_cw.map(pos_w));

            // 更新路标点的世界坐标
            lm->set_pos_in_world(corrected_pos_w);
            // 更新路标点的平均观测方向和观测尺度方差
            lm->update_mean_normal_and_obs_scale_variance();
        }
    }
}

} // namespace optimize
} // namespace stella_vslam
