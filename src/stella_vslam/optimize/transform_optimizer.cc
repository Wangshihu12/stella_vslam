#include "stella_vslam/data/keyframe.h"
#include "stella_vslam/data/landmark.h"
#include "stella_vslam/optimize/transform_optimizer.h"
#include "stella_vslam/optimize/internal/sim3/transform_vertex.h"
#include "stella_vslam/optimize/internal/sim3/mutual_reproj_edge_wrapper.h"

#include <g2o/core/solver.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/core/optimization_algorithm_levenberg.h>

namespace stella_vslam {
namespace optimize {

transform_optimizer::transform_optimizer(const bool fix_scale, const unsigned int num_iter)
    : fix_scale_(fix_scale), num_iter_(num_iter) {}

/**
 * [功能描述]：优化两个关键帧之间的Sim3变换
 *            通过最小化双向重投影误差来精化Sim3估计，同时剔除外点
 * @param keyfrm_1：关键帧1（当前帧）
 * @param keyfrm_2：关键帧2（候选帧/回环帧）
 * @param matched_lms_in_keyfrm_2：[输入/输出] 关键帧1特征点与关键帧2路标点的匹配，外点会被置为nullptr
 * @param g2o_Sim3_12：[输入/输出] 从关键帧2到关键帧1的Sim3变换，优化后更新
 * @param chi_sq：卡方检验阈值，用于判断内点/外点
 * @return unsigned int：优化后的内点数量
 */
unsigned int transform_optimizer::optimize(const std::shared_ptr<data::keyframe>& keyfrm_1, const std::shared_ptr<data::keyframe>& keyfrm_2,
                                           std::vector<std::shared_ptr<data::landmark>>& matched_lms_in_keyfrm_2,
                                           ::g2o::Sim3& g2o_Sim3_12, const float chi_sq) const {
    // 计算卡方阈值的平方根，用于鲁棒核函数
    const float sqrt_chi_sq = std::sqrt(chi_sq);

    // ==================== 步骤1：构建优化器 ====================

    // 创建线性求解器（使用Eigen求解）
    auto linear_solver = stella_vslam::make_unique<g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>>();
    // 创建块求解器
    auto block_solver = stella_vslam::make_unique<g2o::BlockSolverX>(std::move(linear_solver));
    // 创建Levenberg-Marquardt优化算法
    auto algorithm = new g2o::OptimizationAlgorithmLevenberg(std::move(block_solver));

    // 创建稀疏优化器并设置算法
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(algorithm);

    // ==================== 步骤2：创建Sim3变换顶点 ====================

    // 创建Sim3变换顶点（待优化变量）
    auto Sim3_12_vtx = new internal::sim3::transform_vertex();
    Sim3_12_vtx->setId(0);
    Sim3_12_vtx->setEstimate(g2o_Sim3_12);  // 设置初始估计值
    Sim3_12_vtx->setFixed(false);            // 设置为可优化
    Sim3_12_vtx->fix_scale_ = fix_scale_;    // 是否固定尺度
    // 设置两个关键帧的世界坐标系位姿（用于将3D点变换到相机坐标系）
    Sim3_12_vtx->rot_1w_ = keyfrm_1->get_rot_cw();
    Sim3_12_vtx->trans_1w_ = keyfrm_1->get_trans_cw();
    Sim3_12_vtx->rot_2w_ = keyfrm_2->get_rot_cw();
    Sim3_12_vtx->trans_2w_ = keyfrm_2->get_trans_cw();
    optimizer.addVertex(Sim3_12_vtx);

    // ==================== 步骤3：添加路标点和约束边 ====================

    // 双向重投影边包装器，包含两条边：
    // - backward (edge_21_)：将关键帧2观测到的3D点重投影到关键帧1（使用关键帧1的相机模型）
    // - forward (edge_12_)：将关键帧1观测到的3D点重投影到关键帧2（使用关键帧2的相机模型）
    using reproj_edge_wrapper = internal::sim3::mutual_reproj_edge_wapper<data::keyframe>;
    std::vector<reproj_edge_wrapper> mutual_edges;

    // 匹配数量
    const unsigned int num_matches = matched_lms_in_keyfrm_2.size();
    mutual_edges.reserve(num_matches);

    // 获取关键帧1观测到的所有路标点
    const auto lms_in_keyfrm_1 = keyfrm_1->get_landmarks();

    // 有效匹配计数
    unsigned int num_valid_matches = 0;

    // 遍历所有匹配，创建约束边
    for (unsigned int idx1 = 0; idx1 < num_matches; ++idx1) {
        // 跳过无匹配的特征点
        if (!matched_lms_in_keyfrm_2.at(idx1)) {
            continue;
        }

        // 获取匹配的路标点对
        const auto& lm_1 = lms_in_keyfrm_1.at(idx1);      // 关键帧1在idx1处观测到的路标点
        const auto& lm_2 = matched_lms_in_keyfrm_2.at(idx1);  // 关键帧2中与之匹配的路标点

        // 检查两个路标点是否都有效
        if (!lm_1 || !lm_2) {
            continue;
        }
        if (lm_1->will_be_erased() || lm_2->will_be_erased()) {
            continue;
        }

        // 获取路标点2在关键帧2中的特征点索引
        const auto idx2 = lm_2->get_index_in_keyframe(keyfrm_2);

        if (idx2 < 0) {
            continue;
        }

        // 创建双向重投影边并添加到优化器
        reproj_edge_wrapper mutual_edge(keyfrm_1, idx1, lm_1, keyfrm_2, idx2, lm_2, Sim3_12_vtx, sqrt_chi_sq);
        optimizer.addEdge(mutual_edge.edge_12_);  // 前向边：1→2
        optimizer.addEdge(mutual_edge.edge_21_);  // 后向边：2→1

        ++num_valid_matches;
        mutual_edges.push_back(mutual_edge);
    }

    // ==================== 步骤4：执行第一轮优化 ====================

    optimizer.initializeOptimization();
    optimizer.optimize(5);  // 进行5次迭代

    // ==================== 步骤5：剔除外点 ====================

    unsigned int num_outliers = 0;
    for (unsigned int i = 0; i < num_valid_matches; ++i) {
        auto edge_12 = mutual_edges.at(i).edge_12_;
        auto edge_21 = mutual_edges.at(i).edge_21_;

        // 内点判断：两个方向的卡方误差都小于阈值
        if (edge_12->chi2() < chi_sq && edge_21->chi2() < chi_sq) {
            continue;
        }

        // 外点剔除：将匹配置为空，并标记为外点
        const auto idx1 = mutual_edges.at(i).idx1_;
        matched_lms_in_keyfrm_2.at(idx1) = nullptr;

        mutual_edges.at(i).set_as_outlier();
        ++num_outliers;
    }

    // 如果剩余内点数少于10个，认为优化失败
    if (num_valid_matches - num_outliers < 10) {
        return 0;
    }

    // ==================== 步骤6：执行第二轮优化 ====================
    // 剔除外点后重新优化，获得更精确的结果

    optimizer.initializeOptimization();
    optimizer.optimize(num_iter_);  // 使用配置的迭代次数

    // ==================== 步骤7：统计最终内点数量 ====================

    unsigned int num_inliers = 0;
    for (unsigned int i = 0; i < num_valid_matches; ++i) {
        auto edge_12 = mutual_edges.at(i).edge_12_;
        auto edge_21 = mutual_edges.at(i).edge_21_;

        // 跳过已标记为外点的边
        if (mutual_edges.at(i).is_outlier()) {
            continue;
        }

        // 再次检查误差是否超过阈值（优化后可能产生新的外点）
        if (chi_sq < edge_12->chi2() || chi_sq < edge_21->chi2()) {
            // 剔除新产生的外点
            unsigned int idx1 = mutual_edges.at(i).idx1_;
            matched_lms_in_keyfrm_2.at(idx1) = nullptr;
            continue;
        }

        ++num_inliers;
    }

    // ==================== 步骤8：输出优化结果 ====================

    // 将优化后的Sim3估计值赋给输出参数
    g2o_Sim3_12 = Sim3_12_vtx->estimate();

    return num_inliers;
}

} // namespace optimize
} // namespace stella_vslam
