在segment分割模型中，我希望对预测的掩码增加如下统计功能：
    -stats = {
        'total_masks': #实例数量
        'individual_areas': # 每个实例的面积列表
        'individual_areas_sum': #所有实例面积之和
        'actual_coverage_area': #实际覆盖的像素数（去重叠）        'largest_mask_area': #最大实例面积
        'smallest_mask_area': #最小实例面积
        'mask_coverage_ratio': #实际覆盖面积与总面积比
        'overlap_info': #重叠面积信息
    }
    -以下信息在gradio输出页面显示出来