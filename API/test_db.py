"""
测试数据库连接和写入功能
"""

import sys
import os

# 添加API目录到系统路径，确保可以导入模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db_utils import save_detection_result, get_detection_history, update_detection_result, delete_detection_record

def test_save_detection():
    """测试保存检测结果到数据库"""
    # 测试数据
    skid_number = "1234"
    cls_name = "正常"
    
    # 调用函数保存数据
    result = save_detection_result(skid_number, cls_name)
    
    # 输出结果
    if result:
        print(f"测试成功: 已将雪橇号 {skid_number} 的检测结果 '{cls_name}' 保存到数据库")
    else:
        print("测试失败: 数据保存失败")
    
    return result

def test_get_history():
    """测试获取检测历史记录"""
    # 获取最近10条记录
    records = get_detection_history(limit=10)
    
    # 输出结果
    print(f"\n获取到 {len(records)} 条历史记录:")
    for i, record in enumerate(records, 1):
        print(f"{i}. ID: {record.get('id')}, 雪橇号: {record.get('skid_nr')}, "
              f"时间: {record.get('dateTime')}, 结果: {record.get('result_cls')}")
    
    return records

def test_update_record(record_id, new_cls_name="异常"):
    """测试更新检测记录"""
    # 更新记录
    result = update_detection_result(record_id, new_cls_name)
    
    # 输出结果
    if result:
        print(f"\n测试成功: 已将ID为 {record_id} 的记录更新为 '{new_cls_name}'")
    else:
        print(f"\n测试失败: 更新ID为 {record_id} 的记录失败")
    
    return result

def test_delete_record(record_id):
    """测试删除检测记录"""
    # 删除记录
    result = delete_detection_record(record_id)
    
    # 输出结果
    if result:
        print(f"\n测试成功: 已删除ID为 {record_id} 的记录")
    else:
        print(f"\n测试失败: 删除ID为 {record_id} 的记录失败")
    
    return result

def run_all_tests():
    """运行所有测试"""
    # 保存测试
    save_result = test_save_detection()
    
    # 获取历史记录
    records = test_get_history()
    
    # 如果有记录且保存成功，则测试更新和删除
    if records and save_result:
        # 获取最新记录的ID
        latest_record_id = records[0].get('id')
        
        # 测试更新记录
        test_update_record(latest_record_id, "异常")
        
        # 再次获取历史记录，确认更新成功
        updated_records = get_detection_history(limit=1)
        if updated_records:
            print(f"更新后的记录: ID: {updated_records[0].get('id')}, "
                  f"结果: {updated_records[0].get('result_cls')}")
        
        # 测试删除记录
        # test_delete_record(latest_record_id)

if __name__ == "__main__":
    run_all_tests()