import os
import re
import sys
import csv

def rename_videos_simple(folder_path):
    """重命名视频文件并保存映射关系"""
    # 支持视频格式
    video_exts = {'.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv'}
    
    # 获取视频文件
    files = [f for f in os.listdir(folder_path) 
             if os.path.isfile(os.path.join(folder_path, f)) and 
             os.path.splitext(f)[1].lower() in video_exts]
    
    if not files:
        print("未找到视频文件")
        return
    
    print(f"找到 {len(files)} 个视频文件")
    
    # 自然排序
    files.sort(key=lambda x: [int(t) if t.isdigit() else t.lower() 
                              for t in re.split(r'(\d+)', x)])
    
    # 显示文件列表
    print("\n文件列表:")
    for i, f in enumerate(files, 1):
        print(f"  [{i}] {f}")
    
    # 设置数字位数
    digit_width = 5
    
    # 获取绝对路径
    abs_folder = os.path.abspath(folder_path)
    
    # 保存映射关系
    mapping_file = os.path.join(folder_path, "original_names.txt")
    
    # 检查是否已有映射文件
    if os.path.exists(mapping_file):
        response = input(f"\n已存在映射文件 {mapping_file}，是否覆盖？(y/n): ").strip().lower()
        if response != 'y':
            print("操作取消")
            return
    
    # 准备映射列表（先不写入文件，先计算所有新文件名）
    mapping_list = []
    
    # 先计算所有新文件名（考虑冲突）
    for i, old_name in enumerate(files, 1):
        ext = os.path.splitext(old_name)[1]
        base_new_name = f"{i:0{digit_width}d}{ext}"
        
        # 初始新文件名
        new_name = base_new_name
        new_path = os.path.join(abs_folder, new_name)
        old_path = os.path.join(abs_folder, old_name)
        
        # 检查是否有文件名冲突
        counter = 0
        # 检查新文件名是否会被已计算的其他文件使用，或者是否已存在
        while (any(m[0] == new_name for m in mapping_list) or 
               (os.path.exists(new_path) and old_path != new_path)):
            counter += 1
            new_name = f"{i:0{digit_width}d}_{counter}{ext}"
            new_path = os.path.join(abs_folder, new_name)
        
        # 记录映射关系
        new_full_path = os.path.join(abs_folder, new_name)
        mapping_list.append((new_name, old_name, new_full_path))
    
    # 现在写入映射文件（使用实际将要使用的新文件名）
    with open(mapping_file, 'w', encoding='utf-8') as f:
        for new_name, old_name, new_full_path in mapping_list:
            # 格式：新文件完整路径,原始文件名
            f.write(f"{new_full_path},{old_name}\n")
    
    print(f"\n映射关系已保存到: {mapping_file}")
    
    # 显示映射关系预览
    print("\n映射关系预览:")
    for i, (new_name, old_name, _) in enumerate(mapping_list[:5], 1):
        print(f"  [{i}] {old_name} -> {new_name}")
    if len(mapping_list) > 5:
        print(f"  ... 还有 {len(mapping_list)-5} 个")
    
    # 确认重命名
    confirm = input(f"\n确认要将 {len(files)} 个文件重命名吗？(y/n): ").strip().lower()
    if confirm != 'y':
        print("操作取消")
        return
    
    # 重命名文件（使用 mapping_list 中的新文件名）
    print("\n开始重命名...")
    success_count = 0
    
    for new_name, old_name, new_full_path in mapping_list:
        old_path = os.path.join(folder_path, old_name)
        new_path = os.path.join(folder_path, new_name)
        
        try:
            os.rename(old_path, new_path)
            print(f"✓ {old_name} -> {new_name}")
            success_count += 1
        except Exception as e:
            print(f"✗ {old_name} -> 重命名失败: {e}")
    
    print(f"\n完成！成功重命名了 {success_count}/{len(files)} 个文件")
    print(f"映射文件: {mapping_file}")

def restore_original_names(folder_path):
    """根据original_names.txt恢复原始文件名"""
    mapping_file = os.path.join(folder_path, "original_names.txt")
    
    if not os.path.exists(mapping_file):
        print(f"错误: 找不到映射文件 {mapping_file}")
        return
    
    print(f"找到映射文件: {mapping_file}")
    
    # 读取映射关系
    mapping = []
    try:
        with open(mapping_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line and ',' in line:
                    new_full_path, old_name = line.split(',', 1)
                    new_full_path = new_full_path.strip()
                    old_name = old_name.strip()
                    
                    # 从完整路径中提取文件名
                    new_filename = os.path.basename(new_full_path)
                    mapping.append((new_filename, old_name))
                elif line:  # 非空行但没有逗号
                    print(f"警告: 第 {line_num} 行格式错误，跳过: {line}")
    except Exception as e:
        print(f"读取映射文件失败: {e}")
        return
    
    if not mapping:
        print("错误: 映射文件中没有有效数据")
        return
    
    print(f"找到 {len(mapping)} 个映射关系")
    
    # 显示映射关系
    print("\n映射关系:")
    for i, (new_name, old_name) in enumerate(mapping[:5], 1):
        print(f"  [{i}] {new_name} -> {old_name}")
    if len(mapping) > 5:
        print(f"  ... 还有 {len(mapping)-5} 个")
    
    # 确认恢复
    confirm = input(f"\n确认要恢复 {len(mapping)} 个文件的原始名称吗？(y/n): ").strip().lower()
    if confirm != 'y':
        print("操作取消")
        return
    
    # 执行恢复
    print("\n开始恢复原始文件名...")
    success_count = 0
    fail_count = 0
    restored_files = []
    
    for new_name, old_name in mapping:
        current_path = os.path.join(folder_path, new_name)
        restored_path = os.path.join(folder_path, old_name)
        
        # 检查新文件是否存在
        if not os.path.exists(current_path):
            print(f"✗ 文件不存在: {new_name}")
            fail_count += 1
            continue
        
        # 检查目标文件是否已存在
        if os.path.exists(restored_path) and current_path != restored_path:
            print(f"✗ 目标文件已存在，跳过: {old_name}")
            fail_count += 1
            continue
        
        try:
            os.rename(current_path, restored_path)
            print(f"✓ {new_name} -> {old_name}")
            success_count += 1
            restored_files.append(old_name)
        except Exception as e:
            print(f"✗ {new_name} -> 恢复失败: {e}")
            fail_count += 1
    
    print(f"\n恢复完成！")
    print(f"✓ 成功恢复: {success_count} 个文件")
    if fail_count > 0:
        print(f"✗ 恢复失败: {fail_count} 个文件")
    
    # 询问是否删除映射文件
    if success_count > 0:
        delete_choice = input(f"\n是否删除映射文件 {mapping_file}？(y/n): ").strip().lower()
        if delete_choice == 'y':
            try:
                os.remove(mapping_file)
                print(f"已删除映射文件")
            except Exception as e:
                print(f"删除映射文件失败: {e}")

def generate_train_csv(folder_path):
    """根据original_names.txt生成train.csv文件（不包含表头）"""
    mapping_file = os.path.join(folder_path, "original_names.txt")
    
    if not os.path.exists(mapping_file):
        print(f"错误: 找不到映射文件 {mapping_file}")
        return
    
    print(f"找到映射文件: {mapping_file}")
    
    # 读取映射关系
    data_lines = []
    try:
        with open(mapping_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line and ',' in line:
                    new_full_path, old_name = line.split(',', 1)
                    new_full_path = new_full_path.strip()
                    old_name = old_name.strip()
                    
                    # 提取标签：原始文件名中第一个下划线前的数字
                    label = "0"  # 默认标签
                    
                    # 查找第一个下划线
                    underscore_index = old_name.find('_')
                    if underscore_index != -1:
                        # 获取下划线前的部分
                        prefix = old_name[:underscore_index]
                        if prefix.isdigit():
                            label = prefix
                        else:
                            numbers = re.findall(r'\d+', old_name)
                            if numbers:
                                label = numbers[0]
                    else:
                        numbers = re.findall(r'\d+', old_name)
                        if numbers:
                            label = numbers[0]
                    
                    data_lines.append((new_full_path, label))
                elif line:
                    print(f"警告: 第 {line_num} 行格式错误，跳过: {line}")
    except Exception as e:
        print(f"读取映射文件失败: {e}")
        return
    
    if not data_lines:
        print("错误: 映射文件中没有有效数据")
        return
    
    print(f"找到 {len(data_lines)} 条数据")
    
    # 生成train.csv文件（不包含表头）
    csv_file = os.path.join(folder_path, "train.csv")
    
    if os.path.exists(csv_file):
        response = input(f"\n已存在文件 {csv_file}，是否覆盖？(y/n): ").strip().lower()
        if response != 'y':
            print("操作取消")
            return
    
    try:
        # 直接写入，每行格式：文件路径,标签
        with open(csv_file, 'w', encoding='utf-8') as f:
            for file_path, label in data_lines:
                f.write(f"{file_path},{label}\n")
        
        print(f"\n✅ train.csv 文件已生成: {csv_file}")
        print(f"   共写入 {len(data_lines)} 条记录")
        
        # 显示生成的内容预览
        print("\n生成的内容预览（前3条）:")
        for i, (file_path, label) in enumerate(data_lines[:3], 1):
            filename = os.path.basename(file_path)
            print(f"  {file_path},{label}")
            print(f"     → 文件名: {filename}, 标签: {label}")
        
    except Exception as e:
        print(f"❌ 生成train.csv文件失败: {e}")

def main():
    """主函数"""
    print("视频文件处理工具")
    print("=" * 50)
    print("1. 重命名视频文件（并保存映射关系）")
    print("2. 恢复原始文件名（根据映射文件）")
    print("3. 生成train.csv文件（根据映射文件）")
    print("=" * 50)
    
    # 选择模式
    while True:
        mode = input("\n请选择模式 (1:重命名, 2:恢复, 3:生成CSV, q:退出): ").strip().lower()
        
        if mode == 'q':
            print("退出程序")
            break
        
        if mode not in ['1', '2', '3']:
            print("请输入 1、2、3 或 q")
            continue
        
        # 获取文件夹路径
        if len(sys.argv) > 1:
            folder_path = sys.argv[1]
            print(f"使用命令行参数: {folder_path}")
        else:
            folder_path = input("请输入文件夹路径: ").strip()
        
        folder_path = folder_path.strip('"\'')
        
        if not os.path.exists(folder_path):
            print(f"错误: 文件夹 '{folder_path}' 不存在！")
            continue
        
        if not os.path.isdir(folder_path):
            print(f"错误: '{folder_path}' 不是文件夹！")
            continue
        
        # 执行对应功能
        if mode == '1':
            rename_videos_simple(folder_path)
        elif mode == '2':
            restore_original_names(folder_path)
        else:  # mode == '3'
            generate_train_csv(folder_path)
        
        # 是否继续
        continue_choice = input("\n是否继续其他操作？(y/n): ").strip().lower()
        if continue_choice != 'y':
            print("感谢使用！")
            break

if __name__ == "__main__":
    main()