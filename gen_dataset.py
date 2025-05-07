import json
import os
from open_thoughts.prompt import format_code_prompt


# 获取脚本所在的目录作为基础路径
base_dir = "/public/data0/NLP/users/wucanhui.volcano/output"
# 定义输出目录
output_dir = base_dir
# 如果输出目录不存在，则创建它
os.makedirs(output_dir, exist_ok=True)

data_name = ["code", "math"]

def merge_data_files():
    all_data = [] # 将 all_data 移到循环外部

    for name in data_name:
        current_type_data_count = 0 # 用于跟踪当前类型添加的数据量
        
        type_count = 0
        shorten_file_name = f"{name}_shorten_iter5_data.jsonl"
        shorten_file_path = os.path.join(base_dir, shorten_file_name) # 使用绝对路径
        if os.path.exists(shorten_file_path):
            count_before = len(all_data)
            try:
                with open(shorten_file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            item = json.loads(line.strip())
                            # 转换为所需的消息格式，并设置 system content
                            formatted_item = format_item_to_messages(item, name, system_content="")
                            all_data.append(formatted_item)
                            type_count += 1
                loaded_count = len(all_data) - count_before
                current_type_data_count += loaded_count
                print(f"从 {shorten_file_path} 加载了 {loaded_count} 条数据")
            except json.JSONDecodeError as e:
                print(f"错误: 解析 {shorten_file_path} 时出错 - {e}")
            except Exception as e:
                 print(f"错误: 读取 {shorten_file_path} 时发生未知错误 - {e}")
        else:
            print(f"警告: {shorten_file_path} 不存在")
        
        # 读取 iter0~6 的 {}_wrong_itern_data.jsonl 文件 (注意文件扩展名)
        for i in range(6):  # 0 到 5
            wrong_file_name = f"{name}_wrong_iter{i}_data.jsonl"
            wrong_file_path = os.path.join(base_dir, wrong_file_name) # 使用绝对路径
            if os.path.exists(wrong_file_path):
                count_before = len(all_data)
                added_count_in_file = 0
                try:
                    with open(wrong_file_path, 'r', encoding='utf-8') as f:
                        # 逐行读取和解析 JSON Lines 文件
                        for line_num, line in enumerate(f, 1):
                            if line.strip():
                                try:
                                    data_item = json.loads(line.strip())
                                    # 转换为所需的消息格式，并设置 system content
                                    formatted_item = format_item_to_messages(data_item, name, system_content="")
                                    all_data.append(formatted_item)
                                    added_count_in_file += 1
                                    type_count += 1
                                except json.JSONDecodeError as e:
                                    print(f"错误: 解析 {wrong_file_path} 第 {line_num} 行时出错 - {e}")
                                    print(f"问题行内容: {line.strip()}")
                        # *** 修改结束 ***
                    current_type_data_count += added_count_in_file
                    print(f"从 {wrong_file_path} 添加了 {added_count_in_file} 条数据")
                except Exception as e:
                     print(f"错误: 读取或处理 {wrong_file_path} 时发生未知错误 - {e}")
            else:
                print(f"警告: {wrong_file_path} 不存在")
        
        print(f"从 {name} 类型的文件中加载了 {type_count} 条数据")
        
        # 读取 {}_verified_data.jsonl 文件
        # verified_file_name = f"{name}_verified_data.jsonl"
        # verified_file_path = os.path.join(base_dir, verified_file_name) # 使用绝对路径
        # if os.path.exists(verified_file_path):
        #     count_before = len(all_data)
        #     try:
        #         with open(verified_file_path, 'r', encoding='utf-8') as f:
        #             for line in f:
        #                 if line.strip():
        #                     item = json.loads(line.strip())
        #                     # 转换为所需的消息格式，verified 数据的 system_content 为空
        #                     formatted_item = format_item_to_messages(item, name, system_content="")
        #                     all_data.append(formatted_item)
        #         loaded_count = len(all_data) - count_before
        #         current_type_data_count += loaded_count
        #         print(f"从 {verified_file_path} 加载了 {loaded_count} 条数据")
        #     except json.JSONDecodeError as e:
        #         print(f"错误: 解析 {verified_file_path} 时出错 - {e}")
        #     except Exception as e:
        #         print(f"错误: 读取 {verified_file_path} 时发生未知错误 - {e}")
        # else:
        #     print(f"警告: {verified_file_path} 不存在")
        
        # print(f"处理完 {name} 类型，共添加 {current_type_data_count} 条数据")

    # 将所有合并后的数据写入一个总的 jsonl 文件
    output_file_name = "all_merged_data_iter5_no_puzzle.jsonl" # 定义总输出文件名
    output_file_path = os.path.join(output_dir, output_file_name) # 使用绝对路径
    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            for item in all_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"\n成功合并所有数据，共 {len(all_data)} 条") # 更新最终的打印信息
        print(f"输出保存到 {output_file_path}") # 更新最终的打印信息
    except Exception as e:
        print(f"错误: 写入输出文件 {output_file_path} 时出错 - {e}")


def format_item_to_messages(item, domain, system_content=""):
    """
    将数据项转换为消息格式
    {"messages": [{"role": "system", "content": "<system>"}, 
                 {"role": "user", "content": "<query1>"}, 
                 {"role": "assistant", "content": "<response1>"}],
     "compression_ratio": <float>}
    
    参数:
    - item: 数据项
    - domain: 领域类型 (puzzle, math, code)
    - system_content: 系统消息内容，默认为空字符串
    """
    # 系统消息
    system_message = {"role": "system", "content": system_content}
    
    # 用户消息
    user_content = ""
    if domain == "puzzle":
        user_content = item['question']
    elif domain == "math":
        user_content = f"Return your final response within \\boxed{{}}. {item['question']}"
    else:  # code
        user_content = format_code_prompt(item)
    
    user_message = {"role": "user", "content": user_content}
    
    # 助手消息
    reasoning = item.get('shorten_reasoning', '')
    solution = item.get('deepseek_solution', '')
    assistant_content = f"<think>\n{reasoning}\n</think>\n{solution}"
    assistant_message = {"role": "assistant", "content": assistant_content}
    
    # 组合消息
    formatted_data = {
        "messages": [system_message, user_message, assistant_message],
        "compression_ratio": item.get('compression_ratio', 0.0)  # 直接从item中获取compression_ratio字段
    }
    
    return formatted_data


if __name__ == "__main__":
    merge_data_files()