from langchain.prompts import PromptTemplate,ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

import re

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import HumanMessagePromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_nvidia_ai_endpoints import ChatNVIDIA

import subprocess
import sys
import os
import json
from pathlib import Path

llm = ChatNVIDIA(
    model="meta/llama-3.2-3b-instruct",
    api_key="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",
    temperature=0.2,
    top_p=0.7,
    max_tokens=1024,
)

user_query="Generate AdvCollisionLoss class for rear-end collision with high severity"

class CodeDebugger:
    def __init__(self, llm):
        self.llm = llm
        self.chain = self.create_chain()

    def debugger_prompt(self):
        debugger_prompt_template = """
        You are an expert Python debugger with strong skills in in code tuning and optimization. 
        Read the original code, identify potential errors, and optimize it based on error information.
        Original Code: {original_code}
        Traceback error information: {error_info}
        Review the given original code and error information to improve the code. Ensure that 
        the improved code remains original motivation and runs without errors. 
        <output format>:
        '''python...'''
        # improved code
        """
        return debugger_prompt_template

    def create_debugger_prompt(self):
        prompt = self.debugger_prompt()
        debugger_prompt = PromptTemplate.from_template(prompt)
        return debugger_prompt

    def create_chain(self):
        prompt = self.create_debugger_prompt()
        chain = (
            {
                "original_code": RunnablePassthrough(),
                'error_info': RunnablePassthrough()
            }
            | prompt
            | self.llm
        )
        return chain

    def debugger(self, original_code, error_info):
        output = self.chain.invoke({'original_code':original_code, 'error_info':error_info})
        improved_code = output.content
        improved_code = extract_python_code(improved_code)
        return improved_code

def pipeline(query):
    test_file_path = '/home/ztj/learn/LD-Scene/scripts/test_diffusion_with_guidance.py'
    guidance_file_path = '/home/ztj/learn/LD-Scene/guidefn/Guidance.py'

    code_generator = CodeGenerator(llm)
    original_code = code_generator.generate_code(query)

    keep_only_imports_regex(guidance_file_path)
    append_code_smart(guidance_file_path, original_code)

    max_cycle_depth = 20
    success = False
    code_debugger = CodeDebugger(llm)

    args = {
        'diffusion_cfg': '/home/ztj/learn/LD-Scene/configs/test/diffusion_model.cfg',
        'traffic_cfg': '/home/ztj/learn/LD-Scene/configs/test/traffic_model.cfg'
    }
    count = 0
    while not success:
        success, _, error_info = run_script_with_args(test_file_path, args)
        if not success:
            original_code = code_debugger.debugger(original_code, error_info)
            keep_only_imports_regex(guidance_file_path)
            append_code_smart(guidance_file_path,original_code)
        if count == max_cycle_depth:
            print("Exceed maximum cycle depth!")


class CodeGenerator:
    def __init__(self, llm):
        self.llm = llm
        self.code_generate_system_prompt = """
        <Role>: You are a senior research scientist specializing in adversarial scenarios generation
        for autonomous driving, and a Python programming expert with strong skills in generating 
        concise, efficient, and reusable code.
        <Task>: Generate a guidance loss class for the diffusion model to generate scenarios that 
        encourages collision consistent with user query between the adversarial vehicle and the ego.
        <Input>: traj is a (NA x T x 4) tensor representing the current trajectories of all vehicles.
        <Instructions>: 
        1. You can use EnvCollLoss as a punishment for vehicles leaving the drivable area， 
        and use class BvCollLoss to penalize collisions between background vehicles to avoid unreasonable 
        collisions.
        2. The provided context contains helper functions for selecting adversarial vehicles and obtaining 
        information such as speed based on vehicle idx. You can only use them to construct class guidance loss 
        and should not use undefined functions in the context.
        3. The ego's index in input traj is 0, and when calculating the relative information between the trajectories 
        of other vehicles and the ego, pay attention to the processing of the input traj to ensure the feasibility of the code.
        4. The goal of class AdvanceCollisionLoss is to guide the adversarial vehicle and the ego to generate 
        collision trajectories consistent with the user query.
        <Output>:
        1. The guidance loss class AdvCollisionLoss for the diffusion model.
        2. The complete class Sample_fn.
        3. The collision type and collision level.
        """
        self.code_generate_reasoning_prompt = """
        Generate code based on the {query} and {context}. You need to output the answers to the following questions step by step. 
        Answer contains reasoning process.
        -step1: Determine the collision type based on user query, which can be selected from 
        'rear_end', 'head_on', 'side_impact' and 'angle'; If there is no specific type, it is any.
        -step2: Define the collision level based on user query, and set the corresponding weights and thresholds.
        -step3: Generate the guidance loss function class with the help of code functions provided in the context. 
        The class should be wrapped in '''python''' in the final code snippet.
        Provide the process of thinking, including determining the type and degree of collision and the logical composition of the loss.
        """
        self.code_generate_format_prompt = """
        Below is the the template code for generating a guidance loss class for a diffusion model. Generate class AdvCollisionLoss code 
        and complete the TODO section of class Sample_fn based on how to use AdvCollisionLoss. 
        Class AdvCollisionLoss(nn.Module):
            def  __init__(self):
             
            
            def forward(self, traj):
            # return the adversarial collision loss of adversarial vehicles against the ego vehicle in the time series T
            
            return collision_loss
            
        class Sample_fn:
            def __init__(self, traffic_model,
                         state_normalizer=None,
                         att_normalizer=None):
                self.traffic_model = traffic_model
                self.state_normalizer = state_normalizer
                self.att_normalizer = att_normalizer
        
            def calculate_loss(self, z, scene_graph, map_feat, past_feat,
                               map_idx=None,
                               map_env=None):
                # compute veh2veh collision
                # z: (sample_size, NA, feat)
                if self.state_normalizer is None or self.att_normalizer is None:
                    print('Must have normalizers to compute collisison loss!')
                    exit()
                # unnormalize
                with torch.enable_grad():
                    full_grad = torch.zeros_like(z)
                    veh_att = self.att_normalizer.unnormalize(scene_graph.lw)
                    z = torch.unbind(z, dim=1)
                    loss = []
                    for i, z_i in enumerate(z):
                        traj = self.traffic_model.decoder(scene_graph, map_feat, past_feat, z_i, map_idx, map_env)
                        traj = self.state_normalizer.unnormalize(traj)
        
                        # build the loss function
                        #Initialize loss function
                        bv_coll_loss = BvCollLoss(veh_att, scene_graph.batch, scene_graph.ptr)
                        env_coll_loss = EnvCollLoss(veh_att, map_idx, map_env, traj.size(1))
        
        
                        traj = traj.contiguous()
                        bv_penalties, num_pairs = bv_coll_loss(traj)
                        env_penalties = env_coll_loss(traj)
        
                        ###TODO###
                        adv_coll_loss = AdvCollisionLoss(……)
                        adv_reward = adv_coll_loss(traj)
            
                        total_loss = bv_penalties.sum() + env_penalties.sum() + adv_reward.sum()
                        ###TODO###
        
                        if not total_loss.requires_grad:
                            total_loss = total_loss.requires_grad_(True)
        
                        loss.append(total_loss)
        
                        grad_i = torch.autograd.grad(total_loss, z[i], retain_graph=True)
                        full_grad[:, i, :] = grad_i[0]
        
                return loss, full_grad
        
            def calculate_grad(self, loss, z):
                grad = torch.autograd.grad(loss, z, retain_graph=True)
                return grad
                    
        class Sample_fn:
            def __init__(self, traffic_model,
                         state_normalizer=None,
                         att_normalizer=None):
                self.traffic_model = traffic_model
                self.state_normalizer = state_normalizer
                self.att_normalizer = att_normalizer
        
            def calculate_loss(self, z, scene_graph, map_feat, past_feat,
                               map_idx=None,
                               map_env=None):
                # compute veh2veh collision
                if self.state_normalizer is None or self.att_normalizer is None:
                    print('Must have normalizers to compute collisison loss!')
                    exit()
                # unnormalize
                with torch.enable_grad():
                    veh_att = self.att_normalizer.unnormalize(scene_graph.lw)
                    traj = self.traffic_model.decoder(scene_graph, map_feat, past_feat, z, map_idx, map_env)
                    traj = self.state_normalizer.unnormalize(traj)
                    # build the loss function
                    #Initialize loss function
                    bv_coll_loss = BvCollLoss(veh_att, scene_graph.batch, scene_graph.ptr)
                    env_coll_loss = EnvCollLoss(veh_att, map_idx, map_env, traj.size(1))
        
                    bv_penalties, num_pairs = bv_coll_loss(traj)
                    env_penalties = env_coll_loss(traj)
                    
                    ###TODO###
                    adv_coll_loss = AdvCollisionLoss(……)
                    adv_reward = adv_coll_loss(traj)
        
                    total_loss = bv_penalties.sum() + env_penalties.sum() + adv_reward.sum()
                    ###TODO###
        
                    if not total_loss.requires_grad:
                        total_loss = total_loss.requires_grad_(True)
        
                return total_loss   
        """

    def create_retriever(self):
        model_name = '/home/ztj/learn/LD-Scene/model_ckpt/all-MiniLM-L6-v2'
        embedding = HuggingFaceEmbeddings(model_name=model_name)
        persist_directory = "./rag_utils"
        helper_functions = Chroma(embedding_function=embedding, persist_directory=persist_directory,
                                  collection_name="code_chunks")
        helper_functions = helper_functions.as_retriever()
        return helper_functions


    def create_chain(self):
        # 使用示例
        retriever = self.create_retriever()
        composer = MultiPromptComposer()

        # 添加多个提示
        composer.add_prompt(
            "system",
            self.code_generate_system_prompt
        ).add_prompt(
            "human",
            self.code_generate_reasoning_prompt,
            input_variables=["context", "query"]
        ).add_prompt(
            "human",
            self.code_generate_format_prompt
        )

        # 创建链
        chain = composer.create_chain(llm=llm, retriever=retriever, output_parser=StrOutputParser())
        return chain

    def generate_code(self, query):
        chain = self.create_chain()
        result = chain.invoke(query)
        extracted_code = extract_python_code(result)
        if extracted_code:
            print(extracted_code)
        else:
            print("No Code is generated!")
        return extracted_code


# rag_chain = ({"context": helper_functions | format_docs, "query": RunnablePassthrough()} | integrate_prompt | llm | StrOutputParser())



def extract_python_code(text: str) -> str:
    """extract python code from the text"""
    code_blocks = re.findall(r'```python\n(.*?)\n```', text, re.DOTALL)
    if code_blocks:
        return code_blocks[0]


class MultiPromptComposer:
    """多提示组合器"""

    def __init__(self):
        self.prompts = []

    def add_prompt(self, prompt_type, template, **kwargs):
        """添加提示模板"""
        if prompt_type == "system":
            prompt = SystemMessagePromptTemplate.from_template(template, **kwargs)
        elif prompt_type == "human":
            prompt = HumanMessagePromptTemplate.from_template(template, **kwargs)
        elif prompt_type == "ai":
            from langchain.prompts import AIMessagePromptTemplate
            prompt = AIMessagePromptTemplate.from_template(template, **kwargs)
        else:
            prompt = PromptTemplate.from_template(template, **kwargs)

        self.prompts.append((prompt_type, prompt))
        return self

    def compose(self, input_variables=None):
        """组合所有提示"""
        messages = []

        for prompt_type, prompt in self.prompts:
            if prompt_type in ["system", "human", "ai"]:
                messages.append(prompt)
            else:
                # 普通提示转换为人类消息
                messages.append(HumanMessagePromptTemplate(prompt=prompt))

        return ChatPromptTemplate.from_messages(messages)

    def create_chain(self, llm, retriever=None, output_parser=None):
        """创建完整的链"""
        # 组合提示
        chat_prompt = self.compose()

        if retriever:
            # 带检索的链
            # def get_context(query):
            #     docs = retriever.get_relevant_documents(query)
            #     return "\n".join([doc.page_content for doc in docs])

            chain = (
                    {
                        "context": retriever,
                        "query": RunnablePassthrough()
                    }
                    | chat_prompt
                    | llm
            )
        else:
            # 不带检索的链
            chain = chat_prompt | llm

        if output_parser:
            chain = chain | output_parser

        return chain




def append_code_smart(file_path, code_string, check_duplicates=True):
    """
    智能追加代码，可选检查重复内容

    Args:
        file_path: 目标文件路径
        code_string: 要追加的代码字符串
        check_duplicates: 是否检查重复代码
    """
    try:
        # 读取现有内容
        with open(file_path, 'r', encoding='utf-8') as f:
            existing_content = f.read()

        # 检查是否已经存在相同的代码
        if check_duplicates and code_string.strip() in existing_content:
            print("⚠️  发现相似代码，跳过追加")
            return False

        # 追加代码
        with open(file_path, 'a', encoding='utf-8') as f:
            # 添加时间戳和分隔符
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            separator = f"\n\n{'#' + '=' * 60}\n"
            separator += f"# AI Generated Code - {timestamp}\n"
            separator += f"{'#' + '=' * 60}\n\n"

            f.write(separator)
            f.write(code_string)

            # 确保代码以换行符结束
            if not code_string.endswith('\n'):
                f.write('\n')

        print(f"✅ 代码已成功追加到文件末尾: {file_path}")
        return True

    except FileNotFoundError:
        # 如果文件不存在，创建新文件
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(code_string)
            if not code_string.endswith('\n'):
                f.write('\n')
        print(f"✅ 文件不存在，已创建新文件: {file_path}")
        return True
    except Exception as e:
        print(f"❌ 追加代码失败: {e}")
        return False




def run_script_with_args(script_path, args=None, config=None, working_dir=None):
    """
    运行带命令行参数的Python脚本
    Args:
        script_path: Python脚本路径
        args: 命令行参数列表，如 ['--input', 'data.txt', '--output', 'result.json']
        config: 配置参数字典（通过环境变量传递）
    Returns:
        tuple: (是否成功, 输出, 错误信息)
    """

    # 检查文件
    if not os.path.exists(script_path):
        return False, "", f"文件不存在: {script_path}"
    # 获取脚本目录
    script_abs_path = os.path.abspath(script_path)
    script_dir = '/home/ztj/learn/LD-Scene/scripts'
    # 准备命令
    cmd = [sys.executable, script_path]
    # 添加命令行参数
    if args:
        cmd.extend(args)
    # 准备环境变量
    env = os.environ.copy()

    # 设置工作目录
    if working_dir is None:
        working_dir = script_dir
    else:
        working_dir = os.path.abspath(working_dir)
    if 'PYTHONPATH' in env:
        env['PYTHONPATH'] = script_dir + os.pathsep + env['PYTHONPATH']
    else:
        env['PYTHONPATH'] = script_dir

    # 传递配置参数
    if config:
        env['APP_CONFIG'] = json.dumps(config)
    # 运行脚本
    result = subprocess.run(
        cmd,
        cwd=working_dir,  # 在脚本目录下运行
        env=env,
        capture_output=True,
        text=True,
        timeout=60
    )
    filtered_stderr = extract_traceback_only(result.stderr)
    success = result.returncode == 0
    return success, result.stdout, filtered_stderr


def extract_traceback_only(error_output):
    """
    只提取Traceback部分
    """
    if not error_output:
        return ""

    lines = error_output.split('\n')
    traceback_lines = []
    in_traceback = False

    for line in lines:
        if 'Traceback (most recent call last):' in line:
            in_traceback = True
            traceback_lines = [line]  # 重新开始
        elif in_traceback:
            traceback_lines.append(line)
            # 如果遇到空行且下一行不是Traceback相关，可能结束
            if line.strip() == '':
                # 检查下一行是否还是Traceback的一部分
                pass
            # 如果遇到非缩进行且不是错误信息，可能结束
            elif not line.startswith(' ') and not line.startswith('File') and 'Error:' not in line:
                in_traceback = False

    return '\n'.join(traceback_lines)

def keep_only_imports_regex(file_path, output_path=None):
    """
    使用正则表达式保留文件中的import语句

    Args:
        file_path: 输入文件路径
        output_path: 输出文件路径，如果为None则覆盖原文件
    """
    if output_path is None:
        output_path = file_path

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 匹配所有import语句（包括import和from ... import）
        import_pattern = r'^\s*(?:import\s+[\w., ]+|from\s+[\w.]+\s+import\s+[\w., *]+).*$'

        lines = content.split('\n')
        import_lines = []

        for line in lines:
            line_stripped = line.strip()

            # 跳过空行和注释
            if not line_stripped or line_stripped.startswith('#'):
                continue

            # 检查是否是import语句
            if re.match(r'^\s*(import|from)\s+', line):
                import_lines.append(line)

        # 写入结果
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(import_lines))

        print(f"✅ 已处理文件: {file_path}")
        print(f"   保留 {len(import_lines)} 个import语句")

        return True

    except Exception as e:
        print(f"❌ 处理失败: {e}")
        return False

pipeline(user_query)
# import os
# class CodeReplacer:
#     def __init__(self, target_file_path):
#         self.target_file_path = target_file_path
#         self.backup_file_path = f"{target_file_path}.backup"
#
#     def create_backup(self):
#         """Create a backup of the target file"""
#         if os.path.exists(self.target_file_path):
#             with open(self.target_file_path, 'r', encoding='utf-8') as src:
#                 with open(self.backup_file_path, 'w', encoding='utf-8') as dst:
#                     dst.write(src.read())
#             print(f"A backup: {self.backup_file_path} is created")
#         else:
#             print(f"Warning: {self.target_file_path} dose not exist")
#
#     def restore_backup(self):
#         """从备份恢复文件"""
#         if os.path.exists(self.backup_file_path):
#             with open(self.backup_file_path, 'r', encoding='utf-8') as src:
#                 with open(self.target_file_path, 'w', encoding='utf-8') as dst:
#                     dst.write(src.read())
#             print(f"✅ 已从备份恢复: {self.target_file_path}")
#         else:
#             print("❌ 备份文件不存在")
#
#     def replace_class_in_file(self, generated_code, class_name="AdvCollisionLoss"):
#         """用生成的代码替换文件中的类"""
#
#         # 1. 读取原始文件内容
#         with open(self.target_file_path, 'r', encoding='utf-8') as f:
#             original_content = f.read()
#
#         # 2. 从生成的代码中提取目标类
#         new_class_code = self.extract_class_code(generated_code, class_name)
#         if not new_class_code:
#             print(f"❌ 无法从生成代码中提取 {class_name} 类")
#             return False
#
#         # 3. 替换文件中的类
#         updated_content = self.replace_class_in_content(original_content, new_class_code, class_name)
#
#         if updated_content == original_content:
#             print(f"❌ 未找到要替换的 {class_name} 类")
#             return False
#
#         # 4. 写入更新后的内容
#         with open(self.target_file_path, 'w', encoding='utf-8') as f:
#             f.write(updated_content)
#
#         print(f"✅ 成功替换 {class_name} 类")
#         return True
#
#     def extract_class_code(self, generated_code, class_name):
#         """从生成的代码中提取特定类的代码"""
#
#         # 方法1: 使用正则表达式提取类定义
#         pattern = rf'(class {class_name}.*?(?=\nclass|\n\nclass|\Z))'
#         match = re.search(pattern, generated_code, re.DOTALL)
#
#         if match:
#             return match.group(1).strip()
#
#         # 方法2: 使用AST解析
#         try:
#             tree = ast.parse(generated_code)
#             for node in tree.body:
#                 if isinstance(node, ast.ClassDef) and node.name == class_name:
#                     # 获取类的源代码行
#                     start_line = node.lineno
#                     end_line = node.end_lineno
#                     lines = generated_code.split('\n')
#                     return '\n'.join(lines[start_line - 1:end_line])
#         except:
#             pass
#
#         return None
#
#     def replace_class_in_content(self, content, new_class_code, class_name):
#         """在文件内容中替换类定义"""
#
#         # 方法1: 使用正则表达式替换
#         pattern = rf'class {class_name}.*?(?=\nclass|\n\nclass|\Z)'
#         replacement = new_class_code
#
#         if re.search(pattern, content, re.DOTALL):
#             updated_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
#             return updated_content
#
#         # 方法2: 如果类不存在，则在文件末尾添加
#         print(f"⚠️ 文件中未找到 {class_name} 类，将在末尾添加")
#         return content + '\n\n' + new_class_code
#
#     def validate_syntax(self):
#         """验证替换后文件的语法"""
#         try:
#             with open(self.target_file_path, 'r', encoding='utf-8') as f:
#                 source_code = f.read()
#             ast.parse(source_code)
#             print("✅ 语法验证通过")
#             return True
#         except SyntaxError as e:
#             print(f"❌ 语法错误: {e}")
#             return False