# 1. 优先配置环境变量（确保JAX找到系统CuDNN和CUDA）
import os
import jax
import jaxlib
import jax.numpy as jnp

# 关键：替换为你的系统CUDA实际路径（可通过`nvcc -V`确认，通常是/usr/local/cuda或/usr/local/cuda-11.8）
CUDA_INSTALL_PATH = "/usr/local/cuda-11.8"

# 配置CUDA和CuDNN路径，强制JAX使用系统库
os.environ["CUDA_HOME"] = CUDA_INSTALL_PATH
os.environ["LD_LIBRARY_PATH"] = f"{CUDA_INSTALL_PATH}/lib64:{os.environ.get('LD_LIBRARY_PATH', '')}"
# 明确告诉JAX CuDNN位置，并启用CuDNN加速
os.environ["XLA_FLAGS"] = f"--xla_gpu_cudnn_dir={CUDA_INSTALL_PATH}/lib64 --xla_gpu_enable_cudnn=true"
# 限制显存占用，避免初始化时显存分配异常
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"  # 仅用50% GPU显存


# 2. 定义完整测试函数
def full_jax_cudnn_test():
    print("=" * 70)
    print("JAX-CuDNN 完整功能测试报告")
    print("=" * 70)

    # 步骤1：验证版本信息（确认jax与jaxlib匹配）
    print("\n【1. 版本兼容性检查】")
    print(f"   - JAX 版本: {jax.__version__}")
    print(f"   - jaxlib 版本: {jaxlib.__version__}")
    # 验证jaxlib是否为CUDA版本（含cuda标识）
    if "cuda" in jaxlib.__version__:
        print("   ✅ jaxlib 为CUDA版本，支持GPU加速")
    else:
        print("   ❌ jaxlib 为CPU版本，需重新安装带CUDA的jaxlib")
        return

    # 步骤2：验证GPU识别（基础环境检查）
    print("\n【2. GPU设备识别检查】")
    print(f"   - 默认计算后端: {jax.default_backend()}")
    devices = jax.devices()
    print(f"   - 可用设备总数: {len(devices)}")
    gpu_devices = [dev for dev in devices if "NVIDIA" in dev.device_kind]
    if gpu_devices:
        for i, dev in enumerate(gpu_devices):
            print(f"   - GPU设备{i}: {dev.device_kind}")
        print("   ✅ GPU设备已成功识别")
    else:
        print("   ❌ 未识别到NVIDIA GPU设备")
        return

    # 步骤3：测试基础CUDA计算（矩阵乘法，不依赖CuDNN）
    print("\n【3. 基础CUDA计算测试（矩阵乘法）】")
    try:
        # 在GPU上生成随机数组
        rng = jax.random.PRNGKey(0)
        x = jax.random.normal(rng, (1024, 1024))  # 自动分配到GPU
        y = jnp.matmul(x, x.T)  # GPU矩阵乘法
        y.block_until_ready()  # 等待计算完成，确保不异步
        print(f"   - 计算结果形状: {y.shape}")
        print(f"   - 计算结果前2x2元素:\n{y[:2, :2]}")
        print("   ✅ 基础CUDA计算正常（不依赖CuDNN）")
    except Exception as e:
        print(f"   ❌ 基础CUDA计算失败: {str(e)[:120]}...")
        return

    # 步骤4：测试CuDNN核心功能（卷积计算，必须依赖CuDNN）
    print("\n【4. CuDNN功能测试（卷积计算）】")
    try:
        # 构造卷积输入（格式：[batch, height, width, in_channels]）
        input_data = jax.random.normal(rng, (1, 64, 64, 3))  # 1个样本，64x64尺寸，3通道
        # 构造卷积核（格式：[kernel_height, kernel_width, in_channels, out_channels]）
        conv_kernel = jax.random.normal(rng, (3, 3, 3, 16))  # 3x3核，3输入通道，16输出通道
        
        # 执行卷积计算（触发CuDNN加速，若CuDNN不可用则报错）
        conv_result = jax.lax.conv_general_dilated(
            lhs=input_data,
            rhs=conv_kernel,
            window_strides=(1, 1),  # 步长1x1
            padding="SAME"  #  SAME padding，输出尺寸与输入一致
        )
        conv_result.block_until_ready()  # 等待卷积计算完成
        
        print(f"   - 卷积输入形状: {input_data.shape}")
        print(f"   - 卷积核形状: {conv_kernel.shape}")
        print(f"   - 卷积输出形状: {conv_result.shape}")
        print("   ✅ CuDNN功能正常！卷积计算成功（JAX-CUDA完全可用）")
    except Exception as e:
        print(f"   ❌ CuDNN功能测试失败: {str(e)[:150]}...")
        print("   提示：若报'CUDNN_STATUS_INTERNAL_ERROR'，请确认环境变量中CuDNN路径正确")

    print("\n" + "=" * 70)


# 执行完整测试
if __name__ == "__main__":
    full_jax_cudnn_test()