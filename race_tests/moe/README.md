# 使用说明


## 1. 文件定位

`custom_fusedmoe.py` 作为对外暴露的 MoE fused kernel 接入文件。

调用和改写：

1. `RoutedMoEKernel` 的初始化和__call__接口固定；
2. 内部实现可重写调整。

## 2. 使用
如下命令可以自动跑功能和性能测试,结果输出在终端
```bash
python fusedmoe_benchmark.py
```
或者
```bash
bash run.sh
```
## 3. 测试用例
可以在`moe_test_config.json`中直接添加测试case，目前有性能和功能各两个
当前默认：input float16, output float32

功能测试正确输出
```bash
✅ Functional test passed for config:
```
功能测试错误输出
```bash
❌ Functional test failed for config:
```

## 3. 需安装
torch \
tilelang \
apache-tvm-ffi