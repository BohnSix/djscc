"""
pip install pyldpc

生成矩阵 G 的形状通常预期为 [k x n]。 k 是信息位数，而 n 是码字的总位数。
生成矩阵用于将 k 个信息位映射到 n 个码字位上，因此它有 k 行和 n 列。
"""

from pyldpc import make_ldpc

codelength = 3072
coderates = [1/3, 1/6, 1/12]

print("*" * 50)
print("*" * 50)
print("*" * 50)

for coderate in coderates:
    n = codelength
    k = int(codelength * coderate)
    print(f"\ncoderate: {coderate:.3f}, n: {n}, k is expected to be {k:3d}.")
    # d_v = 3
    # d_c = int(d_v / (1-coderate))
    d_c = 12
    d_v = int(d_c * (1-coderate))
    print(f"d_v: {d_v}, d_c: {d_c}")

    H, G = make_ldpc(n, d_v, d_c, systematic=True, sparse=True)
    print(f"The shape of G is {G.shape}, k is {G.shape[1]}.")

print()
print("*" * 50)
print("*" * 50)
print("*" * 50)
