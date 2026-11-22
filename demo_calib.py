#!/usr/bin/env python3
"""复核演示：校准分析员审查共用参考误差的读数。

场景：仪表 X、Y 受约束 Y = 1，先验 X ~ N(1, 1)。
三条读数分别观测 X、Y、Y，噪声协方差 R 全为 1 —— 三条读数共用
同一个参考误差。因此两条 Y 读数必须完全一致：逐条看都合理的
读数组，整组可能自相矛盾。复核的目标就是拦住这种证据。

Run: python3 demo_calib.py
"""

import calib


def show(title, result):
    status, rows, mean, cov = result
    print("  %-10s status = %s" % (title, status))
    print("  %-10s rows    = %s" % ("", rows))
    if status == "ok":
        print("  %-10s mean    = %s" % ("", tuple(str(v) for v in mean)))
        print("  %-10s cov     =" % "")
        for name, row in zip(rows, cov):
            print("      %-6s %s" % (name, tuple(str(v) for v in row)))
    print()


def main():
    print("=" * 64)
    print("校准复核演示：共用参考误差读数")
    print("=" * 64)

    # 登记模型：密度 ∝ exp(-xᵀPx/2 + hᵀx) 于 Ax=b 上。
    # P = diag(1, 0), h = (1, 0)，约束 Y = 1  =>  X ~ N(1, 1), Y == 1。
    model = calib.compile(
        ["X", "Y"],
        [[1, 0], [0, 0]],   # P：对称半正定，且在 ker(A)=span{(1,0)} 上正定
        [1, 0],             # h
        [[0, 1]],           # A
        [1],                # b
    )
    print("模型已登记：P=diag(1,0), h=(1,0), 约束 Y=1（先验 X~N(1,1)）")
    print("模型可复用，以下各次分析相互独立。\n")

    H = {"读数X": {"X": 1}, "读数Y1": {"Y": 1}, "读数Y2": {"Y": 1}}
    R = [[1, 1, 1],
         [1, 1, 1],
         [1, 1, 1]]        # 共用参考误差：三条读数噪声完全相同
    L = {"X": {"X": 1}, "Y": {"Y": 1}, "X+Y": {"X": 1, "Y": 1}}

    print("[1] 无观测基线（空 H、y、R）")
    show("prior", calib.evaluate(model, {}, {}, [], L))

    print("[2] 读数 (3, 2, 2)：逐条合理，整组也自洽")
    print("    两条 Y 读数一致，共用参考误差被整组抵消，(X,Y) 被固定。")
    show("y=(3,2,2)",
         calib.evaluate(model, H, {"读数X": 3, "读数Y1": 2, "读数Y2": 2},
                        R, L))

    print("[3] 篡改末值 -> 读数 (3, 2, 3)：逐条都合理，整组矛盾")
    print("    两条 Y 读数共享同一参考误差，必须相等；2 与 3 不可实现。")
    show("y=(3,2,3)",
         calib.evaluate(model, H, {"读数X": 3, "读数Y1": 2, "读数Y2": 3},
                        R, L))

    print("[4] 复核结论：证据组 (3,2,3) 被拦下（inconsistent），")
    print("    数值项按约定返回空元组；合法组 (3,2,2) 给出精确分数结果。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
