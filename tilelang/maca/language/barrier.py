from tvm import tirx

__all__ = ["mxc_barrier_inst", "mxc_arrive_gvmcnt"]


def mxc_barrier_inst() -> tirx.Stmt:
    return tirx.call_intrin("void", tirx.op.Op.get("tl.mxc_barrier_inst"))


def mxc_arrive_gvmcnt(cnt: int = 0) -> tirx.Stmt:
    return tirx.call_intrin("void", tirx.op.Op.get("tl.mxc_arrive_gvmcnt"), cnt)
