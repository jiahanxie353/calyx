from calyx.builder import (
    Builder,
    add_comp_params,
    invoke,
    while_with,
    par,
)
from typing import Literal
from math import log2
import json

# In general, ports to the wrapper are uppercase, internal registers are lower case.

yxi_input = """
{
  "toplevel": "main",
  "memories": [
    {
      "name": "A0",
      "width": 32,
      "size": 8
    },
    {
      "name": "B0",
      "width": 32,
      "size": 8
    },
    {
      "name": "v0",
      "width": 32,
      "size": 1
    }
  ]
}
"""

yxi = json.loads(yxi_input)
mems = yxi["memories"]


def add_arread_channel(prog, mem):
    _add_m_to_s_address_channel(prog, mem, "AR")


def add_awwrite_channel(prog, mem):
    awwrite_channel = _add_m_to_s_address_channel(prog, mem, "AW")
    max_transfers = awwrite_channel.reg("max_transfers", 8, is_ref=True)

    # TODO(nathanielnrn): We eventually want to move beyond
    # the implicit 1 transaction that is the size of the memory
    # How should we store this?
    # Recall this goes to write channel as number of transfers it expectes to do before
    # setting WLAST high
    with awwrite_channel.get_group("do_aw_transfer"):
        max_transfers.in_ = mem["size"] - 1
        max_transfers.write_en = 1


def _add_m_to_s_address_channel(prog, mem, prefix: Literal["AW", "AR"]):
    """Adds a manager to subordinate
    channel to the program. Uses `prefix` to name the channels
    appropriately. Expected to be either "AW" or "AR."
    Contains all of the channels shared between AW and AR channels.

    Returns a component builder in case there are additional
    cells/wires/groups that need to be added to the component.
    """

    assert prefix in ["AW", "AR"], "Prefix must be either AW or AR."

    # Following Arm's notation of denoting `xVALID` and `xREADY` signals
    # `x` stands for the prefix of the channel, i.e. `AW` or `AR`
    lc_x = prefix.lower()
    x = prefix
    # Inputs/outputs
    m_to_s_address_channel = prog.component(f"m_{lc_x}_channel")
    channel_inputs = [("ARESETn", 1), (f"{x}READY", 1)]
    channel_outputs = [
        (f"{x}VALID", 1),
        (f"{x}ADDR", 64),
        (f"{x}SIZE", 3),  # bytes used in transfer
        (f"{x}LEN", 8),  # number of transfers in transaction
        (f"{x}BURST", 2),  # for XRT should be tied to 2'b01 for WRAP burst
        (f"{x}PROT", 3),  # tied to be priviliged, nonsecure, data access request
    ]
    add_comp_params(m_to_s_address_channel, channel_inputs, channel_outputs)

    # Cells
    xvalid = m_to_s_address_channel.reg(f"{lc_x}valid", 1)
    xvalid_was_high = m_to_s_address_channel.reg(f"{lc_x}valid_was_high", 1)
    base_addr = m_to_s_address_channel.reg("base_addr", 64, is_ref=True)
    xlen = m_to_s_address_channel.reg(f"{lc_x}len", 8)

    # Number of txns we want to occur before m_arread_channel is done
    # TODO: parameterize
    txn_n = m_to_s_address_channel.const("txn_n", 32, 1)
    txn_count = m_to_s_address_channel.reg("txn_count", 32)
    txn_adder = m_to_s_address_channel.add(32, "txn_adder")

    # Need to put block_transfer register here to avoid combinational loops
    bt_reg = m_to_s_address_channel.reg("bt_reg", 1)

    # Wires
    with m_to_s_address_channel.continuous:
        m_to_s_address_channel.this()[f"{x}VALID"] = xvalid.out

    # Groups
    # Responsible for asserting ARVALID, and deasserting it a cycle after the handshake.
    # This is necesarry because of the way transitions between groups work.
    # See #1828 https://github.com/calyxir/calyx/issues/1828
    with m_to_s_address_channel.group(f"do_{lc_x}_transfer") as do_x_transfer:
        xREADY = m_to_s_address_channel.this()[f"{x}READY"]
        # TODO: Can we simplify this?
        # See comments #1846 https://github.com/calyxir/calyx/pull/1846
        # Assert arvalid if it was not previously high
        xvalid.in_ = ~xvalid_was_high.out @ 1
        # Deassert in the next cycle once it is high
        xvalid.in_ = (xvalid.out & xREADY & xvalid_was_high.out) @ 0
        xvalid.write_en = 1

        xvalid_was_high.in_ = (~(xvalid.out & xREADY) & ~xvalid_was_high.out) @ 1
        xvalid_was_high.write_en = (~(xvalid.out & xREADY) & ~xvalid_was_high.out) @ 1

        # Drive output signals for transfer
        m_to_s_address_channel.this()[f"{x}ADDR"] = base_addr.out
        # This is taken from mem size, we assume the databus width is the size
        # of our memory cell and that width is a power of 2
        # TODO(nathanielnrn): convert to binary instead of decimal
        m_to_s_address_channel.this()[f"{x}SIZE"] = width_xsize(mem["width"])
        # TODO(nathanielnrn): Figure our how to set arlen. For now set to size of mem.
        m_to_s_address_channel.this()[f"{x}LEN"] = xlen.out
        m_to_s_address_channel.this()[f"{x}BURST"] = 1  # Must be INCR for XRT
        # Required by spec, we hardcode to privileged, non-secure, data access
        m_to_s_address_channel.this()[f"{x}PROT"] = 0b110

        # control block_transfer reg to go low after one cycle
        bt_reg.in_ = (xREADY & xvalid.out) @ 1
        bt_reg.in_ = ~(xREADY & xvalid.out) @ 0
        bt_reg.write_en = 1
        do_x_transfer.done = bt_reg.out

    with m_to_s_address_channel.group("incr_txn_count") as incr_txn_count:
        txn_adder.left = txn_count.out
        txn_adder.right = 1
        txn_count.in_ = txn_adder.out
        txn_count.write_en = 1
        incr_txn_count.done = txn_count.done

    # Control
    # check if txn_count == txn_n
    cellname = "perform_reads" if prefix == "AR" else "perform_writes"
    check_transactions_done = m_to_s_address_channel.neq_use(
        txn_count.out, txn_n.out, signed=False, cellname=cellname, width=32
    )
    # with arread_channel.comb_group("check_reads_done") as check_reads_done:
    #     perform_reads.left = txn_count.out
    #     perform_reads.right = txn_n.out

    invoke_txn_count = invoke(txn_count, in_in=0)
    # ARLEN must be between 0-255, make sure to subtract 1 from yxi
    # size when assigning to ARLEN
    assert mem["size"] < 256, "Memory size must be less than 256"
    invoke_xlen = invoke(xlen, in_in=mem["size"] - 1)

    while_body = [
        par(
            invoke(bt_reg, in_in=0),
            invoke(xvalid_was_high, in_in=0),
        ),
        do_x_transfer,
        invoke(xvalid, in_in=0),
        incr_txn_count,
    ]

    while_loop = while_with(check_transactions_done, while_body)
    m_to_s_address_channel.control += [invoke_txn_count, invoke_xlen, while_loop]
    return m_to_s_address_channel


# Helper functions
def width_in_bytes(width: int):
    assert width % 8 == 0, "Width must be a multiple of 8."
    return width // 8


def width_xsize(width: int):
    log = log2(width_in_bytes(width))
    assert log.is_integer(), "Width must be a power of 2."
    return int(log)


def clog2(x):
    """Ceiling log2"""
    if x <= 0:
        raise ValueError("x must be positive")
    return (x - 1).bit_length()


def build():
    prog = Builder()
    # add_arread_channel(prog, mems[0])
    add_awwrite_channel(prog, mems[0])
    # add_read_channel(prog, mems[0])
    return prog.program


if __name__ == "__main__":
    build().emit()