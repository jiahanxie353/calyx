#!/usr/bin/env python3
"""A standalone tool for executing compiled Xilinx XRT bitstreams.

This tool can be invoked as a subprocess to run a compiled `.xclbin`, which may
be compiled either for RTL simulation or for actual on-FPGA execution. It
consumes and produces fud-style JSON input/output data files but is otherwise
isolated from the rest of fud and can be invoked as a standalone program. This
separate-process model is important so the user (or parent process) can set the
*required* environment variables that the Xilinx toolchain needs to control its
execution mode and to find its support files.

This tool currently uses the `PYNQ`_ Python library, which is meant for
high-level application interaction but is also a fairly stable wrapper around
the underlying XRT libraries. In the future, we can consider replcaing PYNQ
with directly using the `pyxrt`_ library, or abandoning Python altogether and
using the native XRT library directly for simplicity.

A bunch of environment variables have to be set to use xclrun. A minimal
invocation of xclrun looks something like this::

    $ source /scratch/opt/Xilinx/Vitis/2020.2/settings64.sh
    $ source /scratch/opt/xilinx/xrt/setup.sh
    $ export EMCONFIG_PATH=`pwd`
    $ XCL_EMULATION_MODE=hw_emu
    $ XRT_INI_PATH=`pwd`/xrt.ini
    $ python -m fud.xclrun something.xclbin data.json

.. _PYNQ: https://github.com/xilinx/pynq
.. _pyxrt: https://github.com/Xilinx/XRT/blob/master/src/python/pybind11/src/pyxrt.cpp
"""
import argparse
import sys
import pyxrt
import numpy as np
import simplejson as sjson
from typing import Mapping, Any, Dict
from pathlib import Path
from fud.stages.verilator.json_to_dat import parse_fp_widths, float_to_fixed
from calyx.numeric_types import InvalidNumericType
import ctypes


def mem_to_buf(device, mem):
    """Convert a fud-style JSON memory object to an XRT buffer."""
    ndarray = np.array(mem["data"], dtype=_dtype(mem["format"]))
    buffer = pyxrt.bo(device, ndarray.nbytes, pyxrt.bo.normal, mem_bank=0) # TODO: adjust mem_bank
    buffer.write(ndarray.tobytes())
    buffer.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    return buffer


def buf_to_mem(fmt, buf):
    """Convert an XRT buffer to a fud-style JSON memory value."""
    buf.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
    data = np.frombuffer(buf.read(), dtype=_dtype(fmt))

    # Converts int representation into fixed point
    if fmt["numeric_type"] == "fixed_point":
        width, int_width = parse_fp_widths(fmt)
        frac_width = width - int_width

        def convert_to_fp(value: float):
            return float_to_fixed(float(value), frac_width)

        data = np.vectorize(convert_to_fp)(data)
        return list(data)
    elif fmt["numeric_type"] in {"bitnum", "floating_point"}:
        return [int(value) if isinstance(value, np.integer) else float(value) for value in data]
    else:
        raise InvalidNumericType('Fud only supports "fixed_point", "bitnum", and "floating_point.')


def run(xclbin: Path, data: Mapping[str, Any]) -> Dict[str, Any]:
    """Takes in a json data output and runs pynq using the data provided
    returns a dictionary that can be converted into json

    `xclbin` is path to relevant xclbin file.
    Assumes that data is a properly formatted calyx data file.
    Data file order must match the expected call signature in terms of order
    Also assume that the data Mapping values type are valid json-type equivalents
    """
    device = pyxrt.device(0)
    xclbin_path = str(xclbin.resolve(strict=True))
    xclbin_obj = pyxrt.xclbin(xclbin_path)
    uuid = device.load_xclbin(xclbin_obj)
    kernel_name =  xclbin_obj.get_kernels()[0].get_name()
    kernel = pyxrt.kernel(device, uuid, kernel_name)
    
    COUNT = 16
    DATA_SIZE = ctypes.sizeof(ctypes.c_int32) * COUNT
    boHandle = pyxrt.bo(device, DATA_SIZE, pyxrt.bo.normal, kernel.group_id(0))
    # buffers = [pyxrt.bo(device, DATA_SIZE, pyxrt.bo.normal, kernel.group_id(0)) for mem in data.values()]

    for buf, mem in zip(buffers, data.values()):
        breakpoint()
        buf.write(mem)


def _dtype(fmt) -> np.dtype:
    # See https://numpy.org/doc/stable/reference/arrays.dtypes.html for typing
    # details
    if (fmt["numeric_type"] == "floating_point"):
        type_string = "f"
    else:
        type_string = "i" if fmt["is_signed"] else "u"
    byte_size = int(fmt["width"] / 8)
    type_string = type_string + str(byte_size)
    return np.dtype(type_string)


def xclrun():
    # Parse command-line arguments.
    parser = argparse.ArgumentParser(
        description="run a compiled XRT program",
    )
    parser.add_argument("bin", metavar="XCLBIN", help="the .xclbin binary file to run")
    parser.add_argument("data", metavar="DATA", help="the JSON input data file")
    parser.add_argument(
        "--out",
        "-o",
        metavar="FILE",
        help="write JSON results to a file instead of stdout",
    )
    args = parser.parse_args()

    # Load the input JSON data file.
    with open(args.data) as f:
        in_data = sjson.load(f, use_decimal=True)

    # Run the program.
    out_data = run(Path(args.bin), in_data)

    # Dump the output JSON data.
    outfile = open(args.out, "w") if args.out else sys.stdout
    sjson.dump(out_data, outfile, indent=2, use_decimal=True)


if __name__ == "__main__":
    xclrun()
