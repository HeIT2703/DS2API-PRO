import os
import ctypes
import logging
import threading
from typing import Optional

from .exceptions import PoWSolverError
from .validation import ensure_positive_int, ensure_string


logger = logging.getLogger(__name__)


class PoWSolver:
    def __init__(self, max_difficulty: Optional[int] = None):
        self.wasm_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sha3_wasm_bg.wasm")
        if not os.path.exists(self.wasm_path):
            raise PoWSolverError(f"Missing WASM file at {self.wasm_path}. Please ensure sha3_wasm_bg.wasm is present.")

        if max_difficulty is not None:
            ensure_positive_int(max_difficulty, "max_difficulty")
        self.max_difficulty = max_difficulty
        self._lock = threading.Lock()
        
        try:
            from wasmtime import Engine, Store, Module, Instance
            self.engine = Engine()
            self.store = Store(self.engine)
            self.module = Module.from_file(self.engine, self.wasm_path)
            self.instance = Instance(self.store, self.module, [])
            self.exports = self.instance.exports(self.store)

            self.wasm_solve_func = self.exports["wasm_solve"]
            self.alloc_func = self.exports["__wbindgen_export_0"]
            self.dealloc_func = self.exports["__wbindgen_export_2"]
            self.add_stack_ptr_func = self.exports["__wbindgen_add_to_stack_pointer"]
            self.memory = self.exports["memory"]
        except ImportError:
            raise PoWSolverError("wasmtime library is missing. Install it via: pip install wasmtime")
        except Exception as e:
            raise PoWSolverError(f"Failed to initialize WASM module: {e}")

    def _write_string(self, s: str) -> tuple[int, int]:
        bytes_data = s.encode("utf-8")
        ptr = self.alloc_func(self.store, len(bytes_data), 1)
        if not ptr:
            raise PoWSolverError("WASM memory allocation failed.")
        mem_slice = self.memory.data_ptr(self.store)
        address = ctypes.cast(mem_slice, ctypes.c_void_p).value
        ctypes.memmove(address + ptr, bytes_data, len(bytes_data))
        return ptr, len(bytes_data)

    def _free_string(self, ptr: int, length: int) -> None:
        if ptr:
            self.dealloc_func(self.store, ptr, length, 1)

    def solve(self, challenge: str, salt: str, expire_at: int, difficulty: int) -> int:
        """
        Solves the PoW challenge using the WebAssembly module.
        """
        challenge = ensure_string(challenge, "challenge", max_length=4096)
        salt = ensure_string(salt, "salt", max_length=1024)
        expire_at = ensure_positive_int(expire_at, "expire_at")
        difficulty = ensure_positive_int(difficulty, "difficulty")
        if self.max_difficulty is not None and difficulty > self.max_difficulty:
            raise PoWSolverError(f"PoW difficulty {difficulty} exceeds max_difficulty {self.max_difficulty}.")

        with self._lock:
            prefix = f"{salt}_{expire_at}_"
            target_ptr = target_len = prefix_ptr = prefix_len = ret_ptr = 0

            try:
                target_ptr, target_len = self._write_string(challenge)
                prefix_ptr, prefix_len = self._write_string(prefix)
                ret_ptr = self.add_stack_ptr_func(self.store, -16)

                self.wasm_solve_func(
                    self.store, ret_ptr, target_ptr, target_len, prefix_ptr, prefix_len, float(difficulty)
                )

                mem_slice = self.memory.data_ptr(self.store)
                address = ctypes.cast(mem_slice, ctypes.c_void_p).value
                is_present = ctypes.c_int32.from_address(address + ret_ptr).value
                nonce = ctypes.c_double.from_address(address + ret_ptr + 8).value

                if is_present == 0:
                    raise PoWSolverError(f"WASM failed to solve PoW within {difficulty} iterations.")

                return int(nonce)
            except PoWSolverError:
                raise
            except Exception as e:
                raise PoWSolverError(f"Execution error inside WASM module: {e}") from e
            finally:
                try:
                    if ret_ptr:
                        self.add_stack_ptr_func(self.store, 16)
                    self._free_string(target_ptr, target_len)
                    self._free_string(prefix_ptr, prefix_len)
                except Exception as cleanup_error:
                    logger.debug("Failed to clean up WASM PoW memory", exc_info=cleanup_error)
