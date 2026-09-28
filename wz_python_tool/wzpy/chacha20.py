"""ChaCha20 stream cipher (IETF / RFC 7539 variant) used by the V2 ``.ms`` packs.

A faithful, dependency-free port of the cipher WzComparerR2 uses in
``Cryptography/ChaCha20CryptoTransform.cs`` (itself a port of
CSharp-ChaCha20-NetStandard). 256-bit key, 96-bit nonce, 32-bit block counter,
20 rounds. Little-endian throughout.

Two consumers, matching the two ways WzComparerR2 drives the cipher:

* :func:`chacha20_xor` — a plain streaming XOR where the block counter advances
  normally (used for the 8-byte file header and for an image's first 1024 bytes).
* :class:`ChaCha20BlockReader` — the quirky reader WzComparerR2 uses for the
  **entry table**: it decrypts the stream in 64-byte blocks but *resets the
  block counter to 0* every time a 64-byte block is fully consumed. Replicated
  byte-for-byte so the entry TOC decodes identically.
"""

from __future__ import annotations

import struct
from typing import List

_SIGMA = b"expand 32-byte k"
_MASK = 0xFFFFFFFF


def _rotl(v: int, c: int) -> int:
    v &= _MASK
    return ((v << c) | (v >> (32 - c))) & _MASK


def _quarter_round(x: List[int], a: int, b: int, c: int, d: int) -> None:
    x[a] = (x[a] + x[b]) & _MASK; x[d] = _rotl(x[d] ^ x[a], 16)
    x[c] = (x[c] + x[d]) & _MASK; x[b] = _rotl(x[b] ^ x[c], 12)
    x[a] = (x[a] + x[b]) & _MASK; x[d] = _rotl(x[d] ^ x[a], 8)
    x[c] = (x[c] + x[d]) & _MASK; x[b] = _rotl(x[b] ^ x[c], 7)


class ChaCha20:
    """ChaCha20 keystream generator.

    ``key`` is 32 bytes, ``nonce`` 12 bytes, ``counter`` the initial 32-bit
    block counter. :meth:`block` emits the next 64-byte keystream block and
    advances the counter; :meth:`reset_counter` forces it back to 0.
    """

    __slots__ = ("state",)

    def __init__(self, key: bytes, nonce: bytes, counter: int = 0):
        if len(key) != 32:
            raise ValueError(f"ChaCha20 key must be 32 bytes, got {len(key)}")
        if len(nonce) != 12:
            raise ValueError(f"ChaCha20 nonce must be 12 bytes, got {len(nonce)}")
        c = struct.unpack("<4I", _SIGMA)
        k = struct.unpack("<8I", key)
        n = struct.unpack("<3I", nonce)
        self.state: List[int] = [
            c[0], c[1], c[2], c[3],
            k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7],
            counter & _MASK, n[0], n[1], n[2],
        ]

    def block(self) -> bytes:
        s = self.state
        x = list(s)
        for _ in range(10):
            _quarter_round(x, 0, 4, 8, 12)
            _quarter_round(x, 1, 5, 9, 13)
            _quarter_round(x, 2, 6, 10, 14)
            _quarter_round(x, 3, 7, 11, 15)
            _quarter_round(x, 0, 5, 10, 15)
            _quarter_round(x, 1, 6, 11, 12)
            _quarter_round(x, 2, 7, 8, 13)
            _quarter_round(x, 3, 4, 9, 14)
        out = bytearray(64)
        for i in range(16):
            struct.pack_into("<I", out, 4 * i, (x[i] + s[i]) & _MASK)
        s[12] = (s[12] + 1) & _MASK
        if s[12] == 0:
            s[13] = (s[13] + 1) & _MASK
        return bytes(out)

    def reset_counter(self) -> None:
        self.state[12] = 0


def chacha20_xor(key: bytes, nonce: bytes, counter: int, data: bytes) -> bytes:
    """XOR ``data`` with a fresh ChaCha20 keystream (counter advances normally)."""
    cipher = ChaCha20(key, nonce, counter)
    out = bytearray(len(data))
    for off in range(0, len(data), 64):
        ks = cipher.block()
        chunk = data[off:off + 64]
        for i in range(len(chunk)):
            out[off + i] = chunk[i] ^ ks[i]
    return bytes(out)


class ChaCha20BlockReader:
    """Sequential reader over a ChaCha20 stream that resets the block counter
    after every fully-consumed 64-byte block.

    Mirrors ``Ms_FileV2.ChaCha20Reader``: reads the ciphertext in 64-byte
    chunks, decrypts each with the current counter, and — crucially — calls
    ``reset_counter`` whenever a read lands exactly on a 64-byte boundary. The
    entry table is serialized against this exact behaviour, so it must be
    reproduced rather than "cleaned up".
    """

    def __init__(self, data: bytes, key: bytes, nonce: bytes):
        self._data = data
        self._pos = 0                       # ciphertext bytes consumed
        self._cipher = ChaCha20(key, nonce, 0)
        self._buf = bytearray(64)
        self._readed = 64                   # forces a refill on first read

    @property
    def pos(self) -> int:
        """Ciphertext offset consumed so far (advances in 64-byte steps)."""
        return self._pos

    def _read(self, count: int) -> bytes:
        out = bytearray(count)
        w = 0
        while w < count:
            if self._readed >= 64:
                block = self._data[self._pos:self._pos + 64]
                self._pos += 64
                ks = self._cipher.block()
                buf = bytearray(64)
                for i in range(len(block)):
                    buf[i] = block[i] ^ ks[i]
                self._buf = buf
                self._readed = 0
            rc = min(count - w, 64 - self._readed)
            out[w:w + rc] = self._buf[self._readed:self._readed + rc]
            w += rc
            self._readed += rc
        if self._readed >= 64:
            self._cipher.reset_counter()
        return bytes(out)

    def read_bytes(self, n: int) -> bytes:
        return self._read(n)

    def read_i32(self) -> int:
        return struct.unpack("<i", self._read(4))[0]

    def read_utf16_string(self) -> str:
        """Length-prefixed (int32 char count) UTF-16LE string."""
        n = self.read_i32()
        if n < 0 or n > 1 << 20:
            raise ValueError(f"implausible entry-name length {n}")
        return self._read(n * 2).decode("utf-16-le")
