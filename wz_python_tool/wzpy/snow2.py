"""SNOW 2.0 stream cipher — Python port of MapleLib's ``Snow2CryptoTransform``.

The legacy V1 ``.ms`` archive format encrypts its header, entry table and image
payloads with SNOW 2.0 used as an **additive** stream cipher over 32-bit
little-endian words::

    cipher_word = (plain_word + keystream_word) & 0xFFFFFFFF    # encrypt
    plain_word  = (cipher_word - keystream_word) & 0xFFFFFFFF   # decrypt

The keystream is produced 16 words at a time by the SNOW 2.0 LFSR + FSM. This
module exposes :class:`Snow2`, a fresh keystream generator for a given key, and
:func:`snow_decrypt`, which subtracts the keystream from a buffer word-by-word.

Fidelity notes (these match MapleLib exactly and are load-bearing):

* ``LoadKey`` reinterprets the key bytes as **signed** bytes before packing them
  big-endian into the initial LFSR state — the sign extension is part of the
  on-disk key schedule, so it must be reproduced verbatim.
* All ``.ms`` keys are 16 bytes and the IV is always empty, but the 32-byte key
  path is kept for completeness.

Credits: Elem8100 (WzComparerR2 / MapleNecrocer), MapleLib.
Tables in :mod:`wzpy._snow2_tables` are auto-extracted from the C# source.
"""

from __future__ import annotations

from ._snow2_tables import (
    snow_T0,
    snow_T1,
    snow_T2,
    snow_T3,
    snow_alpha_mul,
    snow_alphainv_mul,
)

_MASK = 0xFFFFFFFF


def _a_mul(w: int) -> int:
    return (((w << 8) & _MASK) ^ snow_alpha_mul[w >> 24]) & _MASK


def _ainv_mul(w: int) -> int:
    return ((w >> 8) ^ snow_alphainv_mul[w & 0xFF]) & _MASK


def _fsm_r2(r1: int) -> int:
    return (
        snow_T0[r1 & 0xFF]
        ^ snow_T1[(r1 >> 8) & 0xFF]
        ^ snow_T2[(r1 >> 16) & 0xFF]
        ^ snow_T3[(r1 >> 24) & 0xFF]
    ) & _MASK


class Snow2:
    """A SNOW 2.0 keystream generator.

    Construct with a 16- or 32-byte ``key`` (and optional 4-byte ``iv``); call
    :meth:`next_word` to pull the next 32-bit keystream word. State mirrors the
    C# ``Snow2CryptoTransform`` after construction (key loaded, 32 initial
    clockings done, first 16 keystream words generated, ``curIndex == 0``).
    """

    __slots__ = ("_s", "_r1", "_r2", "_ks", "_idx")

    def __init__(self, key: bytes, iv: bytes = b""):
        if len(key) not in (16, 32):
            raise ValueError("SNOW2 key must be 16 or 32 bytes")
        if iv and len(iv) != 4:
            raise ValueError("SNOW2 IV must be empty or 4 bytes")
        # s[0]..s[15] LFSR state.
        self._s = [0] * 16
        self._r1 = 0
        self._r2 = 0
        self._ks = [0] * 16
        self._load_key(key, iv)
        self._refresh()
        self._idx = 0

    # ── key schedule ────────────────────────────────────────────────────
    @staticmethod
    def _word_be_signed(b0: int, b1: int, b2: int, b3: int) -> int:
        """Pack four key bytes big-endian, sign-extending each (matches the
        ``MemoryMarshal.Cast<byte, sbyte>`` in C# ``LoadKey``)."""
        def sx(b: int) -> int:
            return b - 256 if b >= 128 else b
        return (
            (sx(b0) << 24) | (sx(b1) << 16) | (sx(b2) << 8) | sx(b3)
        ) & _MASK

    def _load_key(self, key: bytes, iv: bytes) -> None:
        s = self._s
        w = self._word_be_signed
        if len(key) == 16:
            s[15] = w(key[0], key[1], key[2], key[3])
            s[14] = w(key[4], key[5], key[6], key[7])
            s[13] = w(key[8], key[9], key[10], key[11])
            s[12] = w(key[12], key[13], key[14], key[15])
            s[11] = ~s[15] & _MASK
            s[10] = ~s[14] & _MASK
            s[9] = ~s[13] & _MASK
            s[8] = ~s[12] & _MASK
            s[7] = s[15]
            s[6] = s[14]
            s[5] = s[13]
            s[4] = s[12]
            s[3] = ~s[15] & _MASK
            s[2] = ~s[14] & _MASK
            s[1] = ~s[13] & _MASK
            s[0] = ~s[12] & _MASK
        else:  # 256-bit key
            for i in range(8):
                s[15 - i] = w(key[i * 4], key[i * 4 + 1],
                              key[i * 4 + 2], key[i * 4 + 3])
            for i in range(8):
                s[7 - i] = ~s[15 - i] & _MASK

        if iv:
            s[15] ^= iv[0]
            s[12] ^= iv[1]
            s[10] ^= iv[2]
            s[9] ^= iv[3]

        self._r1 = 0
        self._r2 = 0
        # 32 initial clockings (two passes of 16 LFSR updates).
        for _ in range(2):
            self._clock_init()

    def _clock_init(self) -> None:
        """One pass of 16 LFSR updates with the FSM output fed back into the
        register (the SNOW 2.0 initialization mode)."""
        s = self._s
        r1, r2 = self._r1, self._r2
        for i in range(16):
            out = ((r1 + s[(i + 15) % 16]) & _MASK) ^ r2
            s[i] = (
                _a_mul(s[i]) ^ s[(i + 2) % 16]
                ^ _ainv_mul(s[(i + 11) % 16]) ^ out
            ) & _MASK
            fsmtmp = (r2 + s[(i + 5) % 16]) & _MASK
            r2 = _fsm_r2(r1)
            r1 = fsmtmp
        self._r1, self._r2 = r1, r2

    # ── keystream ───────────────────────────────────────────────────────
    def _refresh(self) -> None:
        """Generate the next 16 keystream words (SNOW 2.0 running mode)."""
        s = self._s
        r1, r2 = self._r1, self._r2
        ks = self._ks
        for i in range(16):
            s[i] = (
                _a_mul(s[i]) ^ s[(i + 2) % 16] ^ _ainv_mul(s[(i + 11) % 16])
            ) & _MASK
            fsmtmp = (r2 + s[(i + 5) % 16]) & _MASK
            r2 = _fsm_r2(r1)
            r1 = fsmtmp
            ks[i] = (((r1 + s[i]) & _MASK) ^ r2 ^ s[(i + 1) % 16]) & _MASK
        self._r1, self._r2 = r1, r2

    def next_word(self) -> int:
        w = self._ks[self._idx]
        self._idx += 1
        if self._idx >= 16:
            self._refresh()
            self._idx = 0
        return w


def snow_decrypt(cipher: bytes, key: bytes, *, double_prefix: int = 0) -> bytes:
    """Decrypt ``cipher`` with a fresh SNOW 2.0 additive keystream from ``key``.

    Each 32-bit little-endian word is decrypted as ``plain = cipher - ks``.
    ``cipher`` is treated word-by-word; a trailing partial word (length not a
    multiple of 4) is decrypted from the bytes present (zero-padded) and the
    low bytes are kept — borrow within a 32-bit subtraction only propagates
    low→high, so the visible low bytes are exact regardless of the pad.

    ``double_prefix`` (bytes): the first ``ceil(min(double_prefix, len)/4)``
    words are decrypted **twice** (``plain = cipher - 2*ks``). ``.ms`` image
    payloads double-encrypt their first 1024 bytes; header and entry-table
    regions use ``double_prefix=0``.
    """
    n = len(cipher)
    n_words = (n + 3) // 4
    if double_prefix <= 0:
        double_words = 0
    else:
        double_words = (min(double_prefix, n) + 3) // 4

    snow = Snow2(key)
    out = bytearray(n_words * 4)
    mv = memoryview(cipher)
    for wi in range(n_words):
        off = wi * 4
        chunk = bytes(mv[off:off + 4])
        if len(chunk) < 4:
            chunk = chunk + b"\x00" * (4 - len(chunk))
        c = int.from_bytes(chunk, "little")
        k = snow.next_word()
        mult = 2 if wi < double_words else 1
        p = (c - mult * k) & _MASK
        out[off:off + 4] = p.to_bytes(4, "little")
    return bytes(out[:n])


class Snow2Decryptor:
    """Sequential SNOW 2.0 additive decryptor for streamed reads.

    Mirrors reading through a .NET ``CryptoStream``: bytes are decrypted
    word-by-word in order and handed out on demand, so a partial-word read
    leaves the unused decrypted bytes buffered for the next call. Used for the
    variable-length ``.ms`` header and entry table, where field sizes (1/4/16
    bytes, UTF-16 names) don't align to word boundaries.
    """

    __slots__ = ("_raw", "_snow", "_word_index", "_buf")

    def __init__(self, raw: bytes, key: bytes):
        self._raw = raw
        self._snow = Snow2(key)
        self._word_index = 0
        self._buf = bytearray()

    def read(self, n: int) -> bytes:
        buf = self._buf
        while len(buf) < n:
            off = self._word_index * 4
            chunk = self._raw[off:off + 4]
            if len(chunk) < 4:
                chunk = bytes(chunk) + b"\x00" * (4 - len(chunk))
            c = int.from_bytes(chunk, "little")
            k = self._snow.next_word()
            p = (c - k) & _MASK
            buf += p.to_bytes(4, "little")
            self._word_index += 1
        out = bytes(buf[:n])
        del buf[:n]
        return out

    def read_i32(self) -> int:
        return int.from_bytes(self.read(4), "little", signed=True)

    def read_u32(self) -> int:
        return int.from_bytes(self.read(4), "little")

    def read_byte(self) -> int:
        return self.read(1)[0]
