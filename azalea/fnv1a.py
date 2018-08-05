
from numba import njit, uint32

# https://softwareengineering.stackexchange.com/a/145633

@njit('uint32(int32[:])')
def fnv1a(seq):
    """32-bit FNV-1a hash for 32-bit sequences
    :param seq: signed 32-bit sequence
    :returns: unsigned 32-bit checksum
    """
    fnv_32_prime = uint32(0x01000193)
    h = uint32(0x811c9dc5)
    for s in seq:
        u = uint32(s)
        h = (h ^ (u & 0xff)) * fnv_32_prime
        h = (h ^ ((u >> 8) & 0xff)) * fnv_32_prime
        h = (h ^ ((u >> 16) & 0xff)) * fnv_32_prime
        h = (h ^ ((u >> 24) & 0xff)) * fnv_32_prime
    return h
